"""Pinned local sentence embeddings, deterministic windows and content caches."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .io import digest, file_hash, write_json


def token_windows(ids: list[int], limit: int, overlap: int):
    """Slice original token IDs, with guaranteed progress and complete coverage."""
    if limit <= 0 or not 0 <= overlap < limit:
        raise ValueError("Require positive content limit and 0 <= overlap < limit")
    start = 0
    while start < len(ids):
        end = min(len(ids), start + limit)
        yield ids[start:end], start, end
        if end == len(ids):
            break
        start = end - overlap


def effective_token_limit(model) -> int:
    limits = [model.max_seq_length, getattr(model.tokenizer, "model_max_length", None)]
    if hasattr(model, "__getitem__"):
        config = getattr(getattr(model[0], "auto_model", None), "config", None)
        limits.append(getattr(config, "max_position_embeddings", None))
    valid = [x for x in limits if isinstance(x, int) and 0 < x < 1_000_000]
    if not valid:
        raise ValueError("Encoder does not declare a finite token limit")
    return min(valid)


def prepare_token_ids(tokenizer, ids: list[int]) -> dict:
    """Support legacy and current HF tokenizers without decoding source IDs."""
    prepare = getattr(tokenizer, "prepare_for_model", None)
    if callable(prepare):
        return dict(
            prepare(
                ids,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=True,
                verbose=False,
            )
        )
    # Transformers 5 removed prepare_for_model. Validate the pinned BERT template
    # before constructing its single-sequence features; reject unknown templates.
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    if (
        cls is None
        or sep is None
        or tokenizer.num_special_tokens_to_add(pair=False) != 2
        or tokenizer.encode("", add_special_tokens=True) != [cls, sep]
    ):
        raise ValueError("Unsupported special-token template for direct-ID embedding")
    features = {"input_ids": [cls, *ids, sep], "attention_mask": [1] * (len(ids) + 2)}
    if "token_type_ids" in tokenizer.model_input_names:
        features["token_type_ids"] = [0] * (len(ids) + 2)
    return features


class Embedder:
    def __init__(self, config: Config, cache: Path, *, allow_download: bool = False):
        self.config, self.cache = config, cache
        self.model = None
        self.metadata: dict = {
            "backend": config.backend,
            "long_text": "direct_token_id_windows_normalized_pool_v3",
        }
        self.long_count = 0
        self.stats: list[dict] = []
        self.last_windows: list[dict] = []
        if config.backend == "lexical":
            return
        if not allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # This Windows Conda environment requires Arrow's native libraries to be
        # initialized before torch's bundled DLLs. No installation or patching.
        if sys.platform == "win32":
            import pyarrow  # noqa: F401
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from sentence_transformers import SentenceTransformer

        revision = config.model_revision
        if revision is None:
            if not allow_download:
                raise ValueError(
                    "Semantic backend needs an immutable --model-revision or --allow-download"
                )
            revision = HfApi().model_info(config.model_name, timeout=30).sha
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError(
                "Model revision must be a 40-character immutable commit hash"
            )
        snapshot = Path(
            snapshot_download(
                config.model_name,
                revision=revision,
                local_files_only=not allow_download,
                allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*"],
                ignore_patterns=["onnx/*", "openvino/*", "*.onnx"],
            )
        )
        torch.set_num_threads(1)
        torch.manual_seed(config.seed)
        torch.use_deterministic_algorithms(True)
        self.model = SentenceTransformer(
            str(snapshot), device="cpu", local_files_only=True
        )
        self.model.eval()
        self.metadata.update(
            model_name=config.model_name,
            model_revision=revision,
            artifacts={
                p.relative_to(snapshot).as_posix(): file_hash(p)
                for p in sorted(snapshot.rglob("*"))
                if p.is_file()
            },
            packages={
                n: importlib.metadata.version(n)
                for n in ("sentence-transformers", "torch", "transformers", "numpy")
            },
            device="cpu",
            batch_size=config.batch_size,
            overlap=config.window_overlap,
            max_sequence_length=self.model.max_seq_length,
            effective_token_limit=effective_token_limit(self.model),
            special_tokens=self.model.tokenizer.num_special_tokens_to_add(pair=False),
        )
        self.cache.mkdir(parents=True, exist_ok=True)

    def encode(self, texts: list[str]) -> np.ndarray | None:
        if self.model is None:
            return None
        if any(not text.strip() for text in texts):
            raise ValueError(
                "Empty/whitespace-only embedding text; clause must be diagnosed, not skipped"
            )
        dimension = self.model.get_embedding_dimension()
        if not texts:
            self.last_windows = []
            return np.empty((0, dimension), dtype=np.float32)
        key = digest({"texts": texts, "model": self.metadata})
        path = self.cache / f"{key}.npy"
        if path.exists():
            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if manifest["artifact_sha256"] != file_hash(path) or manifest["key"] != key:
                raise ValueError(f"Embedding cache hash mismatch: {path}")
            result = np.load(path, allow_pickle=False)
            if result.shape != (len(texts), dimension) or not np.isfinite(result).all():
                raise ValueError(f"Invalid embedding cache: {path}")
            self.last_windows = manifest["windows"]
            self.stats.append({**manifest["statistics"], "cache_hit": True})
            self.long_count += manifest["statistics"]["long_clauses"]
            return result
        tokenizer = self.model.tokenizer
        effective_limit = effective_token_limit(self.model)
        limit = effective_limit - tokenizer.num_special_tokens_to_add(pair=False)
        step = limit - self.config.window_overlap
        if step <= 0:
            raise ValueError("Embedding overlap must be smaller than token window")
        prepared, owners, boundaries = [], [], []
        long_clauses, total_tokens, max_tokens = 0, 0, 0
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if not ids:
                raise ValueError(f"Tokenizer emitted no content for text index {i}")
            total_tokens += len(ids)
            max_tokens = max(max_tokens, len(ids))
            if len(ids) > limit:
                long_clauses += 1
            for window, start, end in token_windows(
                ids, limit, self.config.window_overlap
            ):
                features = prepare_token_ids(tokenizer, window)
                encoded_length = len(features["input_ids"])
                if encoded_length > effective_limit:
                    raise ValueError(
                        f"Prepared token window {i}:{start}:{end} exceeds effective limit {effective_limit}"
                    )
                prepared.append(dict(features))
                owners.append(i)
                boundaries.append(
                    {
                        "text_index": i,
                        "token_start": start,
                        "token_end": end,
                        "content_token_sha256": digest(window),
                        "encoded_tokens": encoded_length,
                    }
                )
        vectors = self._encode_prepared(prepared, effective_limit)
        result = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        for owner, vector in zip(owners, vectors):
            result[owner] += vector
        result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite encoder output")
        statistics = {
            "clauses": len(texts),
            "long_clauses": long_clauses,
            "source_tokens": total_tokens,
            "max_clause_tokens": max_tokens,
            "windows": len(boundaries),
            "effective_token_limit": effective_limit,
            "content_token_limit": limit,
            "max_encoded_tokens": max(w["encoded_tokens"] for w in boundaries),
        }
        self.last_windows = boundaries
        self.long_count += long_clauses
        self.stats.append({**statistics, "cache_hit": False})
        with path.open("xb") as handle:
            np.save(handle, result, allow_pickle=False)
        write_json(
            path.with_suffix(".json"),
            {
                "key": key,
                "artifact_sha256": file_hash(path),
                "shape": list(result.shape),
                "windows": boundaries,
                "statistics": statistics,
            },
        )
        return result

    def _encode_prepared(
        self, prepared: list[dict], effective_limit: int
    ) -> np.ndarray:
        """Forward already-tokenized features through the full ST module stack."""
        import torch

        vectors = []
        with torch.inference_mode():
            for start in range(0, len(prepared), self.config.batch_size):
                batch = self.model.tokenizer.pad(
                    prepared[start : start + self.config.batch_size],
                    padding=True,
                    return_tensors="pt",
                )
                if batch["input_ids"].shape[1] > effective_limit:
                    raise ValueError(
                        "Padded embedding batch exceeds effective token limit"
                    )
                result = self.model(dict(batch))["sentence_embedding"]
                vectors.append(
                    torch.nn.functional.normalize(result, p=2, dim=1).cpu().numpy()
                )
        return np.concatenate(vectors, axis=0)
