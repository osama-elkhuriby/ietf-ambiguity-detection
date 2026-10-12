"""Target-preserving tokenization for transformer ambiguity experiments."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT, require_training_ready

DEFAULT_DATASET_DIR = (
    ROOT / "corpus/processed/transformer_partitioned/v1/development"
)
DEFAULT_MODEL = "microsoft/deberta-v3-small"
DEFAULT_REVISION = "a36c739020e01763fe789b4b85e2df55d6180012"


class TokenizerLike(Protocol):
    sep_token_id: int | None
    cls_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def num_special_tokens_to_add(self, *, pair: bool) -> int: ...

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]: ...

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]: ...


@dataclass(frozen=True)
class EncodedExample:
    input_ids: list[int]
    attention_mask: list[int]
    token_type_ids: list[int]
    target_token_count: int
    previous_token_count: int
    next_token_count: int
    target_tokens_kept: int
    previous_tokens_kept: int
    next_tokens_kept: int

    @property
    def target_truncated(self) -> bool:
        return self.target_tokens_kept < self.target_token_count

    @property
    def context_truncated(self) -> bool:
        return (
            self.previous_tokens_kept < self.previous_token_count
            or self.next_tokens_kept < self.next_token_count
        )


def _head_tail(tokens: list[int], limit: int) -> list[int]:
    if limit < 0:
        raise ValueError("Token limit cannot be negative")
    if len(tokens) <= limit:
        return list(tokens)
    head = math.ceil(limit / 2)
    return [*tokens[:head], *tokens[-(limit - head) :]] if limit > head else tokens[:head]


def _balanced_context(
    previous: list[int], following: list[int], budget: int, separator: int | None
) -> tuple[list[int], list[int], list[int]]:
    if budget <= 0:
        return [], [], []
    internal_separator = 1 if previous and following and separator is not None else 0
    content_budget = max(0, budget - internal_separator)
    previous_quota = content_budget // 2
    next_quota = content_budget - previous_quota
    previous_kept = previous[:previous_quota]
    next_kept = following[:next_quota]
    unused = content_budget - len(previous_kept) - len(next_kept)
    if unused and len(previous_kept) < len(previous):
        take = min(unused, len(previous) - len(previous_kept))
        previous_kept = previous[: len(previous_kept) + take]
        unused -= take
    if unused and len(next_kept) < len(following):
        take = min(unused, len(following) - len(next_kept))
        next_kept = following[: len(next_kept) + take]
    context = list(previous_kept)
    if previous_kept and next_kept and separator is not None:
        context.append(separator)
    context.extend(next_kept)
    return context, previous_kept, next_kept


def _build_model_inputs(
    tokenizer: TokenizerLike,
    target_ids: list[int],
    context_ids: list[int] | None,
) -> tuple[list[int], list[int]]:
    build = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    token_types = getattr(tokenizer, "create_token_type_ids_from_sequences", None)
    if callable(build):
        input_ids = build(target_ids, context_ids)
        types = token_types(target_ids, context_ids) if callable(token_types) else []
        return input_ids, types or [0] * len(input_ids)
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise RuntimeError("Tokenizer does not expose a usable CLS/SEP input contract")
    input_ids = [tokenizer.cls_token_id, *target_ids, tokenizer.sep_token_id]
    if context_ids is not None:
        input_ids.extend([*context_ids, tokenizer.sep_token_id])
    return input_ids, [0] * len(input_ids)


def encode_example(
    tokenizer: TokenizerLike,
    *,
    target: str,
    previous: str = "",
    following: str = "",
    variant: str = "target_only",
    max_length: int = 256,
) -> EncodedExample:
    """Encode one example while trimming context before target text."""
    if variant not in {"target_only", "local_context"}:
        raise ValueError(f"Unsupported input variant: {variant}")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    previous_ids = tokenizer.encode(previous, add_special_tokens=False)
    next_ids = tokenizer.encode(following, add_special_tokens=False)

    use_pair = variant == "local_context"
    special_count = tokenizer.num_special_tokens_to_add(pair=use_pair)
    target_budget = max_length - special_count
    if target_budget < 1:
        raise ValueError("max_length leaves no room for target tokens")
    if use_pair and len(target_ids) > target_budget:
        # A pair consumes an additional separator. Drop all context when doing so
        # is sufficient to retain the complete target.
        use_pair = False
        special_count = tokenizer.num_special_tokens_to_add(pair=False)
        target_budget = max_length - special_count
    target_kept = _head_tail(target_ids, target_budget)
    context_ids: list[int] | None = None
    previous_kept: list[int] = []
    next_kept: list[int] = []
    if use_pair:
        context_budget = max(0, max_length - special_count - len(target_kept))
        context_ids, previous_kept, next_kept = _balanced_context(
            previous_ids, next_ids, context_budget, tokenizer.sep_token_id
        )
    input_ids, token_types = _build_model_inputs(
        tokenizer, target_kept, context_ids
    )
    if len(input_ids) > max_length:
        raise RuntimeError("Tokenizer policy exceeded max_length")
    return EncodedExample(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        token_type_ids=token_types,
        target_token_count=len(target_ids),
        previous_token_count=len(previous_ids),
        next_token_count=len(next_ids),
        target_tokens_kept=len(target_kept),
        previous_tokens_kept=len(previous_kept),
        next_tokens_kept=len(next_kept),
    )


def _percentiles(values: list[int]) -> dict[str, float]:
    import numpy as np

    return {
        name: round(float(np.percentile(values, percentile)), 3)
        for name, percentile in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99), ("max", 100))
    }


def audit_token_lengths(
    tokenizer: TokenizerLike, frame: pd.DataFrame, *, max_length: int
) -> dict[str, Any]:
    target_lengths: list[int] = []
    pair_lengths: list[int] = []
    target_truncated = 0
    context_truncated = 0
    for row in frame.itertuples(index=False):
        target_ids = tokenizer.encode(str(row.target_clause), add_special_tokens=False)
        previous_ids = tokenizer.encode(str(row.previous_clause), add_special_tokens=False)
        next_ids = tokenizer.encode(str(row.next_clause), add_special_tokens=False)
        target_lengths.append(
            len(target_ids) + tokenizer.num_special_tokens_to_add(pair=False)
        )
        pair_lengths.append(
            len(target_ids)
            + len(previous_ids)
            + len(next_ids)
            + tokenizer.num_special_tokens_to_add(pair=True)
            + (1 if previous_ids and next_ids and tokenizer.sep_token_id is not None else 0)
        )
        encoded = encode_example(
            tokenizer,
            target=str(row.target_clause),
            previous=str(row.previous_clause),
            following=str(row.next_clause),
            variant="local_context",
            max_length=max_length,
        )
        target_truncated += int(encoded.target_truncated)
        context_truncated += int(encoded.context_truncated)
    return {
        "rows": len(frame),
        "max_length": max_length,
        "target_with_special_tokens": _percentiles(target_lengths),
        "local_context_with_special_tokens": _percentiles(pair_lengths),
        "target_truncated_rows": target_truncated,
        "context_truncated_rows": context_truncated,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit transformer token lengths.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require_training_ready(manifest)
    try:
        train_file = manifest["split_files"]["train"]
    except KeyError as error:
        raise RuntimeError("Manifest does not define an isolated training split") from error
    dataset_path = args.dataset_dir / train_file["path"]
    if sha256_file(dataset_path) != train_file["sha256"]:
        raise RuntimeError("Training split hash does not match its manifest")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=not args.allow_download,
        use_fast=True,
    )
    train = pd.read_csv(dataset_path).fillna("")
    if set(train["split"]) != {"train"}:
        raise RuntimeError("The isolated training file contains non-training rows")
    report = {
        "schema_version": 1,
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "train_split_sha256": train_file["sha256"],
        "model": args.model,
        "revision": args.revision,
        "split": "train",
        **audit_token_lengths(tokenizer, train, max_length=args.max_length),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
