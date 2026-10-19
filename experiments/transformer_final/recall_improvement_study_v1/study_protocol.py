"""Freeze family-disjoint inner folds and intervention specifications."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from models.transformer_final.common import (
    read_json,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
    write_json,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.experiment2 import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PROTOCOL,
    load_annotation_ready_protocol,
)
from models.transformer_final.leakage import audit_leakage, build_fold_manifests

STUDY_ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = STUDY_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
SPLITS_PATH = STUDY_ROOT / "protocols" / "nested_inner_splits_v1.json"
EXPECTED_BASE_PROTOCOL = (
    "242f98c56d88a11dea3b355c4c3b14c693029e3d8a607fdd7a067f29af630b36"
)


def _driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    versions = {value.strip() for value in result.stdout.splitlines() if value.strip()}
    if len(versions) != 1:
        raise RuntimeError(f"Expected one NVIDIA driver version, found {versions}")
    return versions.pop()


def _study_source_hashes() -> dict[str, str]:
    files = {}
    for path in sorted(STUDY_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files[path.relative_to(STUDY_ROOT.parents[2]).as_posix()] = sha256_file(path)
    return files


def make_inner_splits(
    frame: Any, outer_fold: int, *, splits: int = 3
) -> list[dict[str, Any]]:
    outer_fit = frame.loc[frame["fold"].astype(int) != outer_fold].copy()
    groups = outer_fit["family_id"].astype(str).to_numpy()
    unique_families = sorted(set(groups))
    if len(unique_families) < splits:
        raise RuntimeError(
            f"Outer fold {outer_fold} has only {len(unique_families)} fitting families"
        )
    splitter = GroupKFold(n_splits=splits)
    inner_frame = outer_fit.reset_index(drop=True)
    inner_frame["fold"] = 0
    family_to_inner: dict[str, int] = {}
    for inner_fold, (_, validation_indices) in enumerate(
        splitter.split(np.zeros(len(inner_frame)), groups=groups), start=1
    ):
        for family in (
            inner_frame.iloc[validation_indices]["family_id"].astype(str).unique()
        ):
            if family in family_to_inner:
                raise RuntimeError(f"Family assigned to multiple inner folds: {family}")
            family_to_inner[family] = inner_fold
    if set(family_to_inner) != set(unique_families):
        raise RuntimeError("Inner group folds do not cover all outer-fitting families")
    inner_frame["fold"] = inner_frame["family_id"].astype(str).map(family_to_inner)
    if inner_frame["fold"].isna().any():
        raise RuntimeError("An outer-fitting row has no inner fold assignment")

    audit = audit_leakage(inner_frame.reset_index(drop=True), near_threshold=0.85)
    if audit["training_blocked"]:
        raise RuntimeError(f"Inner-fold leakage audit blocked outer fold {outer_fold}")
    manifests = build_fold_manifests(inner_frame, audit)
    result: list[dict[str, Any]] = []
    scoring_ids: list[str] = []
    for manifest in manifests:
        fold = int(manifest["fold"])
        fit_ids = set(manifest["fitting_record_ids"])
        score_ids = set(manifest["scoring_record_ids"])
        fitting = inner_frame.loc[inner_frame["record_id"].astype(str).isin(fit_ids)]
        scoring = inner_frame.loc[inner_frame["record_id"].astype(str).isin(score_ids)]
        if set(fitting["family_id"]) & set(scoring["family_id"]):
            raise RuntimeError("Inner fitting and scoring families overlap")
        if not score_ids or not fit_ids:
            raise RuntimeError("An inner fold has an empty fit or scoring partition")
        scoring_ids.extend(sorted(score_ids))
        result.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": fold,
                "fitting_record_ids": sorted(fit_ids),
                "scoring_record_ids": sorted(score_ids),
                "fitting_families": manifest["fitting_families"],
                "held_out_families": manifest["held_out_families"],
                "fitting_exclusions": manifest["fitting_exclusions"],
                "context_sanitizations": manifest["context_sanitizations"],
                "counts": manifest["counts"],
                "fitting_positive_labels": int(fitting["label"].astype(int).sum()),
                "scoring_positive_labels": int(scoring["label"].astype(int).sum()),
                "scoring_negative_labels": int(
                    (scoring["label"].astype(int) == 0).sum()
                ),
                "leakage_audit_counts": audit["counts"],
            }
        )
    if len(scoring_ids) != len(set(scoring_ids)) or set(scoring_ids) != set(
        inner_frame["record_id"].astype(str)
    ):
        raise RuntimeError("Inner OOF splits do not score each fitting record once")
    return result


def freeze_study_protocol() -> Path:
    if PROTOCOL_PATH.exists() or SPLITS_PATH.exists():
        raise FileExistsError("Study protocol artifacts are immutable")
    config = load_config(DEFAULT_EXPERIMENT_CONFIG)
    baseline, baseline_hash = load_annotation_ready_protocol(DEFAULT_PROTOCOL, config)
    if baseline_hash != EXPECTED_BASE_PROTOCOL:
        raise RuntimeError("Frozen Experiment 2 baseline protocol hash changed")
    frame, dataset_manifest, dataset_dir = load_effective_dataset(config)

    inner_splits: list[dict[str, Any]] = []
    for outer_fold in range(1, 6):
        inner_splits.extend(make_inner_splits(frame, outer_fold, splits=3))
    if len(inner_splits) != 15:
        raise RuntimeError("Expected exactly 15 nested calibration training cells")
    splits_json = json.dumps(inner_splits, indent=2, sort_keys=True) + "\n"

    from transformers import AutoTokenizer

    from models.transformer_final.tokenization import audit_lengths
    from models.transformer_final.train import initialize_pretrained_components

    model_path = Path(baseline["model"]["local_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=baseline["model"]["revision"],
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
        fix_mistral_regex=False,
    )
    length_audit = audit_lengths(
        tokenizer, frame, max_length=256, input_mode="local_context"
    )
    delimiters = {
        text: tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
        for text in ("[TARGET]", "[SECTION]", "[PREVIOUS]", "[NEXT]")
    }
    if length_audit["target_truncated_rows"] != 0:
        raise RuntimeError("Context length policy would truncate at least one target")

    _, model = initialize_pretrained_components(
        model_path, revision=baseline["model"]["revision"]
    )
    layers = model.deberta.encoder.layer
    if len(layers) != 12:
        raise RuntimeError(f"Expected 12 DeBERTa encoder layers, found {len(layers)}")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_counts: dict[str, Any] = {}
    named_parameters = list(model.named_parameters())
    for last_layers in (12, 4, 2):
        trainable = 0
        frozen = 0
        for name, parameter in named_parameters:
            layer_index = None
            prefix = "deberta.encoder.layer."
            if name.startswith(prefix):
                layer_index = int(name[len(prefix) :].split(".", 1)[0])
            is_trainable = (
                name.startswith(("classifier.", "pooler."))
                or (layer_index is not None and layer_index >= 12 - last_layers)
                or (
                    last_layers == 12
                    and (name.startswith(("deberta.embeddings.", "deberta.encoder.")))
                )
            )
            if is_trainable:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        if trainable + frozen != total_parameters:
            raise RuntimeError("Trainable/frozen parameter count accounting failed")
        trainable_counts[str(last_layers)] = {
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
            "total_parameters": total_parameters,
            "trainable_percent": 100 * trainable / total_parameters,
        }
    del model

    import torch

    core_hashes = source_file_hashes()
    study_hashes = _study_source_hashes()
    source_snapshot = {
        "core_files": core_hashes,
        "study_files": study_hashes,
    }
    environment = {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": importlib.metadata.version("torch"),
        "transformers": importlib.metadata.version("transformers"),
        "numpy": importlib.metadata.version("numpy"),
        "pandas": importlib.metadata.version("pandas"),
        "scikit-learn": importlib.metadata.version("scikit-learn"),
        "tokenizers": importlib.metadata.version("tokenizers"),
        "sentencepiece": importlib.metadata.version("sentencepiece"),
        "safetensors": importlib.metadata.version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "driver": _driver_version(),
    }
    protocol = {
        "schema_version": 1,
        "study_id": "recall_improvement_study_v1",
        "baseline_protocol_path": str(DEFAULT_PROTOCOL.resolve()),
        "baseline_protocol_sha256": baseline_hash,
        "config_path": str(Path(config["_config_path"]).resolve()),
        "config_sha256": config["_config_sha256"],
        "dataset": {
            "source_sha256": baseline["dataset"]["source_sha256"],
            "effective_dataset_sha256": dataset_manifest["effective_dataset_sha256"],
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
            "fold_manifest_sha256": {
                str(fold): sha256_file(dataset_dir / "folds" / f"fold_{fold}.json")
                for fold in range(1, 6)
            },
            "rows": len(frame),
            "labels": {
                "positive": int(frame["label"].sum()),
                "negative": int((frame["label"] == 0).sum()),
            },
            "family_fold_source_sha256": baseline["dataset"]["fold_source_sha256"],
            "reserved_families": baseline["dataset"]["reserved_families"],
        },
        "model": {
            "local_path": str(model_path.resolve()),
            "source_sha256": baseline["model"]["source_sha256"],
            "model_file_hashes": baseline["model"]["model_file_hashes"],
            "tokenizer_file_hashes": baseline["model"]["tokenizer_file_hashes"],
            "architecture_class": "DebertaV2ForSequenceClassification",
            "encoder_layers": len(layers),
            "full_parameter_count": total_parameters,
            "partial_trainable_parameter_counts": trainable_counts,
            "precision": "fp32",
        },
        "training": baseline["training"],
        "inner_cv": {
            "algorithm": "GroupKFold",
            "splits_per_outer_fold": 3,
            "groups": "canonical family_id",
            "assignment_uses_labels": False,
            "leakage_policy": "existing label-blind exact/near/lineage/BCP14 exclusions and context sanitization",
            "manifests_path": str(SPLITS_PATH.resolve()),
            "manifests_sha256": sha256_text(splits_json),
            "cell_count": len(inner_splits),
            "threshold_candidates": [
                0.001,
                0.002,
                0.003,
                0.005,
                0.01,
                0.02,
                0.03,
                0.05,
                0.075,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "primary": "maximize inner OOF recall subject to precision >= 0.20",
            "tie_breaking": "higher precision, then higher threshold",
            "fallback": "if no feasible threshold, maximize F2, then precision, then threshold",
            "calibration": "none; threshold-only policy leaves probabilities unchanged",
        },
        "interventions": {
            "context": {
                "input_mode": "local_context",
                "max_length": 256,
                "token_length_audit": length_audit,
                "delimiter_token_ids": delimiters,
                "vocabulary_expansion": False,
                "target_truncation_rows": 0,
                "context_truncation_rows": length_audit["context_truncated_rows"],
                "policy": "reuse existing target-prioritized deterministic allocation; no vocabulary changes",
            },
            "partial_finetuning": {
                "partial_4": "freeze embeddings, shared encoder-level non-layer backbone parameters, and layers 0-7; train layers 8-11, pooler, and classifier",
                "partial_2": "freeze embeddings, shared encoder-level non-layer backbone parameters, and layers 0-9; train layers 10-11, pooler, and classifier",
                "optimizer": "AdamW receives only parameters with requires_grad=True",
                "dropout": "all modules retain the existing model.train() behavior; frozen weights do not disable dropout",
            },
            "focal": {
                "gamma": 2.0,
                "alpha": "existing fitting-side class weights (No=1; Yes=negative/positive)",
                "formula": "-alpha_y * (1 - p_y)^gamma * log(p_y), batch mean",
            },
        },
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": sha256_text(stable_json(source_snapshot)),
        "environment": environment,
        "budget": read_json(STUDY_ROOT / "time_budget.json"),
    }
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_bytes(splits_json.encode("utf-8"))
    PROTOCOL_PATH.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    inventory_path = STUDY_ROOT / "EXPERIMENT_INVENTORY.json"
    inventory = read_json(inventory_path)
    inventory["protocol_sha256"] = sha256_file(PROTOCOL_PATH)
    write_json(inventory_path, inventory)
    return PROTOCOL_PATH


if __name__ == "__main__":
    print(f"Frozen study protocol: {freeze_study_protocol()}")
