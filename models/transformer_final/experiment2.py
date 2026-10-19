"""Dataset and frozen-protocol support for the annotation-ready FP32 pilot."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    hash_directory,
    read_json,
    resolve_path,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
    write_json,
)
from .config import load_config
from .dataset import prepare_dataset, verify_dataset

EXPERIMENT_ID = "final_annotation_ready_weighted_ce_fp32_pilot_v1"
DEFAULT_EXPERIMENT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "transformer_final"
    / "final_annotation_ready_pilot_v1"
    / "configs"
    / "experiment_v1.json"
)
DEFAULT_PROTOCOL = DEFAULT_EXPERIMENT_CONFIG.with_name("fp32_protocol_v1.json")
THRESHOLDS = (
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,
    0.50,
)
TOP_K_VALUES = (10, 25, 50, 100, 200)


def _cuda_driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    if len(versions) != 1:
        raise RuntimeError(f"Expected one NVIDIA driver version, found {versions}")
    return versions.pop()


def audit_annotation_ready_source(config: dict[str, Any]) -> dict[str, Any]:
    source_path = resolve_path(config["data"]["source_csv"])
    if sha256_file(source_path) != config["data"]["source_sha256"]:
        raise RuntimeError("Annotation-ready dataset hash mismatch")
    frame = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    required = {
        "item_id",
        "document",
        "target_clause",
        "final_label",
        "final_source",
        "final_status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Annotation-ready schema is missing columns: {sorted(missing)}")
    if len(frame) != int(config["data"]["expected_source_rows"]):
        raise RuntimeError(
            f"Annotation-ready row count changed: expected "
            f"{config['data']['expected_source_rows']}, found {len(frame)}"
        )
    if frame["item_id"].str.strip().eq("").any() or frame["item_id"].duplicated().any():
        raise RuntimeError("Annotation-ready item_id values are blank or duplicated")
    if frame["target_clause"].str.strip().eq("").any():
        raise RuntimeError("Annotation-ready source contains blank target clauses")
    if frame["document"].str.strip().eq("").any():
        raise RuntimeError("Annotation-ready source contains blank document names")

    statuses = frame["final_status"].value_counts().to_dict()
    unresolved_mask = (
        frame["final_status"].str.casefold().ne("resolved")
        | frame["final_source"].str.casefold().eq("unresolved")
    )
    unresolved_ids = sorted(frame.loc[unresolved_mask, "item_id"].astype(str))
    expected_unresolved = sorted(
        str(value) for value in config["data"]["expected_unresolved_item_ids"]
    )
    if unresolved_ids != expected_unresolved or statuses != {
        "resolved": len(frame) - len(expected_unresolved),
        "unresolved_blank_manual_verdict": len(expected_unresolved),
    }:
        raise RuntimeError(
            f"Unresolved annotation set changed: {unresolved_ids}; statuses={statuses}"
        )
    resolved = frame.loc[~unresolved_mask]
    if set(resolved["final_label"]) != {"Yes", "No"}:
        raise RuntimeError("Resolved final_label values must be exactly Yes and No")
    if set(resolved["final_source"]) - {
        "model_agreement",
        "manual_conflict_verdict",
    }:
        raise RuntimeError("Resolved labels have an undocumented final_source")
    if frame.loc[unresolved_mask, "final_label"].tolist() != ["No"]:
        raise RuntimeError(
            "The unresolved record's placeholder label changed; inspect before training"
        )
    private_key = pd.read_csv(
        resolve_path(config["data"]["private_key_csv"]),
        usecols=["item_id", "family_id"],
        dtype=str,
        keep_default_na=False,
    )
    family_rows = frame[["item_id"]].merge(
        private_key, on="item_id", how="left", validate="one_to_one", indicator=True
    )
    if family_rows["_merge"].ne("both").any() or family_rows["family_id"].eq("").any():
        raise RuntimeError("Annotation-ready source has an unmapped family")
    family_counts = family_rows["family_id"].value_counts().sort_index()

    return {
        "schema_version": 1,
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "rows": len(frame),
        "label_authority": (
            "final_label is the only binary Yes/No outcome column in the source "
            "schema; final_source/final_status are provenance/status fields."
        ),
        "raw_final_label_values": {
            str(label): int(count)
            for label, count in frame["final_label"].value_counts().sort_index().items()
        },
        "resolved_label_counts": {
            str(label): int(count)
            for label, count in resolved["final_label"].value_counts().sort_index().items()
        },
        "final_status_counts": {
            str(status): int(count)
            for status, count in sorted(statuses.items())
        },
        "final_source_counts": {
            str(source): int(count)
            for source, count in frame["final_source"].value_counts().sort_index().items()
        },
        "unresolved_item_ids": unresolved_ids,
        "excluded_unresolved_count": len(unresolved_ids),
        "duplicate_item_ids": int(frame["item_id"].duplicated().sum()),
        "blank_target_clauses": int(frame["target_clause"].str.strip().eq("").sum()),
        "document_versions": int(frame["document"].nunique()),
        "canonical_family_counts": {
            str(family): int(count) for family, count in family_counts.items()
        },
    }


def _write_or_verify_source_audit(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        if read_json(path) != report:
            raise RuntimeError(f"Existing dataset source audit differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def prepare_annotation_ready_dataset(config_path: Path) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    source_report = audit_annotation_ready_source(config)
    dataset_dir = resolve_path(config["outputs"]["dataset_dir"])
    if dataset_dir.exists():
        manifest = verify_dataset(dataset_dir)
        if (
            manifest["original_dataset_sha256"] != config["data"]["source_sha256"]
            or manifest["config_sha256"] != config["_config_sha256"]
        ):
            raise RuntimeError("Existing Experiment 2 dataset manifest does not match")
    else:
        prepare_dataset(config_path, dataset_dir)
        manifest = verify_dataset(dataset_dir)
    validation = read_json(dataset_dir / manifest["validation_report"])
    effective = pd.read_csv(
        dataset_dir / manifest["dataset_file"],
        dtype={"item_id": str, "record_id": str},
        keep_default_na=False,
    )
    folds: list[dict[str, Any]] = []
    for fold in range(1, int(config["folds"]["count"]) + 1):
        fold_manifest = read_json(dataset_dir / "folds" / f"fold_{fold}.json")
        folds.append(
            {
                "fold": fold,
                "held_out_families": fold_manifest["held_out_families"],
                **fold_manifest["counts"],
            }
        )
    source_report.update(
        {
            "effective_cohort_rows": len(effective),
            "effective_yes": int(effective["label"].astype(int).sum()),
            "effective_no": int(
                len(effective) - effective["label"].astype(int).sum()
            ),
            "effective_prevalence": float(effective["label"].astype(int).mean()),
            "effective_families": sorted(effective["family_id"].unique().tolist()),
            "reserved_partition_counts": validation["reserved_partition_counts"],
            "annotated_cohort_omissions": validation["annotated_cohort_omissions"],
            "unresolved_exclusions": validation["unresolved_exclusions"],
            "folds": folds,
            "leakage_audit_counts": read_json(
                dataset_dir / manifest["leakage_audit"]
            )["counts"],
        }
    )
    audit_path = (
        resolve_path(config["outputs"]["audit_dir"]) / "dataset_source_audit.json"
    )
    _write_or_verify_source_audit(audit_path, source_report)
    return dataset_dir, source_report


def _environment(torch: Any) -> dict[str, Any]:
    packages = (
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "tokenizers",
        "sentencepiece",
        "safetensors",
    )
    dependencies = {name: importlib.metadata.version(name) for name in packages}
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The frozen Experiment 2 pilot requires exactly one CUDA GPU")
    environment = {
        "interpreter": str(Path(sys.executable).resolve()),
        "python": platform.python_version(),
        **dependencies,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "driver": _cuda_driver_version(),
    }
    return environment


def freeze_annotation_ready_protocol(
    *,
    config_path: Path = DEFAULT_EXPERIMENT_CONFIG,
    protocol_path: Path = DEFAULT_PROTOCOL,
    model_path: Path = Path(r"D:\RFC\models\deberta-v3-base"),
) -> Path:
    config = load_config(config_path)
    dataset_dir, source_audit = prepare_annotation_ready_dataset(config_path)
    dataset_manifest = verify_dataset(dataset_dir)
    model_hash, model_files = hash_directory(model_path.resolve())
    tokenizer_files = {
        name: digest
        for name, digest in model_files.items()
        if any(
            marker in Path(name).name.casefold()
            for marker in (
                "tokenizer",
                "spm",
                "special_tokens",
                "vocab",
                "merges",
            )
        )
    }
    if not tokenizer_files:
        raise RuntimeError("Cannot identify tokenizer files in the pinned local model")
    source_hashes = source_file_hashes()
    source_files_hash = sha256_text(stable_json(source_hashes))

    conda_library_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
    os.environ["PATH"] = (
        str(conda_library_bin) + os.pathsep + os.environ.get("PATH", "")
    )
    import torch

    folds = {
        str(fold): sha256_file(dataset_dir / "folds" / f"fold_{fold}.json")
        for fold in range(1, int(config["folds"]["count"]) + 1)
    }
    payload = {
        "schema_version": 1,
        "protocol_id": EXPERIMENT_ID,
        "config_path": str(Path(config["_config_path"]).resolve()),
        "config_sha256": config["_config_sha256"],
        "dataset": {
            "source_path": str(resolve_path(config["data"]["source_csv"])),
            "source_sha256": config["data"]["source_sha256"],
            "source_audit_sha256": sha256_file(
                resolve_path(config["outputs"]["audit_dir"])
                / "dataset_source_audit.json"
            ),
            "effective_dataset_sha256": dataset_manifest[
                "effective_dataset_sha256"
            ],
            "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
            "fold_source_sha256": config["folds"]["fold_source_sha256"],
            "fold_manifest_sha256": folds,
            "rows": int(dataset_manifest["effective_summary"]["rows"]),
            "label_counts": dataset_manifest["effective_summary"]["labels"],
            "label_column": config["data"]["label_column"],
            "label_mapping": {"No": 0, "Yes": 1},
            "exclusion_policy": config["data"]["unresolved_exclusion_policy"],
            "reserved_families": [
                config["partitions"]["tcp_family"],
                config["partitions"]["dns_family"],
            ],
        },
        "model": {
            "repository": config["model"]["repository"],
            "local_path": str(model_path.resolve()),
            "revision": config["model"].get("revision"),
            "source_sha256": model_hash,
            "model_file_hashes": model_files,
            "tokenizer_file_hashes": tokenizer_files,
            "parameter_storage_dtype": "float32",
            "forward_precision": "float32_no_autocast",
            "cross_entropy_input_dtype": "float32",
            "initialization": "fresh local base weights independently per fold",
        },
        "training": {
            "condition": "A2",
            "loss": "weighted_ce",
            "seed": 2718,
            "folds": [1, 2, 3, 4, 5],
            "input_mode": "target_only",
            "max_length": 256,
            "precision": "fp32",
            "optimizer": "AdamW",
            "learning_rate": 0.00001,
            "weight_decay": 0.01,
            "scheduler": "linear",
            "warmup_ratio": 0.1,
            "epochs": 5,
            "early_stopping": False,
            "train_batch_size": 8,
            "eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "gradient_clip_norm": 1.0,
            "class_weight_formula": "No=1.0; Yes=fitting_negative_count/fitting_positive_count",
            "threshold_reference": 0.5,
            "diagnostic_numerics": True,
            "diagnostics_affect_updates": False,
        },
        "thresholds": list(THRESHOLDS),
        "top_k_values": list(TOP_K_VALUES),
        "threshold_analysis_policy": (
            "Exploratory only on held-out outer-fold predictions; no threshold "
            "selection, calibration, or training adjustment."
        ),
        "source_snapshot": {
            "files": source_hashes,
            "sha256": source_files_hash,
        },
        "environment": _environment(torch),
        "dataset_source_audit": source_audit,
    }
    protocol_path = protocol_path.resolve()
    if protocol_path.exists():
        existing = read_json(protocol_path)
        if existing != payload:
            raise FileExistsError(
                f"Frozen protocol already exists and differs: {protocol_path}"
            )
        return protocol_path
    write_json(protocol_path, payload)
    return protocol_path


def load_annotation_ready_protocol(
    path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    protocol = read_json(path)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != EXPERIMENT_ID
        or protocol.get("config_sha256") != config["_config_sha256"]
    ):
        raise RuntimeError(f"Unsupported or mismatched Experiment 2 protocol: {path}")
    if protocol.get("source_snapshot", {}).get("files") != source_file_hashes():
        raise RuntimeError("Experiment 2 source files differ from the frozen snapshot")
    if (
        protocol["source_snapshot"].get("sha256")
        != sha256_text(stable_json(protocol["source_snapshot"]["files"]))
    ):
        raise RuntimeError("Experiment 2 source snapshot digest is invalid")
    environment = protocol["environment"]
    for name in (
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "tokenizers",
        "sentencepiece",
        "safetensors",
    ):
        expected = environment[name]
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(
                f"Experiment 2 dependency mismatch for {name}: "
                f"expected {expected}, found {actual}"
            )
    return protocol, sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--model-path", type=Path, default=Path(r"D:\RFC\models\deberta-v3-base")
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = freeze_annotation_ready_protocol(
        config_path=args.config, protocol_path=args.protocol, model_path=args.model_path
    )
    print(f"Experiment 2 frozen protocol: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
