"""CPU-only preparation, provenance, preflight, and validation for V1.2.

This module deliberately does not import torch or transformers.  The only
training entry point is :mod:`runner`, and it accepts V1.2 inner-cell specs only.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.transformer_final.common import sha256_file, stable_json
from models.transformer_final.leakage import audit_leakage, build_fold_manifests

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
V11_ROOT = EXPERIMENT_ROOT.parent / "final_annotation_661_v1_1"
CONFIG_PATH = EXPERIMENT_ROOT / "config.json"
PROTOCOL_PATH = EXPERIMENT_ROOT / "EXPERIMENT_PROTOCOL.json"
SOURCE_PROVENANCE_PATH = EXPERIMENT_ROOT / "SOURCE_PROVENANCE.json"
DATASET_DIR = V11_ROOT / "manifests" / "dataset_v1"
INNER_DIR = EXPERIMENT_ROOT / "manifests" / "inner"
CELL_SPECS_DIR = EXPERIMENT_ROOT / "cell_specs"
RUNS_DIR = EXPERIMENT_ROOT / "runs"
AUDITS_DIR = EXPERIMENT_ROOT / "audits"
LOGS_DIR = EXPERIMENT_ROOT / "logs"
FAILURES_DIR = EXPERIMENT_ROOT / "failure_manifests"
COMPLETIONS_DIR = EXPERIMENT_ROOT / "completion_manifests"
REUSE_PATH = EXPERIMENT_ROOT / "OUTER_REUSE_MANIFEST.json"
IMMUTABILITY_PATH = EXPERIMENT_ROOT / "V1_1_IMMUTABILITY.json"
FROZEN_PATH = EXPERIMENT_ROOT / "FROZEN.json"

METHODS = ("focal", "partial4")
EXPECTED_DATASET_SHA256 = "8e395e83087c28cdc8e5b514f6fb4eca96e54a31fb525be770810db6f9e22e1d"
EXPECTED_EFFECTIVE_SHA256 = "19364a405048c279c2959ed6110a2e6868f2b4759ea21c659399038e6fbcec23"
EXPECTED_MODEL_SHA256 = "206ee7bb2c7805bcc8fd3bbbc18a09e185bafc19d4e112f31d4a088136717dd7"
EXPECTED_V11_PROTOCOL_SHA256 = "94024e67ddd61fc5562a7a3f09a4f6b136c765a0f590fbea5330ce26baddb4c0"
EXPECTED_V11_DATASET_MANIFEST_SHA256 = "c4b7de8ffada5bb75d37eedfba651dbf7612f159b43ecde094aff9b889c67676"
FAILED_CELL_ID = "inner-focal-outer1-inner1-seed2718"
MANIFEST_SCHEMA_FIELDS = {
    "schema_version", "created_at", "git_commit", "experiment", "config_path",
    "config_sha256", "source_path", "original_dataset_sha256",
    "approved_metadata_hashes", "fold_source_path", "fold_source_sha256",
    "dataset_file", "effective_dataset_sha256", "dataset_columns",
    "validation_report", "validation_report_sha256", "leakage_audit",
    "leakage_audit_sha256", "unresolved_exclusions", "label_mapping",
    "reserved_partitions", "effective_summary", "private_key_reporting",
}
DATASET_COLUMNS = [
    "item_id", "record_id", "family_id", "document_id", "revision",
    "clause_ids", "section_ids", "lineage_ids", "target_clause",
    "section_heading", "previous_clause", "next_clause", "context_source_ids",
    "context_resolution_status", "alignment_source_path", "source_hashes",
    "label", "final_source", "final_status", "fold",
]


def as_path(value: str | os.PathLike[str] | Path) -> Path:
    return Path(value).expanduser().resolve()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: str | Path) -> Any:
    with as_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def _write_text(path: str | Path, value: str, *, exclusive: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        handle.write(value)


def _repo_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def load_local_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    path = as_path(path)
    config = _read_json(path)
    if config.get("experiment") != "final_annotation_661_v1_2":
        raise RuntimeError("Wrong experiment configuration")
    if config.get("schema_version") != 1:
        raise RuntimeError("Unsupported configuration schema")
    if config["data"]["source_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Authoritative dataset identity changed")
    if config["model"]["source_sha256"] != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Approved model identity changed")
    if config["training"] != _read_json(V11_ROOT / "config.json")["training"]:
        raise RuntimeError("Training configuration differs from V1.1")
    for section in ("data", "partitions", "folds", "model", "tokenization", "methods", "calibration", "ensemble", "evaluation", "conditions"):
        if config[section] != _read_json(V11_ROOT / "config.json")[section]:
            raise RuntimeError(f"Scientific configuration differs from V1.1: {section}")
    config["_config_path"] = str(path)
    config["_config_sha256"] = sha256_file(path)
    return config


def _read_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"item_id": str, "record_id": str}, keep_default_na=False)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame


def _labels(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label"].value_counts().to_dict()
    return {"0": int(counts.get(0, 0)), "1": int(counts.get(1, 0))}


def _write_assignments(root: Path, frame: pd.DataFrame, fold: dict[str, Any]) -> None:
    number = int(fold["fold"])
    values = frame[["item_id", "record_id", "family_id", "fold"]].copy()
    exclusions = {
        item["record_id"]: "|".join(item["reasons"])
        for item in fold["fitting_exclusions"]
    }
    values["role"] = values["fold"].map(
        lambda value: "evaluation" if int(value) == number else "fitting"
    )
    values["included"] = ~values["record_id"].isin(exclusions)
    values.loc[values["role"] == "evaluation", "included"] = True
    values["exclusion_reason"] = values["record_id"].map(exclusions).fillna("")
    values.to_csv(root / "folds" / f"fold_{number}_assignments.csv", index=False, lineterminator="\n")


def _build_inner_dataset(outer_fold: int, config: dict[str, Any]) -> dict[str, Any]:
    destination = INNER_DIR / f"outer_{outer_fold}"
    if destination.exists():
        raise FileExistsError(f"Immutable inner dataset already exists: {destination}")
    global_frame = _read_dataset(DATASET_DIR / "dataset.csv")
    outer_manifest_path = DATASET_DIR / "folds" / f"fold_{outer_fold}.json"
    outer_manifest = _read_json(outer_manifest_path)
    authoritative_ids = set(outer_manifest["fitting_record_ids"])
    old_inner_path = V11_ROOT / "manifests" / "inner" / f"outer_{outer_fold}" / "dataset.csv"
    old_inner = _read_dataset(old_inner_path)
    assignment = dict(zip(old_inner["record_id"], old_inner["fold"], strict=True))
    if not authoritative_ids <= set(assignment):
        raise RuntimeError(f"V1.1 inner assignments do not cover outer {outer_fold} fitting IDs")
    frame = global_frame.loc[global_frame["record_id"].isin(authoritative_ids)].copy()
    frame["fold"] = frame["record_id"].map(assignment).astype(int)
    frame = frame.loc[:, DATASET_COLUMNS].sort_values("item_id", key=lambda s: s.astype(int)).reset_index(drop=True)
    destination.mkdir(parents=True)
    (destination / "folds").mkdir()
    dataset_path = destination / "dataset.csv"
    frame.to_csv(dataset_path, index=False, encoding="utf-8", lineterminator="\n")
    audit = audit_leakage(frame, near_threshold=float(config["folds"]["near_duplicate_threshold"]))
    if audit["training_blocked"]:
        raise RuntimeError(f"Inner outer-{outer_fold} leakage audit blocked training")
    _write_json(destination / "leakage_audit.json", audit, exclusive=True)
    folds = build_fold_manifests(frame, audit)
    if [int(item["fold"]) for item in folds] != [1, 2, 3]:
        raise RuntimeError(f"Inner outer-{outer_fold} does not have exactly three folds")
    summaries = []
    for fold in folds:
        fold_path = destination / "folds" / f"fold_{fold['fold']}.json"
        _write_json(fold_path, fold, exclusive=True)
        _write_assignments(destination, frame, fold)
        summaries.append({"fold": fold["fold"], **fold["counts"], "held_out_families": fold["held_out_families"]})
    parent_candidate = set(global_frame.loc[global_frame["fold"] != outer_fold, "record_id"])
    removed = sorted(parent_candidate - authoritative_ids)
    validation = {
        "schema_version": 1,
        "derivation": "outer-fold fitting_record_ids, retaining V1.1 inner family-fold assignments",
        "parent_outer_fold": outer_fold,
        "parent_outer_manifest_path": _repo_path(outer_manifest_path),
        "parent_outer_manifest_sha256": sha256_file(outer_manifest_path),
        "parent_candidate_rows": len(parent_candidate),
        "effective_rows": len(frame),
        "effective_labels": _labels(frame),
        "outer_fitting_exclusions_removed": outer_manifest["fitting_exclusions"],
        "removed_record_ids": removed,
        "unresolved_exclusions": [],
        "reserved_partition_counts": {"tcp_rows": 0, "dns_rows": 0},
    }
    _write_json(destination / "validation_report.json", validation, exclusive=True)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "git_commit": _git_commit(),
        "experiment": "final_annotation_661_v1_2",
        "config_path": _repo_path(CONFIG_PATH),
        "config_sha256": config["_config_sha256"],
        "source_path": config["data"]["source_csv"],
        "original_dataset_sha256": EXPECTED_DATASET_SHA256,
        "approved_metadata_hashes": config["data"]["approved_source_hashes"],
        "fold_source_path": _repo_path(old_inner_path),
        "fold_source_sha256": sha256_file(old_inner_path),
        "dataset_file": "dataset.csv",
        "effective_dataset_sha256": sha256_file(dataset_path),
        "dataset_columns": DATASET_COLUMNS,
        "validation_report": "validation_report.json",
        "validation_report_sha256": sha256_file(destination / "validation_report.json"),
        "leakage_audit": "leakage_audit.json",
        "leakage_audit_sha256": sha256_file(destination / "leakage_audit.json"),
        "unresolved_exclusions": [],
        "label_mapping": {"No": 0, "Yes": 1},
        "reserved_partitions": {
            "tcp": {"family": config["partitions"]["tcp_family"], "use": "reserved_no_training_tuning_or_calibration"},
            "dns": {"family": config["partitions"]["dns_family"], "use": "disabled_incompatible_sealed_evaluation"},
        },
        "effective_summary": {
            "rows": len(frame), "labels": _labels(frame),
            "families": sorted(frame["family_id"].unique()), "folds": summaries,
        },
        "private_key_reporting": "Only hashes, join counts, and required stable identifiers are persisted; votes and private annotations are omitted.",
    }
    if set(manifest) != MANIFEST_SCHEMA_FIELDS:
        raise RuntimeError("Inner manifest differs from authoritative historical schema")
    _write_json(destination / "manifest.json", manifest, exclusive=True)
    return manifest


def _static_source_hashes() -> dict[str, str]:
    paths = [
        CONFIG_PATH, EXPERIMENT_ROOT / "__init__.py", EXPERIMENT_ROOT / "core.py",
        EXPERIMENT_ROOT / "runner.py", EXPERIMENT_ROOT / "analysis.py",
        EXPERIMENT_ROOT / "cli.py",
        ROOT / "models" / "transformer_final" / "dataset.py",
        ROOT / "models" / "transformer_final" / "leakage.py",
        ROOT / "models" / "transformer_final" / "train.py",
        ROOT / "models" / "transformer_final" / "common.py",
    ]
    return {_repo_path(path): sha256_file(path) for path in paths}


def _failed_attempt_files() -> dict[str, str]:
    parent = V11_ROOT / "runs" / "inner" / "focal" / "outer_1" / "inner_1"
    failures = sorted(parent.glob(f".attempt_{FAILED_CELL_ID}.failed-*"))
    if len(failures) != 1:
        raise RuntimeError(f"Expected exactly one preserved failed attempt, found {len(failures)}")
    result: dict[str, str] = {}
    for path in sorted(failures[0].rglob("*")):
        if path.is_file():
            result[_repo_path(path)] = sha256_file(path)
    return result


def _v11_immutability() -> dict[str, Any]:
    paths = [
        V11_ROOT / "config.json", V11_ROOT / "EXPERIMENT_PROTOCOL.json",
        V11_ROOT / "FROZEN.json", V11_ROOT / "SOURCE_PROVENANCE.json",
        DATASET_DIR / "manifest.json", DATASET_DIR / "dataset.csv",
        V11_ROOT / "failure_manifests" / f"{FAILED_CELL_ID}.json",
    ]
    for spec in sorted((V11_ROOT / "cell_specs").glob("*.json"))[:10]:
        value = _read_json(spec)
        paths.extend([
            spec, Path(value["completion_path"]), Path(value["audit_path"]),
            Path(value["run_path"]) / "run_manifest.json",
            Path(value["run_path"]) / "run_config.json",
            Path(value["run_path"]) / "predictions" / "held_out_predictions.csv",
            Path(value["run_path"]) / "checkpoints" / "checkpoint_metadata.json",
        ])
    hashes = {_repo_path(path): sha256_file(path) for path in paths}
    failure_hashes = _failed_attempt_files()
    return {
        "schema_version": 1, "source_experiment": "final_annotation_661_v1_1",
        "files": hashes, "failed_attempt_files": failure_hashes,
        "failed_attempt_file_count": len(failure_hashes),
    }


def _assert_hashes(mapping: dict[str, str], label: str) -> None:
    for relative, expected in mapping.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"{label} changed: {relative}")


def _semantic_config(config: dict[str, Any]) -> dict[str, Any]:
    keys = ("data", "partitions", "folds", "model", "tokenization", "training", "methods", "calibration", "ensemble", "evaluation", "conditions")
    return {key: config[key] for key in keys}


def prediction_path(spec: dict[str, Any]) -> Path:
    return Path(spec["run_path"]) / "predictions" / "held_out_predictions.csv"


def _validate_outer_spec(spec: dict[str, Any], *, hash_checkpoint: bool = True) -> dict[str, Any]:
    if spec["kind"] != "outer" or spec.get("reuse_by_reference") is not True:
        raise RuntimeError("Outer V1.2 spec is not an immutable reference")
    cell = spec["cell_id"]
    run = Path(spec["run_path"])
    completion = _read_json(spec["completion_path"])
    audit = _read_json(spec["audit_path"])
    run_manifest_path = run / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    predictions_path = prediction_path(spec)
    if completion["cell_id"] != cell or completion["automatic_retry"] is not False:
        raise RuntimeError(f"Invalid V1.1 completion semantics: {cell}")
    expected_completion = {
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "audit_sha256": sha256_file(Path(spec["audit_path"])),
    }
    for key, actual in expected_completion.items():
        if completion.get(key) != actual:
            raise RuntimeError(f"Completion hash mismatch for {cell}: {key}")
    if completion["protocol_sha256"] != EXPECTED_V11_PROTOCOL_SHA256:
        raise RuntimeError(f"V1.1 completion protocol mismatch: {cell}")
    if audit.get("protocol_sha256") != EXPECTED_V11_PROTOCOL_SHA256 or audit.get("split_manifest_sha256") != spec["split_manifest_sha256"]:
        raise RuntimeError(f"V1.1 audit provenance mismatch: {cell}")
    expected_manifest = {
        "cell_id": cell, "cell_kind": "outer", "analysis_method": spec["method"],
        "outer_fold": spec["outer_fold"], "inner_fold": None, "seed": 2718,
        "precision": "fp32", "input_mode": "target_only", "condition": "A2",
        "protocol_sha256": EXPECTED_V11_PROTOCOL_SHA256,
        "dataset_manifest_sha256": EXPECTED_V11_DATASET_MANIFEST_SHA256,
        "effective_dataset_sha256": EXPECTED_EFFECTIVE_SHA256,
        "original_dataset_sha256": EXPECTED_DATASET_SHA256,
        "model_source_sha256": EXPECTED_MODEL_SHA256,
        "split_manifest_sha256": spec["split_manifest_sha256"],
        "fold_manifest_sha256": spec["split_manifest_sha256"],
        "fresh_base_initialization": True, "resume_checkpoint": None,
        "epoch_policy": "fixed_5_no_early_stopping", "epochs_completed": 5,
        "outer_fold_evaluations_during_training": 0, "outer_fold_evaluations_after_training": 1,
    }
    for key, expected in expected_manifest.items():
        if run_manifest.get(key) != expected:
            raise RuntimeError(f"Run provenance mismatch for {cell}: {key}")
    if spec["method"] == "focal" and run_manifest.get("loss") != "focal_gamma_2":
        raise RuntimeError(f"Focal semantics mismatch: {cell}")
    if spec["method"] == "partial4" and (run_manifest.get("fine_tuning_policy") != "partial_4" or run_manifest.get("trainable_last_layers") != 4):
        raise RuntimeError(f"Partial-4 semantics mismatch: {cell}")
    for artifact in run_manifest["artifact_hashes"].values():
        path = run / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Run artifact mismatch for {cell}: {artifact['path']}")
    split = _read_json(spec["split_manifest_path"])
    if sha256_file(Path(spec["split_manifest_path"])) != spec["split_manifest_sha256"]:
        raise RuntimeError(f"Outer split changed: {cell}")
    dataset = _read_dataset(DATASET_DIR / "dataset.csv")
    fitting = dataset.loc[dataset["record_id"].isin(split["fitting_record_ids"])]
    scoring = dataset.loc[dataset["record_id"].isin(split["scoring_record_ids"])]
    if len(fitting) != run_manifest["fitting_rows"] or len(scoring) != run_manifest["held_out_rows"]:
        raise RuntimeError(f"Outer partition counts differ: {cell}")
    predictions = pd.read_csv(predictions_path, dtype={"item_id": str, "record_id": str}, keep_default_na=False)
    if set(predictions["record_id"]) != set(split["scoring_record_ids"]) or predictions["record_id"].duplicated().any():
        raise RuntimeError(f"Prediction IDs differ from held-out IDs: {cell}")
    expected_rows = scoring.set_index("record_id")
    actual_rows = predictions.set_index("record_id")
    for column in ("item_id", "family_id", "label"):
        left = actual_rows.loc[expected_rows.index, column].astype(str)
        right = expected_rows[column].astype(str)
        if not left.equals(right):
            raise RuntimeError(f"Prediction {column} provenance mismatch: {cell}")
    values = predictions["probability_yes"].astype(float).to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise RuntimeError(f"Invalid prediction probabilities: {cell}")
    checkpoint_meta = _read_json(run / "checkpoints" / "checkpoint_metadata.json")
    if checkpoint_meta.get("epoch") != 5 or checkpoint_meta.get("policy") != "final epoch only; no checkpoint selection":
        raise RuntimeError(f"Checkpoint policy mismatch: {cell}")
    if hash_checkpoint:
        for name, expected in checkpoint_meta["files"].items():
            if sha256_file(run / "checkpoints" / "final_epoch" / name) != expected:
                raise RuntimeError(f"Checkpoint file mismatch for {cell}: {name}")
    run_config = _read_json(run / "run_config.json")
    if run_config["protocol_sha256"] != EXPECTED_V11_PROTOCOL_SHA256 or run_config["cell_id"] != cell:
        raise RuntimeError(f"Run configuration provenance mismatch: {cell}")
    if _semantic_config(run_config["base_configuration"]) != _semantic_config(_read_json(V11_ROOT / "config.json")):
        raise RuntimeError(f"Run scientific configuration mismatch: {cell}")
    return {
        "cell_id": cell, "source_protocol_sha256": EXPECTED_V11_PROTOCOL_SHA256,
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "predictions_sha256": sha256_file(predictions_path),
        "completion_sha256": sha256_file(Path(spec["completion_path"])),
        "checkpoint_metadata_sha256": sha256_file(run / "checkpoints" / "checkpoint_metadata.json"),
        "fitting_rows": len(fitting), "held_out_rows": len(scoring),
        "fitting_ids_sha256": hashlib.sha256("\n".join(sorted(split["fitting_record_ids"])).encode()).hexdigest(),
        "held_out_ids_sha256": hashlib.sha256("\n".join(sorted(split["scoring_record_ids"])).encode()).hexdigest(),
    }


def _make_specs(protocol_hash: str) -> None:
    CELL_SPECS_DIR.mkdir(parents=True)
    v11_specs = {_read_json(path)["cell_id"]: _read_json(path) for path in sorted((V11_ROOT / "cell_specs").glob("*.json"))}
    config = load_local_config()
    index = 0
    for method in METHODS:
        for outer in range(1, 6):
            index += 1
            cell = f"outer-{method}-fold{outer}-seed2718"
            source = v11_specs[cell]
            spec = dict(source)
            spec.update(
                protocol_path=str(PROTOCOL_PATH), protocol_sha256=protocol_hash,
                artifact_protocol_sha256=EXPECTED_V11_PROTOCOL_SHA256,
                reuse_by_reference=True, source_experiment="final_annotation_661_v1_1",
                dataset_manifest_sha256=EXPECTED_V11_DATASET_MANIFEST_SHA256,
                effective_dataset_sha256=EXPECTED_EFFECTIVE_SHA256,
            )
            _write_json(CELL_SPECS_DIR / f"{index:02d}_{cell}.json", spec, exclusive=True)
    for method in METHODS:
        for outer in range(1, 6):
            root = INNER_DIR / f"outer_{outer}"
            manifest = _read_json(root / "manifest.json")
            for inner in range(1, 4):
                index += 1
                cell = f"inner-{method}-outer{outer}-inner{inner}-seed2718"
                split_path = root / "folds" / f"fold_{inner}.json"
                spec = {
                    "schema_version": 1, "cell_id": cell, "kind": "inner",
                    "method": method, "outer_fold": outer, "inner_fold": inner,
                    "train_fold": inner, "seed": 2718, "precision": "fp32",
                    "input_mode": "target_only", "reuse_by_reference": False,
                    "automatic_retry": False, "fresh_base_initialization": True,
                    "config_path": str(CONFIG_PATH), "protocol_path": str(PROTOCOL_PATH),
                    "protocol_sha256": protocol_hash,
                    "dataset_path": str(root / "dataset.csv"),
                    "dataset_manifest_path": str(root / "manifest.json"),
                    "dataset_manifest_sha256": sha256_file(root / "manifest.json"),
                    "effective_dataset_sha256": manifest["effective_dataset_sha256"],
                    "split_manifest_path": str(split_path),
                    "split_manifest_sha256": sha256_file(split_path),
                    "model_path": str(as_path(config["model"]["default_local_path"])),
                    "model_source_sha256": EXPECTED_MODEL_SHA256,
                    "run_path": str(RUNS_DIR / "inner" / method / f"outer_{outer}" / f"inner_{inner}" / f"attempt_{cell}"),
                    "audit_path": str(AUDITS_DIR / f"{cell}.json"),
                    "completion_path": str(COMPLETIONS_DIR / f"{cell}.json"),
                }
                _write_json(CELL_SPECS_DIR / f"{index:02d}_{cell}.json", spec, exclusive=True)


def iter_specs(kind: str | None = None) -> list[tuple[Path, dict[str, Any]]]:
    result = [(path, _read_json(path)) for path in sorted(CELL_SPECS_DIR.glob("*.json"))]
    return [item for item in result if kind is None or item[1]["kind"] == kind]


def validate_inner_dataset(outer_fold: int) -> dict[str, Any]:
    root = INNER_DIR / f"outer_{outer_fold}"
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    if set(manifest) != MANIFEST_SCHEMA_FIELDS or manifest["schema_version"] != 1:
        raise RuntimeError(f"Inner manifest schema mismatch: outer {outer_fold}")
    for key, hash_key in (("dataset_file", "effective_dataset_sha256"), ("validation_report", "validation_report_sha256"), ("leakage_audit", "leakage_audit_sha256")):
        if sha256_file(root / manifest[key]) != manifest[hash_key]:
            raise RuntimeError(f"Inner manifest hash mismatch: outer {outer_fold} {key}")
    if manifest["config_sha256"] != sha256_file(CONFIG_PATH) or manifest["original_dataset_sha256"] != EXPECTED_DATASET_SHA256:
        raise RuntimeError(f"Inner source/config provenance mismatch: outer {outer_fold}")
    if sha256_file(ROOT / manifest["fold_source_path"]) != manifest["fold_source_sha256"]:
        raise RuntimeError(f"Inner fold assignment source changed: outer {outer_fold}")
    frame = _read_dataset(root / "dataset.csv")
    outer = _read_json(DATASET_DIR / "folds" / f"fold_{outer_fold}.json")
    if set(frame["record_id"]) != set(outer["fitting_record_ids"]):
        raise RuntimeError(f"Inner dataset is not the actual outer fitting partition: {outer_fold}")
    if len(frame) != manifest["effective_summary"]["rows"] or _labels(frame) != manifest["effective_summary"]["labels"]:
        raise RuntimeError(f"Inner summary mismatch: outer {outer_fold}")
    seen_scoring: set[str] = set()
    fold_hashes: dict[str, str] = {}
    for inner in range(1, 4):
        path = root / "folds" / f"fold_{inner}.json"
        fold = _read_json(path)
        scoring = set(fold["scoring_record_ids"])
        fitting = set(fold["fitting_record_ids"])
        if scoring != set(frame.loc[frame["fold"] == inner, "record_id"]):
            raise RuntimeError(f"Inner scoring IDs mismatch: outer {outer_fold}, inner {inner}")
        if scoring & fitting or not (scoring | fitting) <= set(frame["record_id"]):
            raise RuntimeError(f"Inner split overlap/out-of-domain IDs: outer {outer_fold}, inner {inner}")
        assignments = pd.read_csv(root / "folds" / f"fold_{inner}_assignments.csv", dtype={"record_id": str}, keep_default_na=False)
        if set(assignments["record_id"]) != set(frame["record_id"]):
            raise RuntimeError(f"Inner assignments incomplete: outer {outer_fold}, inner {inner}")
        seen_scoring |= scoring
        fold_hashes[str(inner)] = sha256_file(path)
    if seen_scoring != set(frame["record_id"]):
        raise RuntimeError(f"Inner folds do not partition outer fitting set: {outer_fold}")
    return {"outer_fold": outer_fold, "rows": len(frame), "labels": _labels(frame), "manifest_sha256": sha256_file(manifest_path), "fold_sha256": fold_hashes}


def verify_frozen() -> dict[str, Any]:
    frozen = _read_json(FROZEN_PATH)
    if sha256_file(PROTOCOL_PATH) != frozen["protocol_sha256"]:
        raise RuntimeError("V1.2 protocol changed")
    actual_sources = _static_source_hashes()
    if stable_json(actual_sources) != stable_json(_read_json(EXPERIMENT_ROOT / "SOURCE_SNAPSHOT.json")["files"]):
        raise RuntimeError("V1.2 source snapshot changed")
    if hashlib.sha256(stable_json(actual_sources).encode()).hexdigest() != frozen["source_snapshot_sha256"]:
        raise RuntimeError("V1.2 source snapshot aggregate mismatch")
    immutable = _read_json(IMMUTABILITY_PATH)
    _assert_hashes(immutable["files"], "V1.1 immutable source")
    _assert_hashes(immutable["failed_attempt_files"], "V1.1 failed-attempt evidence")
    if sha256_file(ROOT / "results" / "annotation" / "final_annotation.csv") != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Authoritative dataset changed")
    if sha256_file(DATASET_DIR / "dataset.csv") != EXPECTED_EFFECTIVE_SHA256:
        raise RuntimeError("Effective global dataset changed")
    inner = [validate_inner_dataset(outer) for outer in range(1, 6)]
    return {"protocol_sha256": frozen["protocol_sha256"], "source_snapshot_sha256": frozen["source_snapshot_sha256"], "inner_datasets": inner}


def validate_cells(kind: str, require_all: bool = False) -> list[dict[str, Any]]:
    if kind not in {"outer", "inner"}:
        raise ValueError("kind must be outer or inner")
    results = []
    for _, spec in iter_specs(kind):
        if kind == "outer":
            detail = _validate_outer_spec(spec, hash_checkpoint=True)
            results.append({"cell_id": spec["cell_id"], "status": "valid_reused", **detail})
            continue
        complete = Path(spec["completion_path"]).is_file()
        if not complete:
            if require_all:
                raise RuntimeError(f"Incomplete cell: {spec['cell_id']}")
            results.append({"cell_id": spec["cell_id"], "status": "pending"})
            continue
        run = Path(spec["run_path"])
        completion = _read_json(spec["completion_path"])
        manifest_path = run / "run_manifest.json"
        predictions = prediction_path(spec)
        if completion.get("automatic_retry") is not False or completion.get("protocol_sha256") != spec["protocol_sha256"]:
            raise RuntimeError(f"Invalid inner completion semantics: {spec['cell_id']}")
        checks = {
            "run_manifest_sha256": sha256_file(manifest_path),
            "predictions_sha256": sha256_file(predictions),
            "audit_sha256": sha256_file(Path(spec["audit_path"])),
        }
        if any(completion.get(key) != value for key, value in checks.items()):
            raise RuntimeError(f"Inner completion hash mismatch: {spec['cell_id']}")
        run_manifest = _read_json(manifest_path)
        expected = {
            "cell_id": spec["cell_id"], "cell_kind": "inner", "analysis_method": spec["method"],
            "outer_fold": spec["outer_fold"], "inner_fold": spec["inner_fold"],
            "protocol_sha256": spec["protocol_sha256"],
            "dataset_manifest_sha256": spec["dataset_manifest_sha256"],
            "effective_dataset_sha256": spec["effective_dataset_sha256"],
            "split_manifest_sha256": spec["split_manifest_sha256"],
            "fold_manifest_sha256": spec["split_manifest_sha256"],
            "fresh_base_initialization": True, "resume_checkpoint": None,
        }
        if any(run_manifest.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Inner run provenance mismatch: {spec['cell_id']}")
        split = _read_json(spec["split_manifest_path"])
        prediction_frame = pd.read_csv(predictions, dtype={"record_id": str}, keep_default_na=False)
        if set(prediction_frame["record_id"]) != set(split["scoring_record_ids"]):
            raise RuntimeError(f"Inner prediction IDs mismatch: {spec['cell_id']}")
        results.append({"cell_id": spec["cell_id"], "status": "valid"})
    return results


def validate_cell(spec: dict[str, Any]) -> dict[str, Any]:
    for result in validate_cells(spec["kind"], require_all=False):
        if result["cell_id"] == spec["cell_id"]:
            if result["status"] == "pending":
                raise RuntimeError(f"Incomplete cell: {spec['cell_id']}")
            return result
    raise RuntimeError(f"Unknown cell: {spec['cell_id']}")


def preflight() -> dict[str, Any]:
    frozen = verify_frozen()
    specs = iter_specs()
    if len(specs) != 40 or len(iter_specs("outer")) != 10 or len(iter_specs("inner")) != 30:
        raise RuntimeError("Expected 10 outer references and 30 inner execution specs")
    outer = validate_cells("outer", require_all=True)
    for _, spec in iter_specs("inner"):
        required = ["dataset_path", "dataset_manifest_path", "split_manifest_path", "config_path", "protocol_path"]
        if any(not Path(spec[key]).is_file() for key in required):
            raise RuntimeError(f"Required inner path missing: {spec['cell_id']}")
        if sha256_file(Path(spec["dataset_manifest_path"])) != spec["dataset_manifest_sha256"]:
            raise RuntimeError(f"Inner manifest hash mismatch in spec: {spec['cell_id']}")
        if sha256_file(Path(spec["dataset_path"])) != spec["effective_dataset_sha256"]:
            raise RuntimeError(f"Inner dataset hash mismatch in spec: {spec['cell_id']}")
        if sha256_file(Path(spec["split_manifest_path"])) != spec["split_manifest_sha256"]:
            raise RuntimeError(f"Inner split hash mismatch in spec: {spec['cell_id']}")
        if spec["reuse_by_reference"] is not False or spec["automatic_retry"] is not False:
            raise RuntimeError(f"Inner fresh/no-retry policy mismatch: {spec['cell_id']}")
    v11_completion = V11_ROOT / "completion_manifests" / f"{FAILED_CELL_ID}.json"
    if v11_completion.exists():
        raise RuntimeError("Failed V1.1 inner cell was incorrectly marked complete")
    result = {
        "schema_version": 1, "checked_at": utc_now(), "status": "pass",
        "protocol_sha256": frozen["protocol_sha256"],
        "source_snapshot_sha256": frozen["source_snapshot_sha256"],
        "valid_reused_outer_cells": len(outer), "inner_dataset_roots": 5,
        "inner_partitions": 15, "inner_execution_specs": 30,
        "failed_cell_complete": False, "training_executed": False,
    }
    _write_json(EXPERIMENT_ROOT / "PREFLIGHT_RESULT.json", result)
    return result


def refresh_status() -> dict[str, Any]:
    cells = []
    for _, spec in iter_specs():
        if spec["kind"] == "outer":
            status = "valid_reused_reference"
        else:
            status = "complete_unvalidated" if Path(spec["completion_path"]).is_file() else "pending_fresh_execution"
        cells.append({"cell_id": spec["cell_id"], "kind": spec["kind"], "status": status})
    result = {"schema_version": 1, "updated_at": utc_now(), "cells": cells}
    _write_json(EXPERIMENT_ROOT / "TRAINING_STATUS.json", result)
    return result


def status() -> dict[str, Any]:
    cells = []
    for _, spec in iter_specs():
        if spec["kind"] == "outer":
            value = "valid_reused_reference"
        else:
            value = "complete_unvalidated" if Path(spec["completion_path"]).is_file() else "pending_fresh_execution"
        cells.append({"cell_id": spec["cell_id"], "kind": spec["kind"], "status": value})
    counts = Counter(item["status"] for item in cells)
    return {"cells": cells, "counts": dict(sorted(counts.items()))}


def prepare() -> Path:
    if PROTOCOL_PATH.exists():
        raise FileExistsError("V1.2 is already prepared and immutable")
    config = load_local_config()
    if sha256_file(ROOT / config["data"]["source_csv"]) != EXPECTED_DATASET_SHA256:
        raise RuntimeError("Authoritative source mismatch")
    if sha256_file(DATASET_DIR / "manifest.json") != EXPECTED_V11_DATASET_MANIFEST_SHA256:
        raise RuntimeError("V1.1 global dataset manifest mismatch")
    if sha256_file(DATASET_DIR / "dataset.csv") != EXPECTED_EFFECTIVE_SHA256:
        raise RuntimeError("V1.1 effective dataset mismatch")
    if sha256_file(V11_ROOT / "EXPERIMENT_PROTOCOL.json") != EXPECTED_V11_PROTOCOL_SHA256:
        raise RuntimeError("V1.1 protocol mismatch")
    global_frame = _read_dataset(DATASET_DIR / "dataset.csv")
    if len(global_frame) != 2125 or _labels(global_frame) != {"0": 1989, "1": 136}:
        raise RuntimeError("Development cohort counts changed")
    for path in (INNER_DIR, RUNS_DIR, AUDITS_DIR, LOGS_DIR, FAILURES_DIR, COMPLETIONS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    inner = [_build_inner_dataset(outer, config) for outer in range(1, 6)]
    sources = _static_source_hashes()
    source_snapshot_hash = hashlib.sha256(stable_json(sources).encode()).hexdigest()
    _write_json(EXPERIMENT_ROOT / "SOURCE_SNAPSHOT.json", {"schema_version": 1, "files": sources, "sha256": source_snapshot_hash}, exclusive=True)
    provenance = {
        "schema_version": 1, "created_at": utc_now(),
        "authoritative_dataset": {"path": config["data"]["source_csv"], "sha256": EXPECTED_DATASET_SHA256},
        "effective_global_dataset": {"path": _repo_path(DATASET_DIR / "dataset.csv"), "sha256": EXPECTED_EFFECTIVE_SHA256},
        "source_experiment": {"path": _repo_path(V11_ROOT), "protocol_sha256": EXPECTED_V11_PROTOCOL_SHA256, "dataset_manifest_sha256": EXPECTED_V11_DATASET_MANIFEST_SHA256},
        "authoritative_manifest_producer": {"path": "models/transformer_final/dataset.py", "function": "prepare_dataset", "source_sha256": sha256_file(ROOT / "models" / "transformer_final" / "dataset.py"), "schema_version": 1, "fields": sorted(MANIFEST_SCHEMA_FIELDS)},
        "inner_derivation": "Each inner root is exactly the corresponding outer fold manifest's fitting_record_ids; V1.1 inner family assignments are retained for those IDs and leakage manifests are regenerated.",
        "historical_interface": "train.run_cell hashes dataset_dir/manifest.json and dataset_dir/folds/fold_N.json; V1.2 adapter returns the complete inner dataset root.",
    }
    _write_json(SOURCE_PROVENANCE_PATH, provenance, exclusive=True)
    immutability = _v11_immutability()
    _write_json(IMMUTABILITY_PATH, immutability, exclusive=True)
    protocol = {
        "schema_version": 2, "protocol_id": "final_annotation_661_v1_2",
        "created_at": utc_now(), "git_commit": _git_commit(),
        "recovery_scope": "inner dataset provenance-manifest compatibility only",
        "source_protocol_sha256": EXPECTED_V11_PROTOCOL_SHA256,
        "source_provenance_sha256": sha256_file(SOURCE_PROVENANCE_PATH),
        "v1_1_immutability_sha256": sha256_file(IMMUTABILITY_PATH),
        "config_sha256": config["_config_sha256"],
        "authoritative_dataset_sha256": EXPECTED_DATASET_SHA256,
        "effective_global_dataset_sha256": EXPECTED_EFFECTIVE_SHA256,
        "model_source_sha256": EXPECTED_MODEL_SHA256,
        "scientific_configuration": _semantic_config(config),
        "outer_policy": "reuse 10 validated V1.1 cells by immutable absolute reference; never copy or retrain",
        "inner_policy": "30 fresh V1.2 cells, explicit user execution only, no automatic retry",
        "failed_v1_1_cell": FAILED_CELL_ID,
        "failed_cell_policy": "preserve evidence; never complete or reuse; V1.2 path is a fresh attempt",
        "inner_datasets": [
            {"outer_fold": index + 1, "manifest_sha256": sha256_file(INNER_DIR / f"outer_{index + 1}" / "manifest.json"), "effective_dataset_sha256": item["effective_dataset_sha256"]}
            for index, item in enumerate(inner)
        ],
        "ensemble_policy_changed": False, "training_executed": False,
    }
    _write_json(PROTOCOL_PATH, protocol, exclusive=True)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    _write_text(EXPERIMENT_ROOT / "PROTOCOL_SHA256.txt", protocol_hash + "\n", exclusive=True)
    _make_specs(protocol_hash)
    outer_details = [_validate_outer_spec(spec, hash_checkpoint=True) for _, spec in iter_specs("outer")]
    reuse = {
        "schema_version": 1, "created_at": utc_now(), "reuse_mode": "immutable_reference",
        "v1_2_protocol_sha256": protocol_hash, "v1_1_protocol_sha256": EXPECTED_V11_PROTOCOL_SHA256,
        "compatible_cells": len(outer_details), "cells": outer_details,
        "compatibility_dimensions": ["dataset_identity", "fitting_and_held_out_ids", "labels", "model_configuration", "source_and_protocol_semantics", "prediction_provenance", "completion_manifests"],
    }
    _write_json(REUSE_PATH, reuse, exclusive=True)
    frozen = {
        "schema_version": 1, "frozen_at": utc_now(), "protocol_sha256": protocol_hash,
        "source_snapshot_sha256": source_snapshot_hash,
        "source_provenance_sha256": sha256_file(SOURCE_PROVENANCE_PATH),
        "v1_1_immutability_sha256": sha256_file(IMMUTABILITY_PATH),
        "outer_reuse_manifest_sha256": sha256_file(REUSE_PATH),
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "effective_dataset_sha256": EXPECTED_EFFECTIVE_SHA256,
        "outer_cells_reused": 10, "inner_cells": 30, "training_executed": False,
    }
    _write_json(FROZEN_PATH, frozen, exclusive=True)
    _write_json(EXPERIMENT_ROOT / "DATASET_MANIFEST.json", _read_json(DATASET_DIR / "manifest.json"), exclusive=True)
    refresh_status()
    return EXPERIMENT_ROOT
