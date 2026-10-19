"""Recover V1 without retraining its first successfully trained GPU cell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.transformer_final.method_specific_calibration_v1 import study as v1
from models.transformer_final.common import (
    hash_directory,
    read_json,
    resolve_path,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
    write_json,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.leakage import audit_leakage

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
V1_CAL_ROOT = ROOT.parent / "method_specific_calibration_v1"
PROTOCOL_PATH = ROOT / "EXPERIMENT_PROTOCOL.json"
PROVENANCE_PATH = ROOT / "SOURCE_PROVENANCE.json"
REUSE_AUDIT_PATH = ROOT / "V1_ARTIFACT_REUSE_AUDIT.json"
STATUS_PATH = ROOT / "TRAINING_STATUS.json"
GPU_MANIFEST_PATH = ROOT / "GPU_EXECUTION_MANIFEST.json"
SPEC_DIR = ROOT / "cell_specs"
RUN_DIR = ROOT / "runs"
LOG_DIR = ROOT / "logs"
AUDIT_DIR = ROOT / "audits"
COMPLETION_DIR = ROOT / "completion_manifests"
FAILURE_DIR = ROOT / "failure_manifests"
COMPATIBILITY_DIR = ROOT / "compatibility_manifests"
V1_PROTOCOL_PATH = V1_CAL_ROOT / "EXPERIMENT_PROTOCOL.json"
V1_STATUS_PATH = V1_CAL_ROOT / "TRAINING_STATUS.json"
V1_GPU_MANIFEST_PATH = V1_CAL_ROOT / "GPU_EXECUTION_MANIFEST.json"
V1_FAILURE_PATH = V1_CAL_ROOT / "failure_manifests" / "cal-v1-focal-loss-outer1-inner1-seed2718.json"
V1_FIRST_SPEC_PATH = V1_CAL_ROOT / "cell_specs" / "01_cal-v1-focal-loss-outer1-inner1-seed2718.json"
V1_FIRST_RUN = V1_CAL_ROOT / "runs" / "focal_loss" / "outer_1" / "inner_1" / "attempt_cal-v1-focal-loss-outer1-inner1-seed2718"
V1_FIRST_AUDIT = V1_CAL_ROOT / "audits" / "cal-v1-focal-loss-outer1-inner1-seed2718.json"
V1_PROTOCOL_SHA256 = "7533462d4958da0f1ee02dd5283b653c17de89de6214e491a391e6b49f8f7eee"
V1_SOURCE_SHA256 = "4c173afdbb648f0a72623abebe06255f9c37e58dae81641650d784d818a94074"
LEGACY_FOLD_HASH = "684cac99acb8eca3d34698265f23a9f22e879de9f1ec25280e02cf61bd87643e"
CUSTOM_SPLIT_HASH = "df7bcbb6af0d6e781f8a55f0101c8820366f3435718e3180fdd12aa501d61b38"
SEED = 2718
METHODS = v1.METHODS


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def _inner_splits() -> list[dict[str, Any]]:
    return v1._inner_splits()


def _frozen_split(outer: int, inner: int) -> dict[str, Any]:
    matches = [
        item
        for item in _inner_splits()
        if int(item["outer_fold"]) == outer and int(item["inner_fold"]) == inner
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one frozen split for outer={outer}, inner={inner}")
    return matches[0]


def _dataset() -> tuple[pd.DataFrame, dict[str, Any], Path, dict[str, Any]]:
    parent = read_json(v1.V1_PROTOCOL)
    config = load_config(parent["config_path"])
    frame, manifest, dataset_dir = load_effective_dataset(config)
    frame = frame.copy()
    frame["record_id"] = frame["record_id"].astype(str)
    frame["fold"] = frame["fold"].astype(int)
    frame["label"] = frame["label"].astype(int)
    if (
        manifest["effective_dataset_sha256"] != v1.EFFECTIVE_SHA256
        or sha256_file(resolve_path(config["data"]["source_csv"])) != v1.SOURCE_SHA256
        or len(frame) != v1.EXPECTED_ROWS
        or int(frame["label"].sum()) != v1.EXPECTED_POSITIVES
    ):
        raise RuntimeError("Authoritative development cohort changed")
    return frame, manifest, dataset_dir, config


def _v1_preserved_files() -> dict[str, str]:
    files: set[Path] = {
        V1_PROTOCOL_PATH,
        V1_STATUS_PATH,
        V1_GPU_MANIFEST_PATH,
        V1_FAILURE_PATH,
        V1_FIRST_SPEC_PATH,
        V1_FIRST_AUDIT,
        V1_FIRST_RUN / "run_manifest.json",
        V1_FIRST_RUN / "run_config.json",
        V1_FIRST_RUN / "predictions" / "held_out_predictions.csv",
        V1_FIRST_RUN / "history" / "training_history.csv",
        V1_CAL_ROOT / "logs" / "cal-v1-focal-loss-outer1-inner1-seed2718.stdout.log",
        V1_CAL_ROOT / "logs" / "cal-v1-focal-loss-outer1-inner1-seed2718.stderr.log",
        V1_CAL_ROOT / "BLOCKER_REPORT.md",
    }
    for path in V1_FIRST_RUN.rglob("*"):
        if path.is_file():
            files.add(path)
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing V1 evidence: {missing[0]}")
    return {str(path.resolve()): sha256_file(path) for path in sorted(files)}


def _verify_v1_preserved(expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Preserved V1 artifact changed: {path}")


_FROZEN_SPLIT_AUDIT: dict[str, Any] | None = None


def _audit_exclusions(audit: dict[str, Any], inner: int) -> list[dict[str, Any]]:
    excluded: dict[str, set[str]] = {}
    sources = (
        ("exact_target_duplicates", "exact_target_of_evaluation"),
        ("near_cross_fold_targets", "near_target_of_evaluation"),
        ("cross_fold_lineage_conflicts", "shared_lineage_with_evaluation"),
        ("cross_fold_bcp14_skeleton_conflicts", "bcp14_skeleton_of_evaluation"),
    )
    for source, reason in sources:
        for pair in audit[source]:
            left_fold, right_fold = int(pair["left_fold"]), int(pair["right_fold"])
            if left_fold == inner and right_fold != inner:
                excluded.setdefault(str(pair["right_record_id"]), set()).add(
                    f"{reason}:{pair['left_record_id']}"
                )
            if right_fold == inner and left_fold != inner:
                excluded.setdefault(str(pair["left_record_id"]), set()).add(
                    f"{reason}:{pair['right_record_id']}"
                )
    return [
        {"record_id": record_id, "reasons": sorted(reasons)}
        for record_id, reasons in sorted(excluded.items())
    ]


def _audit_context_sanitizations(
    audit: dict[str, Any], inner: int
) -> list[dict[str, Any]]:
    sanitizations: dict[tuple[str, str, str], set[str]] = {}
    for item in audit["cross_fold_target_context_overlaps"]:
        context_fold, target_fold = int(item["context_fold"]), int(item["target_fold"])
        if target_fold == inner:
            key = (str(item["context_record_id"]), str(item["context_field"]), "fitting")
            sanitizations.setdefault(key, set()).add(
                f"matches_evaluation_target:{item['target_record_id']}"
            )
        if context_fold == inner:
            key = (str(item["context_record_id"]), str(item["context_field"]), "evaluation")
            sanitizations.setdefault(key, set()).add(
                f"matches_fitting_target:{item['target_record_id']}"
            )
    return [
        {
            "record_id": key[0],
            "field": key[1],
            "role": key[2],
            "reasons": sorted(reasons),
        }
        for key, reasons in sorted(sanitizations.items())
    ]


def _verify_reconstructed_splits(frame: pd.DataFrame) -> dict[str, Any]:
    """Semantically validate the immutable splits against their source records.

    V1 froze context sanitizations before a later infrastructure cleanup stopped
    listing sanitizations for fitting candidates that had already been excluded.
    Those entries cannot affect target-only training.  We therefore reconstruct
    the original label-blind evidence rather than require byte equality with the
    current manifest builder.
    """

    global _FROZEN_SPLIT_AUDIT
    if _FROZEN_SPLIT_AUDIT is not None:
        return _FROZEN_SPLIT_AUDIT
    frozen = _inner_splits()
    if len(frozen) != 15:
        raise RuntimeError("Expected exactly 15 immutable inner splits")
    indexed = frame.set_index("record_id")
    summaries: dict[str, Any] = {}
    blocking_count_names = (
        "exact_target_duplicates_cross_fold",
        "near_cross_fold_targets",
        "cross_fold_lineage_conflicts",
        "cross_fold_bcp14_skeleton_conflicts",
        "cross_fold_document_overlaps",
        "families_assigned_to_multiple_folds",
    )
    for outer in range(1, 6):
        frozen_outer = sorted(
            (item for item in frozen if int(item["outer_fold"]) == outer),
            key=lambda item: int(item["inner_fold"]),
        )
        if [int(item["inner_fold"]) for item in frozen_outer] != [1, 2, 3]:
            raise RuntimeError(f"Outer {outer} does not have inner folds 1-3")
        outer_fit = frame.loc[frame["fold"] != outer].copy().reset_index(drop=True)
        outer_ids = set(outer_fit["record_id"])
        scoring_ids = [
            str(record_id)
            for item in frozen_outer
            for record_id in item["scoring_record_ids"]
        ]
        if len(scoring_ids) != len(set(scoring_ids)) or set(scoring_ids) != outer_ids:
            raise RuntimeError(f"Inner scoring coverage changed for outer {outer}")
        record_to_inner = {
            str(record_id): int(item["inner_fold"])
            for item in frozen_outer
            for record_id in item["scoring_record_ids"]
        }
        family_inner_counts = (
            outer_fit.assign(
                _inner=outer_fit["record_id"].astype(str).map(record_to_inner)
            )
            .groupby("family_id")["_inner"]
            .nunique()
        )
        if (family_inner_counts != 1).any():
            raise RuntimeError(f"A family crosses inner folds for outer {outer}")
        inner_frame = outer_fit.copy()
        inner_frame["fold"] = inner_frame["record_id"].astype(str).map(record_to_inner)
        audit = audit_leakage(inner_frame, near_threshold=0.85)
        if audit["training_blocked"]:
            raise RuntimeError(f"Leakage audit blocked frozen outer {outer}")

        for split in frozen_outer:
            inner = int(split["inner_fold"])
            score = set(split["scoring_record_ids"])
            candidate = outer_ids - score
            exclusions = _audit_exclusions(audit, inner)
            excluded = {item["record_id"] for item in exclusions}
            fitting = set(split["fitting_record_ids"])
            if stable_json(exclusions) != stable_json(split["fitting_exclusions"]):
                raise RuntimeError(
                    f"Lineage/duplicate exclusions changed for outer {outer}, inner {inner}"
                )
            if fitting != candidate - excluded or fitting & score:
                raise RuntimeError(
                    f"Frozen membership changed for outer {outer}, inner {inner}"
                )
            sanitizations = _audit_context_sanitizations(audit, inner)
            if stable_json(sanitizations) != stable_json(split["context_sanitizations"]):
                raise RuntimeError(
                    f"Context isolation evidence changed for outer {outer}, inner {inner}"
                )
            fitting_families = set(indexed.loc[list(fitting), "family_id"].astype(str))
            candidate_families = set(indexed.loc[list(candidate), "family_id"].astype(str))
            scoring_families = set(indexed.loc[list(score), "family_id"].astype(str))
            if fitting_families & scoring_families:
                raise RuntimeError(
                    f"Family leakage in outer {outer}, inner {inner}"
                )
            if (
                candidate_families != set(split["fitting_families"])
                or scoring_families != set(split["held_out_families"])
            ):
                raise RuntimeError(
                    f"Frozen family identities changed for outer {outer}, inner {inner}"
                )
            counts = split["counts"]
            expected_counts = {
                "candidate_fitting_rows": len(candidate),
                "excluded_fitting_rows": len(excluded),
                "fitting_rows": len(fitting),
                "fitting_positive_labels": int(indexed.loc[list(fitting), "label"].sum()),
                "held_out_rows": len(score),
                "held_out_positive_labels": int(indexed.loc[list(score), "label"].sum()),
                "scoring_rows": len(score),
                "context_fields_sanitized": len(sanitizations),
            }
            if (
                counts != expected_counts
                or int(split["fitting_positive_labels"])
                != expected_counts["fitting_positive_labels"]
                or int(split["scoring_positive_labels"])
                != expected_counts["held_out_positive_labels"]
                or int(split["scoring_negative_labels"])
                != len(score) - expected_counts["held_out_positive_labels"]
                or split["leakage_audit_counts"] != audit["counts"]
            ):
                raise RuntimeError(
                    f"Frozen row counts changed for outer {outer}, inner {inner}"
                )
            final = frame.loc[frame["record_id"].isin(fitting | score)].copy()
            final["fold"] = final["record_id"].map(
                {**{value: 1 for value in fitting}, **{value: 2 for value in score}}
            )
            final_audit = audit_leakage(final.reset_index(drop=True), near_threshold=0.85)
            if any(final_audit["counts"][name] for name in blocking_count_names):
                raise RuntimeError(
                    f"Residual target leakage in outer {outer}, inner {inner}"
                )
            summaries[f"outer_{outer}/inner_{inner}"] = {
                "fitting_rows": len(fitting),
                "scoring_rows": len(score),
                "excluded_fitting_rows": len(excluded),
                "context_fields_documented": len(sanitizations),
                "residual_target_leakage_counts": {
                    name: final_audit["counts"][name]
                    for name in blocking_count_names
                },
            }
    _FROZEN_SPLIT_AUDIT = summaries
    return summaries


def semantic_validate(
    spec: dict[str, Any],
    *,
    run_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """Validate legacy and custom hashes according to their actual semantics."""

    frame, dataset_manifest, dataset_dir, config = _dataset()
    outer = int(spec["outer_fold"])
    inner = int(spec["inner_fold"])
    frozen = _frozen_split(outer, inner)
    frozen_hash = sha256_text(stable_json(frozen))
    if spec["fold_manifest"] != frozen or spec["fold_manifest_sha256"] != frozen_hash:
        raise RuntimeError("Custom inner-split specification differs from frozen assignment")
    run = (run_path or Path(spec["run_path"])).resolve()
    audit_file = (audit_path or Path(spec["audit_path"])).resolve()
    manifest = read_json(run / "run_manifest.json")
    run_config = read_json(run / "run_config.json")
    audit = read_json(audit_file)

    # The generic trainer always hashes dataset/folds/fold_{train_fold}.json.
    legacy_fold_file = dataset_dir / "folds" / f"fold_{inner}.json"
    legacy_hash = sha256_file(legacy_fold_file)
    if manifest.get("fold_manifest_sha256") != legacy_hash:
        raise RuntimeError("Legacy fold-file hash does not identify its exact dataset fold file")
    if audit.get("custom_split_manifest_sha256") != frozen_hash:
        raise RuntimeError("Treatment audit does not identify the immutable custom inner split")

    expected_fit = set(frozen["fitting_record_ids"])
    expected_score = set(frozen["scoring_record_ids"])
    if (
        expected_fit != set(spec["fitting_record_ids"])
        or expected_score != set(spec["scoring_record_ids"])
        or expected_fit & expected_score
        or int(manifest.get("fitting_rows", -1)) != len(expected_fit)
        or int(manifest.get("held_out_rows", -1)) != len(expected_score)
        or audit.get("custom_fitting_rows") != len(expected_fit)
        or audit.get("custom_scoring_rows") != len(expected_score)
    ):
        raise RuntimeError("Fitting/scoring identities or row counts changed")

    indexed = frame.set_index("record_id")
    fitting_families = set(indexed.loc[list(expected_fit), "family_id"].astype(str))
    scoring_families = set(indexed.loc[list(expected_score), "family_id"].astype(str))
    if fitting_families & scoring_families:
        raise RuntimeError("Inner fitting and scoring families overlap")
    if fitting_families != set(frozen["fitting_families"]) or scoring_families != set(frozen["held_out_families"]):
        raise RuntimeError("Inner family identities changed")
    if any(
        marker in family.casefold()
        for family in fitting_families | scoring_families
        for marker in v1.RESERVED_FAMILY_MARKERS
    ):
        raise RuntimeError("Reserved TCP/DNS family entered calibration")

    split_audit = _verify_reconstructed_splits(frame)
    split_key = f"outer_{outer}/inner_{inner}"
    if split_key not in split_audit:
        raise RuntimeError("Frozen split has no semantic leakage validation")

    predictions = pd.read_csv(
        run / "predictions" / "held_out_predictions.csv",
        dtype={"record_id": str},
        keep_default_na=False,
    )
    if (
        len(predictions) != len(expected_score)
        or predictions["record_id"].duplicated().any()
        or set(predictions["record_id"]) != expected_score
    ):
        raise RuntimeError("Prediction identities are duplicate, missing, or unexpected")
    aligned = indexed.loc[predictions["record_id"]]
    if (
        not np.array_equal(predictions["label"].astype(int), aligned["label"])
        or not np.array_equal(predictions["family_id"].astype(str), aligned["family_id"].astype(str))
        or set(predictions["label"].astype(int).unique()) - {0, 1}
    ):
        raise RuntimeError("Prediction label convention or family identity changed")
    scores = pd.to_numeric(predictions["probability_yes"], errors="raise").to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError("Prediction scores are non-finite or outside [0,1]")

    parent = read_json(v1.V1_PROTOCOL)
    calibration_parent = read_json(V1_PROTOCOL_PATH)
    expected_loss = "focal_gamma_2" if spec["analysis_method"] == "focal_loss" else "weighted_ce"
    expected_full = spec["analysis_method"] == "focal_loss"
    if (
        manifest.get("condition") != "A2"
        or int(manifest.get("fold", 0)) != inner
        or int(manifest.get("seed", 0)) != SEED
        or manifest.get("loss") != expected_loss
        or manifest.get("precision") != "fp32"
        or manifest.get("input_mode") != "target_only"
        or manifest.get("epochs_completed") != 5
        or manifest.get("resume_checkpoint") is not None
        or manifest.get("fresh_base_initialization") is not True
        or manifest.get("full_encoder_fine_tuned") is not expected_full
        or manifest.get("effective_dataset_sha256") != v1.EFFECTIVE_SHA256
        or manifest.get("original_dataset_sha256") != v1.SOURCE_SHA256
        or manifest.get("config_sha256") != parent["config_sha256"]
        or manifest.get("dataset_manifest_sha256") != parent["dataset"]["dataset_manifest_sha256"]
        or manifest.get("model_source_sha256") != parent["model"]["source_sha256"]
        or manifest.get("model_file_hashes") != parent["model"]["model_file_hashes"]
        or manifest.get("source_file_hashes")
        != calibration_parent["source_snapshot"]["core_training_files"]
    ):
        raise RuntimeError("Model configuration or frozen provenance changed")
    if spec["analysis_method"] == "focal_loss" and (
        manifest.get("focal_gamma") != 2.0
        or manifest.get("focal_alpha") != "existing fitting-side class weights"
    ):
        raise RuntimeError("Focal Loss configuration changed")
    if spec["analysis_method"] == "partial_4" and (
        manifest.get("fine_tuning_policy") != "partial_4"
        or manifest.get("trainable_last_layers") != 4
        or audit.get("frozen_parameters_unchanged") is not True
        or audit.get("frozen_parameters_have_no_gradients") is not True
        or audit.get("trainable_parameters_finite") is not True
    ):
        raise RuntimeError("Partial-4 freeze policy validation failed")
    is_v1_reuse = spec.get("reuse_by_reference") is True or spec.get(
        "study_id"
    ) == "method_specific_calibration_v1"
    if is_v1_reuse:
        if (
            manifest.get("calibration_study_id")
            != "method_specific_calibration_v1"
            or manifest.get("calibration_attempt_id")
            != "cal-v1-focal-loss-outer1-inner1-seed2718"
        ):
            raise RuntimeError("Reused artifact lacks its original V1 identity")
    elif (
        manifest.get("calibration_study_id")
        != "method_specific_calibration_v1_1"
        or manifest.get("calibration_protocol_sha256")
        != spec["protocol_sha256"]
        or manifest.get("calibration_cell_id") != spec["cell_id"]
        or manifest.get("calibration_attempt_id") != spec["attempt_id"]
        or manifest.get("calibration_outer_fold") != outer
        or manifest.get("calibration_inner_fold") != inner
        or manifest.get("calibration_automatic_retry") is not False
        or manifest.get("calibration_reuse_by_reference") is not False
        or manifest.get("compatibility_fix_version") != "V1.1"
    ):
        raise RuntimeError("New V1.1 execution identity or provenance changed")
    resolved = run_config.get("resolved", {})
    if resolved.get("loss") != expected_loss or resolved.get("precision") != "fp32" or resolved.get("input_mode") != "target_only":
        raise RuntimeError("Resolved training configuration changed")
    if sha256_file(resolve_path(config["data"]["source_csv"])) != dataset_manifest["original_dataset_sha256"]:
        raise RuntimeError("Source CSV no longer matches dataset manifest")
    model_hash, model_files = hash_directory(Path(parent["model"]["local_path"]))
    if model_hash != parent["model"]["source_sha256"] or model_files != parent["model"]["model_file_hashes"]:
        raise RuntimeError("Local pretrained model changed")
    for artifact in manifest.get("artifact_hashes", {}).values():
        path = (run / artifact["path"]).resolve()
        if run not in path.parents or not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError("Saved run artifact hash validation failed")
    return {
        "legacy_fold_file": str(legacy_fold_file.resolve()),
        "legacy_fold_file_sha256": legacy_hash,
        "custom_inner_split_sha256": frozen_hash,
        "fitting_rows": len(expected_fit),
        "scoring_rows": len(expected_score),
        "prediction_rows": len(predictions),
        "fitting_families": sorted(fitting_families),
        "scoring_families": sorted(scoring_families),
        "leakage_audit_counts": frozen["leakage_audit_counts"],
        "fitting_exclusions": frozen["fitting_exclusions"],
        "context_sanitizations": frozen["context_sanitizations"],
        "run_manifest_sha256": sha256_file(run / "run_manifest.json"),
        "prediction_sha256": sha256_file(run / "predictions" / "held_out_predictions.csv"),
        "audit_sha256": sha256_file(audit_file),
        "runtime_seconds": float(manifest["runtime_seconds"]),
        "peak_gpu_memory_allocated_mb": float(manifest["peak_gpu_memory_allocated_mb"]),
        "peak_gpu_memory_reserved_mb": float(manifest["peak_gpu_memory_reserved_mb"]),
    }


def _source_snapshot() -> dict[str, Any]:
    return {
        "v1_1_files": {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in sorted(ROOT.glob("*.py"))
        },
        "core_training_files": source_file_hashes(),
        "v1_calibration_protocol_sha256": sha256_file(V1_PROTOCOL_PATH),
        "v1_calibration_source_snapshot_sha256": read_json(V1_PROTOCOL_PATH)[
            "source_snapshot_sha256"
        ],
        "v1_first_spec_sha256": sha256_file(V1_FIRST_SPEC_PATH),
        "v1_first_run_manifest_sha256": sha256_file(
            V1_FIRST_RUN / "run_manifest.json"
        ),
        "v1_first_audit_sha256": sha256_file(V1_FIRST_AUDIT),
        "parent_inner_splits_sha256": sha256_file(v1.INNER_SPLITS),
        "parent_v1_cell_launcher_sha256": sha256_file(
            v1.V1_ROOT / "cell_launcher.py"
        ),
    }


def _protocol_payload(reuse: dict[str, Any]) -> dict[str, Any]:
    parent = read_json(V1_PROTOCOL_PATH)
    source_snapshot = _source_snapshot()
    return {
        "schema_version": 1,
        "study_id": "method_specific_calibration_v1_1",
        "created_at": _now(),
        "correction_scope": "compatibility validation and execution infrastructure only",
        "parent_v1": {
            "path": str(V1_CAL_ROOT.resolve()),
            "protocol_sha256": V1_PROTOCOL_SHA256,
            "source_snapshot_sha256": V1_SOURCE_SHA256,
            "preserved_failed_attempt": "cal-v1-focal-loss-outer1-inner1-seed2718",
        },
        "compatibility_correction": {
            "legacy_hash_semantics": "SHA-256 of canonical dataset/folds/fold_{train_fold}.json written by generic trainer",
            "custom_hash_semantics": "SHA-256 of stable JSON for the injected outer-specific inner-fold manifest",
            "legacy_hash_validated_against_legacy_file": True,
            "custom_hash_validated_against_frozen_split": True,
            "hash_assertion_removed": False,
            "model_or_method_change": False,
            "threshold_or_evaluation_change": False,
        },
        "dataset": parent["dataset"],
        "model": parent["model"],
        "training": parent["training"],
        "methods": parent["methods"],
        "inner_cv": parent["inner_cv"],
        "threshold_selection": parent["threshold_selection"],
        "review_budget": parent["review_budget"],
        "ensemble": parent["ensemble"],
        "execution": {
            **parent["execution"],
            "logical_cells": 30,
            "reused_cells": 1,
            "new_gpu_cells": 29,
            "remaining_focal_loss": 14,
            "remaining_partial_4": 15,
        },
        "first_cell_reuse": reuse,
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": sha256_text(stable_json(source_snapshot)),
    }


def _new_specs(protocol_hash: str) -> list[dict[str, Any]]:
    parent_spec = read_json(V1_FIRST_SPEC_PATH)
    specs: list[dict[str, Any]] = []
    sequence = 1
    for method in METHODS:
        for split in _inner_splits():
            outer = int(split["outer_fold"])
            inner = int(split["inner_fold"])
            is_reuse = method == "focal_loss" and outer == 1 and inner == 1
            if is_reuse:
                attempt = "compat-v1_1-reuse-cal-v1-focal-loss-outer1-inner1-seed2718"
                run_path = str(V1_FIRST_RUN.resolve())
                audit_path = str(V1_FIRST_AUDIT.resolve())
            else:
                attempt = (
                    f"cal-v1_1-{method.replace('_', '-')}-"
                    f"outer{outer}-inner{inner}-seed{SEED}"
                )
                run_path = str(
                    (
                        RUN_DIR
                        / method
                        / f"outer_{outer}"
                        / f"inner_{inner}"
                        / f"attempt_{attempt}"
                    ).resolve()
                )
                audit_path = str((AUDIT_DIR / f"{attempt}.json").resolve())
            specs.append(
                {
                    "schema_version": 1,
                    "study_id": "method_specific_calibration_v1_1",
                    "protocol_sha256": protocol_hash,
                    "cell_id": f"{method}/outer_{outer}/inner_{inner}/seed_{SEED}",
                    "attempt_id": attempt,
                    "method": v1.METHOD_TO_LAUNCHER[method],
                    "analysis_method": method,
                    "outer_fold": outer,
                    "inner_fold": inner,
                    "train_fold": inner,
                    "seed": SEED,
                    "config_path": parent_spec["config_path"],
                    "model_path": parent_spec["model_path"],
                    "input_mode": "target_only",
                    "fold_manifest": split,
                    "fold_manifest_sha256": sha256_text(stable_json(split)),
                    "fitting_record_ids": split["fitting_record_ids"],
                    "scoring_record_ids": split["scoring_record_ids"],
                    "expected_score_ids": split["scoring_record_ids"],
                    "run_path": run_path,
                    "audit_path": audit_path,
                    "log_base": (
                        str((LOG_DIR / attempt).resolve()) if not is_reuse else None
                    ),
                    "automatic_retry": False,
                    "fresh_base_initialization": True,
                    "reuse_by_reference": is_reuse,
                    "historical_v1_attempt_id": (
                        "cal-v1-focal-loss-outer1-inner1-seed2718"
                        if is_reuse
                        else None
                    ),
                    "sequence": sequence,
                }
            )
            sequence += 1
    return specs


def prepare() -> dict[str, Any]:
    outputs = (
        PROTOCOL_PATH,
        PROVENANCE_PATH,
        REUSE_AUDIT_PATH,
        STATUS_PATH,
        GPU_MANIFEST_PATH,
    )
    if any(path.exists() for path in outputs):
        raise FileExistsError("V1.1 preparation artifacts already exist")
    if sha256_file(V1_PROTOCOL_PATH) != V1_PROTOCOL_SHA256:
        raise RuntimeError("V1 protocol changed")
    if read_json(V1_PROTOCOL_PATH)["source_snapshot_sha256"] != V1_SOURCE_SHA256:
        raise RuntimeError("V1 source snapshot identity changed")
    preserved = _v1_preserved_files()
    frame, _, _, _ = _dataset()
    _verify_reconstructed_splits(frame)
    original_spec = read_json(V1_FIRST_SPEC_PATH)
    reuse_validation = semantic_validate(
        original_spec, run_path=V1_FIRST_RUN, audit_path=V1_FIRST_AUDIT
    )
    if (
        reuse_validation["legacy_fold_file_sha256"] != LEGACY_FOLD_HASH
        or reuse_validation["custom_inner_split_sha256"] != CUSTOM_SPLIT_HASH
        or reuse_validation["fitting_rows"] != 910
        or reuse_validation["scoring_rows"] != 439
        or reuse_validation["prediction_rows"] != 439
    ):
        raise RuntimeError("First V1 cell evidence differs from reported identity")
    v1_failure = read_json(V1_FAILURE_PATH)
    if (
        v1_failure.get("status") != "FAILED_VALIDATION"
        or v1_failure.get("attempt_id")
        != "cal-v1-focal-loss-outer1-inner1-seed2718"
        or read_json(V1_STATUS_PATH)["cells"][original_spec["cell_id"]]["status"]
        != "FAILED_VALIDATION"
    ):
        raise RuntimeError("Original V1 failure status changed")
    reuse = {
        "logical_cell_id": original_spec["cell_id"],
        "historical_attempt_id": original_spec["attempt_id"],
        "historical_spec_path": str(V1_FIRST_SPEC_PATH.resolve()),
        "historical_spec_sha256": sha256_file(V1_FIRST_SPEC_PATH),
        "historical_run_path": str(V1_FIRST_RUN.resolve()),
        "historical_failure_manifest": str(V1_FAILURE_PATH.resolve()),
        "training_state": "TRAINING_COMPLETED",
        "v1_validation_state": "V1_VALIDATION_FAILED",
        "v1_1_state": "V1_1_COMPATIBILITY_VALIDATED",
        "new_gpu_execution_fabricated": False,
        **reuse_validation,
    }
    protocol = _protocol_payload(reuse)
    _write_json_exclusive(PROTOCOL_PATH, protocol)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    specs = _new_specs(protocol_hash)
    SPEC_DIR.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    reuse_spec_path: Path | None = None
    for spec in specs:
        path = SPEC_DIR / f"{spec['sequence']:02d}_{spec['attempt_id']}.json"
        _write_json_exclusive(path, spec)
        if spec["reuse_by_reference"]:
            reuse_spec_path = path
        records[spec["cell_id"]] = {
            "sequence": spec["sequence"],
            "attempt_id": spec["attempt_id"],
            "spec_path": str(path.resolve()),
            "spec_sha256": sha256_file(path),
            "status": (
                "COMPLETE" if spec["reuse_by_reference"] else "PLANNED"
            ),
            "reuse_by_reference": spec["reuse_by_reference"],
        }
    if reuse_spec_path is None:
        raise RuntimeError("V1.1 reuse specification was not generated")
    reuse_spec = read_json(reuse_spec_path)
    reuse_check = semantic_validate(
        reuse_spec, run_path=V1_FIRST_RUN, audit_path=V1_FIRST_AUDIT
    )
    compatibility = {
        "schema_version": 1,
        "created_at": _now(),
        "protocol_sha256": protocol_hash,
        "historical_v1_artifact_hashes": preserved,
        "reuse": reuse,
        "v1_1_spec_path": str(reuse_spec_path.resolve()),
        "v1_1_spec_sha256": sha256_file(reuse_spec_path),
        "v1_1_semantic_validation": reuse_check,
    }
    _write_json_exclusive(REUSE_AUDIT_PATH, compatibility)
    compatibility_path = COMPATIBILITY_DIR / "first_cell_v1_1.json"
    _write_json_exclusive(
        compatibility_path,
        {
            "cell_id": reuse_spec["cell_id"],
            "historical_attempt_id": reuse_spec["historical_v1_attempt_id"],
            "training_state": "TRAINING_COMPLETED",
            "v1_validation_state": "V1_VALIDATION_FAILED",
            "v1_1_compatibility_state": "V1_1_COMPATIBILITY_VALIDATED",
            "retrained": False,
            "new_gpu_execution": False,
            "semantic_validation": reuse_check,
            "created_at": _now(),
        },
    )
    records[reuse_spec["cell_id"]].update(
        {
            "training_state": "TRAINING_COMPLETED",
            "v1_validation_state": "V1_VALIDATION_FAILED",
            "v1_1_compatibility_state": "V1_1_COMPATIBILITY_VALIDATED",
            "compatibility_manifest": str(compatibility_path.resolve()),
            "run_manifest_sha256": reuse_check["run_manifest_sha256"],
        }
    )
    _write_json_exclusive(
        STATUS_PATH,
        {
            "schema_version": 1,
            "study_id": "method_specific_calibration_v1_1",
            "protocol_sha256": protocol_hash,
            "updated_at": _now(),
            "automatic_retry": False,
            "stop_on_first_failure": True,
            "cells": records,
        },
    )
    _write_json_exclusive(
        GPU_MANIFEST_PATH,
        {
            "schema_version": 1,
            "protocol_sha256": protocol_hash,
            "logical_cells": 30,
            "planned_new_gpu_cells": 29,
            "reused_cells": 1,
            "completed_cells": 1,
            "failed_cells": 0,
            "skipped_cells": 0,
            "sequential": True,
            "automatic_retry": False,
            "created_at": _now(),
            "executions": [
                {
                    "cell_id": reuse_spec["cell_id"],
                    "attempt_id": reuse_spec["attempt_id"],
                    "historical_attempt_id": reuse_spec["historical_v1_attempt_id"],
                    "origin": "V1_IMMUTABLE_REUSE",
                    "status": "COMPLETE",
                    "reused": True,
                    "new_gpu_execution": False,
                    "training_runtime_seconds": reuse_check["runtime_seconds"],
                    "peak_gpu_memory_allocated_mb": reuse_check[
                        "peak_gpu_memory_allocated_mb"
                    ],
                    "peak_gpu_memory_reserved_mb": reuse_check[
                        "peak_gpu_memory_reserved_mb"
                    ],
                    "run_manifest_sha256": reuse_check["run_manifest_sha256"],
                }
            ],
        },
    )
    _write_json_exclusive(
        PROVENANCE_PATH,
        {
            "schema_version": 1,
            "created_at": _now(),
            "source_csv_sha256": v1.SOURCE_SHA256,
            "effective_dataset_sha256": v1.EFFECTIVE_SHA256,
            "v1_protocol_sha256": V1_PROTOCOL_SHA256,
            "v1_1_protocol_sha256": protocol_hash,
            "v1_preserved_artifact_count": len(preserved),
            "reserved_sets_consumed": False,
        },
    )
    _prepare_existing_cpu_outputs()
    _write_text_exclusive(ROOT / "COMPATIBILITY_FIX_REPORT.md", _compatibility_report(reuse))
    return {
        "status": "PASS",
        "protocol_sha256": protocol_hash,
        "reused_cells": 1,
        "remaining_gpu_cells": 29,
        "cuda_used": False,
    }


def _prepare_existing_cpu_outputs() -> None:
    effective, methods, _ = v1.load_existing()
    existing = v1._existing_analysis(effective, methods)
    joined, _summary = v1._complementarity(methods)
    exploratory = v1._exploratory_ensembles(joined)
    for name, frame in existing.items():
        v1._write_csv(ROOT / name, frame)
    v1._write_csv(ROOT / "PREDICTION_COMPLEMENTARITY.csv", joined)
    v1._write_csv(ROOT / "EXPLORATORY_ENSEMBLE.csv", exploratory)


def _compatibility_report(reuse: dict[str, Any]) -> str:
    return f"""# Method-Specific Calibration V1.1 — Compatibility Fix

## Scope

V1.1 changes only compatibility validation and execution infrastructure. Dataset
labels, cohorts, folds, model methods, hyperparameters, thresholds, and ensemble
methodology are unchanged.

## Hash semantics

- Legacy hash `{LEGACY_FOLD_HASH}` is the SHA-256 of the canonical dataset fold
  file loaded by the generic trainer for numeric `train_fold=1`.
- Custom hash `{CUSTOM_SPLIT_HASH}` is the SHA-256 of the stable JSON representation
  of the immutable outer-1/inner-1 fitting/scoring manifest.
- These values correctly identify different artifacts. Both are independently
  validated; no assertion was merely removed.

## Frozen split compatibility

All 15 frozen inner splits are checked against their authoritative records. The
check reconstructs family assignments, fitting exclusions, lineage and duplicate
evidence, context-overlap evidence, row and label counts, and complete one-time
OOF coverage. A later infrastructure cleanup omits context-sanitization entries
for fitting candidates that are already excluded; V1 retains those documented
historical entries. They cannot affect this study because every calibration cell
uses target-only input. No assignment or scientific setting was changed.

## First-cell disposition

- Historical attempt: `{reuse['historical_attempt_id']}`.
- Training: `TRAINING_COMPLETED`.
- Original V1 validation: `V1_VALIDATION_FAILED` (preserved).
- V1.1 compatibility: `V1_1_COMPATIBILITY_VALIDATED`.
- Retrained: no.
- New GPU execution fabricated: no.
- Fitting/scoring/prediction rows: {reuse['fitting_rows']}/{reuse['scoring_rows']}/{reuse['prediction_rows']}.
- Runtime: {reuse['runtime_seconds']:.4f} seconds.
- Peak allocated/reserved GPU memory: {reuse['peak_gpu_memory_allocated_mb']:.2f}/{reuse['peak_gpu_memory_reserved_mb']:.2f} MiB.

TCP and DNS remain sealed.
"""


def _verify_source_snapshot(protocol: dict[str, Any]) -> None:
    expected = protocol["source_snapshot"]
    current = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(ROOT.glob("*.py"))
    }
    if current != expected["v1_1_files"]:
        raise RuntimeError("Frozen V1.1 source files changed")
    if source_file_hashes() != expected["core_training_files"]:
        raise RuntimeError("Core transformer source changed")
    if (
        sha256_file(V1_PROTOCOL_PATH)
        != expected["v1_calibration_protocol_sha256"]
        or read_json(V1_PROTOCOL_PATH)["source_snapshot_sha256"]
        != expected["v1_calibration_source_snapshot_sha256"]
        or sha256_file(v1.INNER_SPLITS)
        != expected["parent_inner_splits_sha256"]
        or sha256_file(v1.V1_ROOT / "cell_launcher.py")
        != expected["parent_v1_cell_launcher_sha256"]
    ):
        raise RuntimeError("Parent protocol or launcher provenance changed")


def validate_pretraining() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    status = read_json(STATUS_PATH)
    if sha256_file(PROTOCOL_PATH) != status["protocol_sha256"]:
        raise RuntimeError("V1.1 protocol hash differs from training status")
    _verify_source_snapshot(protocol)
    reuse_audit = read_json(REUSE_AUDIT_PATH)
    _verify_v1_preserved(reuse_audit["historical_v1_artifact_hashes"])
    frame, _, _, _ = _dataset()
    _verify_reconstructed_splits(frame)
    attempts: set[str] = set()
    new_run_paths: set[str] = set()
    specs: list[dict[str, Any]] = []
    for record in sorted(status["cells"].values(), key=lambda value: value["sequence"]):
        spec_path = Path(record["spec_path"])
        if not spec_path.is_file() or sha256_file(spec_path) != record["spec_sha256"]:
            raise RuntimeError(f"V1.1 cell specification changed: {spec_path}")
        spec = read_json(spec_path)
        if spec["protocol_sha256"] != sha256_file(PROTOCOL_PATH):
            raise RuntimeError("Cell specification points to another protocol")
        if spec["attempt_id"] in attempts:
            raise RuntimeError("V1.1 attempt identity is not unique")
        attempts.add(spec["attempt_id"])
        if not spec["reuse_by_reference"]:
            if spec["run_path"] in new_run_paths:
                raise RuntimeError("V1.1 output path is not unique")
            if V1_CAL_ROOT.resolve() in Path(spec["run_path"]).resolve().parents:
                raise RuntimeError("New V1.1 output points into preserved V1")
            new_run_paths.add(spec["run_path"])
        specs.append(spec)
    if len(specs) != 30 or sum(bool(spec["reuse_by_reference"]) for spec in specs) != 1:
        raise RuntimeError("Expected 30 logical cells with exactly one immutable reuse")
    reuse_spec = next(spec for spec in specs if spec["reuse_by_reference"])
    semantic_validate(reuse_spec, run_path=V1_FIRST_RUN, audit_path=V1_FIRST_AUDIT)
    return {
        "status": "PASS",
        "logical_cells": 30,
        "reused_cells": 1,
        "new_gpu_cells": 29,
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "cuda_used": False,
    }


def _update_status(status: dict[str, Any]) -> None:
    status["updated_at"] = _now()
    write_json(STATUS_PATH, status)


def _update_gpu_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _now()
    write_json(GPU_MANIFEST_PATH, manifest)


def run_one(spec_path: Path) -> dict[str, Any]:
    validate_pretraining()
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    if spec["reuse_by_reference"]:
        raise RuntimeError("The reused first cell must never be launched")
    status = read_json(STATUS_PATH)
    record = status["cells"].get(spec["cell_id"])
    if (
        record is None
        or Path(record["spec_path"]).resolve() != spec_path
        or sha256_file(spec_path) != record["spec_sha256"]
    ):
        raise RuntimeError("Unrecognized V1.1 cell specification")
    if record["status"] != "PLANNED":
        raise RuntimeError(f"No-retry gate: cell status is {record['status']}")
    run = Path(spec["run_path"])
    stdout = Path(spec["log_base"] + ".stdout.log")
    stderr = Path(spec["log_base"] + ".stderr.log")
    completion = COMPLETION_DIR / f"{spec['attempt_id']}.json"
    failure = FAILURE_DIR / f"{spec['attempt_id']}.json"
    if any(
        path.exists()
        for path in (run, stdout, stderr, completion, failure, Path(spec["audit_path"]))
    ):
        raise FileExistsError("No-retry gate: an attempt artifact already exists")
    gpu = v1.gpu_preflight()
    record.update({"status": "IN_PROGRESS", "started_at": _now(), "gpu_preflight": gpu})
    _update_status(status)
    gpu_manifest = read_json(GPU_MANIFEST_PATH)
    execution = {
        "cell_id": spec["cell_id"],
        "attempt_id": spec["attempt_id"],
        "origin": "V1_1_NEW_GPU_EXECUTION",
        "reused": False,
        "started_at": record["started_at"],
        "gpu_preflight": gpu,
        "status": "IN_PROGRESS",
    }
    gpu_manifest["executions"].append(execution)
    _update_gpu_manifest(gpu_manifest)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    library_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
    environment["PATH"] = str(library_bin) + os.pathsep + environment.get("PATH", "")
    command = [
        sys.executable,
        "-m",
        "experiments.transformer_final.method_specific_calibration_v1_1.cell_launcher",
        "--spec",
        str(spec_path),
    ]
    started = time.monotonic()
    try:
        with stdout.open("x", encoding="utf-8") as out, stderr.open(
            "x", encoding="utf-8"
        ) as err:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=out,
                stderr=err,
                timeout=900,
                check=False,
            )
    except Exception as exc:
        elapsed = time.monotonic() - started
        record.update(
            {"status": "FAILED", "elapsed_seconds": elapsed, "failure": repr(exc)}
        )
        execution.update(
            {"status": "FAILED", "elapsed_seconds": elapsed, "failure": repr(exc)}
        )
        gpu_manifest["failed_cells"] += 1
        _update_status(status)
        _update_gpu_manifest(gpu_manifest)
        _write_json_exclusive(
            failure,
            {**execution, "automatic_retry": False, "preserved": True},
        )
        raise
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        record.update(
            {
                "status": "FAILED",
                "elapsed_seconds": elapsed,
                "return_code": result.returncode,
            }
        )
        execution.update(
            {
                "status": "FAILED",
                "elapsed_seconds": elapsed,
                "return_code": result.returncode,
            }
        )
        gpu_manifest["failed_cells"] += 1
        _update_status(status)
        _update_gpu_manifest(gpu_manifest)
        _write_json_exclusive(
            failure,
            {
                **execution,
                "stderr_path": str(stderr),
                "automatic_retry": False,
                "preserved": True,
            },
        )
        raise RuntimeError(
            f"V1.1 cell failed; no retry: {spec['cell_id']}; see {stderr}"
        )
    try:
        validated = semantic_validate(spec)
    except Exception as exc:
        record.update(
            {
                "status": "FAILED_VALIDATION",
                "elapsed_seconds": elapsed,
                "return_code": 0,
                "failure": repr(exc),
            }
        )
        execution.update(
            {
                "status": "FAILED_VALIDATION",
                "elapsed_seconds": elapsed,
                "failure": repr(exc),
            }
        )
        gpu_manifest["failed_cells"] += 1
        _update_status(status)
        _update_gpu_manifest(gpu_manifest)
        _write_json_exclusive(
            failure,
            {
                **execution,
                "stderr_path": str(stderr),
                "automatic_retry": False,
                "preserved": True,
            },
        )
        raise
    record.update(
        {
            "status": "COMPLETE",
            "completed_at": _now(),
            "elapsed_seconds": elapsed,
            "return_code": 0,
            "run_manifest_sha256": validated["run_manifest_sha256"],
        }
    )
    execution.update(
        {
            "status": "COMPLETE",
            "completed_at": record["completed_at"],
            "elapsed_seconds": elapsed,
            "training_runtime_seconds": validated["runtime_seconds"],
            "peak_gpu_memory_allocated_mb": validated[
                "peak_gpu_memory_allocated_mb"
            ],
            "peak_gpu_memory_reserved_mb": validated[
                "peak_gpu_memory_reserved_mb"
            ],
            "run_manifest_sha256": validated["run_manifest_sha256"],
        }
    )
    gpu_manifest["completed_cells"] += 1
    _update_status(status)
    _update_gpu_manifest(gpu_manifest)
    _write_json_exclusive(
        completion,
        {
            **execution,
            "spec_sha256": sha256_file(spec_path),
            "protocol_sha256": spec["protocol_sha256"],
            "semantic_validation": validated,
        },
    )
    return execution


def run_all() -> dict[str, Any]:
    validation = validate_pretraining()
    status = read_json(STATUS_PATH)
    for record in sorted(status["cells"].values(), key=lambda value: value["sequence"]):
        if record["status"] == "COMPLETE":
            continue
        if record["status"] != "PLANNED":
            raise RuntimeError(
                f"Sequence stopped at {record['status']}: {record['attempt_id']}"
            )
        run_one(Path(record["spec_path"]))
    return {
        **validation,
        "completed_cells": read_json(GPU_MANIFEST_PATH)["completed_cells"],
    }


def validate_completed(*, require_all: bool = False) -> dict[str, Any]:
    base = validate_pretraining()
    status = read_json(STATUS_PATH)
    counts: dict[str, int] = {}
    for record in status["cells"].values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        if record["status"] == "COMPLETE":
            spec = read_json(Path(record["spec_path"]))
            validated = semantic_validate(spec)
            if validated["run_manifest_sha256"] != record["run_manifest_sha256"]:
                raise RuntimeError(f"Completed run manifest changed: {spec['cell_id']}")
    if require_all and counts.get("COMPLETE", 0) != 30:
        raise RuntimeError(f"All 30 logical cells required; statuses={counts}")
    reuse = read_json(REUSE_AUDIT_PATH)
    _verify_v1_preserved(reuse["historical_v1_artifact_hashes"])
    return {
        **base,
        "cell_statuses": counts,
        "v1_historical_artifacts_unchanged": True,
    }


def _load_inner_predictions() -> tuple[
    dict[tuple[str, int], pd.DataFrame], pd.DataFrame
]:
    status = read_json(STATUS_PATH)
    grouped: dict[tuple[str, int], list[pd.DataFrame]] = {}
    rows: list[dict[str, Any]] = []
    for record in sorted(status["cells"].values(), key=lambda value: value["sequence"]):
        if record["status"] != "COMPLETE":
            continue
        spec = read_json(record["spec_path"])
        frame = pd.read_csv(
            Path(spec["run_path"]) / "predictions" / "held_out_predictions.csv",
            dtype={"record_id": str},
        )
        frame["outer_fold"] = int(spec["outer_fold"])
        frame["inner_fold"] = int(spec["inner_fold"])
        frame["analysis_method"] = spec["analysis_method"]
        frame["artifact_origin"] = (
            "V1_IMMUTABLE_REUSE"
            if spec["reuse_by_reference"]
            else "V1_1_NEW_GPU_EXECUTION"
        )
        grouped.setdefault(
            (spec["analysis_method"], int(spec["outer_fold"])), []
        ).append(frame)
        rows.append(
            v1.metric_row(
                frame,
                frame["probability_yes"].to_numpy(float) >= 0.5,
                method=spec["analysis_method"],
                outer_fold=int(spec["outer_fold"]),
                inner_fold=int(spec["inner_fold"]),
                operating_point="threshold_0.5",
                artifact_origin=frame["artifact_origin"].iloc[0],
            )
        )
    combined: dict[tuple[str, int], pd.DataFrame] = {}
    frozen = _inner_splits()
    for key, pieces in grouped.items():
        method, outer = key
        frame = pd.concat(pieces, ignore_index=True)
        expected = {
            record
            for item in frozen
            if int(item["outer_fold"]) == outer
            for record in item["scoring_record_ids"]
        }
        if (
            len(frame) != len(expected)
            or frame["record_id"].duplicated().any()
            or set(frame["record_id"]) != expected
        ):
            raise RuntimeError(
                f"V1.1 inner OOF reconstruction failed: {method} outer {outer}"
            )
        combined[key] = frame
    return combined, pd.DataFrame(rows)


def _write_reports_v1_1(
    final_metrics: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    top_k: pd.DataFrame,
    families: pd.DataFrame,
    uncertainty: pd.DataFrame,
    gpu_time: pd.DataFrame,
) -> None:
    del thresholds, families, uncertainty
    combined = pd.concat([final_metrics, ensemble_metrics], ignore_index=True, sort=False)
    existing = pd.read_csv(ROOT / "CURRENT_MODEL_COMPARISON.csv").set_index("method")
    focal_ap = float(existing.loc["focal_loss", "average_precision"])
    partial_ap = float(existing.loc["partial_4", "average_precision"])
    ensemble_scores = pd.read_csv(
        ROOT / "final_predictions" / "ensemble_ranking_scores.csv"
    )
    from sklearn.metrics import average_precision_score, roc_auc_score

    ensemble_ap = float(
        average_precision_score(
            ensemble_scores["label"], ensemble_scores["probability_yes"]
        )
    )
    ensemble_auc = float(
        roc_auc_score(
            ensemble_scores["label"], ensemble_scores["probability_yes"]
        )
    )
    total_time = float(gpu_time["training_runtime_seconds"].sum())
    new_time = float(
        gpu_time.loc[
            gpu_time["origin"] == "V1_1_NEW_GPU_EXECUTION",
            "training_runtime_seconds",
        ].sum()
    )
    peak = float(gpu_time["peak_gpu_memory_allocated_mb"].max())
    report = [
        "# Final Report — Method-Specific Calibration V1.1",
        "",
        "## Dataset and governance",
        "",
        (
            f"The unchanged development cohort contains {v1.EXPECTED_ROWS} clauses, "
            f"{v1.EXPECTED_POSITIVES} positive and {v1.EXPECTED_NEGATIVES} negative. "
            "All outer/inner family assignments and leakage exclusions are inherited "
            "unchanged. TCP and DNS were not consumed."
        ),
        "",
        "## Compatibility recovery",
        "",
        (
            "The first Focal outer-1/inner-1 cell was reused by immutable reference "
            "after semantic validation of both hash fields, record identities, family "
            "isolation, leakage exclusions, predictions, configuration, and provenance. "
            "Its original V1 `FAILED_VALIDATION` evidence remains unchanged. It was not retrained."
        ),
        "",
        "## Training execution",
        "",
        (
            f"Logical cells: 30; immutable V1 reuse: 1; new sequential V1.1 GPU cells: 29. "
            f"Total recorded training time including the reused cell: {total_time:.1f} seconds; "
            f"new V1.1 time: {new_time:.1f} seconds; peak allocated memory: {peak:.1f} MiB. "
            "Automatic retries: zero."
        ),
        "",
        "## Valid nested operating points",
        "",
        *v1._format_metric_table(combined),
        "",
        (
            "Precision targets are fitting-side objectives, not guarantees on unseen families. "
            "Infeasible objectives remain explicit and have no always-positive fallback."
        ),
        "",
        "## Ranking and ensemble",
        "",
        (
            f"Frozen outer prediction AP: Focal {focal_ap:.4f}, Partial-4 {partial_ap:.4f}. "
            f"The pre-registered equal-weight fitting-side ECDF ensemble achieved AP "
            f"{ensemble_ap:.4f} and ROC-AUC {ensemble_auc:.4f}. No ensemble weight was fit "
            "on outer labels."
        ),
        "",
        (
            "Top-K budgets in `TOP_K_COMPARISON.csv` are global pooled development OOF "
            "rankings with deterministic record-ID tie breaking."
        ),
        "",
        "## Cross-family robustness and uncertainty",
        "",
        (
            "Family metrics are reported in `FAMILY_ROBUSTNESS.csv`. Family-clustered "
            "bootstrap intervals are reported in `UNCERTAINTY_INTERVALS.csv`; only nine "
            "family clusters exist, and the intervals do not measure single-seed training variability."
        ),
        "",
        "## Annotation and deployment limitations",
        "",
        (
            "The labels are an operational AI-assisted reference, not an independent "
            "expert gold standard. Use the models as a ranked human-review assistant with "
            "an uncertain/review band; do not certify clauses automatically. A separately "
            "double-annotated expert sample and multi-seed confirmation remain necessary "
            "before TCP transfer validation and the one-time sealed DNS evaluation."
        ),
    ]
    _write_text_exclusive(ROOT / "FINAL_REPORT.md", "\n".join(report) + "\n")
    conclusions = [
        "# Scientific Conclusions",
        "",
        f"- Focal outer ranking AP: {focal_ap:.4f}.",
        f"- Partial-4 outer ranking AP: {partial_ap:.4f}.",
        f"- Pre-registered ECDF ensemble AP/AUC: {ensemble_ap:.4f}/{ensemble_auc:.4f}.",
        "- Method-specific thresholds correct the earlier cross-method threshold transfer.",
        "- Choose an operating policy by review capacity and false-positive tolerance, not ordinary accuracy.",
        "- Single-seed variability and AI-assisted annotation validity remain unresolved.",
    ]
    _write_text_exclusive(
        ROOT / "SCIENTIFIC_CONCLUSIONS.md", "\n".join(conclusions) + "\n"
    )
    validation = [
        "# Validation Report",
        "",
        "- V1 historical artifact immutability: PASS.",
        "- Legacy fold-file hash semantic validation: PASS.",
        "- Custom inner-split hash semantic validation: PASS.",
        "- Dataset, fold, family, lineage, near-duplicate, and context controls: PASS.",
        "- First cell immutable reuse: PASS; no retraining.",
        "- Logical inner cells validated: 30/30.",
        "- New sequential GPU cells validated: 29/29.",
        "- Automatic retries: 0.",
        "- TCP/DNS consumed: no.",
        f"- V1.1 protocol SHA-256: `{sha256_file(PROTOCOL_PATH)}`.",
    ]
    _write_text_exclusive(
        ROOT / "VALIDATION_REPORT.md", "\n".join(validation) + "\n"
    )


def analyze_final() -> dict[str, Any]:
    # Reuse the already tested V1 analysis algorithms with only their output
    # namespace and validation callbacks replaced. No scientific policy changes.
    replacements = {
        "ROOT": ROOT,
        "PROTOCOL_PATH": PROTOCOL_PATH,
        "STATUS_PATH": STATUS_PATH,
        "GPU_MANIFEST_PATH": GPU_MANIFEST_PATH,
        "SPEC_DIR": SPEC_DIR,
        "RUN_DIR": RUN_DIR,
        "LOG_DIR": LOG_DIR,
        "AUDIT_DIR": AUDIT_DIR,
        "COMPLETION_DIR": COMPLETION_DIR,
        "FAILURE_DIR": FAILURE_DIR,
        "validate_completed": validate_completed,
        "_load_inner_predictions": _load_inner_predictions,
        "_write_reports": _write_reports_v1_1,
    }
    original = {name: getattr(v1, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(v1, name, value)
        return v1.analyze_final()
    finally:
        for name, value in original.items():
            setattr(v1, name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("validate")
    one = commands.add_parser("run-one")
    one.add_argument("--spec", type=Path, required=True)
    commands.add_parser("run-all")
    commands.add_parser("analyze")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "validate":
        result = validate_completed(require_all=False)
    elif args.command == "run-one":
        result = run_one(args.spec)
    elif args.command == "run-all":
        result = run_all()
    else:
        result = analyze_final()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



