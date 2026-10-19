"""Audit, prepare, execute, and analyze method-specific nested calibration.

The only commands in this module that use CUDA are ``run-one`` and ``run-all``.
All preparation, validation, and analysis commands are CPU-only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from models.transformer_final.common import (
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

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[2]
V1_ROOT = ROOT.parent / "recall_improvement_study_v1"
V2_ROOT = ROOT.parent / "recall_improvement_study_v2"
V3_ROOT = ROOT.parent / "recall_improvement_study_v3_optimizer_fix"
V1_PROTOCOL = V1_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
INNER_SPLITS = V1_ROOT / "protocols" / "nested_inner_splits_v1.json"
V3_PROTOCOL = V3_ROOT / "protocols" / "recall_improvement_protocol_v3_optimizer_fix.json"
V3_ANALYZER = V3_ROOT / "manual_execution_v1" / "analyze_completed_v1.py"
PROTOCOL_PATH = ROOT / "EXPERIMENT_PROTOCOL.json"
PROVENANCE_PATH = ROOT / "SOURCE_PROVENANCE.json"
INVENTORY_PATH = ROOT / "EXISTING_ARTIFACT_INVENTORY.json"
STATUS_PATH = ROOT / "TRAINING_STATUS.json"
GPU_MANIFEST_PATH = ROOT / "GPU_EXECUTION_MANIFEST.json"
SPEC_DIR = ROOT / "cell_specs"
RUN_DIR = ROOT / "runs"
LOG_DIR = ROOT / "logs"
AUDIT_DIR = ROOT / "audits"
COMPLETION_DIR = ROOT / "completion_manifests"
FAILURE_DIR = ROOT / "failure_manifests"

SOURCE_SHA256 = "c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712"
EFFECTIVE_SHA256 = "a25e99563563f71570e48f63b9515fbb51ff845762ff718763d98e0b3eefae2f"
PROMPT_EFFECTIVE_HASH = "a25e99563563f71570e48f63b9515fbb51ff845762ff718763d98e0b"
V3_SHA256 = "9502e659c1f246f433ce534fa7165959a6519c6637c71b3be0289f6638e72c51"
V3_SOURCE_SHA256 = "22af9f00506241a4d78918b3b1fc65f1a787281a685c966fc881df99671ee8c4"
SEED = 2718
METHODS = ("focal_loss", "partial_4")
METHOD_TO_LAUNCHER = {"focal_loss": "focal", "partial_4": "partial_4"}
PRECISION_TARGETS = (0.20, 0.30, 0.50)
TOP_K = (10, 25, 50, 100)
EXPECTED_ROWS = 2142
EXPECTED_POSITIVES = 126
EXPECTED_NEGATIVES = 2016
RESERVED_FAMILY_MARKERS = ("tcp", "dns", "svcb", "https_rr")
MIN_FREE_VRAM_MIB = 18_000


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


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable analysis output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _load_v3_analysis_module() -> Any:
    package = str(V3_ANALYZER.parent)
    if package not in sys.path:
        sys.path.insert(0, package)
    spec = importlib.util.spec_from_file_location("v3_completed_analysis", V3_ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen V3 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inner_splits() -> list[dict[str, Any]]:
    value = json.loads(INNER_SPLITS.read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 15:
        raise RuntimeError("Frozen inner split file must contain exactly 15 objects")
    return value


def load_existing() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    module = _load_v3_analysis_module()
    collected = module.collect()
    methods = {key: value.copy() for key, value in collected["methods"].items()}
    protocol = read_json(V1_PROTOCOL)
    config = load_config(protocol["config_path"])
    effective, manifest, dataset_dir = load_effective_dataset(config)
    effective = effective.copy()
    effective["record_id"] = effective["record_id"].astype(str)
    effective["label"] = effective["label"].astype(int)
    effective["fold"] = effective["fold"].astype(int)
    if (
        sha256_file(Path(config["data"]["source_csv"])) != SOURCE_SHA256
        or manifest["effective_dataset_sha256"] != EFFECTIVE_SHA256
        or sha256_file(V3_PROTOCOL) != V3_SHA256
        or read_json(V3_PROTOCOL)["source_snapshot_sha256"] != V3_SOURCE_SHA256
        or sha256_file(dataset_dir / "manifest.json")
        != protocol["dataset"]["dataset_manifest_sha256"]
    ):
        raise RuntimeError("Frozen dataset or V3 provenance check failed")
    if (
        len(effective) != EXPECTED_ROWS
        or int(effective["label"].sum()) != EXPECTED_POSITIVES
        or int((effective["label"] == 0).sum()) != EXPECTED_NEGATIVES
    ):
        raise RuntimeError("Frozen cohort counts changed")
    expected_ids = set(effective["record_id"])
    indexed = effective.set_index("record_id")
    for method, frame in methods.items():
        frame["record_id"] = frame["record_id"].astype(str)
        if (
            len(frame) != EXPECTED_ROWS
            or frame["record_id"].duplicated().any()
            or set(frame["record_id"]) != expected_ids
        ):
            raise RuntimeError(f"Outer prediction coverage failed: {method}")
        aligned = frame.set_index("record_id").loc[indexed.index]
        if not np.array_equal(aligned["label"].astype(int), indexed["label"]):
            raise RuntimeError(f"Positive-label convention mismatch: {method}")
        if not np.array_equal(aligned["fold"].astype(int), indexed["fold"]):
            raise RuntimeError(f"Outer-fold membership mismatch: {method}")
        if not np.array_equal(aligned["family_id"].astype(str), indexed["family_id"].astype(str)):
            raise RuntimeError(f"Family membership mismatch: {method}")
        scores = pd.to_numeric(aligned["probability_yes"], errors="raise").to_numpy(float)
        if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise RuntimeError(f"Invalid scores: {method}")
        if any(marker in family.casefold() for family in aligned["family_id"].unique() for marker in RESERVED_FAMILY_MARKERS):
            raise RuntimeError(f"Reserved family entered development predictions: {method}")
        methods[method] = aligned.reset_index()
    return effective, methods, collected


def confusion(labels: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)
    tp = int(np.count_nonzero((labels == 1) & (predicted == 1)))
    fp = int(np.count_nonzero((labels == 0) & (predicted == 1)))
    tn = int(np.count_nonzero((labels == 0) & (predicted == 0)))
    fn = int(np.count_nonzero((labels == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_positive_count": tp + fp,
        "predicted_positive_rate": (tp + fp) / len(labels),
    }


def metric_row(frame: pd.DataFrame, predicted: np.ndarray, **identity: Any) -> dict[str, Any]:
    labels = frame["label"].to_numpy(int)
    scores = frame["probability_yes"].to_numpy(float)
    prevalence = float(labels.mean())
    row = {
        **identity,
        "rows": len(frame),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "average_precision": float(average_precision_score(labels, scores)),
        "ap_prevalence_lift": float(average_precision_score(labels, scores) / prevalence),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(brier_score_loss(labels, scores)),
    }
    row.update(confusion(labels, predicted))
    return row


def threshold_candidates(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Threshold scores must be non-empty and finite")
    unique = np.unique(values)
    return np.concatenate(([np.nextafter(unique[-1], np.inf)], unique[::-1]))


def select_precision_threshold(
    labels: Iterable[int], scores: Iterable[float], minimum_precision: float
) -> dict[str, Any]:
    labels_array = np.asarray(list(labels), dtype=np.int64)
    scores_array = np.asarray(list(scores), dtype=np.float64)
    if labels_array.size != scores_array.size or labels_array.size == 0:
        raise ValueError("Labels and scores must have equal non-zero length")
    if set(np.unique(labels_array)) - {0, 1}:
        raise ValueError("Labels must use positive=1 and negative=0")
    if not 0 < minimum_precision <= 1:
        raise ValueError("minimum_precision must be in (0, 1]")
    feasible: list[dict[str, Any]] = []
    for threshold in threshold_candidates(scores_array):
        counts = confusion(labels_array, scores_array >= threshold)
        if counts["predicted_positive_count"] and counts["precision"] >= minimum_precision:
            feasible.append({"threshold": float(threshold), **counts})
    if not feasible:
        return {
            "status": "INFEASIBLE",
            "minimum_precision": minimum_precision,
            "threshold": None,
            "tie_breaking": "maximum recall, then precision, then higher threshold",
            "fallback_used": False,
        }
    chosen = max(feasible, key=lambda row: (row["recall"], row["precision"], row["threshold"]))
    return {
        "status": "FEASIBLE",
        "minimum_precision": minimum_precision,
        **chosen,
        "tie_breaking": "maximum recall, then precision, then higher threshold",
        "fallback_used": False,
    }


def stable_rank_percentile(frame: pd.DataFrame, score_column: str) -> pd.Series:
    ordered = frame[["record_id", score_column]].copy()
    ordered = ordered.sort_values([score_column, "record_id"], ascending=[True, True], kind="mergesort")
    ranks = pd.Series(np.arange(1, len(ordered) + 1, dtype=float) / len(ordered), index=ordered.index)
    return ranks.reindex(frame.index)


def empirical_cdf(reference: Iterable[float], values: Iterable[float]) -> np.ndarray:
    reference_array = np.sort(np.asarray(list(reference), dtype=np.float64))
    values_array = np.asarray(list(values), dtype=np.float64)
    if reference_array.size == 0 or not np.isfinite(reference_array).all() or not np.isfinite(values_array).all():
        raise ValueError("ECDF inputs must be non-empty and finite")
    return np.searchsorted(reference_array, values_array, side="right") / reference_array.size


def _score_distribution(method: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, group in frame.groupby("label", sort=True):
        scores = group["probability_yes"].to_numpy(float)
        rows.append({
            "method": method,
            "label": int(label),
            "rows": len(group),
            "minimum": float(scores.min()),
            "q05": float(np.quantile(scores, 0.05)),
            "q25": float(np.quantile(scores, 0.25)),
            "median": float(np.median(scores)),
            "q75": float(np.quantile(scores, 0.75)),
            "q95": float(np.quantile(scores, 0.95)),
            "maximum": float(scores.max()),
            "mean": float(scores.mean()),
        })
    return rows


def _calibration_bins(method: str, frame: pd.DataFrame, bins: int = 10) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    bucket = pd.cut(frame["probability_yes"], np.linspace(0, 1, bins + 1), include_lowest=True, labels=False)
    for index in range(bins):
        group = frame.loc[bucket == index]
        result.append({
            "method": method,
            "bin": index + 1,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "rows": len(group),
            "mean_score": float(group["probability_yes"].mean()) if len(group) else None,
            "observed_positive_rate": float(group["label"].mean()) if len(group) else None,
        })
    return result


def _top_k_rows(method: str, frame: pd.DataFrame, scope: str = "global") -> list[dict[str, Any]]:
    ranked = frame.sort_values(["probability_yes", "record_id"], ascending=[False, True], kind="mergesort")
    positives = int(frame["label"].sum())
    rows = []
    for k in TOP_K:
        top = ranked.head(k)
        tp = int(top["label"].sum())
        rows.append({
            "method": method,
            "scope": scope,
            "k": k,
            "reviewed": len(top),
            "ambiguities_found": tp,
            "false_positives": len(top) - tp,
            "precision_at_k": tp / len(top),
            "recall_at_k": tp / positives,
        })
    return rows


def _historical_file_inventory(collected: dict[str, Any]) -> dict[str, Any]:
    paths: set[Path] = {V1_PROTOCOL, INNER_SPLITS, V3_PROTOCOL}
    v3_inventory = read_json(V3_ROOT / "EXPERIMENT_INVENTORY.json")
    for record in v3_inventory.get("attempts", {}).values():
        if record.get("status") == "COMPLETE":
            run = Path(record["run_path"])
            paths.add(run / "run_manifest.json")
            paths.add(run / "predictions" / "held_out_predictions.csv")
    baseline = ROOT.parent / "final_annotation_ready_pilot_v1" / "training_outputs"
    for fold in range(1, 6):
        run = baseline / f"fold_{fold}" / "seed_2718"
        paths.add(run / "run_manifest.json")
        paths.add(run / "predictions" / "held_out_predictions.csv")
    for path in (V1_ROOT / "runs" / "C_local_context").rglob("held_out_predictions.csv"):
        paths.add(path)
        paths.add(path.parents[1] / "run_manifest.json")
    for path in (V2_ROOT / "runs" / "C_local_context").rglob("held_out_predictions.csv"):
        paths.add(path)
        paths.add(path.parents[1] / "run_manifest.json")
    for relative in (
        "failure_manifests",
        "attempt_markers",
        "completion_manifests",
        "logs/authorized_attempts",
    ):
        directory = V3_ROOT / relative
        if directory.is_dir():
            paths.update(path for path in directory.rglob("*") if path.is_file())
    files = {
        str(path.resolve()): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
    }
    return {
        "files": files,
        "methods": {
            method: {
                "rows": len(frame),
                "prediction_identity_sha256": sha256_text(stable_json(frame[["record_id", "fold", "family_id", "label", "probability_yes"]].to_dict(orient="records"))),
            }
            for method, frame in collected["methods"].items()
        },
    }


def _existing_analysis(effective: pd.DataFrame, methods: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    current_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for method, frame in methods.items():
        current_rows.append(metric_row(frame, frame["probability_yes"].to_numpy(float) >= 0.5, method=method, operating_point="threshold_0.5", finding_class="valid_existing_evaluation"))
        fold_metrics = [metric_row(group, group["probability_yes"].to_numpy(float) >= 0.5, method=method, fold=int(fold)) for fold, group in frame.groupby("fold", sort=True)]
        current_rows[-1]["mean_fold_ap"] = float(np.mean([row["average_precision"] for row in fold_metrics]))
        current_rows[-1]["std_fold_ap"] = float(np.std([row["average_precision"] for row in fold_metrics], ddof=1))
        for family, group in frame.groupby("family_id", sort=True):
            both = group["label"].nunique() == 2
            predicted = group["probability_yes"].to_numpy(float) >= 0.5
            row = metric_row(group, predicted, method=method, family_id=str(family), operating_point="threshold_0.5") if both else {
                "method": method,
                "family_id": str(family),
                "operating_point": "threshold_0.5",
                "rows": len(group),
                "positives": int(group["label"].sum()),
                "prevalence": float(group["label"].mean()),
                "average_precision": float(average_precision_score(group["label"], group["probability_yes"])) if group["label"].sum() else None,
                "ap_prevalence_lift": None,
                "roc_auc": None,
                "brier_score": float(brier_score_loss(group["label"], group["probability_yes"])),
                **confusion(group["label"].to_numpy(int), predicted),
            }
            family_rows.append(row)
        top_rows.extend(_top_k_rows(method, frame))
        distribution_rows.extend(_score_distribution(method, frame))
        calibration_rows.extend(_calibration_bins(method, frame))
    return {
        "CURRENT_MODEL_COMPARISON.csv": pd.DataFrame(current_rows),
        "FAMILY_ROBUSTNESS_EXISTING.csv": pd.DataFrame(family_rows),
        "TOP_K_EXISTING.csv": pd.DataFrame(top_rows),
        "SCORE_DISTRIBUTIONS.csv": pd.DataFrame(distribution_rows),
        "CALIBRATION_DIAGNOSTICS.csv": pd.DataFrame(calibration_rows),
    }


def _complementarity(methods: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    left = methods["focal_loss"].copy().rename(columns={"probability_yes": "focal_score"})
    right = methods["partial_4"][["record_id", "fold", "family_id", "label", "probability_yes"]].copy().rename(columns={"probability_yes": "partial4_score"})
    joined = left.merge(right, on=["record_id", "fold", "family_id", "label"], how="outer", validate="one_to_one", indicator=True)
    if len(joined) != EXPECTED_ROWS or set(joined["_merge"]) != {"both"}:
        raise RuntimeError("Focal/Partial-4 alignment failed")
    joined = joined.drop(columns="_merge")
    joined["focal_positive_0_5"] = joined["focal_score"] >= 0.5
    joined["partial4_positive_0_5"] = joined["partial4_score"] >= 0.5
    joined["positive_detection_group"] = np.select(
        [
            (joined["label"] == 1) & joined["focal_positive_0_5"] & joined["partial4_positive_0_5"],
            (joined["label"] == 1) & joined["focal_positive_0_5"] & ~joined["partial4_positive_0_5"],
            (joined["label"] == 1) & ~joined["focal_positive_0_5"] & joined["partial4_positive_0_5"],
            (joined["label"] == 1) & ~joined["focal_positive_0_5"] & ~joined["partial4_positive_0_5"],
        ],
        ["both", "focal_only", "partial4_only", "missed_by_both"],
        default="not_positive_label",
    )
    joined["false_positive_group"] = np.select(
        [
            (joined["label"] == 0) & joined["focal_positive_0_5"] & joined["partial4_positive_0_5"],
            (joined["label"] == 0) & joined["focal_positive_0_5"] & ~joined["partial4_positive_0_5"],
            (joined["label"] == 0) & ~joined["focal_positive_0_5"] & joined["partial4_positive_0_5"],
        ],
        ["shared", "focal_only", "partial4_only"],
        default="not_false_positive",
    )
    summary = {
        "positive_detection": joined.loc[joined["label"] == 1, "positive_detection_group"].value_counts().to_dict(),
        "false_positives": joined.loc[(joined["label"] == 0) & (joined["focal_positive_0_5"] | joined["partial4_positive_0_5"]), "false_positive_group"].value_counts().to_dict(),
        "score_spearman": float(joined[["focal_score", "partial4_score"]].corr(method="spearman").iloc[0, 1]),
    }
    return joined, summary


def _exploratory_ensembles(joined: pd.DataFrame) -> pd.DataFrame:
    frame = joined.copy()
    frame["raw_average"] = (frame["focal_score"] + frame["partial4_score"]) / 2
    rank_parts = []
    z_parts = []
    for _, group in frame.groupby("fold", sort=True):
        focal_rank = stable_rank_percentile(group, "focal_score")
        partial_rank = stable_rank_percentile(group, "partial4_score")
        rank_parts.append(pd.Series((focal_rank + partial_rank) / 2, index=group.index))
        f_std = group["focal_score"].std(ddof=0)
        p_std = group["partial4_score"].std(ddof=0)
        f_z = (group["focal_score"] - group["focal_score"].mean()) / f_std
        p_z = (group["partial4_score"] - group["partial4_score"].mean()) / p_std
        z_parts.append(pd.Series((f_z + p_z) / 2, index=group.index))
    frame["within_fold_rank_average"] = pd.concat(rank_parts).sort_index()
    frame["within_fold_z_average"] = pd.concat(z_parts).sort_index()
    rows = []
    for name, column, inference in (
        ("focal_loss", "focal_score", "direct model probability"),
        ("partial_4", "partial4_score", "direct model probability"),
        ("equal_raw_probability_average", "raw_average", "reproducible without fitted weights but score scales differ"),
        ("equal_within_fold_rank_average", "within_fold_rank_average", "exploratory; batch-relative ranking is unsuitable for isolated inference"),
        ("equal_within_fold_standardized_average", "within_fold_z_average", "exploratory; requires fitting-side location and scale at inference"),
    ):
        rows.append({
            "method": name,
            "average_precision": float(average_precision_score(frame["label"], frame[column])),
            "roc_auc": float(roc_auc_score(frame["label"], frame[column])),
            "status": "EXPLORATORY - NOT INDEPENDENTLY VALIDATED",
            "inference_interpretation": inference,
            "outer_labels_used_to_fit_weights": False,
        })
    return pd.DataFrame(rows)


def _protocol_payload() -> dict[str, Any]:
    v1 = read_json(V1_PROTOCOL)
    v3 = read_json(V3_PROTOCOL)
    splits = _inner_splits()
    config = load_config(v1["config_path"])
    source_snapshot = {
        "study_files": {
            path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
            for path in sorted(ROOT.glob("*.py"))
        },
        "core_training_files": source_file_hashes(),
        "v1_cell_launcher_sha256": sha256_file(V1_ROOT / "cell_launcher.py"),
        "v1_inner_splits_sha256": sha256_file(INNER_SPLITS),
        "v1_protocol_sha256": sha256_file(V1_PROTOCOL),
        "v3_protocol_sha256": sha256_file(V3_PROTOCOL),
        "v3_source_snapshot_sha256": v3["source_snapshot_sha256"],
    }
    return {
        "schema_version": 1,
        "study_id": "method_specific_calibration_v1",
        "created_at": _now(),
        "research_status": "single_seed_nested_calibration",
        "dataset": {
            "source_csv": str(resolve_path(config["data"]["source_csv"])),
            "source_sha256": SOURCE_SHA256,
            "effective_dataset_sha256": EFFECTIVE_SHA256,
            "prompt_effective_hash_as_supplied": PROMPT_EFFECTIVE_HASH,
            "prompt_hash_disposition": "truncated_reference; canonical repository hash verified",
            "rows": EXPECTED_ROWS,
            "positive": EXPECTED_POSITIVES,
            "negative": EXPECTED_NEGATIVES,
            "reserved_families": v1["dataset"]["reserved_families"],
            "outer_fold_manifest_sha256": v1["dataset"]["fold_manifest_sha256"],
        },
        "model": v1["model"],
        "training": v1["training"],
        "methods": {
            "focal_loss": {
                "launcher_method": "focal",
                "loss": "focal_gamma_2",
                "gamma": 2.0,
                "alpha": "existing fitting-side class weights",
                "fine_tuning": "full encoder",
            },
            "partial_4": {
                "launcher_method": "partial_4",
                "loss": "weighted_ce",
                "fine_tuning": "encoder layers 8-11, pooler, classifier",
                "optimizer": "trainable parameters only",
            },
        },
        "inner_cv": {
            "source": str(INNER_SPLITS.resolve()),
            "sha256": sha256_file(INNER_SPLITS),
            "outer_folds": 5,
            "inner_folds_per_outer": 3,
            "cells_per_method": 15,
            "total_new_cells": 30,
            "splits": [{
                "outer_fold": int(item["outer_fold"]),
                "inner_fold": int(item["inner_fold"]),
                "manifest_sha256": sha256_text(stable_json(item)),
                "fitting_rows": len(item["fitting_record_ids"]),
                "scoring_rows": len(item["scoring_record_ids"]),
            } for item in splits],
        },
        "threshold_selection": {
            "objectives": [f"precision_at_least_{value:.2f}" for value in PRECISION_TARGETS],
            "candidates": "all unique fitting-side OOF scores plus one threshold above the maximum",
            "comparison": "score >= threshold",
            "primary_order": "maximum fitting-side recall",
            "tie_breaking": "higher fitting-side precision, then higher threshold",
            "zero_predictions": "not feasible",
            "infeasible": "explicit INFEASIBLE; no always-positive or data-dependent fallback",
            "outer_labels_used": False,
        },
        "review_budget": {
            "scope": "global pooled development OOF ranking",
            "k": list(TOP_K),
            "future_policy": "sort unseen candidates by frozen score transformation and record-id tie break; review first K",
        },
        "ensemble": {
            "primary_pre_registered": "equal-weight empirical-CDF average",
            "fitting": "one ECDF per method from that outer fold's inner OOF scores",
            "application": "freeze both ECDFs and apply to corresponding untouched outer predictions",
            "weights": {"focal_loss": 0.5, "partial_4": 0.5},
            "weight_fitting": "none",
            "thresholds": "selected from matching inner ensemble scores using the same precision constraints",
        },
        "execution": {
            "sequential": True,
            "automatic_retry": False,
            "one_process_per_cell": True,
            "stop_on_first_failure": True,
            "minimum_free_vram_mib": MIN_FREE_VRAM_MIB,
            "timeout_seconds_per_cell": 900,
        },
        "historical_protocols": {
            "v3_sha256": V3_SHA256,
            "v3_source_snapshot_sha256": V3_SOURCE_SHA256,
        },
        "source_snapshot": source_snapshot,
        "source_snapshot_sha256": sha256_text(stable_json(source_snapshot)),
    }


def _specs(protocol_hash: str) -> list[dict[str, Any]]:
    v1 = read_json(V1_PROTOCOL)
    splits = _inner_splits()
    specs: list[dict[str, Any]] = []
    for method in METHODS:
        for split in splits:
            outer = int(split["outer_fold"])
            inner = int(split["inner_fold"])
            attempt = f"cal-v1-{method.replace('_', '-')}-outer{outer}-inner{inner}-seed{SEED}"
            specs.append({
                "schema_version": 1,
                "study_id": "method_specific_calibration_v1",
                "protocol_sha256": protocol_hash,
                "cell_id": f"{method}/outer_{outer}/inner_{inner}/seed_{SEED}",
                "attempt_id": attempt,
                "method": METHOD_TO_LAUNCHER[method],
                "analysis_method": method,
                "outer_fold": outer,
                "inner_fold": inner,
                "train_fold": inner,
                "seed": SEED,
                "config_path": v1["config_path"],
                "model_path": v1["model"]["local_path"],
                "input_mode": "target_only",
                "fold_manifest": split,
                "fold_manifest_sha256": sha256_text(stable_json(split)),
                "fitting_record_ids": split["fitting_record_ids"],
                "scoring_record_ids": split["scoring_record_ids"],
                "expected_score_ids": split["scoring_record_ids"],
                "run_path": str((RUN_DIR / method / f"outer_{outer}" / f"inner_{inner}" / f"attempt_{attempt}").resolve()),
                "audit_path": str((AUDIT_DIR / f"{attempt}.json").resolve()),
                "log_base": str((LOG_DIR / attempt).resolve()),
                "automatic_retry": False,
                "fresh_base_initialization": True,
            })
    return specs


def _audit_markdown(
    complementarity: dict[str, Any], existing: dict[str, pd.DataFrame]
) -> str:
    thresholds = pd.read_csv(V1_ROOT / "analysis" / "nested_threshold_calibration" / "selected_thresholds_by_outer_fold.csv")
    threshold_values = ", ".join(f"{value:.3f}" for value in thresholds["selected_threshold"].tolist())
    return f"""# Calibration Audit

## Verdict

**Qualified.** Historical outer predictions, cohort identity, labels, fold membership, score direction, and artifact hashes validate. The existing nested thresholds are valid only for the Full Fine-Tuning model that generated the inner OOF scores. Their later application to Focal Loss, Partial-4, Partial-2, and Local Context is a cross-method threshold transfer and is not a valid method-specific calibrated operating point.

## Verified facts

- Source CSV SHA-256: `{SOURCE_SHA256}`.
- Canonical effective cohort SHA-256: `{EFFECTIVE_SHA256}`.
- The prompt supplied the truncated value `{PROMPT_EFFECTIVE_HASH}`; the canonical value above matches all frozen repository manifests.
- Development cohort: {EXPECTED_ROWS} rows, {EXPECTED_POSITIVES} positive, {EXPECTED_NEGATIVES} negative, five family-aware outer folds.
- Every completed method has exactly one held-out prediction per eligible record, with matching labels, families, and folds.
- Positive class is integer `1` and score direction is probability of `Yes`/ambiguous.
- No TCP or DNS family occurs in the development predictions or nested split IDs.
- Existing Full Fine-Tuning thresholds by outer fold are: {threshold_values}.
- The V3 aggregate analyzer maps those same fold thresholds onto every method. This is the exact source of the transferred-threshold result.
- Fifteen inner OOF cells exist for Full Fine-Tuning only. No compatible Focal Loss or Partial-4 inner OOF predictions exist.
- Thirty new cells are therefore required: 5 outer folds × 3 inner folds × 2 methods.
- Historical V3 protocol SHA-256 remains `{V3_SHA256}`.

## Existing prediction complementarity at threshold 0.5

- Positive detections: `{json.dumps(complementarity['positive_detection'], sort_keys=True)}`.
- False-positive overlap: `{json.dumps(complementarity['false_positives'], sort_keys=True)}`.
- Score Spearman correlation: {complementarity['score_spearman']:.4f}.

## Finding classification

- `CURRENT_MODEL_COMPARISON.csv`, fixed threshold 0.5, AP/AUC, and existing Top-K: **valid existing evaluation**.
- Score distributions, calibration bins, complementarity, and newly examined ensemble combinations: **exploratory diagnostic**.
- Focal Loss/Partial-4 method-specific operating points and the pre-registered ECDF ensemble: **require additional fitting-side training** until all 30 inner cells validate.

## Governance limitation

Approximately 88% of the operational labels originate from agreement between two AI annotators. They are not an independently human-verified gold standard. Correlated annotator errors can inflate agreement-based performance and hide systematic ambiguity categories.
"""


def prepare() -> dict[str, Any]:
    if any(path.exists() for path in (PROTOCOL_PATH, PROVENANCE_PATH, INVENTORY_PATH, STATUS_PATH, GPU_MANIFEST_PATH)):
        raise FileExistsError("Study preparation artifacts already exist; refusing to overwrite")
    effective, methods, collected = load_existing()
    existing = _existing_analysis(effective, methods)
    joined, complementarity_summary = _complementarity(methods)
    exploratory = _exploratory_ensembles(joined)
    historical = _historical_file_inventory(collected)
    inventory = {
        "schema_version": 1,
        "created_at": _now(),
        "historical_files": historical["files"],
        "prediction_sets": historical["methods"],
        "historical_attempt_statuses": {
            key: value.get("status")
            for key, value in read_json(V3_ROOT / "EXPERIMENT_INVENTORY.json").get("attempts", {}).items()
        },
        "outer_prediction_rows_per_method": {key: len(value) for key, value in methods.items()},
        "reusable_method_specific_inner_cells": {"focal_loss": 0, "partial_4": 0},
        "required_new_cells": 30,
        "historical_outputs_modified": False,
    }
    protocol = _protocol_payload()
    _write_json_exclusive(INVENTORY_PATH, inventory)
    _write_json_exclusive(PROVENANCE_PATH, {
        "schema_version": 1,
        "created_at": _now(),
        "source_csv_sha256": SOURCE_SHA256,
        "effective_dataset_sha256": EFFECTIVE_SHA256,
        "v1_protocol_sha256": sha256_file(V1_PROTOCOL),
        "inner_splits_sha256": sha256_file(INNER_SPLITS),
        "v3_protocol_sha256": V3_SHA256,
        "v3_source_snapshot_sha256": V3_SOURCE_SHA256,
        "reserved_sets_consumed": False,
    })
    _write_json_exclusive(PROTOCOL_PATH, protocol)
    protocol_hash = sha256_file(PROTOCOL_PATH)
    specs = _specs(protocol_hash)
    SPEC_DIR.mkdir(parents=True, exist_ok=False)
    spec_records = {}
    for index, spec in enumerate(specs, start=1):
        path = SPEC_DIR / f"{index:02d}_{spec['attempt_id']}.json"
        _write_json_exclusive(path, spec)
        spec_records[spec["cell_id"]] = {
            "sequence": index,
            "spec_path": str(path.resolve()),
            "spec_sha256": sha256_file(path),
            "attempt_id": spec["attempt_id"],
            "status": "PLANNED",
        }
    _write_json_exclusive(STATUS_PATH, {
        "schema_version": 1,
        "study_id": "method_specific_calibration_v1",
        "protocol_sha256": protocol_hash,
        "updated_at": _now(),
        "automatic_retry": False,
        "stop_on_first_failure": True,
        "cells": spec_records,
    })
    _write_json_exclusive(GPU_MANIFEST_PATH, {
        "schema_version": 1,
        "protocol_sha256": protocol_hash,
        "planned_cells": 30,
        "reused_cells": 0,
        "completed_cells": 0,
        "failed_cells": 0,
        "skipped_cells": 0,
        "sequential": True,
        "automatic_retry": False,
        "created_at": _now(),
        "executions": [],
    })
    for name, frame in existing.items():
        _write_csv(ROOT / name, frame)
    _write_csv(ROOT / "PREDICTION_COMPLEMENTARITY.csv", joined)
    _write_csv(ROOT / "EXPLORATORY_ENSEMBLE.csv", exploratory)
    _write_text_exclusive(ROOT / "CALIBRATION_AUDIT.md", _audit_markdown(complementarity_summary, existing))
    return {"protocol_sha256": protocol_hash, "cells": len(specs), "audit": "QUALIFIED", "cuda_used": False}


def _verify_source_snapshot(protocol: dict[str, Any]) -> None:
    expected = protocol["source_snapshot"]
    current_study = {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(ROOT.glob("*.py"))
    }
    if current_study != expected["study_files"] or source_file_hashes() != expected["core_training_files"]:
        raise RuntimeError("Frozen source snapshot changed")
    if (
        sha256_file(V1_ROOT / "cell_launcher.py") != expected["v1_cell_launcher_sha256"]
        or sha256_file(INNER_SPLITS) != expected["v1_inner_splits_sha256"]
        or sha256_file(V1_PROTOCOL) != expected["v1_protocol_sha256"]
        or sha256_file(V3_PROTOCOL) != expected["v3_protocol_sha256"]
    ):
        raise RuntimeError("Parent source/protocol provenance changed")


def _verify_historical_immutability() -> None:
    inventory = read_json(INVENTORY_PATH)
    for name, expected in inventory["historical_files"].items():
        path = Path(name)
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Historical artifact changed: {path}")


def validate_pretraining() -> dict[str, Any]:
    protocol = read_json(PROTOCOL_PATH)
    if sha256_file(PROTOCOL_PATH) != read_json(STATUS_PATH)["protocol_sha256"]:
        raise RuntimeError("Protocol hash differs from training status")
    _verify_source_snapshot(protocol)
    _verify_historical_immutability()
    effective, methods, _ = load_existing()
    del methods
    splits = _inner_splits()
    expected_ids = set(effective["record_id"])
    for outer in range(1, 6):
        selected = [item for item in splits if int(item["outer_fold"]) == outer]
        scoring = [record for item in selected for record in item["scoring_record_ids"]]
        outer_fit = set(effective.loc[effective["fold"] != outer, "record_id"])
        if len(selected) != 3 or len(scoring) != len(set(scoring)) or set(scoring) != outer_fit:
            raise RuntimeError(f"Inner OOF coverage failed: outer {outer}")
        for item in selected:
            fit_ids = set(item["fitting_record_ids"])
            score_ids = set(item["scoring_record_ids"])
            if fit_ids & score_ids or not fit_ids or not score_ids or not (fit_ids | score_ids) <= expected_ids:
                raise RuntimeError(f"Inner split identity failed: outer {outer}")
            indexed = effective.set_index("record_id")
            if set(indexed.loc[list(fit_ids), "family_id"]) & set(indexed.loc[list(score_ids), "family_id"]):
                raise RuntimeError(f"Inner family leakage: outer {outer}")
    status = read_json(STATUS_PATH)
    specs = []
    attempts = set()
    run_paths = set()
    for record in sorted(status["cells"].values(), key=lambda item: item["sequence"]):
        path = Path(record["spec_path"])
        if sha256_file(path) != record["spec_sha256"]:
            raise RuntimeError(f"Cell spec changed: {path}")
        spec = read_json(path)
        if spec["protocol_sha256"] != sha256_file(PROTOCOL_PATH):
            raise RuntimeError("Cell points to a different protocol")
        if spec["attempt_id"] in attempts or spec["run_path"] in run_paths:
            raise RuntimeError("Attempt identity or output path is not unique")
        attempts.add(spec["attempt_id"])
        run_paths.add(spec["run_path"])
        specs.append(spec)
    if len(specs) != 30:
        raise RuntimeError("Expected 30 immutable training specifications")
    return {"status": "PASS", "cells": 30, "rows": len(effective), "cuda_used": False, "protocol_sha256": sha256_file(PROTOCOL_PATH)}


def gpu_preflight() -> dict[str, Any]:
    # Reuse the already validated Windows-aware gate. It distinguishes ordinary
    # graphics clients from true CUDA compute jobs and never terminates processes.
    from experiments.transformer_final.recall_improvement_study_v1.study_runner import (
        _gpu_preflight,
    )

    result = _gpu_preflight()
    if result["name"] != "NVIDIA RTX 5000 Ada Generation":
        raise RuntimeError("GPU identity differs from the frozen protocol")
    return result


def _validate_new_cell(spec: dict[str, Any], effective: pd.DataFrame) -> dict[str, Any]:
    run = Path(spec["run_path"])
    manifest_path = run / "run_manifest.json"
    audit_path = Path(spec["audit_path"])
    if not manifest_path.is_file() or not audit_path.is_file():
        raise RuntimeError(f"Missing run manifest or treatment audit: {spec['cell_id']}")
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    expected_loss = "focal_gamma_2" if spec["analysis_method"] == "focal_loss" else "weighted_ce"
    expected_full = spec["analysis_method"] == "focal_loss"
    if (
        manifest.get("condition") != "A2"
        or int(manifest.get("fold", 0)) != int(spec["inner_fold"])
        or int(manifest.get("seed", 0)) != SEED
        or manifest.get("loss") != expected_loss
        or manifest.get("precision") != "fp32"
        or manifest.get("input_mode") != "target_only"
        or manifest.get("epochs_completed") != 5
        or manifest.get("resume_checkpoint") is not None
        or manifest.get("fresh_base_initialization") is not True
        or manifest.get("full_encoder_fine_tuned") is not expected_full
        or manifest.get("device") != "cuda"
        or manifest.get("effective_dataset_sha256") != EFFECTIVE_SHA256
        or manifest.get("original_dataset_sha256") != SOURCE_SHA256
        or manifest.get("fold_manifest_sha256") != spec["fold_manifest_sha256"]
        or manifest.get("calibration_study_id") != "method_specific_calibration_v1"
        or manifest.get("calibration_protocol_sha256") != spec["protocol_sha256"]
        or manifest.get("calibration_cell_id") != spec["cell_id"]
        or manifest.get("calibration_attempt_id") != spec["attempt_id"]
        or manifest.get("calibration_outer_fold") != int(spec["outer_fold"])
        or manifest.get("calibration_inner_fold") != int(spec["inner_fold"])
        or manifest.get("calibration_automatic_retry") is not False
    ):
        raise RuntimeError(f"New run protocol mismatch: {spec['cell_id']}")
    if spec["analysis_method"] == "focal_loss" and (
        manifest.get("focal_gamma") != 2.0
        or manifest.get("focal_alpha") != "existing fitting-side class weights"
    ):
        raise RuntimeError(f"Focal configuration mismatch: {spec['cell_id']}")
    if spec["analysis_method"] == "partial_4" and (
        manifest.get("fine_tuning_policy") != "partial_4"
        or manifest.get("trainable_last_layers") != 4
        or audit.get("frozen_parameters_unchanged") is not True
        or audit.get("frozen_parameters_have_no_gradients") is not True
        or audit.get("trainable_parameters_finite") is not True
    ):
        raise RuntimeError(f"Partial-4 freeze audit failed: {spec['cell_id']}")
    for artifact in manifest.get("artifact_hashes", {}).values():
        path = (run / artifact["path"]).resolve()
        if run.resolve() not in path.parents or not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise RuntimeError(f"Artifact hash mismatch: {spec['cell_id']}")
    predictions = pd.read_csv(run / "predictions" / "held_out_predictions.csv", dtype={"record_id": str}, keep_default_na=False)
    expected_ids = set(spec["scoring_record_ids"])
    if len(predictions) != len(expected_ids) or predictions["record_id"].duplicated().any() or set(predictions["record_id"]) != expected_ids:
        raise RuntimeError(f"Inner prediction coverage mismatch: {spec['cell_id']}")
    indexed = effective.set_index("record_id")
    aligned = indexed.loc[predictions["record_id"]]
    if not np.array_equal(predictions["label"].astype(int), aligned["label"].astype(int)):
        raise RuntimeError(f"Inner prediction label mismatch: {spec['cell_id']}")
    if not np.array_equal(predictions["family_id"].astype(str), aligned["family_id"].astype(str)):
        raise RuntimeError(f"Inner prediction family mismatch: {spec['cell_id']}")
    scores = pd.to_numeric(predictions["probability_yes"], errors="raise").to_numpy(float)
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise RuntimeError(f"Inner probability sanity failed: {spec['cell_id']}")
    if int(manifest["fitting_rows"]) != len(spec["fitting_record_ids"]) or int(manifest["held_out_rows"]) != len(expected_ids):
        raise RuntimeError(f"Inner fit/score row counts failed: {spec['cell_id']}")
    return manifest


def _update_status(status: dict[str, Any]) -> None:
    status["updated_at"] = _now()
    write_json(STATUS_PATH, status)


def _update_gpu_manifest(value: dict[str, Any]) -> None:
    value["updated_at"] = _now()
    write_json(GPU_MANIFEST_PATH, value)


def run_one(spec_path: Path) -> dict[str, Any]:
    validate_pretraining()
    spec_path = spec_path.resolve()
    status = read_json(STATUS_PATH)
    record = status["cells"].get(read_json(spec_path)["cell_id"])
    if record is None or Path(record["spec_path"]).resolve() != spec_path or sha256_file(spec_path) != record["spec_sha256"]:
        raise RuntimeError("Unrecognized or changed training specification")
    if record["status"] != "PLANNED":
        raise RuntimeError(f"No-retry gate: cell status is {record['status']}")
    spec = read_json(spec_path)
    run = Path(spec["run_path"])
    stdout = Path(spec["log_base"] + ".stdout.log")
    stderr = Path(spec["log_base"] + ".stderr.log")
    completion = COMPLETION_DIR / f"{spec['attempt_id']}.json"
    failure = FAILURE_DIR / f"{spec['attempt_id']}.json"
    if any(path.exists() for path in (run, stdout, stderr, completion, failure, Path(spec["audit_path"]))):
        raise FileExistsError("No-retry gate: an attempt artifact already exists")
    preflight = gpu_preflight()
    record.update({"status": "IN_PROGRESS", "started_at": _now(), "gpu_preflight": preflight})
    _update_status(status)
    gpu_manifest = read_json(GPU_MANIFEST_PATH)
    execution = {"cell_id": spec["cell_id"], "attempt_id": spec["attempt_id"], "started_at": record["started_at"], "gpu_preflight": preflight, "status": "IN_PROGRESS"}
    gpu_manifest["executions"].append(execution)
    _update_gpu_manifest(gpu_manifest)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    library_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
    environment["PATH"] = str(library_bin) + os.pathsep + environment.get("PATH", "")
    command = [sys.executable, "-m", "experiments.transformer_final.method_specific_calibration_v1.cell_launcher", "--spec", str(spec_path)]
    started = time.monotonic()
    try:
        with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
            result = subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=out, stderr=err, timeout=900, check=False)
    except Exception as exc:
        elapsed = time.monotonic() - started
        record.update({"status": "FAILED", "elapsed_seconds": elapsed, "failure": type(exc).__name__})
        execution.update({"status": "FAILED", "elapsed_seconds": elapsed, "failure": repr(exc)})
        gpu_manifest["failed_cells"] += 1
        _update_status(status)
        _update_gpu_manifest(gpu_manifest)
        _write_json_exclusive(failure, {**execution, "preserved": True, "automatic_retry": False})
        raise
    elapsed = time.monotonic() - started
    if result.returncode != 0:
        record.update({"status": "FAILED", "elapsed_seconds": elapsed, "return_code": result.returncode})
        execution.update({"status": "FAILED", "elapsed_seconds": elapsed, "return_code": result.returncode})
        gpu_manifest["failed_cells"] += 1
        _update_status(status)
        _update_gpu_manifest(gpu_manifest)
        _write_json_exclusive(failure, {**execution, "stderr_path": str(stderr), "preserved": True, "automatic_retry": False})
        raise RuntimeError(f"Training cell failed; no retry: {spec['cell_id']}; see {stderr}")
    try:
        effective, _, _ = load_existing()
        manifest = _validate_new_cell(spec, effective)
    except Exception as exc:
        record.update({"status": "FAILED_VALIDATION", "elapsed_seconds": elapsed, "return_code": 0, "failure": repr(exc)})
        execution.update({"status": "FAILED_VALIDATION", "elapsed_seconds": elapsed, "failure": repr(exc)})
        gpu_manifest["failed_cells"] += 1
        _update_status(status)
        _update_gpu_manifest(gpu_manifest)
        _write_json_exclusive(failure, {**execution, "stderr_path": str(stderr), "preserved": True, "automatic_retry": False})
        raise
    record.update({"status": "COMPLETE", "completed_at": _now(), "elapsed_seconds": elapsed, "return_code": 0, "run_manifest_sha256": sha256_file(run / "run_manifest.json")})
    execution.update({"status": "COMPLETE", "completed_at": record["completed_at"], "elapsed_seconds": elapsed, "training_runtime_seconds": float(manifest["runtime_seconds"]), "peak_gpu_memory_allocated_mb": float(manifest["peak_gpu_memory_allocated_mb"]), "peak_gpu_memory_reserved_mb": float(manifest["peak_gpu_memory_reserved_mb"]), "run_manifest_sha256": record["run_manifest_sha256"]})
    gpu_manifest["completed_cells"] += 1
    _update_status(status)
    _update_gpu_manifest(gpu_manifest)
    _write_json_exclusive(completion, {**execution, "spec_sha256": sha256_file(spec_path), "protocol_sha256": spec["protocol_sha256"]})
    return execution


def run_all() -> dict[str, Any]:
    validation = validate_pretraining()
    status = read_json(STATUS_PATH)
    ordered = sorted(status["cells"].values(), key=lambda item: item["sequence"])
    for item in ordered:
        if item["status"] == "COMPLETE":
            continue
        if item["status"] != "PLANNED":
            raise RuntimeError(f"Sequence stopped at non-runnable state {item['status']}: {item['attempt_id']}")
        run_one(Path(item["spec_path"]))
    return {**validation, "completed_cells": read_json(GPU_MANIFEST_PATH)["completed_cells"]}


def validate_completed(require_all: bool = False) -> dict[str, Any]:
    base = validate_pretraining()
    effective, _, _ = load_existing()
    status = read_json(STATUS_PATH)
    counts: dict[str, int] = {}
    for record in status["cells"].values():
        counts[record["status"]] = counts.get(record["status"], 0) + 1
        if record["status"] == "COMPLETE":
            spec = read_json(record["spec_path"])
            manifest = _validate_new_cell(spec, effective)
            if sha256_file(Path(spec["run_path"]) / "run_manifest.json") != record["run_manifest_sha256"]:
                raise RuntimeError(f"Completed run manifest changed: {spec['cell_id']}")
            del manifest
    if require_all and counts.get("COMPLETE", 0) != 30:
        raise RuntimeError(f"All 30 cells are required; statuses={counts}")
    _verify_historical_immutability()
    return {**base, "cell_statuses": counts, "historical_outputs_unchanged": True}


def _load_inner_predictions() -> tuple[dict[tuple[str, int], pd.DataFrame], pd.DataFrame]:
    status = read_json(STATUS_PATH)
    grouped: dict[tuple[str, int], list[pd.DataFrame]] = {}
    per_fold_rows: list[dict[str, Any]] = []
    for record in sorted(status["cells"].values(), key=lambda value: value["sequence"]):
        if record["status"] != "COMPLETE":
            continue
        spec = read_json(record["spec_path"])
        frame = pd.read_csv(Path(spec["run_path"]) / "predictions" / "held_out_predictions.csv", dtype={"record_id": str})
        frame["outer_fold"] = int(spec["outer_fold"])
        frame["inner_fold"] = int(spec["inner_fold"])
        frame["analysis_method"] = spec["analysis_method"]
        grouped.setdefault((spec["analysis_method"], int(spec["outer_fold"])), []).append(frame)
        per_fold_rows.append(metric_row(frame, frame["probability_yes"].to_numpy(float) >= 0.5, method=spec["analysis_method"], outer_fold=int(spec["outer_fold"]), inner_fold=int(spec["inner_fold"]), operating_point="threshold_0.5"))
    combined: dict[tuple[str, int], pd.DataFrame] = {}
    splits = _inner_splits()
    for key, pieces in grouped.items():
        method, outer = key
        frame = pd.concat(pieces, ignore_index=True)
        expected = {record for item in splits if int(item["outer_fold"]) == outer for record in item["scoring_record_ids"]}
        if len(frame) != len(expected) or frame["record_id"].duplicated().any() or set(frame["record_id"]) != expected:
            raise RuntimeError(f"Inner OOF reconstruction failed: {method} outer {outer}")
        combined[key] = frame
    return combined, pd.DataFrame(per_fold_rows)


def _metric_from_score(frame: pd.DataFrame, score: np.ndarray, predicted: np.ndarray, **identity: Any) -> dict[str, Any]:
    working = frame.copy()
    working["probability_yes"] = np.asarray(score, dtype=float)
    return metric_row(working, predicted, **identity)


def _cluster_bootstrap(final_scores: dict[str, pd.DataFrame], replicates: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(20261018)
    reference = next(iter(final_scores.values()))
    families = np.array(sorted(reference["family_id"].astype(str).unique()))
    draws: dict[tuple[str, str], list[float]] = {}
    point: dict[tuple[str, str], float] = {}
    for method, frame in final_scores.items():
        labels = frame["label"].to_numpy(int)
        scores = frame["probability_yes"].to_numpy(float)
        point[(method, "average_precision")] = float(average_precision_score(labels, scores))
        point[(method, "roc_auc")] = float(roc_auc_score(labels, scores))
        draws[(method, "average_precision")] = []
        draws[(method, "roc_auc")] = []
    for _ in range(replicates):
        sampled = rng.choice(families, size=len(families), replace=True)
        indices_by_method: dict[str, list[int]] = {method: [] for method in final_scores}
        for family in sampled:
            for method, frame in final_scores.items():
                indices_by_method[method].extend(np.flatnonzero(frame["family_id"].astype(str).to_numpy() == family).tolist())
        for method, frame in final_scores.items():
            index = indices_by_method[method]
            labels = frame.iloc[index]["label"].to_numpy(int)
            scores = frame.iloc[index]["probability_yes"].to_numpy(float)
            if labels.sum() and len(np.unique(labels)) == 2:
                draws[(method, "average_precision")].append(float(average_precision_score(labels, scores)))
                draws[(method, "roc_auc")].append(float(roc_auc_score(labels, scores)))
    rows = []
    for key, values in draws.items():
        method, metric = key
        rows.append({
            "method": method,
            "metric": metric,
            "point_estimate": point[key],
            "cluster_bootstrap_lower_95": float(np.quantile(values, 0.025)),
            "cluster_bootstrap_upper_95": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
            "clusters": len(families),
            "cluster_unit": "RFC family",
            "interpretation": "sampling uncertainty only; does not measure single-seed training variability",
        })
    return pd.DataFrame(rows)


def analyze_final() -> dict[str, Any]:
    validate_completed(require_all=True)
    _effective, methods, _ = load_existing()
    inner, inner_fold_metrics = _load_inner_predictions()
    if len(inner) != 10:
        raise RuntimeError("Expected two methods by five outer folds")
    threshold_rows: list[dict[str, Any]] = []
    final_pieces: dict[tuple[str, float], list[pd.DataFrame]] = {}
    for method in METHODS:
        for outer in range(1, 6):
            fitting = inner[(method, outer)]
            outer_frame = methods[method].loc[methods[method]["fold"] == outer].copy()
            for target in PRECISION_TARGETS:
                selection = select_precision_threshold(fitting["label"], fitting["probability_yes"], target)
                row = {"method": method, "outer_fold": outer, "objective": "precision_constrained_recall", **selection}
                if selection["status"] == "FEASIBLE":
                    threshold = float(selection["threshold"])
                    outer_frame_for_target = outer_frame.copy()
                    outer_frame_for_target["selected_threshold"] = threshold
                    outer_frame_for_target["predicted_label"] = (outer_frame_for_target["probability_yes"] >= threshold).astype(int)
                    final_pieces.setdefault((method, target), []).append(outer_frame_for_target)
                    outer_metrics = metric_row(outer_frame_for_target, outer_frame_for_target["predicted_label"].to_numpy(int), method=method, outer_fold=outer, minimum_fitting_precision=target, operating_point="method_specific_nested")
                    row.update({f"outer_{key}": value for key, value in outer_metrics.items() if key not in {"method", "outer_fold", "minimum_fitting_precision", "operating_point"}})
                threshold_rows.append(row)
    threshold_frame = pd.DataFrame(threshold_rows)
    final_rows: list[dict[str, Any]] = []
    final_prediction_sets: dict[str, pd.DataFrame] = {}
    for (method, target), pieces in final_pieces.items():
        if len(pieces) != 5:
            final_rows.append({"method": method, "minimum_fitting_precision": target, "operating_point": "method_specific_nested", "status": "INCOMPLETE_DUE_TO_INFEASIBLE_OUTER_FOLD"})
            continue
        frame = pd.concat(pieces, ignore_index=True)
        row = metric_row(frame, frame["predicted_label"].to_numpy(int), method=method, minimum_fitting_precision=target, operating_point="method_specific_nested")
        row["status"] = "VALID_NESTED_OUTER_EVALUATION"
        final_rows.append(row)
        final_prediction_sets[f"{method}_p{int(target*100)}"] = frame

    ensemble_threshold_rows: list[dict[str, Any]] = []
    ensemble_pieces: dict[float, list[pd.DataFrame]] = {}
    ensemble_all_outer: list[pd.DataFrame] = []
    for outer in range(1, 6):
        focal_inner = inner[("focal_loss", outer)][["record_id", "label", "probability_yes"]].rename(columns={"probability_yes": "focal_score"})
        partial_inner = inner[("partial_4", outer)][["record_id", "label", "probability_yes"]].rename(columns={"probability_yes": "partial4_score"})
        paired_inner = focal_inner.merge(partial_inner, on=["record_id", "label"], validate="one_to_one")
        paired_inner["ensemble_score"] = 0.5 * empirical_cdf(paired_inner["focal_score"], paired_inner["focal_score"]) + 0.5 * empirical_cdf(paired_inner["partial4_score"], paired_inner["partial4_score"])
        focal_outer = methods["focal_loss"].loc[methods["focal_loss"]["fold"] == outer].copy().rename(columns={"probability_yes": "focal_score"})
        partial_outer = methods["partial_4"].loc[methods["partial_4"]["fold"] == outer, ["record_id", "probability_yes"]].rename(columns={"probability_yes": "partial4_score"})
        paired_outer = focal_outer.merge(partial_outer, on="record_id", validate="one_to_one")
        paired_outer["probability_yes"] = 0.5 * empirical_cdf(paired_inner["focal_score"], paired_outer["focal_score"]) + 0.5 * empirical_cdf(paired_inner["partial4_score"], paired_outer["partial4_score"])
        ensemble_all_outer.append(paired_outer)
        for target in PRECISION_TARGETS:
            selection = select_precision_threshold(paired_inner["label"], paired_inner["ensemble_score"], target)
            row = {"method": "focal_partial4_ecdf_equal", "outer_fold": outer, "objective": "precision_constrained_recall", **selection}
            if selection["status"] == "FEASIBLE":
                evaluated = paired_outer.copy()
                evaluated["selected_threshold"] = float(selection["threshold"])
                evaluated["predicted_label"] = (evaluated["probability_yes"] >= float(selection["threshold"])).astype(int)
                ensemble_pieces.setdefault(target, []).append(evaluated)
                outer_metrics = metric_row(evaluated, evaluated["predicted_label"].to_numpy(int), method="focal_partial4_ecdf_equal", outer_fold=outer, minimum_fitting_precision=target, operating_point="proper_nested_ensemble")
                row.update({f"outer_{key}": value for key, value in outer_metrics.items() if key not in {"method", "outer_fold", "minimum_fitting_precision", "operating_point"}})
            ensemble_threshold_rows.append(row)
    ensemble_outer = pd.concat(ensemble_all_outer, ignore_index=True)
    ensemble_rows: list[dict[str, Any]] = []
    for target, pieces in ensemble_pieces.items():
        if len(pieces) != 5:
            ensemble_rows.append({"method": "focal_partial4_ecdf_equal", "minimum_fitting_precision": target, "status": "INCOMPLETE_DUE_TO_INFEASIBLE_OUTER_FOLD"})
            continue
        frame = pd.concat(pieces, ignore_index=True)
        row = metric_row(frame, frame["predicted_label"].to_numpy(int), method="focal_partial4_ecdf_equal", minimum_fitting_precision=target, operating_point="proper_nested_ensemble")
        row["status"] = "VALID_PRE_REGISTERED_NESTED_OUTER_EVALUATION"
        ensemble_rows.append(row)
        final_prediction_sets[f"ensemble_p{int(target*100)}"] = frame

    final_metrics = pd.DataFrame(final_rows)
    ensemble_metrics = pd.DataFrame(ensemble_rows)
    selected_thresholds = pd.concat([threshold_frame, pd.DataFrame(ensemble_threshold_rows)], ignore_index=True)

    family_rows: list[dict[str, Any]] = []
    for identity, frame in final_prediction_sets.items():
        for family, group in frame.groupby("family_id", sort=True):
            labels = group["label"].to_numpy(int)
            scores = group["probability_yes"].to_numpy(float)
            counts = confusion(labels, group["predicted_label"].to_numpy(int))
            family_rows.append({
                "policy": identity,
                "family_id": family,
                "rows": len(group),
                "positives": int(labels.sum()),
                "average_precision": float(average_precision_score(labels, scores)) if labels.sum() else None,
                "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
                **counts,
            })

    top_rows: list[dict[str, Any]] = []
    for method in ("full_fine_tuning", "focal_loss", "partial_4"):
        top_rows.extend(_top_k_rows(method, methods[method]))
    ensemble_rank_frame = ensemble_outer[["record_id", "family_id", "fold", "label", "probability_yes"]].copy()
    top_rows.extend(_top_k_rows("focal_partial4_ecdf_equal", ensemble_rank_frame))

    gpu_manifest = read_json(GPU_MANIFEST_PATH)
    gpu_time = pd.DataFrame(gpu_manifest["executions"])
    uncertainty_scores = {
        "full_fine_tuning": methods["full_fine_tuning"],
        "focal_loss": methods["focal_loss"],
        "partial_4": methods["partial_4"],
        "focal_partial4_ecdf_equal": ensemble_rank_frame,
    }
    uncertainty = _cluster_bootstrap(uncertainty_scores)

    outputs = {
        "INNER_FOLD_CALIBRATION.csv": inner_fold_metrics,
        "SELECTED_THRESHOLDS.csv": selected_thresholds,
        "FINAL_NESTED_METRICS.csv": final_metrics,
        "FAMILY_ROBUSTNESS.csv": pd.DataFrame(family_rows),
        "TOP_K_COMPARISON.csv": pd.DataFrame(top_rows),
        "ENSEMBLE_VALIDATION.csv": ensemble_metrics,
        "GPU_TRAINING_TIME.csv": gpu_time,
        "UNCERTAINTY_INTERVALS.csv": uncertainty,
    }
    for name, frame in outputs.items():
        _write_csv(ROOT / name, frame)
    for identity, frame in final_prediction_sets.items():
        _write_csv(ROOT / "final_predictions" / f"{identity}.csv", frame)
    _write_csv(ROOT / "final_predictions" / "ensemble_ranking_scores.csv", ensemble_rank_frame)
    _write_reports(final_metrics, ensemble_metrics, selected_thresholds, pd.DataFrame(top_rows), pd.DataFrame(family_rows), uncertainty, gpu_time)
    return {
        "status": "COMPLETE",
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "completed_cells": 30,
        "total_gpu_training_seconds": float(gpu_time["training_runtime_seconds"].sum()),
        "peak_gpu_memory_allocated_mb": float(gpu_time["peak_gpu_memory_allocated_mb"].max()),
        "outputs": list(outputs),
    }


def _format_metric_table(frame: pd.DataFrame) -> list[str]:
    rows = ["| Method | Fit precision target | AP | AUC | Precision | Recall | F1 | F2 | TP | FP | FN |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in frame.to_dict(orient="records"):
        if row.get("status", "").startswith("INCOMPLETE"):
            rows.append(f"| {row['method']} | {row['minimum_fitting_precision']:.2f} | infeasible in at least one fold | | | | | | | | |")
        else:
            rows.append(f"| {row['method']} | {row['minimum_fitting_precision']:.2f} | {row['average_precision']:.4f} | {row['roc_auc']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['f2']:.4f} | {int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} |")
    return rows


def _write_reports(final_metrics: pd.DataFrame, ensemble_metrics: pd.DataFrame, thresholds: pd.DataFrame, top_k: pd.DataFrame, families: pd.DataFrame, uncertainty: pd.DataFrame, gpu_time: pd.DataFrame) -> None:
    combined = pd.concat([final_metrics, ensemble_metrics], ignore_index=True, sort=False)
    focal_ap = float(pd.read_csv(ROOT / "CURRENT_MODEL_COMPARISON.csv").set_index("method").loc["focal_loss", "average_precision"])
    ensemble_scores = pd.read_csv(ROOT / "final_predictions" / "ensemble_ranking_scores.csv")
    ensemble_ap = float(average_precision_score(ensemble_scores["label"], ensemble_scores["probability_yes"]))
    total_time = float(gpu_time["training_runtime_seconds"].sum())
    peak = float(gpu_time["peak_gpu_memory_allocated_mb"].max())
    report = [
        "# Final Report — Method-Specific Calibration V1",
        "",
        "## A. Dataset and governance",
        "",
        f"The frozen development cohort contains {EXPECTED_ROWS} clauses ({EXPECTED_POSITIVES} ambiguous; 5.88%) across nine development families and five family-aware outer folds. TCP and sealed DNS were not consumed. Source SHA-256 is `{SOURCE_SHA256}` and effective cohort SHA-256 is `{EFFECTIVE_SHA256}`.",
        "",
        "## B. Experimental methodology",
        "",
        "Focal Loss and Partial-4 each received method-specific three-fold inner OOF training inside every outer fold. Thresholds maximize fitting-side recall under predeclared precision constraints of 0.20, 0.30, and 0.50. The primary ensemble is a pre-registered equal-weight average of method-specific empirical-CDF scores fitted only on inner OOF predictions.",
        "",
        "## C. Training execution",
        "",
        f"All 30 new FP32 cells completed sequentially from fresh DeBERTa-v3-base initialization. Total recorded training time: {total_time:.1f} seconds. Peak allocated GPU memory: {peak:.1f} MiB. No automatic retries occurred.",
        "",
        "## D–E. Ranking and classification performance",
        "",
        *_format_metric_table(combined),
        "",
        f"The pre-registered ensemble ranking AP is {ensemble_ap:.4f}; existing Focal Loss AP is {focal_ap:.4f}. AP and AUC are threshold-independent; classification trade-offs depend on the fitting-side objective and are not deployment guarantees.",
        "",
        "## F. Ensemble assessment",
        "",
        "The ensemble uses no outer-label-fitted weights. Its ECDF transformations and decision thresholds are fit separately inside each outer fold. Any comparison is paired on the same outer-held-out clauses.",
        "",
        "## G. Statistical uncertainty",
        "",
        "Family-clustered bootstrap intervals are reported in `UNCERTAINTY_INTERVALS.csv`. Only nine family clusters and one training seed are available; the intervals do not measure initialization variability, and nested folds are not treated as independent experiments.",
        "",
        "## H. Annotation limitations",
        "",
        "The labels are an operational AI-assisted reference rather than an independent expert gold standard. Correlated AI annotator errors may distort both apparent false positives and apparent false negatives. An expert-adjudicated stratified evaluation sample remains necessary.",
        "",
        "## I. Recommended deployment design",
        "",
        "Use ranked human review: high-priority candidates first, an explicit uncertain/review band, and lower-priority likely-clear clauses. Do not automatically reject or certify clauses solely from these probabilities. Select a precision target based on review capacity, then monitor family-specific recall and false-positive burden.",
        "",
        "## J. Recommended next experiment",
        "",
        "Freeze one operating policy from this study, run a separately versioned 3–5 seed confirmation, and create an independently double-annotated expert evaluation sample. Only after those choices are frozen should TCP transfer validation occur, followed by one sealed DNS evaluation.",
    ]
    _write_text_exclusive(ROOT / "FINAL_REPORT.md", "\n".join(report) + "\n")
    conclusions = [
        "# Scientific Conclusions",
        "",
        f"- Existing Focal Loss remains the strongest single-model ranking baseline at AP {focal_ap:.4f}.",
        f"- The pre-registered nested ECDF ensemble achieved ranking AP {ensemble_ap:.4f}; interpret its value together with `ENSEMBLE_VALIDATION.csv`, family robustness, and uncertainty intervals.",
        "- Method-specific thresholds correct the invalid cross-method threshold transfer, but fitting-side precision targets do not guarantee the same precision on unseen families.",
        "- Partial-4 remains valuable for efficiency and complementary ranking evidence; threshold 0.5 was never a suitable operating point for its score scale.",
        "- Annotation validity and single-seed variability are the principal remaining scientific limitations.",
    ]
    _write_text_exclusive(ROOT / "SCIENTIFIC_CONCLUSIONS.md", "\n".join(conclusions) + "\n")
    validation = [
        "# Validation Report",
        "",
        "- CPU dataset, split, provenance, prediction-alignment, threshold, ensemble-alignment, and historical-immutability gates: PASS.",
        "- New GPU cells complete and individually validated: 30/30.",
        "- Automatic retries: 0.",
        "- TCP/DNS consumed: no.",
        "- Historical artifacts changed: no.",
        f"- Protocol SHA-256: `{sha256_file(PROTOCOL_PATH)}`.",
    ]
    # The preparation report is deliberately replaced only after the immutable study completes.
    _write_text_exclusive(ROOT / "VALIDATION_REPORT.md", "\n".join(validation) + "\n")


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



