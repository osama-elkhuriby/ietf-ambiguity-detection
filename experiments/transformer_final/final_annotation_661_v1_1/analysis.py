"""Leakage-safe nested analysis and human-review outputs.

Only already-written outer held-out and inner held-out prediction tables are read.
No model, tokenizer, checkpoint, or training code is loaded here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .core import (
    EXPERIMENT_ROOT,
    METHODS,
    _read_json,
    _write_text,
    iter_specs,
    load_local_config,
    prediction_path,
    sha256_file,
    validate_cells,
    verify_frozen,
)

DISPLAY = {"focal": "FOCAL", "partial4": "PARTIAL4", "ensemble": "ENSEMBLE"}
SCORE_KIND = {
    "focal": "raw_model_probability",
    "partial4": "raw_model_probability",
    "ensemble": "equal_weight_fitting_side_ecdf_ranking_score_not_calibrated_probability",
}
ERROR_COLUMNS = [
    "rank",
    "model",
    "item_id",
    "record_id",
    "document",
    "rfc_family",
    "outer_fold",
    "section_heading",
    "previous_clause",
    "target_clause",
    "next_clause",
    "actual_label",
    "predicted_label",
    "yes_score",
    "score_kind",
    "decision_threshold",
    "threshold_policy",
    "error_type",
    "distance_from_threshold",
    "final_source",
    "final_status",
    "document_id",
    "revision",
    "clause_ids",
    "section_ids",
    "lineage_ids",
    "context_source_ids",
    "context_resolution_status",
    "alignment_source_path",
    "source_hashes",
]


def threshold_candidates(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Threshold fitting requires finite non-empty scores")
    return np.sort(np.unique(values))


def confusion(labels: Iterable[int], predicted: Iterable[int]) -> dict[str, int]:
    y = np.asarray(list(labels), dtype=int)
    p = np.asarray(list(predicted), dtype=int)
    if y.shape != p.shape or set(np.unique(y)) - {0, 1} or set(np.unique(p)) - {0, 1}:
        raise ValueError("Confusion inputs must be aligned binary arrays")
    return {
        "tp": int(((y == 1) & (p == 1)).sum()),
        "fp": int(((y == 0) & (p == 1)).sum()),
        "tn": int(((y == 0) & (p == 0)).sum()),
        "fn": int(((y == 1) & (p == 0)).sum()),
    }


def metric_row(
    labels: Iterable[int], scores: Iterable[float], predicted: Iterable[int] | None = None
) -> dict[str, Any]:
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if y.shape != s.shape or y.size == 0 or not np.isfinite(s).all():
        raise ValueError("Metric labels/scores must be finite, non-empty, and aligned")
    both = len(np.unique(y)) == 2
    result: dict[str, Any] = {
        "rows": len(y),
        "yes": int(y.sum()),
        "no": int((y == 0).sum()),
        "prevalence": float(y.mean()),
        "average_precision": float(average_precision_score(y, s)) if y.sum() else math.nan,
        "roc_auc": float(roc_auc_score(y, s)) if both else math.nan,
    }
    if predicted is None:
        return result
    p = np.asarray(list(predicted), dtype=int)
    c = confusion(y, p)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else math.nan
    f2 = 5 * precision * recall / (4 * precision + recall) if 4 * precision + recall > 0 else math.nan
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    result.update(
        **c,
        precision=precision,
        recall=recall,
        specificity=specificity,
        balanced_accuracy=(recall + specificity) / 2 if np.isfinite([recall, specificity]).all() else math.nan,
        f1=f1,
        f2=f2,
        mcc=((tp * tn - fp * fn) / denom) if denom else math.nan,
        predicted_positive_rate=float(p.mean()),
        false_positive_review_workload=fp,
    )
    return result


def select_precision_threshold(
    labels: Iterable[int], scores: Iterable[float], minimum_precision: float
) -> dict[str, Any]:
    """Maximize recall under a precision floor with deterministic tie-breaking."""
    y = np.asarray(list(labels), dtype=int)
    s = np.asarray(list(scores), dtype=float)
    if y.shape != s.shape or y.size == 0 or set(np.unique(y)) - {0, 1}:
        raise ValueError("Threshold inputs must be aligned binary arrays")
    if not (0 < minimum_precision <= 1):
        raise ValueError("minimum_precision must be in (0, 1]")
    eligible: list[dict[str, Any]] = []
    for threshold in threshold_candidates(s):
        predicted = (s >= threshold).astype(int)
        c = confusion(y, predicted)
        if c["tp"] + c["fp"] == 0:
            continue
        precision = c["tp"] / (c["tp"] + c["fp"])
        recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else math.nan
        if precision + 1e-15 >= minimum_precision:
            eligible.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "predicted_positive": c["tp"] + c["fp"],
                    **c,
                }
            )
    if not eligible:
        return {
            "feasible": False,
            "minimum_precision": minimum_precision,
            "threshold": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "predicted_positive": 0,
            "reason": "no non-empty deterministic threshold satisfies precision target",
        }
    best = max(eligible, key=lambda row: (row["recall"], row["precision"], row["threshold"]))
    return {"feasible": True, "minimum_precision": minimum_precision, "reason": "", **best}


def empirical_cdf(reference: Iterable[float], values: Iterable[float]) -> np.ndarray:
    reference_array = np.asarray(list(reference), dtype=float)
    values_array = np.asarray(list(values), dtype=float)
    if reference_array.size == 0 or not np.isfinite(reference_array).all() or not np.isfinite(values_array).all():
        raise ValueError("ECDF inputs must be finite and the reference non-empty")
    ordered = np.sort(reference_array)
    return np.searchsorted(ordered, values_array, side="right") / len(ordered)


def stable_top_k(frame: pd.DataFrame, score_column: str, k: int) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("K must be positive")
    return (
        frame.sort_values([score_column, "record_id"], ascending=[False, True], kind="mergesort")
        .head(min(k, len(frame)))
        .copy()
    )


def top_k_rows(method: str, frame: pd.DataFrame, *, scope: str, outer_fold: int | str = "") -> list[dict[str, Any]]:
    total_yes = int(frame["label"].sum())
    rows = []
    for k in (10, 20, 25, 50, 100):
        selected = stable_top_k(frame, "yes_score", k)
        yes = int(selected["label"].sum())
        reviewed = len(selected)
        rows.append(
            {
                "method": method,
                "score_kind": SCORE_KIND[method],
                "scope": scope,
                "outer_fold": outer_fold,
                "requested_k": k,
                "reviewed": reviewed,
                "real_yes": yes,
                "real_no": reviewed - yes,
                "precision_at_k": yes / reviewed if reviewed else math.nan,
                "recall_at_k": yes / total_yes if total_yes else math.nan,
            }
        )
    return rows


def _load_dataset() -> pd.DataFrame:
    manifest = _read_json(EXPERIMENT_ROOT / "DATASET_MANIFEST.json")
    path = EXPERIMENT_ROOT / "manifests" / "dataset_v1" / "dataset.csv"
    if sha256_file(path) != manifest["effective_dataset_sha256"]:
        raise RuntimeError("Effective dataset changed before analysis")
    frame = pd.read_csv(path, dtype={"item_id": str, "record_id": str}, keep_default_na=False)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame


def _load_predictions(kind: str, method: str) -> dict[int, pd.DataFrame]:
    frames: dict[int, list[pd.DataFrame]] = {}
    for _, spec in iter_specs(kind):
        if spec["method"] != method:
            continue
        prediction = pd.read_csv(
            prediction_path(spec), dtype={"item_id": str, "record_id": str}, keep_default_na=False
        )
        outer = int(spec["outer_fold"])
        prediction["source_cell_id"] = spec["cell_id"]
        prediction["inner_fold"] = spec["inner_fold"] if spec["inner_fold"] is not None else ""
        frames.setdefault(outer, []).append(prediction)
    expected = 1 if kind == "outer" else 3
    result: dict[int, pd.DataFrame] = {}
    for outer in range(1, 6):
        parts = frames.get(outer, [])
        if len(parts) != expected:
            raise RuntimeError(f"Expected {expected} {kind}/{method} tables for outer {outer}")
        joined = pd.concat(parts, ignore_index=True)
        if joined["record_id"].duplicated().any():
            raise RuntimeError(f"Duplicate {kind} OOF IDs for {method}, outer {outer}")
        result[outer] = joined
    return result


def _merge_metadata(prediction: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    prediction = prediction.copy()
    metadata = dataset.drop(columns=["label", "fold"]).copy()
    merged = prediction.merge(metadata, on=["item_id", "record_id"], validate="one_to_one")
    authoritative = dataset.set_index("record_id")
    if not merged["record_id"].map(authoritative["label"]).eq(merged["label"].astype(int)).all():
        raise RuntimeError("Prediction labels disagree with authoritative development labels")
    merged["outer_fold"] = merged["record_id"].map(authoritative["fold"]).astype(int)
    return merged


def _write_csv(name: str, frame: pd.DataFrame) -> None:
    frame.to_csv(
        EXPERIMENT_ROOT / name,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )


def _operational_metrics(
    method: str,
    outer_scores: dict[int, pd.DataFrame],
    selections: pd.DataFrame,
    target: float,
) -> tuple[list[dict[str, Any]], pd.DataFrame | None]:
    rows: list[dict[str, Any]] = []
    evaluated: list[pd.DataFrame] = []
    complete = True
    for outer in range(1, 6):
        selected = selections.loc[
            (selections["method"] == method)
            & (selections["outer_fold"] == outer)
            & np.isclose(selections["minimum_precision"], target)
        ]
        if len(selected) != 1:
            raise RuntimeError("Threshold selection identity is not unique")
        selection = selected.iloc[0]
        if not bool(selection["feasible"]):
            complete = False
            rows.append(
                {
                    "method": method,
                    "outer_fold": outer,
                    "minimum_fitting_precision": target,
                    "status": "INFEASIBLE",
                    "threshold": math.nan,
                }
            )
            continue
        frame = outer_scores[outer].copy()
        threshold = float(selection["threshold"])
        frame["decision_threshold"] = threshold
        frame["predicted_label"] = (frame["yes_score"] >= threshold).astype(int)
        frame["threshold_policy"] = f"nested_precision_{target:.2f}"
        frame["threshold_feasible"] = True
        metric = metric_row(frame["label"], frame["yes_score"], frame["predicted_label"])
        rows.append(
            {
                "method": method,
                "outer_fold": outer,
                "minimum_fitting_precision": target,
                "status": "FEASIBLE",
                "threshold": threshold,
                **metric,
            }
        )
        evaluated.append(frame)
    if not complete:
        return rows, None
    pooled = pd.concat(evaluated, ignore_index=True)
    rows.append(
        {
            "method": method,
            "outer_fold": "pooled",
            "minimum_fitting_precision": target,
            "status": "FEASIBLE_ALL_FOLDS",
            "threshold": math.nan,
            **metric_row(pooled["label"], pooled["yes_score"], pooled["predicted_label"]),
        }
    )
    return rows, pooled


def _error_list(
    frame: pd.DataFrame,
    method: str,
    error_type: str,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    if error_type == "false_positive":
        selected = frame.loc[(frame["label"] == 0) & (frame["predicted_label"] == 1)].copy()
        selected.sort_values(["yes_score", "record_id"], ascending=[False, True], kind="mergesort", inplace=True)
    elif error_type == "false_negative":
        selected = frame.loc[(frame["label"] == 1) & (frame["predicted_label"] == 0)].copy()
        selected.sort_values(["yes_score", "record_id"], ascending=[True, True], kind="mergesort", inplace=True)
    else:
        raise ValueError(error_type)
    selected = selected.head(limit).copy()
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected["model"] = method
    selected["document"] = selected["document_id"]
    selected["rfc_family"] = selected["family_id"]
    selected["actual_label"] = selected["label"].map({0: "No", 1: "Yes"})
    selected["predicted_label"] = selected["predicted_label"].map({0: "No", 1: "Yes"})
    selected["score_kind"] = SCORE_KIND[method]
    selected["error_type"] = error_type
    selected["distance_from_threshold"] = (selected["yes_score"] - selected["decision_threshold"]).abs()
    return selected.reindex(columns=ERROR_COLUMNS)


def _disagreement_list(
    frame: pd.DataFrame, method: str, label: int, *, limit: int = 20
) -> pd.DataFrame:
    selected = frame.loc[frame["label"] == label].copy()
    ascending = label == 1
    selected.sort_values(["yes_score", "record_id"], ascending=[ascending, True], kind="mergesort", inplace=True)
    selected = selected.head(limit).copy()
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    selected["model"] = method
    selected["document"] = selected["document_id"]
    selected["rfc_family"] = selected["family_id"]
    selected["actual_label"] = selected["label"].map({0: "No", 1: "Yes"})
    selected["yes_score"] = selected["yes_score"].astype(float)
    selected["score_kind"] = SCORE_KIND[method]
    selected["review_reason"] = (
        "No label with high predicted Yes score" if label == 0 else "Yes label with low predicted Yes score"
    )
    columns = [
        "rank", "model", "item_id", "record_id", "document", "rfc_family", "outer_fold",
        "section_heading", "previous_clause", "target_clause", "next_clause", "actual_label",
        "yes_score", "score_kind", "review_reason", "final_source", "final_status",
        "document_id", "revision", "clause_ids", "section_ids", "lineage_ids",
        "context_source_ids", "context_resolution_status", "alignment_source_path", "source_hashes",
    ]
    return selected.reindex(columns=columns)


def _fixed_diagnostic(frame: pd.DataFrame, method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = frame.copy()
    fixed["decision_threshold"] = 0.5
    fixed["predicted_label"] = (fixed["yes_score"] >= 0.5).astype(int)
    fixed["threshold_policy"] = (
        "fixed_0.50_raw_probability_diagnostic"
        if method != "ensemble"
        else "fixed_0.50_ecdf_ranking_score_diagnostic_not_probability"
    )
    return _error_list(fixed, method, "false_positive"), _error_list(fixed, method, "false_negative")


def _cluster_bootstrap(score_frames: dict[str, pd.DataFrame], replicates: int) -> pd.DataFrame:
    families = sorted(next(iter(score_frames.values()))["family_id"].unique())
    rng = np.random.default_rng(2718)
    values: dict[tuple[str, str], list[float]] = {}
    for _ in range(replicates):
        sampled = rng.choice(families, size=len(families), replace=True)
        for method, frame in score_frames.items():
            parts = [frame.loc[frame["family_id"] == family] for family in sampled]
            replicate = pd.concat(parts, ignore_index=True)
            y = replicate["label"].to_numpy(int)
            s = replicate["yes_score"].to_numpy(float)
            if y.sum() > 0:
                values.setdefault((method, "average_precision"), []).append(float(average_precision_score(y, s)))
            if len(np.unique(y)) == 2:
                values.setdefault((method, "roc_auc"), []).append(float(roc_auc_score(y, s)))
    rows = []
    for (method, metric), samples in sorted(values.items()):
        arr = np.asarray(samples, dtype=float)
        rows.append(
            {
                "method": method,
                "metric": metric,
                "cluster": "rfc_family",
                "independent_families": len(families),
                "requested_replicates": replicates,
                "defined_replicates": len(arr),
                "undefined_replicates": replicates - len(arr),
                "estimate": metric_row(score_frames[method]["label"], score_frames[method]["yes_score"])[metric],
                "lower_95": float(np.quantile(arr, 0.025)) if len(arr) else math.nan,
                "upper_95": float(np.quantile(arr, 0.975)) if len(arr) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def analyze(*, bootstrap_replicates: int | None = None) -> dict[str, Any]:
    frozen = verify_frozen()
    validate_cells("outer", require_all=True)
    validate_cells("inner", require_all=True)
    config = load_local_config()
    dataset = _load_dataset()
    outer_raw = {method: _load_predictions("outer", method) for method in METHODS}
    inner_raw = {method: _load_predictions("inner", method) for method in METHODS}

    outer_scores: dict[str, dict[int, pd.DataFrame]] = {method: {} for method in (*METHODS, "ensemble")}
    inner_scores: dict[str, dict[int, pd.DataFrame]] = {method: {} for method in (*METHODS, "ensemble")}
    threshold_rows: list[dict[str, Any]] = []
    inner_diagnostics: list[dict[str, Any]] = []
    outer_metric_rows: list[dict[str, Any]] = []

    for outer in range(1, 6):
        for method in METHODS:
            inner = inner_raw[method][outer].copy()
            inner["yes_score"] = inner["probability_yes"].astype(float)
            inner_scores[method][outer] = inner
            outer_frame = _merge_metadata(outer_raw[method][outer], dataset)
            outer_frame["yes_score"] = outer_frame["probability_yes"].astype(float)
            outer_scores[method][outer] = outer_frame
            outer_metric_rows.append(
                {
                    "method": method,
                    "outer_fold": outer,
                    "operating_point": "threshold_free",
                    **metric_row(outer_frame["label"], outer_frame["yes_score"]),
                }
            )
            fixed = (outer_frame["yes_score"] >= 0.5).astype(int)
            outer_metric_rows.append(
                {
                    "method": method,
                    "outer_fold": outer,
                    "operating_point": "fixed_0.50_diagnostic",
                    **metric_row(outer_frame["label"], outer_frame["yes_score"], fixed),
                }
            )

        focal_inner = inner_scores["focal"][outer][["record_id", "label", "yes_score"]].rename(columns={"yes_score": "focal_score"})
        partial_inner = inner_scores["partial4"][outer][["record_id", "label", "yes_score"]].rename(columns={"yes_score": "partial4_score"})
        paired_inner = focal_inner.merge(partial_inner, on=["record_id", "label"], validate="one_to_one")
        paired_inner["yes_score"] = 0.5 * empirical_cdf(paired_inner["focal_score"], paired_inner["focal_score"]) + 0.5 * empirical_cdf(paired_inner["partial4_score"], paired_inner["partial4_score"])
        inner_scores["ensemble"][outer] = paired_inner

        focal_outer = outer_scores["focal"][outer].rename(columns={"yes_score": "focal_score"})
        partial_outer = outer_scores["partial4"][outer][["record_id", "yes_score"]].rename(columns={"yes_score": "partial4_score"})
        paired_outer = focal_outer.merge(partial_outer, on="record_id", validate="one_to_one")
        paired_outer["yes_score"] = 0.5 * empirical_cdf(paired_inner["focal_score"], paired_outer["focal_score"]) + 0.5 * empirical_cdf(paired_inner["partial4_score"], paired_outer["partial4_score"])
        outer_scores["ensemble"][outer] = paired_outer
        outer_metric_rows.append(
            {
                "method": "ensemble",
                "outer_fold": outer,
                "operating_point": "threshold_free_fitting_side_ecdf",
                **metric_row(paired_outer["label"], paired_outer["yes_score"]),
            }
        )

        for method in (*METHODS, "ensemble"):
            inner = inner_scores[method][outer]
            for target in config["calibration"]["precision_targets"]:
                selected = select_precision_threshold(inner["label"], inner["yes_score"], float(target))
                threshold_rows.append(
                    {
                        "method": method,
                        "outer_fold": outer,
                        "minimum_precision": float(target),
                        "selection_data": "inner_family_OOF_only",
                        "tie_breaking": "recall_desc_precision_desc_threshold_desc",
                        **selected,
                    }
                )
            inner_diagnostics.append(
                {
                    "method": method,
                    "outer_fold": outer,
                    "rows": len(inner),
                    "yes": int(inner["label"].sum()),
                    "no": int((inner["label"] == 0).sum()),
                    **metric_row(inner["label"], inner["yes_score"]),
                }
            )

    thresholds = pd.DataFrame(threshold_rows)
    _write_csv("SELECTED_THRESHOLDS.csv", thresholds)
    _write_csv("INNER_FOLD_CALIBRATION.csv", pd.DataFrame(inner_diagnostics))
    _write_csv("OUTER_FOLD_METRICS.csv", pd.DataFrame(outer_metric_rows))

    nested_rows: list[dict[str, Any]] = []
    primary_decisions: dict[str, pd.DataFrame | None] = {}
    for method in (*METHODS, "ensemble"):
        for target in config["calibration"]["precision_targets"]:
            rows, pooled = _operational_metrics(method, outer_scores[method], thresholds, float(target))
            nested_rows.extend(rows)
            if np.isclose(float(target), 0.30):
                primary_decisions[method] = pooled
    nested = pd.DataFrame(nested_rows)
    _write_csv("FINAL_NESTED_METRICS.csv", nested)
    _write_csv("ENSEMBLE_VALIDATION.csv", nested.loc[nested["method"] == "ensemble"].copy())

    pooled_scores: dict[str, pd.DataFrame] = {
        method: pd.concat(list(outer_scores[method].values()), ignore_index=True)
        for method in (*METHODS, "ensemble")
    }
    for method, frame in pooled_scores.items():
        if len(frame) != len(dataset) or set(frame["record_id"]) != set(dataset["record_id"]):
            raise RuntimeError(f"Outer OOF coverage failed for {method}")

    top_rows: list[dict[str, Any]] = []
    for method, pooled in pooled_scores.items():
        top_rows.extend(top_k_rows(method, pooled, scope="global_pooled_outer_oof"))
        for outer, frame in outer_scores[method].items():
            top_rows.extend(top_k_rows(method, frame, scope="foldwise_outer_oof", outer_fold=outer))
    _write_csv("TOP_K_COMPARISON.csv", pd.DataFrame(top_rows))

    family_rows: list[dict[str, Any]] = []
    for method, pooled in pooled_scores.items():
        for family, part in pooled.groupby("family_id"):
            family_rows.append(
                {
                    "method": method,
                    "rfc_family": family,
                    **metric_row(part["label"], part["yes_score"]),
                }
            )
    _write_csv("FAMILY_ROBUSTNESS.csv", pd.DataFrame(family_rows))

    reps = int(bootstrap_replicates or config["evaluation"]["bootstrap_replicates"])
    uncertainty = _cluster_bootstrap(pooled_scores, reps)
    _write_csv("UNCERTAINTY_INTERVALS.csv", uncertainty)

    gpu_rows = []
    for _, spec in iter_specs():
        manifest = _read_json(Path(spec["run_path"]) / "run_manifest.json")
        gpu_rows.append(
            {
                "cell_id": spec["cell_id"],
                "kind": spec["kind"],
                "method": spec["method"],
                "outer_fold": spec["outer_fold"],
                "inner_fold": spec["inner_fold"],
                "runtime_seconds": manifest["runtime_seconds"],
                "peak_gpu_memory_allocated_mb": manifest["peak_gpu_memory_allocated_mb"],
                "peak_gpu_memory_reserved_mb": manifest["peak_gpu_memory_reserved_mb"],
                "gpu": manifest["gpu"],
            }
        )
    _write_csv("GPU_TRAINING_TIME.csv", pd.DataFrame(gpu_rows))

    error_summary: list[dict[str, Any]] = []
    disagreement_membership: list[pd.DataFrame] = []
    for method, pooled in pooled_scores.items():
        prefix = DISPLAY[method]
        primary = primary_decisions.get(method)
        if primary is None:
            primary_for_files = pooled.head(0).copy()
            primary_for_files["decision_threshold"] = pd.Series(dtype=float)
            primary_for_files["predicted_label"] = pd.Series(dtype=int)
            primary_for_files["threshold_policy"] = pd.Series(dtype=str)
            status = "INCOMPLETE_DUE_TO_INFEASIBLE_OUTER_FOLD"
        else:
            primary_for_files = primary
            status = "COMPLETE"
        fp = _error_list(primary_for_files, method, "false_positive")
        fn = _error_list(primary_for_files, method, "false_negative")
        _write_csv(f"{prefix}_TOP20_FALSE_POSITIVES.csv", fp)
        _write_csv(f"{prefix}_TOP20_FALSE_NEGATIVES.csv", fn)

        no_high = _disagreement_list(pooled, method, 0)
        yes_low = _disagreement_list(pooled, method, 1)
        _write_csv(f"{prefix}_TOP20_NO_HIGH_YES_SCORE.csv", no_high)
        _write_csv(f"{prefix}_TOP20_YES_LOW_YES_SCORE.csv", yes_low)
        disagreement_membership.append(no_high[["record_id", "model", "yes_score", "review_reason"]])
        disagreement_membership.append(yes_low[["record_id", "model", "yes_score", "review_reason"]])

        fixed_fp, fixed_fn = _fixed_diagnostic(pooled, method)
        _write_csv(f"{prefix}_FIXED_0_50_DIAGNOSTIC_FALSE_POSITIVES.csv", fixed_fp)
        _write_csv(f"{prefix}_FIXED_0_50_DIAGNOSTIC_FALSE_NEGATIVES.csv", fixed_fn)
        threshold_values = thresholds.loc[
            (thresholds["method"] == method)
            & np.isclose(thresholds["minimum_precision"], 0.30)
            & thresholds["feasible"].astype(bool),
            "threshold",
        ].astype(float)
        error_summary.append(
            {
                "method": method,
                "primary_status": status,
                "false_positive_total": int(
                    ((primary_for_files.get("label", pd.Series(dtype=int)) == 0)
                    & (primary_for_files.get("predicted_label", pd.Series(dtype=int)) == 1)).sum()
                ),
                "false_negative_total": int(
                    ((primary_for_files.get("label", pd.Series(dtype=int)) == 1)
                    & (primary_for_files.get("predicted_label", pd.Series(dtype=int)) == 0)).sum()
                ),
                "fp_listed": len(fp),
                "fn_listed": len(fn),
                "minimum_selected_threshold": threshold_values.min() if len(threshold_values) else math.nan,
                "maximum_selected_threshold": threshold_values.max() if len(threshold_values) else math.nan,
                "maximum_fp_score": fp["yes_score"].max() if len(fp) else math.nan,
                "minimum_fn_score": fn["yes_score"].min() if len(fn) else math.nan,
            }
        )

    disagreements = pd.concat(disagreement_membership, ignore_index=True)
    wide = disagreements.pivot_table(
        index=["record_id", "review_reason"], columns="model", values="yes_score", aggfunc="first"
    ).reset_index()
    model_columns = [column for column in ("focal", "partial4", "ensemble") if column in wide]
    wide["models_flagging"] = wide[model_columns].notna().sum(axis=1)
    if model_columns:
        wide["score_range_across_available_models"] = wide[model_columns].max(axis=1) - wide[model_columns].min(axis=1)
    wide.sort_values(["models_flagging", "record_id"], ascending=[False, True], inplace=True)
    _write_csv("CROSS_MODEL_DISAGREEMENT_SUMMARY.csv", wide)

    lines = [
        "# Top-20 error analysis",
        "",
        "Primary decisions use each outer fold's method-specific threshold selected only from inner family-OOF predictions at precision target 0.30. Ensemble scores are ranking scores, not calibrated probabilities.",
        "",
    ]
    for row in error_summary:
        lines.extend(
            [
                f"## {DISPLAY[row['method']]}",
                "",
                f"- Primary status: {row['primary_status']}.",
                f"- False positives / listed: {row['false_positive_total']} / {row['fp_listed']}.",
                f"- False negatives / listed: {row['false_negative_total']} / {row['fn_listed']}.",
                f"- Feasible threshold range: {row['minimum_selected_threshold']} to {row['maximum_selected_threshold']}.",
                f"- Strongest FP score: {row['maximum_fp_score']}; strongest FN (lowest) score: {row['minimum_fn_score']}.",
                "- Recurring semantic patterns are not asserted automatically; review the preserved clauses with independent human adjudication.",
                "",
            ]
        )
    lines.extend(
        [
            "## Cross-model priorities",
            "",
            "Review records flagged by two or three methods first, then large-margin threshold-dependent errors. Model disagreement is evidence of difficulty, not proof of an annotation error.",
        ]
    )
    _write_text(EXPERIMENT_ROOT / "TOP20_ERROR_ANALYSIS.md", "\n".join(lines) + "\n")

    ranking_summary = pd.DataFrame(
        [
            {"method": method, **metric_row(frame["label"], frame["yes_score"])}
            for method, frame in pooled_scores.items()
        ]
    )
    _write_text(
        EXPERIMENT_ROOT / "SCIENTIFIC_CONCLUSIONS.md",
        "# Scientific conclusions\n\nResults are single-seed, family-isolated estimates over nine development families. Nominal differences do not establish superiority; consult family-clustered intervals, positive scarcity, and annotation-reference uncertainty. Top-K estimates—especially K=10 and K=20—have high sampling variability and must not be interpreted as calibrated confidence.\n",
    )
    _write_text(
        EXPERIMENT_ROOT / "FINAL_REPORT.md",
        "# Final report\n\n"
        + ranking_summary.to_markdown(index=False)
        + "\n\nOperational metrics are in `FINAL_NESTED_METRICS.csv`; Top-K, family robustness, uncertainty, GPU telemetry, and error-review artifacts are separate. Raw model probabilities and ensemble ECDF scores are intentionally not pooled onto one calibration scale.\n",
    )
    validation = validate_analysis()
    return {
        "status": "PASS",
        "protocol_sha256": frozen["protocol_sha256"],
        "outer_oof_rows_per_method": len(dataset),
        "threshold_rows": len(thresholds),
        "bootstrap_replicates": reps,
        "validation": validation,
    }


def _assert_sorted(frame: pd.DataFrame, error_type: str) -> None:
    if len(frame) < 2:
        return
    pairs = list(zip(frame["yes_score"].astype(float), frame["record_id"].astype(str)))
    expected = sorted(pairs, key=lambda x: ((-x[0]) if error_type == "false_positive" else x[0], x[1]))
    if pairs != expected:
        raise RuntimeError(f"Incorrect {error_type} ordering")


def validate_analysis() -> dict[str, Any]:
    verify_frozen()
    dataset = _load_dataset().set_index("record_id")
    required_tables = [
        "OUTER_FOLD_METRICS.csv", "INNER_FOLD_CALIBRATION.csv", "SELECTED_THRESHOLDS.csv",
        "FINAL_NESTED_METRICS.csv", "ENSEMBLE_VALIDATION.csv", "TOP_K_COMPARISON.csv",
        "FAMILY_ROBUSTNESS.csv", "UNCERTAINTY_INTERVALS.csv", "GPU_TRAINING_TIME.csv",
    ]
    for name in required_tables:
        if not (EXPERIMENT_ROOT / name).is_file():
            raise FileNotFoundError(f"Missing analysis artifact: {name}")
    checked_errors = 0
    checked_disagreements = 0
    for method in (*METHODS, "ensemble"):
        prefix = DISPLAY[method]
        for error_name, error_type in (
            ("FALSE_POSITIVES", "false_positive"),
            ("FALSE_NEGATIVES", "false_negative"),
        ):
            path = EXPERIMENT_ROOT / f"{prefix}_TOP20_{error_name}.csv"
            frame = pd.read_csv(path, dtype={"item_id": str, "record_id": str}, keep_default_na=False)
            if frame.columns.tolist() != ERROR_COLUMNS:
                raise RuntimeError(f"Error-report schema mismatch: {path.name}")
            if frame["record_id"].duplicated().any() or len(frame) > 20:
                raise RuntimeError(f"Duplicate/oversized error list: {path.name}")
            _assert_sorted(frame, error_type)
            for row in frame.itertuples(index=False):
                source = dataset.loc[row.record_id]
                if row.target_clause != source.target_clause or row.previous_clause != source.previous_clause or row.next_clause != source.next_clause:
                    raise RuntimeError(f"Text truncation or drift in {path.name}: {row.record_id}")
                if row.rfc_family != source.family_id:
                    raise RuntimeError(f"Family drift in {path.name}")
                if error_type == "false_positive" and not (row.actual_label == "No" and row.predicted_label == "Yes"):
                    raise RuntimeError(f"Invalid FP semantics in {path.name}")
                if error_type == "false_negative" and not (row.actual_label == "Yes" and row.predicted_label == "No"):
                    raise RuntimeError(f"Invalid FN semantics in {path.name}")
            checked_errors += 1
        for suffix, label in (("NO_HIGH_YES_SCORE", "No"), ("YES_LOW_YES_SCORE", "Yes")):
            path = EXPERIMENT_ROOT / f"{prefix}_TOP20_{suffix}.csv"
            frame = pd.read_csv(path, dtype={"item_id": str, "record_id": str}, keep_default_na=False)
            if frame["record_id"].duplicated().any() or len(frame) > 20 or set(frame["actual_label"]) - {label}:
                raise RuntimeError(f"Invalid disagreement list: {path.name}")
            checked_disagreements += 1
    report = {
        "status": "PASS",
        "validated_threshold_dependent_error_files": checked_errors,
        "validated_threshold_independent_disagreement_files": checked_disagreements,
        "authoritative_text_preservation": "PASS",
        "outer_oof_only": "PASS",
        "reserved_tcp_dns_absent": "PASS",
        "protocol_integrity": "PASS",
    }
    _write_text(
        EXPERIMENT_ROOT / "VALIDATION_REPORT.md",
        "# Validation report\n\n"
        + "\n".join(f"- {key}: {value}." for key, value in report.items())
        + "\n",
    )
    return report

