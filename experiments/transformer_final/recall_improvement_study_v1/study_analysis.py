"""CPU-only analysis for the isolated recall-improvement study."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from models.transformer_final.common import (
    read_json,
    resolve_path,
    sha256_file,
    write_json,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.experiment2 import (
    DEFAULT_EXPERIMENT_CONFIG,
)
from models.transformer_final.experiment2_analysis import (
    THRESHOLDS,
    TOP_K_VALUES,
    _threshold_metrics,
)

STUDY_ROOT = Path(__file__).resolve().parent
ANALYSIS_ROOT = STUDY_ROOT / "analysis"
BASELINE_ROOT = resolve_path(
    "results/transformer/final_studies/final_annotation_ready_pilot_v1/aggregate_metrics"
)
STAGE_TO_MODEL = {
    "C_local_context": ("context", "C_local_context"),
    "G_partial_4": ("partial_4", "G_partial_4"),
    "D_focal_loss": ("focal_gamma_2", "D_focal_loss"),
    "G2_partial_2": ("partial_2", "G2_partial_2"),
}
THRESHOLD_CUES = {
    "BCP14 modality": re.compile(
        r"\b(must|shall|should|may|must not|shall not|should not|required|optional)\b",
        re.IGNORECASE,
    ),
    "negation or exception": re.compile(
        r"\b(not|no|never|unless|except|only|without|neither|nor)\b",
        re.IGNORECASE,
    ),
    "conditional": re.compile(
        r"\b(if|when|while|provided|subject to)\b", re.IGNORECASE
    ),
    "temporal": re.compile(
        r"\b(before|after|until|within|during|prior|following|immediately)\b",
        re.IGNORECASE,
    ),
    "comparison or quantity": re.compile(
        r"\b(less|more|greater|at least|at most|exactly|one or more|each|any)\b",
        re.IGNORECASE,
    ),
}


def _load_effective() -> pd.DataFrame:
    config = load_config(DEFAULT_EXPERIMENT_CONFIG)
    frame, _, _ = load_effective_dataset(config)
    frame["record_id"] = frame["record_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    return frame


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _metric_row(
    frame: pd.DataFrame, threshold: float, **identity: Any
) -> dict[str, Any]:
    y = frame["label"].to_numpy(dtype=np.int64)
    score = frame["probability_yes"].to_numpy(dtype=np.float64)
    result = _threshold_metrics(frame, threshold)
    result.update(
        {
            "average_precision": float(average_precision_score(y, score)),
            "roc_auc": float(roc_auc_score(y, score)),
            "brier_score": float(np.mean((score - y) ** 2)),
        }
    )
    result.update(identity)
    return result


def _choose_threshold(frame: pd.DataFrame, candidates: list[float]) -> dict[str, Any]:
    rows = [_threshold_metrics(frame, threshold) for threshold in candidates]
    feasible = [row for row in rows if row["precision"] >= 0.20]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (row["recall"], row["precision"], row["threshold"]),
        )
        selection_policy = "maximum_recall_at_precision_at_least_0.20"
    else:
        selected = max(
            rows,
            key=lambda row: (row["f2"], row["precision"], row["threshold"]),
        )
        selection_policy = "fallback_maximum_f2_then_precision_then_threshold"
    return {**selected, "selection_policy": selection_policy}


def _deployed_metric_row(frame: pd.DataFrame, **identity: Any) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    scores = frame["probability_yes"].to_numpy(dtype=np.float64)
    predicted = frame["predicted_label_nested_threshold"].to_numpy(dtype=np.int64)
    tp = int(np.count_nonzero((labels == 1) & (predicted == 1)))
    fp = int(np.count_nonzero((labels == 0) & (predicted == 1)))
    tn = int(np.count_nonzero((labels == 0) & (predicted == 0)))
    fn = int(np.count_nonzero((labels == 1) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if 4 * precision + recall
        else 0.0
    )
    mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    row = {
        "threshold": "fold_specific",
        "rows": len(frame),
        "actual_yes": int(labels.sum()),
        "actual_no": int((labels == 0).sum()),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "predicted_yes": tp + fp,
        "predicted_no": tn + fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "f2": f2,
        "balanced_accuracy": (recall + specificity) / 2,
        "mcc": ((tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier_score": float(np.mean((scores - labels) ** 2)),
    }
    row.update(identity)
    return row


def _read_nested_oof() -> tuple[pd.DataFrame, dict[int, dict[str, Any]]]:
    protocol = read_json(
        STUDY_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
    )
    splits = json.loads(
        (STUDY_ROOT / "protocols" / "nested_inner_splits_v1.json").read_text(
            encoding="utf-8"
        )
    )
    effective = _load_effective()
    expected = set(effective["record_id"].astype(str))
    output_frames: list[pd.DataFrame] = []
    thresholds: dict[int, dict[str, Any]] = {}
    for outer_fold in range(1, 6):
        outer_expected = set(
            effective.loc[effective["fold"] != outer_fold, "record_id"].astype(str)
        )
        outer_frames: list[pd.DataFrame] = []
        seen_ids: set[str] = set()
        for split in [item for item in splits if item["outer_fold"] == outer_fold]:
            inner_fold = int(split["inner_fold"])
            run_path = (
                STUDY_ROOT
                / "runs"
                / "B_nested_threshold_calibration"
                / f"outer_{outer_fold}_inner_{inner_fold}"
            )
            run_manifest = read_json(run_path / "run_manifest.json")
            predictions = pd.read_csv(
                run_path / "predictions" / "held_out_predictions.csv",
                dtype={"record_id": str},
                keep_default_na=False,
            )
            if (
                run_manifest.get("condition") != "A2"
                or run_manifest.get("precision") != "fp32"
                or run_manifest.get("input_mode") != "target_only"
                or set(predictions["record_id"].astype(str))
                != set(split["scoring_record_ids"])
                or seen_ids & set(predictions["record_id"].astype(str))
            ):
                raise RuntimeError(
                    f"Nested predictions failed identity checks: outer={outer_fold}, "
                    f"inner={inner_fold}"
                )
            predictions["outer_fold"] = outer_fold
            predictions["inner_fold"] = inner_fold
            predictions["label"] = predictions["label"].astype(int)
            predictions["probability_yes"] = predictions["probability_yes"].astype(
                float
            )
            seen_ids.update(predictions["record_id"].astype(str))
            outer_frames.append(predictions)
        inner_oof = pd.concat(outer_frames, ignore_index=True)
        if seen_ids != outer_expected or seen_ids & expected.intersection(
            effective.loc[effective["fold"] == outer_fold, "record_id"].astype(str)
        ):
            raise RuntimeError(
                f"Nested OOF coverage is invalid for outer fold {outer_fold}"
            )
        verified = inner_oof.merge(
            effective[["record_id", "label", "family_id", "fold"]],
            on="record_id",
            suffixes=("", "_dataset"),
            validate="one_to_one",
        )
        if (
            (
                verified["label"].astype(int) != verified["label_dataset"].astype(int)
            ).any()
            or (
                verified["family_id"].astype(str)
                != verified["family_id_dataset"].astype(str)
            ).any()
            or (
                verified["fold"].astype(int) != verified["fold_dataset"].astype(int)
            ).any()
            or not np.isfinite(
                verified["probability_yes"].to_numpy(dtype=np.float64)
            ).all()
            or (
                verified["probability_yes"].lt(0) | verified["probability_yes"].gt(1)
            ).any()
        ):
            raise RuntimeError(
                f"Nested OOF labels or probabilities invalid for fold {outer_fold}"
            )
        inner_oof["outer_fold"] = outer_fold
        inner_oof["family_id"] = verified["family_id"].to_numpy()
        inner_oof["outer_fold_assignment"] = verified["fold"].to_numpy(dtype=int)
        output_frames.append(inner_oof)
        thresholds[outer_fold] = _choose_threshold(
            inner_oof,
            [float(value) for value in protocol["inner_cv"]["threshold_candidates"]],
        )
    return pd.concat(output_frames, ignore_index=True), thresholds


def build_nested_threshold_analysis() -> Path:
    destination = ANALYSIS_ROOT / "nested_threshold_calibration"
    if destination.exists():
        raise FileExistsError(f"Analysis output is immutable: {destination}")
    destination.mkdir(parents=True)
    oof, selected = _read_nested_oof()
    effective = _load_effective()
    baseline_path = BASELINE_ROOT / "pooled_oof_predictions.csv"
    baseline = pd.read_csv(
        baseline_path, dtype={"record_id": str}, keep_default_na=False
    )
    if baseline["record_id"].duplicated().any() or set(
        baseline["record_id"].astype(str)
    ) != set(effective["record_id"].astype(str)):
        raise RuntimeError("Frozen baseline OOF predictions do not cover the cohort")
    baseline["label"] = baseline["label"].astype(int)
    baseline["fold"] = baseline["fold"].astype(int)
    baseline["probability_yes"] = baseline["probability_yes"].astype(float)
    baseline_verified = baseline.merge(
        effective[["record_id", "label", "fold"]],
        on="record_id",
        suffixes=("", "_dataset"),
        validate="one_to_one",
    )
    if (
        (
            baseline_verified["label"].astype(int)
            != baseline_verified["label_dataset"].astype(int)
        ).any()
        or (
            baseline_verified["fold"].astype(int)
            != baseline_verified["fold_dataset"].astype(int)
        ).any()
        or not np.isfinite(
            baseline_verified["probability_yes"].to_numpy(dtype=np.float64)
        ).all()
        or (
            (baseline_verified["probability_yes"] < 0)
            | (baseline_verified["probability_yes"] > 1)
        ).any()
    ):
        raise RuntimeError("Frozen baseline OOF rows fail label/fold/score validation")
    thresholds = pd.DataFrame(
        [
            {
                "outer_fold": fold,
                "selected_threshold": value["threshold"],
                "inner_precision": value["precision"],
                "inner_recall": value["recall"],
                "inner_f2": value["f2"],
                "inner_false_positives": value["false_positives"],
                "inner_false_negatives": value["false_negatives"],
                "selection_policy": value["selection_policy"],
                "inner_oof_rows": value["rows"],
                "inner_oof_positive": value["actual_yes"],
            }
            for fold, value in sorted(selected.items())
        ]
    )
    deployed = baseline.copy()
    deployed["selected_threshold"] = deployed["fold"].map(
        {fold: value["threshold"] for fold, value in selected.items()}
    )
    if deployed["selected_threshold"].isna().any():
        raise RuntimeError("No fitting-side threshold for one or more outer folds")
    deployed["predicted_label_nested_threshold"] = (
        deployed["probability_yes"] >= deployed["selected_threshold"]
    ).astype(int)
    metric_rows = [
        _metric_row(
            baseline,
            0.5,
            evaluation="pooled_outer_oof",
            method="weighted_ce_0.5",
        )
    ]
    deployed_for_metrics = deployed.rename(
        columns={"selected_threshold": "threshold_for_metric"}
    )
    for fold, group in deployed_for_metrics.groupby("fold", sort=True):
        group = group.copy()
        group["predicted_label_nested_threshold"] = (
            group["probability_yes"] >= group["threshold_for_metric"]
        ).astype(int)
        metric_rows.append(
            _deployed_metric_row(
                group,
                evaluation="outer_fold",
                method="nested_threshold",
                fold=int(fold),
            )
        )
    deployed["predicted_label_nested_threshold"] = (
        deployed["probability_yes"] >= deployed["selected_threshold"]
    ).astype(int)
    metric_rows.append(
        _deployed_metric_row(
            deployed,
            evaluation="pooled_outer_oof",
            method="nested_threshold",
        )
    )
    _write_csv(destination / "inner_oof_predictions.csv", oof)
    _write_csv(destination / "selected_thresholds_by_outer_fold.csv", thresholds)
    _write_csv(destination / "outer_oof_predictions_nested_threshold.csv", deployed)
    _write_csv(destination / "outer_metrics.csv", pd.DataFrame(metric_rows))
    write_json(
        destination / "analysis_manifest.json",
        {
            "protocol_sha256": sha256_file(
                STUDY_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
            ),
            "nested_splits_sha256": sha256_file(
                STUDY_ROOT / "protocols" / "nested_inner_splits_v1.json"
            ),
            "baseline_predictions_sha256": sha256_file(baseline_path),
            "nested_scoring_rows": len(oof),
            "nested_rows_per_outer_fold": {
                str(fold): int((oof["outer_fold"] == fold).sum())
                for fold in range(1, 6)
            },
            "threshold_policy": (
                "per outer fold, maximum inner-OOF recall at precision >= 0.20; "
                "if unavailable, maximum F2 then precision then threshold"
            ),
            "thresholds_use_outer_labels": False,
        },
    )
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(thresholds["outer_fold"].astype(str), thresholds["selected_threshold"])
    axis.set(
        title="Nested Fitting-Side Threshold by Outer Fold",
        xlabel="Outer fold",
        ylabel="Selected threshold",
    )
    fig.tight_layout()
    fig.savefig(destination / "thresholds_by_outer_fold.png", dpi=160)
    plt.close(fig)
    report_lines = [
        "# Nested Fitting-Side Threshold Calibration",
        "",
        (
            "This analysis changes only each outer fold's decision threshold. Model "
            "weights, scores, and outer-fold labels were not used to choose thresholds."
        ),
        "",
        "| Outer fold | Inner OOF rows | Inner precision | Inner recall | Selected threshold | Selection rule |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in thresholds.to_dict(orient="records"):
        report_lines.append(
            f"| {row['outer_fold']} | {row['inner_oof_rows']} | "
            f"{row['inner_precision']:.4f} | {row['inner_recall']:.4f} | "
            f"{row['selected_threshold']:.4f} | {row['selection_policy']} |"
        )
    report_lines.extend(
        [
            "",
            (
                "The outer-fold metrics are descriptive evaluation only. Per-fold "
                "thresholds are selected on the other four families' inner OOF "
                "predictions, after the frozen leakage exclusions."
            ),
            "",
            (
                "Probability-based average precision, ROC-AUC, and Brier score are "
                "unchanged by threshold deployment. The pooled fold-specific "
                "precision/recall counts are computed using one selected cutoff per "
                "outer fold; there is no single pooled cutoff."
            ),
        ]
    )
    (destination / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return destination


def _model_predictions(
    model_id: str, stage: str | None, effective: pd.DataFrame
) -> pd.DataFrame | None:
    if stage is None:
        path = BASELINE_ROOT / "pooled_oof_predictions.csv"
        frame = pd.read_csv(path, dtype={"record_id": str}, keep_default_na=False)
    else:
        inventory = read_json(STUDY_ROOT / "EXPERIMENT_INVENTORY.json")
        if inventory.get("stages", {}).get(stage, {}).get("status") != "COMPLETE":
            return None
        records = [
            value
            for key, value in inventory.get("cells", {}).items()
            if key.startswith(stage + "/")
        ]
        expected = 15 if stage == "B_nested_threshold_calibration" else 5
        if len(records) != expected or any(
            record.get("status") != "COMPLETE" for record in records
        ):
            raise RuntimeError(f"Cannot analyze incomplete stage {stage}")
        pieces = []
        for fold in range(1, 6):
            run = STUDY_ROOT / "runs" / stage / f"fold_{fold}" / f"seed_{record_seed()}"
            piece = pd.read_csv(
                run / "predictions" / "held_out_predictions.csv",
                dtype={"record_id": str},
                keep_default_na=False,
            )
            piece["fold"] = fold
            pieces.append(piece)
        frame = pd.concat(pieces, ignore_index=True)
    frame["record_id"] = frame["record_id"].astype(str)
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    frame["probability_yes"] = frame["probability_yes"].astype(float)
    if frame["record_id"].duplicated().any() or set(frame["record_id"]) != set(
        effective["record_id"]
    ):
        raise RuntimeError(f"OOF predictions do not cover the cohort: {model_id}")
    frame = frame.merge(
        effective[["record_id", "label", "fold", "family_id", "target_clause"]],
        on="record_id",
        suffixes=("", "_dataset"),
        validate="one_to_one",
    )
    if (frame["label"].astype(int) != frame["label_dataset"].astype(int)).any() or (
        frame["fold"].astype(int) != frame["fold_dataset"].astype(int)
    ).any():
        raise RuntimeError(f"OOF identity mismatch: {model_id}")
    frame["label"] = frame["label_dataset"].astype(int)
    frame["fold"] = frame["fold_dataset"].astype(int)
    return frame


def record_seed() -> int:
    protocol = read_json(
        STUDY_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
    )
    return int(protocol["training"]["seed"])


def _cue_summary(text: str) -> str:
    return (
        "; ".join(
            label for label, pattern in THRESHOLD_CUES.items() if pattern.search(text)
        )
        or "no listed lexical cue"
    )


def build_error_analysis() -> Path:
    destination = ANALYSIS_ROOT / "error_analysis"
    if destination.exists():
        raise FileExistsError(f"Analysis output is immutable: {destination}")
    destination.mkdir(parents=True)
    effective = _load_effective()
    methods: list[tuple[str, str | None, str | None]] = [
        ("weighted_ce_at_0.5", None, None)
    ]
    inventory = read_json(STUDY_ROOT / "EXPERIMENT_INVENTORY.json")
    completed = [
        (model, stage)
        for stage, (model, _) in STAGE_TO_MODEL.items()
        if inventory.get("stages", {}).get(stage, {}).get("status") == "COMPLETE"
    ]
    methods.extend((model, stage, stage) for model, stage in completed)
    threshold_file = (
        ANALYSIS_ROOT
        / "nested_threshold_calibration"
        / "selected_thresholds_by_outer_fold.csv"
    )
    if threshold_file.is_file():
        methods.append(
            ("weighted_ce_nested_threshold", "B_nested_threshold_calibration", None)
        )
    examples: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for model_id, stage, _ in methods:
        if stage == "B_nested_threshold_calibration":
            predictions = pd.read_csv(
                ANALYSIS_ROOT
                / "nested_threshold_calibration"
                / "outer_oof_predictions_nested_threshold.csv",
                dtype={"record_id": str},
            )
            predictions = predictions.merge(
                effective[["record_id", "family_id", "target_clause"]],
                on="record_id",
                suffixes=("", "_dataset"),
                validate="one_to_one",
            )
            if (
                predictions["family_id"].astype(str)
                != predictions["family_id_dataset"].astype(str)
            ).any():
                raise RuntimeError("Nested threshold family IDs changed")
            predictions["threshold"] = predictions["selected_threshold"]
        else:
            predictions = _model_predictions(model_id, stage, effective)
            if predictions is None:
                continue
            predictions["threshold"] = 0.5
        predictions["error"] = (
            predictions["probability_yes"] >= predictions["threshold"]
        ).astype(int) != predictions["label"].astype(int)
        for fold, group in predictions.groupby("fold", sort=True):
            fn_count = int(
                (
                    (group["label"] == 1)
                    & (group["probability_yes"] < group["threshold"])
                ).sum()
            )
            fp_count = int(
                (
                    (group["label"] == 0)
                    & (group["probability_yes"] >= group["threshold"])
                ).sum()
            )
            summaries.append(
                {
                    "model": model_id,
                    "fold": int(fold),
                    "threshold": float(group["threshold"].iloc[0]),
                    "false_negatives": fn_count,
                    "false_positives": fp_count,
                    "records": len(group),
                }
            )
            for label, subset in (
                ("false_negative", group.loc[group["label"] == 1]),
                ("false_positive", group.loc[group["label"] == 0]),
            ):
                incorrect = subset.loc[subset["error"]].copy()
                incorrect["threshold_distance"] = (
                    incorrect["probability_yes"] - incorrect["threshold"]
                ).abs()
                for row in (
                    incorrect.sort_values(
                        ["threshold_distance", "record_id"], kind="mergesort"
                    )
                    .head(3)
                    .to_dict(orient="records")
                ):
                    examples.append(
                        {
                            "model": model_id,
                            "fold": int(fold),
                            "error_type": label,
                            "record_id": row["record_id"],
                            "family_id": row["family_id"],
                            "gold_label": int(row["label"]),
                            "probability_yes": float(row["probability_yes"]),
                            "threshold": float(row["threshold"]),
                            "threshold_distance": float(row["threshold_distance"]),
                            "lexical_review_cues": _cue_summary(row["target_clause"]),
                            "target_clause_for_manual_review": row["target_clause"],
                        }
                    )
    _write_csv(destination / "fold_error_counts.csv", pd.DataFrame(summaries))
    _write_csv(destination / "selected_error_examples.csv", pd.DataFrame(examples))
    write_json(
        destination / "analysis_manifest.json",
        {
            "effective_dataset_sha256": read_json(
                STUDY_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
            )["dataset"]["effective_dataset_sha256"],
            "baseline_predictions_sha256": sha256_file(
                BASELINE_ROOT / "pooled_oof_predictions.csv"
            ),
            "example_policy": (
                "up to three nearest-threshold false positives and false negatives "
                "per fold and model; lexical cues are heuristic, not manual labels"
            ),
            "models": [value[0] for value in methods],
            "outer_labels_used_for_threshold_selection": False,
        },
    )
    return destination


def build_final_analysis() -> Path:
    destination = ANALYSIS_ROOT / "final"
    if destination.exists():
        raise FileExistsError(f"Analysis output is immutable: {destination}")
    destination.mkdir(parents=True)
    effective = _load_effective()
    methods: list[tuple[str, str | None]] = [("weighted_ce", None)]
    inventory = read_json(STUDY_ROOT / "EXPERIMENT_INVENTORY.json")
    for stage, (model, _) in STAGE_TO_MODEL.items():
        if inventory.get("stages", {}).get(stage, {}).get("status") == "COMPLETE":
            methods.append((model, stage))
    metric_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    top_k_rows: list[dict[str, Any]] = []
    pooled: dict[str, pd.DataFrame] = {}
    for model_id, stage in methods:
        predictions = _model_predictions(model_id, stage, effective)
        if predictions is None:
            continue
        pooled[model_id] = predictions
        metric_rows.append(
            _metric_row(
                predictions,
                0.5,
                model=model_id,
                operating_point="threshold_0.5",
            )
        )
        for family, group in predictions.groupby("family_id", sort=True):
            labels = group["label"].to_numpy(dtype=np.int64)
            scores = group["probability_yes"].to_numpy(dtype=np.float64)
            family_rows.append(
                {
                    "model": model_id,
                    "family_id": str(family),
                    "rows": len(group),
                    "positive_labels": int(labels.sum()),
                    "positive_prevalence": float(labels.mean()),
                    "average_precision": (
                        float(average_precision_score(labels, scores))
                        if len(np.unique(labels)) == 2
                        else None
                    ),
                    "roc_auc": (
                        float(roc_auc_score(labels, scores))
                        if len(np.unique(labels)) == 2
                        else None
                    ),
                    "brier_score": float(np.mean((scores - labels) ** 2)),
                    "recall_at_0_5": float(
                        ((scores >= 0.5) & (labels == 1)).sum() / max(1, labels.sum())
                    ),
                }
            )
        ranked = predictions.sort_values(
            ["probability_yes", "record_id"],
            ascending=[False, True],
            kind="mergesort",
        )
        positives = int(predictions["label"].sum())
        prevalence = float(predictions["label"].mean())
        for k in TOP_K_VALUES:
            top = ranked.head(k)
            retrieved = int(top["label"].sum())
            top_k_rows.append(
                {
                    "model": model_id,
                    "k": k,
                    "true_positives_retrieved": retrieved,
                    "precision_at_k": retrieved / len(top),
                    "recall_at_k": retrieved / positives if positives else 0.0,
                    "lift_over_prevalence": (
                        retrieved / len(top) / prevalence if prevalence else None
                    ),
                }
            )
    nested_path = (
        ANALYSIS_ROOT
        / "nested_threshold_calibration"
        / "outer_oof_predictions_nested_threshold.csv"
    )
    if nested_path.is_file():
        nested = pd.read_csv(nested_path, dtype={"record_id": str})
        for fold, group in nested.groupby("fold", sort=True):
            group = group.copy()
            group["predicted_label_nested_threshold"] = (
                group["probability_yes"] >= group["selected_threshold"]
            ).astype(int)
            metric_rows.append(
                _deployed_metric_row(
                    group,
                    model="weighted_ce",
                    operating_point="nested_fold_threshold",
                    fold=int(fold),
                )
            )
        nested["predicted_label_nested_threshold"] = (
            nested["probability_yes"] >= nested["selected_threshold"]
        ).astype(int)
        metric_rows.append(
            _deployed_metric_row(
                nested,
                model="weighted_ce",
                operating_point="nested_fold_threshold",
            )
        )
    _write_csv(destination / "model_comparison.csv", pd.DataFrame(metric_rows))
    _write_csv(destination / "family_metrics.csv", pd.DataFrame(family_rows))
    _write_csv(destination / "top_k_comparison.csv", pd.DataFrame(top_k_rows))

    baseline = pooled["weighted_ce"]
    threshold_table = pd.concat(
        [
            pd.DataFrame(
                [
                    {"model": "weighted_ce", **_threshold_metrics(baseline, threshold)}
                    for threshold in THRESHOLDS
                ]
            )
        ],
        ignore_index=True,
    )
    _write_csv(destination / "weighted_ce_threshold_sweep.csv", threshold_table)
    plot_dir = destination / "plots"
    plot_dir.mkdir()
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for model, predictions in pooled.items():
        precision, recall, _ = precision_recall_curve(
            predictions["label"], predictions["probability_yes"]
        )
        axis.plot(recall, precision, label=model)
    axis.set(
        title="Precision-Recall Curves by Study Method",
        xlabel="Recall",
        ylabel="Precision",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "precision_recall_by_method.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    comparison = pd.DataFrame(metric_rows)
    comparison = comparison.loc[comparison["operating_point"] == "threshold_0.5"]
    axis.bar(comparison["model"], comparison["recall"])
    axis.set(title="Recall at Fixed Threshold 0.5", ylabel="Recall", ylim=(0, 1))
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(plot_dir / "recall_at_0_5_by_method.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8.2, 5.2))
    for model, predictions in pooled.items():
        rows = [_threshold_metrics(predictions, threshold) for threshold in THRESHOLDS]
        axis.plot(
            [row["threshold"] for row in rows],
            [row["recall"] for row in rows],
            marker="o",
            label=model,
        )
    axis.set(
        title="Recall Across the Frozen Threshold Grid",
        xlabel="Threshold",
        ylabel="Recall",
        xscale="log",
        ylim=(0, 1),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "recall_vs_threshold.png", dpi=160)
    plt.close(figure)

    report_lines = [
        "# Recall Improvement Study — Final Report",
        "",
        "## Scope and status",
        "",
        (
            "All methods use the frozen, family-aware Dataset A cohort. The reported "
            "fold-specific operating point is chosen only from baseline inner OOF "
            "predictions; other ablations are compared at fixed threshold 0.5. Outer "
            "labels are reserved for evaluation. All results are exploratory and based "
            "on one seed; they do not establish generalization beyond the frozen families."
        ),
        "",
        "## Method comparison",
        "",
        "| Model | Operating point | AP | ROC-AUC | Precision | Recall | F2 | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in metric_rows:
        report_lines.append(
            f"| {row.get('model', 'weighted_ce')} | {row.get('operating_point', 'threshold_0.5')} | "
            f"{row['average_precision']:.4f} | {row['roc_auc']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f2']:.4f} | {row['brier_score']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Stage execution status",
            "",
            "| Stage | Status |",
            "| --- | --- |",
            *[
                f"| {stage} | {value.get('status')} |"
                for stage, value in inventory.get("stages", {}).items()
            ],
            "",
            "",
            "## Stage A diagnostic",
            "",
            (
                "Experiment 2 baseline Fold 2 has AP 0.2280 and ROC-AUC 0.8011, but "
                "all positive scores are below 0.5 (median 0.0128, maximum 0.0205), "
                "so threshold 0.5 yields zero recall. This supports testing fitting-side "
                "thresholds; it does not prove probability calibration."
            ),
            "",
            "## Retrieval at fixed review budgets",
            "",
            (
                "See `top_k_comparison.csv` for model-specific top-K precision, recall, "
                "and lift at the predeclared review budgets."
            ),
            "",
            "## Family-stratified metrics",
            "",
            (
                "See `family_metrics.csv`. Family-level AP/AUC are undefined when a "
                "family has only one label class and are recorded as null."
            ),
            "",
            "## Error analysis",
            "",
            (
                "See `../error_analysis/selected_error_examples.csv` for threshold-near "
                "false positives and false negatives. Lexical cues are deterministic "
                "review aids, not expert-assigned ambiguity categories."
            ),
            "",
            "## Study limitations",
            "",
            "- One random seed and five family folds limit precision of method comparisons.",
            "- Grouped family cross-validation is not an external deployment evaluation.",
            (
                "- Inner thresholds optimize a fixed finite candidate grid and a declared "
                "precision constraint; fallback thresholds maximize F2 when infeasible."
            ),
            "- Partial freezing preserves the model's standard training-mode dropout.",
            (
                "- CPU synthetic tests verify study helpers only; they do not verify GPU "
                "optimizer or full-model behavior."
            ),
            "",
            "## Artifacts and reproducibility",
            "",
            (
                "The frozen protocol, split manifests, per-cell run manifests, complete "
                "artifact hashes, treatment audits, logs, and execution inventory are "
                "preserved under the study namespace. No historical run artifacts are "
                "modified."
            ),
        ]
    )
    (destination / "REPORT.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    write_json(
        destination / "analysis_manifest.json",
        {
            "protocol_sha256": sha256_file(
                STUDY_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
            ),
            "execution_inventory": {
                "overall_status": inventory.get("overall_status"),
                "gpu_cells_started": inventory.get("gpu_cells_started"),
                "gpu_cells_completed": inventory.get("gpu_cells_completed"),
                "stage_statuses": {
                    key: value.get("status")
                    for key, value in inventory.get("stages", {}).items()
                },
            },
            "methods_included": list(pooled),
            "rows_per_method": {key: len(value) for key, value in pooled.items()},
            "outer_labels_used_for_threshold_selection": False,
        },
    )
    budget = read_json(STUDY_ROOT / "time_budget.json")
    write_json(
        destination / "execution_summary.json",
        {
            "authorization_window_start": budget["authorization_window_start"],
            "authorization_window_deadline": budget["authorization_window_deadline"],
            "deadline_remaining_seconds": budget.get("deadline_remaining_seconds"),
            "estimated_total_seconds": budget["estimated_total_seconds"],
            "actual_budget": budget.get("actual"),
            "inventory_status": inventory.get("overall_status"),
            "completed_stages": inventory.get("completed_stages", []),
            "stage_statuses": {
                key: value.get("status")
                for key, value in inventory.get("stages", {}).items()
            },
            "training_cells_started": inventory.get("gpu_cells_started", 0),
            "training_cells_completed": inventory.get("gpu_cells_completed", 0),
        },
    )
    return destination


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nested",
        action="store_true",
        help="Build the nested threshold analysis from completed inner runs.",
    )
    parser.add_argument(
        "--errors",
        action="store_true",
        help="Build CPU-only selected error examples.",
    )
    parser.add_argument(
        "--final",
        action="store_true",
        help="Build the consolidated final report and comparison artifacts.",
    )
    args = parser.parse_args()
    requested = sum((args.nested, args.errors, args.final))
    if requested != 1:
        parser.error("select exactly one analysis mode")
    if args.nested:
        print(build_nested_threshold_analysis())
    elif args.errors:
        print(build_error_analysis())
    else:
        print(build_final_analysis())
