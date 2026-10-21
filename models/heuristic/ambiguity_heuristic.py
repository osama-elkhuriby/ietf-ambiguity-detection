"""Rule-based ambiguity baseline using engineered annotation features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from models.classical.annotation_logreg_experiment2 import (
    DEFAULT_INPUT,
    N_SPLITS,
    PREDICTION_THRESHOLD,
    ROOT,
    _binary_metrics,
    build_validation_groups,
    extract_v2_features,
)

DEFAULT_OUTPUT = ROOT / "results/heuristic/final_annotation_v1"
MAX_RULE_SCORE = 9.0

RULES: dict[str, Any] = {
    "conditional_without_exception_marker": {
        "points": 3.0,
        "condition": (
            "condition_marker_count is positive and exception_marker_count is zero"
        ),
    },
    "deictic_expression": {
        "points": 2.0,
        "condition": "deictic_expression_count, capped at two points",
    },
    "mixed_modal_levels": {
        "points": 2.0,
        "condition": "more than one of MUST, SHOULD, and MAY is present",
    },
    "subjective_terms": {
        "points_per_term": 1.0,
        "maximum_points": 2.0,
        "condition": "subjective_term_count, capped at two points",
    },
    "exception_marker": {
        "points": 1.0,
        "condition": "one or more exception_marker_count terms",
    },
    "numeric_bound": {
        "points": -2.0,
        "condition": "subtract points when numeric_bound_count is positive",
    },
    "target_reference": {
        "points": -1.0,
        "condition": "subtract a point when reference_count_target is positive",
    },
    "score": "clamp total points to [0, 9] and divide by 9",
    "prediction_threshold": PREDICTION_THRESHOLD,
}


def score_ambiguity(features: pd.DataFrame) -> pd.Series:
    """Return a deterministic [0, 1] ambiguity score from engineered features."""
    required = {
        "condition_marker_count",
        "exception_marker_count",
        "must_count",
        "should_count",
        "may_count",
        "subjective_term_count",
        "deictic_expression_count",
        "numeric_bound_count",
        "reference_count_target",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing heuristic input features: {sorted(missing)}")

    positive_score = (
        3.0
        * (
            features["condition_marker_count"].gt(0)
            & features["exception_marker_count"].eq(0)
        ).astype(float)
        + features["deictic_expression_count"].clip(upper=2)
        + 2.0
        * (
            features[["must_count", "should_count", "may_count"]]
            .gt(0)
            .sum(axis=1)
            .gt(1)
        ).astype(float)
        + features["subjective_term_count"]
        .clip(upper=2)
        + features["exception_marker_count"].gt(0).astype(float)
    )
    negative_score = (
        2.0 * features["numeric_bound_count"].gt(0).astype(float)
        + features["reference_count_target"].gt(0).astype(float)
    )
    return ((positive_score - negative_score).clip(lower=0, upper=MAX_RULE_SCORE) / MAX_RULE_SCORE).rename(
        "heuristic_score"
    )


def evaluate(
    input_path: Path,
    output_dir: Path,
    *,
    fold_column: str | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )
    required = {
        "item_id",
        "document",
        "target_clause",
        "previous_clause",
        "next_clause",
        "final_label",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Input annotation file contains no rows")
    if frame["item_id"].duplicated().any():
        raise ValueError("Input contains duplicate item_id values")
    if frame["document"].str.strip().eq("").any():
        raise ValueError("Every row must have a document value for grouped validation")
    if frame["target_clause"].str.strip().eq("").any():
        raise ValueError("Every row must have a target_clause")
    normalized_labels = frame["final_label"].str.strip().str.casefold()
    if not normalized_labels.isin({"yes", "no"}).all():
        raise ValueError("final_label must contain only yes/no")
    labels = normalized_labels.eq("yes").astype(np.int8).to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError("Evaluation requires both yes and no labels")

    features = extract_v2_features(frame)
    scores = score_ambiguity(features).to_numpy(dtype=np.float64)
    groups = build_validation_groups(frame)
    group_count = len(np.unique(groups))
    positive_group_count = len(np.unique(groups[labels == 1]))
    if group_count < N_SPLITS or positive_group_count < N_SPLITS:
        raise ValueError(
            f"{N_SPLITS}-fold grouped validation requires at least "
            f"{N_SPLITS} total and positive groups; found {group_count} and "
            f"{positive_group_count}"
        )
    if fold_column is not None:
        if fold_column not in frame.columns:
            raise ValueError(f"Input is missing requested fold column: {fold_column}")
        try:
            supplied_folds = frame[fold_column].astype(int).to_numpy()
        except ValueError as exc:
            raise ValueError(f"{fold_column} must contain integer fold IDs") from exc
        fold_ids = sorted(np.unique(supplied_folds).tolist())
        if len(fold_ids) < 2:
            raise ValueError("Predefined validation requires at least two folds")
        if fold_ids != list(range(1, len(fold_ids) + 1)):
            raise ValueError(
                f"{fold_column} must use consecutive fold IDs starting at 1; "
                f"found {fold_ids}"
            )
        for group in np.unique(groups):
            if np.unique(supplied_folds[groups == group]).size != 1:
                raise RuntimeError(
                    f"Predefined folds split validation group {group!r} across folds"
                )
        splits = [
            (
                np.flatnonzero(supplied_folds != fold),
                np.flatnonzero(supplied_folds == fold),
            )
            for fold in fold_ids
        ]
        validation_strategy = f"predefined family folds from input column {fold_column}"
        validation_random_state: int | None = None
    else:
        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=42,
        )
        splits = list(splitter.split(frame, labels, groups))
        validation_strategy = (
            "3-fold StratifiedGroupKFold; draft revisions grouped by family"
        )
        validation_random_state = 42
    fold_metrics: list[dict[str, int | float]] = []
    fold_numbers = np.zeros(len(frame), dtype=np.int8)
    for fold, (train_indices, test_indices) in enumerate(splits, start=1):
        if (
            np.unique(labels[train_indices]).size != 2
            or np.unique(labels[test_indices]).size != 2
        ):
            raise ValueError(f"Fold {fold} does not contain both label classes")
        if np.intersect1d(groups[train_indices], groups[test_indices]).size:
            raise RuntimeError(f"Fold {fold} leaks draft families across partitions")
        fold_numbers[test_indices] = fold
        metrics = _binary_metrics(labels[test_indices], scores[test_indices])
        fold_metrics.append(
            {
                "fold": fold,
                "validation_rows": len(test_indices),
                "validation_positives": int(labels[test_indices].sum()),
                "validation_groups": len(np.unique(groups[test_indices])),
                **metrics,
                "roc_auc": float(
                    roc_auc_score(labels[test_indices], scores[test_indices])
                ),
            }
        )
    if np.any(fold_numbers == 0):
        raise RuntimeError("Grouped validation did not assign every row to a fold")

    pooled_metrics = _binary_metrics(labels, scores)
    family_ids = frame["document"].str.replace(r"-\d+$", "", regex=True)
    metrics_summary: dict[str, Any] = {
        "model": "deterministic engineered-feature heuristic",
        "rows": len(frame),
        "positive_rows": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "draft_families": int(family_ids.nunique()),
        "validation": {
            "strategy": validation_strategy,
            "random_state": validation_random_state,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "mean_fold_pr_auc": float(
                np.mean([fold["pr_auc"] for fold in fold_metrics])
            ),
            "std_fold_pr_auc": float(
                np.std([fold["pr_auc"] for fold in fold_metrics])
            ),
            "pooled_oof_metrics": {
                **pooled_metrics,
                "roc_auc": float(roc_auc_score(labels, scores)),
            },
            "folds": fold_metrics,
        },
        "rules": RULES,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        output_dir / "engineered_features.csv",
        index=False,
        lineterminator="\n",
    )
    predictions = pd.DataFrame(
        {
            "item_id": frame["item_id"],
            "document": frame["document"],
            "final_label": frame["final_label"],
            "validation_fold": fold_numbers,
            "heuristic_score": scores,
            "predicted_label": np.where(
                scores >= PREDICTION_THRESHOLD,
                "yes",
                "no",
            ),
        }
    )
    predictions.to_csv(
        output_dir / "predictions.csv",
        index=False,
        lineterminator="\n",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source_path": str(input_path.resolve()),
        "feature_set": "engineered features from annotation_logreg_experiment2",
        "score_method": "fixed weighted rules; no fitting or label-based tuning",
        "threshold": PREDICTION_THRESHOLD,
        "metrics_file": "metrics.json",
        "predictions_file": "predictions.csv",
        "fold_column": fold_column,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics_summary, indent=2))
    return metrics_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the engineered-feature ambiguity heuristic."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--fold-column",
        help=(
            "Use integer fold IDs from this input column instead of constructing "
            "the historical 3-fold split. Groups must belong to exactly one fold."
        ),
    )
    args = parser.parse_args()
    evaluate(args.input, args.output_dir, fold_column=args.fold_column)


if __name__ == "__main__":
    main()
