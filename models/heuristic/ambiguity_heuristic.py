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

from models.classical.annotation_logreg_experiment import (
    DEFAULT_INPUT,
    N_SPLITS,
    PREDICTION_THRESHOLD,
    ROOT,
    _binary_metrics,
    build_validation_groups,
    extract_v2_features,
)

DEFAULT_OUTPUT = ROOT / "results/heuristic/final_annotation_v1"
MAX_RULE_SCORE = 12.0

RULES: dict[str, Any] = {
    "has_if_without_otherwise": {
        "points": 3.0,
        "condition": "conditional present with no else/otherwise/unless in target or next clause",
    },
    "pronoun_without_noun_before": {
        "points": 2.0,
        "condition": "anaphoric subject pronoun at clause start",
    },
    "mixed_modal_levels": {
        "points": 2.0,
        "condition": "modal_level_count greater than one",
    },
    "vague_terms": {
        "points_per_term": 1.0,
        "maximum_points": 2.0,
        "condition": "add points for vague or hedge terms",
    },
    "repeated_anaphora": {
        "points_per_occurrence": 1.0,
        "maximum_points": 2.0,
        "condition": "anaphora_count, capped at two points",
    },
    "exception": {
        "points": 1.0,
        "condition": "one or more exception terms",
    },
    "numeric_value": {
        "points": -2.0,
        "condition": "subtract points when a concrete numeric value is present",
    },
    "context_overlap": {
        "points": -1.0,
        "minimum_jaccard": 0.25,
        "condition": "subtract points when target overlaps its neighboring clause",
    },
    "section_reference": {
        "points": -1.0,
        "condition": "subtract a point when an explicit section reference is present",
    },
    "score": "clamp total points to [0, 12] and divide by 12",
    "prediction_threshold": PREDICTION_THRESHOLD,
}


def score_ambiguity(features: pd.DataFrame) -> pd.Series:
    """Return a deterministic [0, 1] ambiguity score from engineered features."""
    required = {
        "has_if_without_otherwise",
        "pronoun_without_noun_before",
        "modal_level_count",
        "vague_term_count",
        "hedge_term_count",
        "anaphora_count",
        "exception_count",
        "has_numeric_value",
        "target_context_overlap",
        "section_reference_count",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing heuristic input features: {sorted(missing)}")

    positive_score = (
        3.0 * features["has_if_without_otherwise"].clip(0, 1)
        + 2.0 * features["pronoun_without_noun_before"].clip(0, 1)
        + 2.0 * features["modal_level_count"].gt(1).astype(float)
        + features[["vague_term_count", "hedge_term_count"]]
        .max(axis=1)
        .clip(upper=2)
        + features["anaphora_count"].clip(upper=2)
        + features["exception_count"].gt(0).astype(float)
    )
    negative_score = (
        2.0 * features["has_numeric_value"].clip(0, 1)
        + features["target_context_overlap"].ge(0.25).astype(float)
        + features["section_reference_count"].gt(0).astype(float)
    )
    return ((positive_score - negative_score).clip(lower=0, upper=MAX_RULE_SCORE) / MAX_RULE_SCORE).rename(
        "heuristic_score"
    )


def evaluate(input_path: Path, output_dir: Path) -> dict[str, Any]:
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
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=42,
    )
    fold_metrics: list[dict[str, int | float]] = []
    fold_numbers = np.zeros(len(frame), dtype=np.int8)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame, labels, groups),
        start=1,
    ):
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
            "strategy": "3-fold StratifiedGroupKFold; draft revisions grouped by family",
            "random_state": 42,
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
        "feature_set": "23 engineered numerical features",
        "score_method": "fixed weighted rules; no fitting or label-based tuning",
        "threshold": PREDICTION_THRESHOLD,
        "metrics_file": "metrics.json",
        "predictions_file": "predictions.csv",
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
    args = parser.parse_args()
    evaluate(args.input, args.output_dir)


if __name__ == "__main__":
    main()
