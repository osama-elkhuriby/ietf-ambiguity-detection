"""Train and test engineered-feature plus TF-IDF logistic regression."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from models.classical.annotation_logreg_experiment import (
    DEFAULT_INPUT,
    PREDICTION_THRESHOLD,
    ROOT,
    V2_COLUMNS,
    _binary_metrics,
    _fit_components,
    _new_classifier,
    _transform_components,
    build_validation_groups,
    extract_v2_features,
)

DEFAULT_OUTPUT = ROOT / "results/annotation/v2_engineered_tfidf_holdout"
FEATURE_MODE = "engineered_tfidf"
TEST_SIZE = 0.25
RANDOM_STATE = 42


def make_holdout_split(
    frame: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by draft family and require both classes in train and test."""
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_indices, test_indices = next(splitter.split(frame, labels, groups))
    if np.unique(labels[train_indices]).size != 2:
        raise ValueError(
            "The training partition must contain both yes and no labels; "
            "try a different split seed."
        )
    if np.unique(labels[test_indices]).size != 2:
        raise ValueError(
            "The held-out test partition must contain both yes and no labels; "
            "try a different split seed."
        )
    return train_indices, test_indices


def train_and_test(input_path: Path, output_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(input_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
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
        raise ValueError("Every row must have a document value for grouped testing")
    if frame["target_clause"].str.strip().eq("").any():
        raise ValueError("Every row must have a target_clause")

    normalized_labels = frame["final_label"].str.strip().str.casefold()
    if not normalized_labels.isin({"yes", "no"}).all():
        raise ValueError("final_label must contain only yes/no")
    labels = normalized_labels.eq("yes").astype(np.int8).to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError("Testing requires both yes and no labels")

    groups = build_validation_groups(frame)
    train_indices, test_indices = make_holdout_split(frame, labels, groups)
    features = extract_v2_features(frame)
    train_features = features.iloc[train_indices]
    test_features = features.iloc[test_indices]
    train_text = frame["target_clause"].iloc[train_indices]
    test_text = frame["target_clause"].iloc[test_indices]

    components, train_matrix = _fit_components(
        FEATURE_MODE,
        train_text,
        train_features,
        None,
    )
    test_matrix = _transform_components(
        FEATURE_MODE,
        components,
        test_text,
        test_features,
        None,
    )
    classifier = _new_classifier().fit(train_matrix, labels[train_indices])
    probabilities = classifier.predict_proba(test_matrix)[:, 1]
    test_metrics = _binary_metrics(labels[test_indices], probabilities)
    test_metrics["roc_auc"] = float(roc_auc_score(labels[test_indices], probabilities))

    predictions = pd.DataFrame(
        {
            "item_id": frame["item_id"].iloc[test_indices].to_numpy(),
            "document": frame["document"].iloc[test_indices].to_numpy(),
            "final_label": frame["final_label"].iloc[test_indices].to_numpy(),
            "probability_yes": probabilities,
            "predicted_label": np.where(
                probabilities >= PREDICTION_THRESHOLD, "yes", "no"
            ),
        }
    )
    family_ids = frame["document"].str.replace(r"-\d+$", "", regex=True).to_numpy()
    train_family_ids = sorted(set(family_ids[train_indices].tolist()))
    test_family_ids = sorted(set(family_ids[test_indices].tolist()))
    metrics: dict[str, Any] = {
        "feature_set": FEATURE_MODE,
        "split_strategy": "single draft-family-grouped holdout",
        "test_size_fraction_of_families": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "rows": len(frame),
        "positive_rows": int(labels.sum()),
        "train_rows": len(train_indices),
        "train_positives": int(labels[train_indices].sum()),
        "test_rows": len(test_indices),
        "test_positives": int(labels[test_indices].sum()),
        "threshold": PREDICTION_THRESHOLD,
        "test_metrics": test_metrics,
        "train_families": train_family_ids,
        "test_families": test_family_ids,
        "train_group_count": len(train_family_ids),
        "test_group_count": len(test_family_ids),
        "feature_selection": "L1 regularization fit on training data only",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            **components,
            "classifier": classifier,
            "feature_mode": FEATURE_MODE,
            "numerical_columns": list(V2_COLUMNS),
        },
        output_dir / "final_model.joblib",
    )
    predictions.to_csv(
        output_dir / "test_predictions.csv",
        index=False,
        lineterminator="\n",
    )
    pd.DataFrame(
        {
            "item_id": frame["item_id"].iloc[train_indices].to_numpy(),
            "document": frame["document"].iloc[train_indices].to_numpy(),
            "final_label": frame["final_label"].iloc[train_indices].to_numpy(),
        }
    ).to_csv(output_dir / "train_rows.csv", index=False, lineterminator="\n")
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source_path": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "feature_set": FEATURE_MODE,
        "engineered_features": list(V2_COLUMNS),
        "fit_scope": "training partition only",
        "test_scope": "held-out draft families; used once for reporting metrics",
        "artifacts": [
            "final_model.joblib",
            "metrics.json",
            "test_predictions.csv",
            "train_rows.csv",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and test engineered-feature plus TF-IDF logistic regression."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train_and_test(args.input, args.output_dir)


if __name__ == "__main__":
    main()
