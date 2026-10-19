"""Grouped-CV annotation experiments for linear SVM and random forest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.classical.annotation_logreg_experiment2 import run_cli

from models.classical.annotation_logreg_experiment2 import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUTS,
    N_SPLITS,
    PREDICTION_THRESHOLD,
    RANDOM_STATE,
    ROOT,
    _binary_metrics,
    _encode_target_embeddings,
    _fit_components,
    _transform_components,
    build_validation_groups,
    extract_v2_features,
)

ClassifierName = Literal["linear_svm", "random_forest"]
CLASSIFIERS: tuple[ClassifierName, ...] = ("linear_svm", "random_forest")
FEATURE_MODES = tuple(DEFAULT_OUTPUTS)
DEFAULT_OUTPUT_ROOT = ROOT / "results/ML/additional_classifiers"
SVM_SCORE_METHOD = "sigmoid of LinearSVC decision_function; 0.5 maps to margin 0"


def _new_classifier(classifier_name: ClassifierName) -> Any:
    if classifier_name == "linear_svm":
        return LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=10_000,
        )
    if classifier_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier: {classifier_name}")


def _positive_scores(classifier: Any, matrix: Any, classifier_name: ClassifierName) -> np.ndarray:
    if classifier_name == "random_forest":
        scores = classifier.predict_proba(matrix)[:, 1]
    else:
        margins = classifier.decision_function(matrix)
        scores = 1.0 / (1.0 + np.exp(-np.clip(margins, -40.0, 40.0)))
    if not np.isfinite(scores).all():
        raise ValueError("Classifier produced non-finite validation scores")
    return np.asarray(scores, dtype=np.float64)


def _validate_input(input_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
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
    return frame, labels


def train_and_validate(
    input_path: Path,
    output_dir: Path,
    *,
    classifier_name: ClassifierName,
    feature_mode: str,
) -> dict[str, Any]:
    if classifier_name not in CLASSIFIERS:
        raise ValueError(f"Unsupported classifier: {classifier_name}")
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    frame, labels = _validate_input(input_path)
    use_engineered = "engineered" in feature_mode
    use_embeddings = "embeddings" in feature_mode
    features = extract_v2_features(frame) if use_engineered else None
    embeddings = (
        _encode_target_embeddings(frame["target_clause"]) if use_embeddings else None
    )
    groups = build_validation_groups(frame)
    group_count = len(np.unique(groups))
    positive_group_count = len(np.unique(groups[labels == 1]))
    if group_count < N_SPLITS or positive_group_count < N_SPLITS:
        raise ValueError(
            f"{N_SPLITS}-fold grouped validation requires at least "
            f"{N_SPLITS} total and positive groups; found "
            f"{group_count} and {positive_group_count}"
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_scores = np.full(len(frame), np.nan, dtype=np.float64)
    fold_metrics: list[dict[str, int | float]] = []
    for fold, (train_indices, valid_indices) in enumerate(
        splitter.split(frame, labels, groups),
        start=1,
    ):
        if (
            np.unique(labels[train_indices]).size != 2
            or np.unique(labels[valid_indices]).size != 2
        ):
            raise ValueError(f"Fold {fold} does not contain both label classes")
        if np.intersect1d(groups[train_indices], groups[valid_indices]).size:
            raise RuntimeError(f"Fold {fold} leaks draft families across partitions")

        train_features = features.iloc[train_indices] if features is not None else None
        valid_features = features.iloc[valid_indices] if features is not None else None
        train_embeddings = embeddings[train_indices] if embeddings is not None else None
        valid_embeddings = embeddings[valid_indices] if embeddings is not None else None
        components, train_matrix = _fit_components(
            feature_mode,
            frame["target_clause"].iloc[train_indices],
            train_features,
            train_embeddings,
        )
        valid_matrix = _transform_components(
            feature_mode,
            components,
            frame["target_clause"].iloc[valid_indices],
            valid_features,
            valid_embeddings,
        )
        classifier = _new_classifier(classifier_name).fit(
            train_matrix,
            labels[train_indices],
        )
        scores = _positive_scores(classifier, valid_matrix, classifier_name)
        oof_scores[valid_indices] = scores
        fold_metrics.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "validation_rows": len(valid_indices),
                "train_positives": int(labels[train_indices].sum()),
                "validation_positives": int(labels[valid_indices].sum()),
                "validation_groups": len(np.unique(groups[valid_indices])),
                "average_precision": float(
                    average_precision_score(labels[valid_indices], scores)
                ),
                "roc_auc": float(roc_auc_score(labels[valid_indices], scores)),
                **_binary_metrics(labels[valid_indices], scores),
            }
        )
    if np.isnan(oof_scores).any():
        raise RuntimeError("Cross-validation did not score every row")

    fold_frame = pd.DataFrame(fold_metrics)
    pooled_metrics = _binary_metrics(labels, oof_scores)
    model_parameters: dict[str, Any] = (
        {
            "class_weight": "balanced",
            "max_iter": 10_000,
            "score_method": SVM_SCORE_METHOD,
        }
        if classifier_name == "linear_svm"
        else {
            "n_estimators": 300,
            "class_weight": "balanced_subsample",
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "score_method": "predict_proba",
        }
    )
    metrics: dict[str, Any] = {
        "rows": len(frame),
        "positive_rows": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "draft_families": int(
            frame["document"].str.replace(r"-\d+$", "", regex=True).nunique()
        ),
        "validation": {
            "strategy": (
                "3-fold StratifiedGroupKFold; draft revisions grouped by family"
            ),
            "random_state": RANDOM_STATE,
            "prediction_threshold": PREDICTION_THRESHOLD,
            "mean_fold_average_precision": float(
                fold_frame["average_precision"].mean()
            ),
            "std_fold_average_precision": float(
                fold_frame["average_precision"].std(ddof=0)
            ),
            "pooled_oof_metrics": {
                **pooled_metrics,
                "roc_auc": float(roc_auc_score(labels, oof_scores)),
            },
            "folds": fold_metrics,
        },
        "model": {
            "type": classifier_name,
            "parameters": model_parameters,
            "feature_set": feature_mode,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_frame.to_csv(output_dir / "fold_metrics.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        {
            "item_id": frame["item_id"],
            "document": frame["document"],
            "final_label": frame["final_label"],
            "validation_group": groups,
            "oof_score_yes": oof_scores,
        }
    ).to_csv(output_dir / "oof_predictions.csv", index=False, lineterminator="\n")

    final_components, final_matrix = _fit_components(
        feature_mode,
        frame["target_clause"],
        features,
        embeddings,
    )
    final_classifier = _new_classifier(classifier_name).fit(final_matrix, labels)
    final_components.update(
        {
            "classifier": final_classifier,
            "classifier_name": classifier_name,
            "feature_mode": feature_mode,
        }
    )
    joblib.dump(final_components, output_dir / "final_model.joblib")
    manifest = {
        "source_path": str(input_path.resolve()),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "classifier": classifier_name,
        "feature_mode": feature_mode,
        "score_method": model_parameters["score_method"],
        "validation_note": (
            "Metrics use out-of-fold scores; final_model.joblib is fitted on all rows."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run grouped-CV annotation classifier experiments."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--classifier",
        choices=CLASSIFIERS,
        required=True,
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODES,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to results/annotation/additional_classifiers/<classifier>/<mode>.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / args.classifier / args.feature_mode
    )
    train_and_validate(
        args.input,
        output_dir,
        classifier_name=args.classifier,
        feature_mode=args.feature_mode,
    )


if __name__ == "__main__":
    main()
