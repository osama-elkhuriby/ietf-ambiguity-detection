from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from models.classical.add_prevalence_baseline import build_annotation
from models.classical.tfidf_comparator import (
    _fit_fold,
    _vectorizer,
    binary_metrics,
    bootstrap_intervals,
    fold_prevalence_tables,
)
from models.transformer.experiment import DevelopmentFold


def test_vectorizer_contract_matches_classical_feature_pipeline() -> None:
    vectorizer = _vectorizer({"min_df": 2, "max_features": 20_000})
    params = vectorizer.get_params()
    assert params["lowercase"] is True
    assert params["strip_accents"] == "unicode"
    assert params["ngram_range"] == (1, 2)
    assert params["sublinear_tf"] is True
    assert params["token_pattern"] == r"(?u)\b[A-Za-z0-9][A-Za-z0-9_-]*\b"


def test_tfidf_vocabulary_is_fitted_without_held_out_text() -> None:
    vectorizer: TfidfVectorizer = _vectorizer({"min_df": 1, "max_features": 100})
    vectorizer.fit(["ordinary training clause", "second training clause"])
    held_out = vectorizer.transform(["validationonlytoken ordinary"])
    assert "validationonlytoken" not in vectorizer.vocabulary_
    assert held_out.nnz == 1


def test_metrics_and_bootstrap_are_deterministic_and_complete() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    probabilities = np.array([0.1, 0.2, 0.8, 0.4, 0.7, 0.9])
    metrics = binary_metrics(labels, probabilities)
    first = bootstrap_intervals(labels, probabilities, repetitions=50, seed=7)
    second = bootstrap_intervals(labels, probabilities, repetitions=50, seed=7)
    assert metrics["no_skill_pr_auc"] == 0.5
    assert first == second
    assert set(first) == {
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "no_skill_pr_auc",
    }
    assert all(value["valid_bootstraps"] > 0 for value in first.values())


def test_prevalence_tables_use_duplicate_safe_scoring_rows() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "family_id": ["one", "one", "two", "two"],
            "label": [0, 1, 1, 0],
        }
    )
    fold = DevelopmentFold(
        number=1,
        train_indices=np.array([2, 3]),
        validation_indices=np.array([0, 1]),
        scoring_indices=np.array([1]),
        held_out_families=("one",),
        excluded_exact_duplicate_record_ids=("a",),
    )
    folds, families = fold_prevalence_tables(frame, [fold])
    assert folds.iloc[0].to_dict()["rows_scored"] == 1
    assert folds.iloc[0].to_dict()["positive_prevalence"] == 1.0
    assert families.iloc[0].to_dict()["excluded_exact_duplicates"] == 1


def test_existing_run_annotation_preserves_source_and_adds_baseline(
    tmp_path: Path,
) -> None:
    predictions = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "ambiguity_probability": [0.1, 0.2, 0.8, 0.9],
            "included_in_scoring": [True, False, True, True],
        }
    )
    predictions_path = tmp_path / "predictions.csv"
    metrics_path = tmp_path / "metrics.json"
    predictions.to_csv(predictions_path, index=False)
    expected = binary_metrics(
        np.array([0, 1, 1]), np.array([0.1, 0.8, 0.9])
    )
    metrics_path.write_text(json.dumps(expected), encoding="utf-8")
    annotation = build_annotation(
        predictions_path,
        metrics_path,
        bootstrap_repetitions=20,
        bootstrap_seed=3,
    )
    metrics = annotation["metrics_with_no_skill_baseline"]
    assert metrics["no_skill_pr_auc"] == 2 / 3
    assert annotation["annotation_type"] == "non_mutating_prevalence_baseline"


def test_fold_run_saves_raw_logits_predictions_and_manifest(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d", "e", "f"],
            "family_id": ["fit", "fit", "fit", "fit", "held", "held"],
            "target_clause": [
                "clear requirement",
                "another clear rule",
                "ambiguous condition",
                "unclear optional behavior",
                "held clear clause",
                "held ambiguous clause",
            ],
            "label": [0, 0, 1, 1, 0, 1],
            "label_tier": ["HIGH"] * 6,
            "sample_weight": [1.0] * 6,
        }
    )
    fold = DevelopmentFold(
        number=1,
        train_indices=np.array([0, 1, 2, 3]),
        validation_indices=np.array([4, 5]),
        scoring_indices=np.array([4, 5]),
        held_out_families=("held",),
        excluded_exact_duplicate_record_ids=(),
    )
    config = {
        "seed": 1729,
        "bootstrap_seed": 11,
        "bootstrap_repetitions": 20,
        "fold_assignment_sha256": "fold-hash",
        "development_train_sha256": "train-hash",
        "tfidf": {"min_df": 1, "max_features": 100},
        "logistic_regression": {"c": 1.0, "solver": "liblinear", "max_iter": 100},
    }
    predictions, metrics = _fit_fold(
        frame, fold, "combined", config, tmp_path / "run"
    )
    assert predictions["record_id"].tolist() == ["e", "f"]
    assert predictions["raw_logit"].notna().all()
    assert metrics["rows"] == 2
    assert (tmp_path / "run" / "validation_predictions.csv").is_file()
    manifest = json.loads(
        (tmp_path / "run" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["training_weight_mean"] == 1.0
