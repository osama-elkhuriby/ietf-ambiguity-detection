from __future__ import annotations

import numpy as np
from scipy import sparse

from models.classical.annotation_classifier_experiments import (
    FEATURE_MODES,
    _new_classifier,
    _positive_scores,
)


def test_additional_classifiers_support_all_existing_feature_modes() -> None:
    assert FEATURE_MODES == (
        "engineered_tfidf",
        "tfidf",
        "embeddings",
        "tfidf_embeddings",
    )


def test_linear_svm_scores_are_bounded_and_use_zero_margin_threshold() -> None:
    matrix = sparse.csr_matrix(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    classifier = _new_classifier("linear_svm").fit(matrix, labels)

    scores = _positive_scores(classifier, matrix, "linear_svm")

    assert np.all((0.0 < scores) & (scores < 1.0))
    assert np.array_equal(scores >= 0.5, classifier.decision_function(matrix) >= 0)


def test_random_forest_scores_use_positive_class_probability() -> None:
    matrix = sparse.csr_matrix(
        [
            [-2.0, 0.0],
            [-1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    classifier = _new_classifier("random_forest").fit(matrix, labels)

    scores = _positive_scores(classifier, matrix, "random_forest")

    assert scores.shape == labels.shape
    assert np.all((0.0 <= scores) & (scores <= 1.0))
    assert np.array_equal(scores, classifier.predict_proba(matrix)[:, 1])
