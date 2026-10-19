from __future__ import annotations

import pandas as pd
import pytest

from models.classical.annotation_logreg_experiment2 import extract_v2_features
from models.heuristic.ambiguity_heuristic import score_ambiguity


def test_score_ambiguity_rewards_direct_ambiguity_signals() -> None:
    features = pd.DataFrame(
        {
            "condition_marker_count": [1],
            "exception_marker_count": [0],
            "must_count": [1],
            "should_count": [1],
            "may_count": [0],
            "subjective_term_count": [2],
            "deictic_expression_count": [2],
            "numeric_bound_count": [0],
            "reference_count_target": [0],
        }
    )

    assert score_ambiguity(features).iloc[0] == pytest.approx(1.0)


def test_concrete_values_context_and_section_references_reduce_score() -> None:
    features = pd.DataFrame(
        {
            "condition_marker_count": [0],
            "exception_marker_count": [0],
            "must_count": [1],
            "should_count": [0],
            "may_count": [0],
            "subjective_term_count": [2],
            "deictic_expression_count": [0],
            "numeric_bound_count": [1],
            "reference_count_target": [1],
        }
    )

    assert score_ambiguity(features).iloc[0] == 0.0


def test_score_ambiguity_accepts_features_from_training_extractor() -> None:
    frame = pd.DataFrame(
        {
            "target_clause": [
                "If the value is reasonable, the server MUST respond."
            ],
            "previous_clause": [""],
            "next_clause": [""],
        }
    )

    features = extract_v2_features(frame)

    assert score_ambiguity(features).between(0, 1).all()


def test_score_ambiguity_rejects_incomplete_feature_rows() -> None:
    with pytest.raises(ValueError, match="Missing heuristic input features"):
        score_ambiguity(pd.DataFrame({"vague_term_count": [1]}))
