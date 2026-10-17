from __future__ import annotations

import pandas as pd
import pytest

from models.heuristic.ambiguity_heuristic import score_ambiguity


def test_score_ambiguity_rewards_direct_ambiguity_signals() -> None:
    features = pd.DataFrame(
        {
            "has_if_without_otherwise": [1],
            "pronoun_without_noun_before": [1],
            "modal_level_count": [2],
            "vague_term_count": [2],
            "hedge_term_count": [1],
            "anaphora_count": [2],
            "exception_count": [1],
            "has_numeric_value": [0],
            "target_context_overlap": [0.0],
            "section_reference_count": [0],
        }
    )

    assert score_ambiguity(features).iloc[0] == 1.0


def test_concrete_values_context_and_section_references_reduce_score() -> None:
    features = pd.DataFrame(
        {
            "has_if_without_otherwise": [0],
            "pronoun_without_noun_before": [0],
            "modal_level_count": [1],
            "vague_term_count": [2],
            "hedge_term_count": [0],
            "anaphora_count": [0],
            "exception_count": [0],
            "has_numeric_value": [1],
            "target_context_overlap": [0.25],
            "section_reference_count": [1],
        }
    )

    assert score_ambiguity(features).iloc[0] == 0.0


def test_score_ambiguity_rejects_incomplete_feature_rows() -> None:
    with pytest.raises(ValueError, match="Missing heuristic input features"):
        score_ambiguity(pd.DataFrame({"vague_term_count": [1]}))
