from __future__ import annotations

from models.transformer.factorial_analysis import select_condition


def test_factorial_selection_prefers_simpler_within_one_se() -> None:
    scores = {
        "hard_target_only": [0.30, 0.31, 0.29, 0.30, 0.30],
        "soft_target_only": [0.31, 0.30, 0.30, 0.31, 0.29],
        "hard_local_context": [0.20] * 5,
        "soft_local_context": [0.19] * 5,
    }
    result = select_condition(scores)
    assert result["top_condition"] == "soft_target_only"
    assert result["difference_is_less_than_one_standard_error"] is True
    assert result["selected_condition"] == "hard_target_only"


def test_factorial_selection_keeps_clear_winner() -> None:
    scores = {
        "hard_target_only": [0.30] * 5,
        "soft_target_only": [0.40] * 5,
        "hard_local_context": [0.20] * 5,
        "soft_local_context": [0.19] * 5,
    }
    result = select_condition(scores)
    assert result["difference_is_less_than_one_standard_error"] is False
    assert result["selected_condition"] == "soft_target_only"
