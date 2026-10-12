"""Tests for the frozen Step 3 weighting analysis."""

import numpy as np
import pandas as pd
import pytest

from models.transformer.weighting_analysis import (
    _ensemble_predictions,
    expected_calibration_error,
    select_weighting,
)


def test_expected_calibration_error_is_zero_for_perfect_predictions() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(labels, probabilities) == pytest.approx(0.0)


def test_weighting_selection_uses_simpler_strategy_within_one_se() -> None:
    scores = {
        "unweighted": [0.30, 0.30, 0.30, 0.30, 0.30],
        "class": [0.31, 0.29, 0.31, 0.29, 0.31],
        "confidence": [0.20] * 5,
        "combined": [0.10] * 5,
    }
    result = select_weighting(scores)
    assert result["top_strategy"] == "class"
    assert result["difference_is_less_than_one_standard_error"] is True
    assert result["selected_strategy"] == "unweighted"


def test_weighting_selection_keeps_clear_winner() -> None:
    scores = {
        "unweighted": [0.30] * 5,
        "class": [0.40] * 5,
        "confidence": [0.20] * 5,
        "combined": [0.10] * 5,
    }
    result = select_weighting(scores)
    assert result["difference_is_less_than_one_standard_error"] is False
    assert result["selected_strategy"] == "class"


def test_ensemble_averages_raw_log_odds_across_three_seeds() -> None:
    frame = pd.DataFrame(
        {
            "strategy": ["unweighted"] * 3,
            "fold": [1] * 3,
            "record_id": ["row-1"] * 3,
            "label": [1] * 3,
            "seed": [1729, 2718, 31415],
            "logit_0": [0.0, 0.0, 0.0],
            "logit_1": [-1.0, 0.0, 1.0],
        }
    )
    result = _ensemble_predictions(frame)
    assert result.loc[0, "ensemble_raw_log_odds"] == pytest.approx(0.0)
    assert result.loc[0, "ambiguity_probability"] == pytest.approx(0.5)
