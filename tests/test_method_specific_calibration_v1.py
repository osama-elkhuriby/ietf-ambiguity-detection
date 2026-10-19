from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.transformer_final.method_specific_calibration_v1.study import (
    confusion,
    empirical_cdf,
    select_precision_threshold,
    stable_rank_percentile,
    threshold_candidates,
)


def test_threshold_candidates_include_zero_prediction_option_and_scores() -> None:
    candidates = threshold_candidates([0.2, 0.8, 0.2])
    assert candidates[0] > 0.8
    assert candidates.tolist()[1:] == [0.8, 0.2]


def test_precision_constrained_threshold_is_deterministic() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    selected = select_precision_threshold(labels, scores, 0.5)
    assert selected["status"] == "FEASIBLE"
    assert selected["threshold"] == 0.7
    assert selected["recall"] == 1.0
    assert selected["precision"] == 2 / 3


def test_infeasible_precision_constraint_has_no_fallback() -> None:
    selected = select_precision_threshold([0, 0, 1], [0.9, 0.8, 0.1], 1.0)
    assert selected["status"] == "INFEASIBLE"
    assert selected["threshold"] is None
    assert selected["fallback_used"] is False


def test_confusion_uses_one_as_positive() -> None:
    result = confusion(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (result["tp"], result["fp"], result["tn"], result["fn"]) == (1, 1, 1, 1)


def test_ecdf_is_frozen_reference_transform() -> None:
    transformed = empirical_cdf([0.1, 0.2, 0.4, 0.8], [0.05, 0.2, 0.5, 1.0])
    assert np.allclose(transformed, [0.0, 0.5, 0.75, 1.0])


def test_stable_rank_tie_breaks_by_record_id() -> None:
    frame = pd.DataFrame({"record_id": ["b", "a", "c"], "score": [0.5, 0.5, 0.9]})
    ranks = stable_rank_percentile(frame, "score")
    assert np.allclose(ranks.to_numpy(), [2 / 3, 1 / 3, 1.0])
