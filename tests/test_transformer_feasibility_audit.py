from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.transformer.feasibility_audit import (
    _oracle_scores,
    logit_average_probability,
    routing_curve,
    routing_point,
)


def test_routing_curve_refers_all_boundary_ties() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.7, 0.7, 0.2, 0.1])
    point = routing_point(labels, scores, 0.5)
    assert point["referred_rows"] == 2
    assert point["referral_rate"] == 0.5
    assert point["ambiguous_recall"] == 0.5
    assert point["local_selective_risk"] == 0.5


def test_no_skill_and_oracle_bound_single_threshold_routing() -> None:
    labels = np.array([1, 0, 0, 1, 0])
    no_skill = np.full(len(labels), labels.mean())
    assert routing_point(labels, no_skill, 0.8)["referral_rate"] == 1.0
    oracle = routing_point(labels, _oracle_scores(labels), 0.5)
    assert oracle["referred_rows"] == 1
    assert oracle["referral_rate"] == 0.2
    curve = routing_curve(labels, _oracle_scores(labels))
    assert curve.iloc[-1]["ambiguous_recall"] == 1.0


def test_logit_average_uses_binary_log_odds_and_aligns_record_ids() -> None:
    first = pd.DataFrame(
        {
            "record_id": ["a", "b"],
            "fold": [1, 1],
            "label": [0, 1],
            "logit_0": [0.0, 1.0],
            "logit_1": [2.0, 0.0],
        }
    )
    second = pd.DataFrame(
        {
            "record_id": ["b", "a"],
            "logit_0": [1.0, 2.0],
            "logit_1": [3.0, 0.0],
        }
    )
    averaged = logit_average_probability(first, second).set_index("record_id")
    assert averaged.loc["a", "ambiguity_probability"] == pytest.approx(0.5)
    assert averaged.loc["b", "ambiguity_probability"] == pytest.approx(
        1.0 / (1.0 + np.exp(-0.5))
    )
