from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.transformer.lr_screen_analysis import (
    _paired_pr_auc_bootstrap,
    _safe_pr_auc,
    lr_slug,
    select_learning_rate,
)
from models.transformer.protocol_audit import history_checkpoint_record
from models.transformer.protocol_comparison import paired_screen_comparison


def test_learning_rate_rule_selects_highest_when_beyond_one_se() -> None:
    decision = select_learning_rate(
        {
            1e-5: [0.20, 0.20, 0.20, 0.20, 0.20],
            2e-5: [0.30, 0.31, 0.29, 0.30, 0.30],
            3e-5: [0.24, 0.25, 0.23, 0.24, 0.24],
        }
    )
    assert decision["selected_learning_rate"] == "2e-5"
    assert decision["second_seed_required"] is False


def test_learning_rate_rule_chooses_lower_and_requests_second_seed_within_se() -> None:
    decision = select_learning_rate(
        {
            1e-5: [0.30, 0.20, 0.31, 0.19, 0.30],
            2e-5: [0.29, 0.22, 0.29, 0.21, 0.29],
            3e-5: [0.10, 0.10, 0.10, 0.10, 0.10],
        }
    )
    assert decision["top_learning_rate"] == "1e-5"
    assert decision["selected_learning_rate"] == "1e-5"
    assert decision["second_seed_required"] is True
    assert decision["second_seed"] == 2718


def test_paired_row_and_cluster_bootstraps_are_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "fold": [1, 1, 1, 1, 2, 2, 2, 2],
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
            "first": [0.1, 0.2, 0.8, 0.9, 0.2, 0.3, 0.7, 0.8],
            "second": [0.2, 0.3, 0.6, 0.7, 0.1, 0.4, 0.6, 0.9],
            "cluster": ["a", "b", "c", "d", "e", "f", "g", "h"],
        }
    )
    row_first = _paired_pr_auc_bootstrap(
        frame, "first", "second", repetitions=50, seed=4
    )
    row_second = _paired_pr_auc_bootstrap(
        frame, "first", "second", repetitions=50, seed=4
    )
    clustered = _paired_pr_auc_bootstrap(
        frame,
        "first",
        "second",
        repetitions=50,
        seed=4,
        cluster_column="cluster",
    )
    assert row_first == row_second
    assert np.isfinite(row_first["point_difference"])
    assert clustered["method"] == "within-fold section-cluster paired bootstrap"


def test_learning_rate_slug_is_stable() -> None:
    assert [lr_slug(rate) for rate in (1e-5, 2e-5, 3e-5)] == [
        "1e-5",
        "2e-5",
        "3e-5",
    ]


def test_tier_pr_auc_is_undefined_without_both_classes() -> None:
    assert _safe_pr_auc(np.array([0, 0]), pd.Series([0.1, 0.2])) is None


def test_protocol_audit_compares_best_final_and_fixed_epochs() -> None:
    history = pd.DataFrame(
        {"epoch": [1, 2, 3, 4, 5], "pr_auc": [0.2, 0.3, 0.4, 0.35, 0.33]}
    )
    record = history_checkpoint_record(history, best_epoch=3)
    assert record["best_minus_final"] == pytest.approx(0.07)
    assert record["best_minus_epoch_3"] == 0.0
    assert record["best_minus_epoch_5"] == pytest.approx(0.07)


def test_paired_screen_comparison_reports_fold_and_pooled_differences() -> None:
    first = pd.DataFrame(
        {
            "record_id": [f"r{fold}-{row}" for fold in range(1, 6) for row in range(4)],
            "fold": [fold for fold in range(1, 6) for _ in range(4)],
            "label": [value for _ in range(5) for value in (0, 0, 1, 1)],
            "ambiguity_probability": [
                value for _ in range(5) for value in (0.1, 0.2, 0.8, 0.9)
            ],
        }
    )
    second = first.copy()
    second["ambiguity_probability"] = [
        value for _ in range(5) for value in (0.4, 0.3, 0.6, 0.5)
    ]
    metadata = pd.DataFrame(
        {
            "record_id": first["record_id"],
            "document_id": first["fold"].map(lambda value: f"doc-{value}"),
            "section_ids": [f"s{row % 2}" for row in range(len(first))],
            "section_cluster": [f"c{row}" for row in range(len(first))],
        }
    )
    result = paired_screen_comparison(
        first,
        second,
        metadata,
        first_name="first",
        second_name="second",
        repetitions=20,
        seed=9,
    )
    assert len(result["per_fold"]) == 5
    assert result["rows"] == 20
    assert result["first"] == "first"
    assert result["second"] == "second"
    assert result["pooled_difference"] == pytest.approx(0.0)
    assert result["section_cluster_bootstrap"]["valid_bootstraps"] == 20
