from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models.transformer.ceiling_audit import (
    add_vote_fraction,
    error_analysis_export,
    leave_one_llm_out_metrics,
    parse_vote_payload,
)


def test_vote_parser_uses_only_valid_binary_votes() -> None:
    payload = json.dumps(
        {
            "gemma3:27b": {"status": "valid", "vote": "yes"},
            "gpt-oss:20b": {"status": "invalid", "vote": "yes"},
            "qwen3:32b": {"status": "valid", "vote": "no"},
        }
    )
    votes = parse_vote_payload(payload)
    assert votes["gemma3_27b"] == 1.0
    assert np.isnan(votes["gpt_oss_20b"])
    assert votes["qwen3_32b"] == 0.0


def test_leave_one_out_requires_two_valid_agreeing_votes() -> None:
    frame = pd.DataFrame(
        {
            "label_tier": ["HIGH", "MEDIUM", "MEDIUM", "HIGH"],
            "gemma3_27b": [1.0, 0.0, 1.0, 0.0],
            "gpt_oss_20b": [1.0, 0.0, 0.0, 0.0],
            "qwen3_32b": [1.0, 0.0, 1.0, 0.0],
        }
    )
    result = leave_one_llm_out_metrics({"pooled": frame})
    gemma = result[
        (result["held_out_llm"] == "gemma3_27b")
        & (result["label_tier"] == "ALL")
    ].iloc[0]
    assert gemma["eligible_rows"] == 3
    assert gemma["excluded_without_two_model_consensus"] == 1
    assert gemma["pr_auc"] == pytest.approx(1.0)


def test_vote_fraction_has_fixed_three_model_denominator() -> None:
    frame = pd.DataFrame(
        {
            "gemma3_27b": [1.0],
            "gpt_oss_20b": [1.0],
            "qwen3_32b": [np.nan],
        }
    )
    result = add_vote_fraction(frame)
    assert result.loc[0, "valid_vote_count"] == 2
    assert result.loc[0, "yes_vote_fraction"] == pytest.approx(2 / 3)


def test_error_export_has_40_rows_per_model_and_slice_without_vote_fields() -> None:
    rows = 100
    train = pd.DataFrame(
        {
            "record_id": [f"r{index}" for index in range(rows)],
            "family_id": ["family"] * rows,
            "label_tier": ["MEDIUM"] * rows,
            "target_clause": [f"clause {index}" for index in range(rows)],
        }
    )
    labels = np.array([0] * 50 + [1] * 50)
    base = pd.DataFrame(
        {
            "record_id": train["record_id"],
            "label": labels,
            "ambiguity_probability": np.linspace(0.0, 1.0, rows),
        }
    )
    output = error_analysis_export(
        train,
        {
            "deberta_1e-5_logit_average": base,
            "tfidf_unweighted": base,
        },
    )
    assert len(output) == 160
    assert output.groupby(["model", "error_slice"]).size().eq(40).all()
    assert not any("vote" in column or "rationale" in column for column in output)
