from __future__ import annotations

import math

import pytest

from models.llm.ollama_client import ScoreUnavailable, extract_binary_score


def test_extract_binary_score_normalizes_yes_no_logprobs() -> None:
    result = extract_binary_score(
        {
            "message": {"content": "yes"},
            "done_reason": "stop",
            "logprobs": [
                {
                    "token": "yes",
                    "logprob": -0.2,
                    "top_logprobs": [
                        {"token": " yes", "logprob": -0.2},
                        {"token": " no", "logprob": -1.2},
                    ],
                }
            ],
        }
    )
    assert result.predicted_label == 1
    assert result.log_odds_yes_no == pytest.approx(1.0)
    assert result.ambiguity_probability == pytest.approx(1 / (1 + math.exp(-1)))


def test_extract_binary_score_rejects_missing_alternative() -> None:
    with pytest.raises(ScoreUnavailable, match="both yes and no"):
        extract_binary_score(
            {
                "message": {"content": "no"},
                "done_reason": "stop",
                "logprobs": [{"token": "no", "logprob": -0.01}],
            }
        )
