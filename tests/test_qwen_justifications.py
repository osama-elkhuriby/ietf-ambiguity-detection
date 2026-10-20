from __future__ import annotations

import json

import pandas as pd
import pytest

from models.llm.justify_qwen_positives import (
    MAX_JUSTIFICATION_CHARS,
    generate_justification,
    predictions_with_justifications,
    render_justification_messages,
    validate_justification,
)


class FakeOllamaClient:
    def __init__(self, content: str = "The condition leaves the required timing open.") -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def chat(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append(payload)
        return {
            "message": {"content": self.content},
            "eval_count": 12,
        }


def test_justification_prompt_contains_only_clause_context() -> None:
    row = pd.Series(
        {
            "record_id": "record-1",
            "target_clause": "A receiver retries when appropriate.",
            "previous_clause": "A request failed.",
            "next_clause": "The connection may close.",
            "label": 1,
            "ambiguity_probability": 0.99,
        }
    )

    messages = render_justification_messages(row)
    rendered = "\n".join(message["content"] for message in messages)
    assert "A receiver retries when appropriate." in rendered
    assert "A request failed." in rendered
    assert "The connection may close." in rendered
    assert "0.99" not in rendered
    assert "label" not in rendered.lower()
    assert "chain-of-thought" in messages[0]["content"]


def test_positive_predictions_get_qwen_justifications_only() -> None:
    predictions = pd.DataFrame(
        {
            "record_id": ["positive", "negative"],
            "predicted_label": [1, 0],
            "ambiguity_probability": [0.9, 0.1],
        }
    )
    dataset = pd.DataFrame(
        {
            "record_id": ["positive", "negative"],
            "target_clause": ["Positive clause.", "Negative clause."],
            "previous_clause": ["Previous positive.", "Previous negative."],
            "next_clause": ["Next positive.", "Next negative."],
        }
    )
    client = FakeOllamaClient()
    config = {
        "model": {"name": "qwen3:32b", "digest": "a" * 64},
        "ollama": {
            "think": False,
            "keep_alive": "30m",
            "temperature": 0.0,
            "seed": 7,
            "num_ctx": 8192,
        },
    }

    result = predictions_with_justifications(predictions, dataset, client, config)

    assert result["justification"].tolist() == [
        "The condition leaves the required timing open.",
        "",
    ]
    assert len(client.calls) == 1
    request = client.calls[0]
    assert request["model"] == "qwen3:32b"
    assert "logprobs" not in request
    assert request["options"]["num_predict"] > 8


def test_justification_cache_is_resumable_and_bound_to_input(tmp_path) -> None:
    row = pd.Series(
        {
            "record_id": "record-1",
            "target_clause": "A receiver retries when appropriate.",
            "previous_clause": "",
            "next_clause": "",
        }
    )
    config = {
        "model": {"name": "qwen3:32b", "digest": "b" * 64},
        "ollama": {
            "think": False,
            "keep_alive": "30m",
            "temperature": 0.0,
            "seed": 7,
            "num_ctx": 8192,
        },
    }
    cache_path = tmp_path / "record.json"
    client = FakeOllamaClient()

    first = generate_justification(client, config, row, cache_path)
    second = generate_justification(client, config, row, cache_path)

    assert first == second
    assert len(client.calls) == 1
    assert json.loads(cache_path.read_text(encoding="utf-8"))["justification"] == first

    changed_row = row.copy()
    changed_row["target_clause"] = "A receiver retries immediately."
    with pytest.raises(RuntimeError, match="cache identity mismatch"):
        generate_justification(client, config, changed_row, cache_path)


def test_justification_rejects_empty_or_overlong_output() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_justification(" \n ")
    with pytest.raises(ValueError, match=str(MAX_JUSTIFICATION_CHARS)):
        validate_justification("x" * (MAX_JUSTIFICATION_CHARS + 1))
