from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gui import rfc_scanner
from gui.app import ScanStore, explain_selected, validate_explanation_input
from gui.rfc_scanner import LoadedClassifier, scan_rfc_text

ROOT = Path(__file__).resolve().parents[1]


class FakeClassifier:
    def __init__(self) -> None:
        self.matrix_shape: tuple[int, ...] | None = None

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        self.matrix_shape = matrix.shape
        scores = np.linspace(0.1, 0.9, len(matrix))
        return np.column_stack((1 - scores, scores))


def test_rfc_scan_ranks_only_normative_clauses(monkeypatch: pytest.MonkeyPatch) -> None:
    classifier = FakeClassifier()

    def transform(
        feature_mode: str,
        components: dict[str, Any],
        texts: Any,
        features: Any,
        embeddings: Any,
    ) -> np.ndarray:
        assert feature_mode == "engineered_tfidf"
        assert len(texts) == len(features)
        return np.zeros((len(texts), 2))

    monkeypatch.setattr(rfc_scanner, "_transform_components", transform)
    loaded = LoadedClassifier(
        components={"classifier": classifier},
        metadata={},
    )
    content = (ROOT / "corpus/raw/rfcs/RFC9460.txt").read_text(encoding="utf-8")

    result = scan_rfc_text(
        filename="RFC9460.txt",
        content=content,
        loaded_model=loaded,
    )

    assert result["document_id"] == "RFC9460"
    assert result["candidate_count"] == result["normative_count"] == 141
    assert result["manual_review_count"] == 465
    assert classifier.matrix_shape == (141, 2)
    assert [row["rank"] for row in result["rows"]] == list(range(1, 142))
    assert result["rows"][0]["relative_score"] == pytest.approx(0.9)
    assert result["rows"][-1]["relative_score"] == pytest.approx(0.1)
    assert all(row["target_clause"].strip() for row in result["rows"])
    assert all(row["line_start"] for row in result["rows"])


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("RFC9460.pdf", "content", "plaintext"),
        ("RFC9460.txt", " ", "empty"),
    ],
)
def test_rfc_scan_rejects_unsupported_or_empty_input(
    filename: str, content: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        scan_rfc_text(
            filename=filename,
            content=content,
            loaded_model=LoadedClassifier(components={}, metadata={}),
        )


def test_explanations_are_limited_and_do_not_send_model_scores() -> None:
    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_explanation_input(
            {"scan_id": "scan", "clause_ids": [str(index) for index in range(6)]}
        )

    row = {
        "clause_id": "RFC9460:1:1",
        "target_clause": "A sender MUST choose an appropriate value.",
        "previous_clause": "Requests are sent in order.",
        "next_clause": "The connection is then closed.",
        "relative_score": 0.99,
        "qwen_justification": "",
    }
    store = ScanStore()
    scan_id = store.add({"rows": [row]})

    class FakeOllamaClient:
        def __init__(self) -> None:
            self.payloads: list[dict[str, Any]] = []

        def tags(self) -> dict[str, dict[str, str]]:
            return {"qwen3:32b": {"digest": "expected"}}

        def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.payloads.append(payload)
            return {"message": {"content": "The word appropriate has no defined criterion."}}

    client = FakeOllamaClient()
    result = explain_selected(
        value={"scan_id": scan_id, "clause_ids": [row["clause_id"]]},
        config={
            "model": {"name": "qwen3:32b", "digest": "expected"},
            "ollama": {
                "think": False,
                "keep_alive": "30m",
                "temperature": 0,
                "seed": 1,
                "num_ctx": 8192,
            },
        },
        client=client,  # type: ignore[arg-type]
        scan_store=store,
    )

    assert result["explanations"] == [
        {
            "clause_id": row["clause_id"],
            "qwen_justification": "The word appropriate has no defined criterion.",
        }
    ]
    assert "0.99" not in str(client.payloads[0]["messages"])
    assert row["qwen_justification"] == result["explanations"][0]["qwen_justification"]
