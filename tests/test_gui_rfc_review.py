from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gui import rfc_scanner
from gui.app import (
    ScanStore,
    explain_selected,
    validate_explanation_input,
)
from gui.rfc_scanner import LoadedClassifier, scan_rfc_text
from models.llm.ollama_client import OllamaError

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
        metadata={"prediction_threshold": 0.5},
    )
    content = (ROOT / "corpus/raw/rfcs/RFC9460.txt").read_text(encoding="utf-8")

    result = scan_rfc_text(
        filename="RFC9460.txt",
        content=content,
        loaded_model=loaded,
    )
    crlf_result = scan_rfc_text(
        filename="RFC9460.txt",
        content=content.replace("\n", "\r\n"),
        loaded_model=loaded,
    )

    assert result["document_id"] == "RFC9460"
    assert result["normative_count"] == 141
    assert result["candidate_count"] == result["normative_count"]
    assert "manual_review_count" not in result
    assert classifier.matrix_shape == (result["scored_count"], 2)
    ambiguous_rows = result["rows"]
    assert ambiguous_rows
    assert result["ambiguous_count"] == len(ambiguous_rows)
    assert all(row["decision"] == "YES" for row in ambiguous_rows)
    assert ambiguous_rows[0]["relative_score"] == pytest.approx(0.9)
    assert all(row["relative_score"] >= 0.5 for row in ambiguous_rows)
    assert all(row["prediction_threshold"] == 0.5 for row in ambiguous_rows)
    assert result["scored_count"] == result["normative_count"]
    assert (
        result["not_ambiguous_count"] + result["ambiguous_count"]
        == result["scored_count"]
    )
    assert all(
        row["candidate_reason"] == "explicit_bcp14_keyword"
        for row in result["rows"]
    )
    assert all(
        row["candidate_type"] == "EXPLICIT_NORMATIVE"
        for row in result["rows"]
    )
    assert all(row["target_clause"].strip() for row in result["rows"])
    assert all(row["line_start"] for row in result["rows"])
    assert [
        (row["clause_id"], row["decision"], row["relative_score"])
        for row in crlf_result["rows"]
    ] == [
        (row["clause_id"], row["decision"], row["relative_score"])
        for row in result["rows"]
    ]


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


@pytest.mark.parametrize("filename", ["RFC2328.txt", "RFC8200.txt"])
def test_rfc_without_explicit_bcp14_terms_has_no_ambiguity_candidates(
    filename: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        rfc_scanner,
        "_transform_components",
        lambda feature_mode, components, texts, features, embeddings: np.zeros(
            (len(texts), 2)
        ),
    )
    classifier = FakeClassifier()
    loaded = LoadedClassifier(
        components={"classifier": classifier},
        metadata={"prediction_threshold": 0.5},
    )
    content = (ROOT / "corpus/raw/rfcs" / filename).read_text(encoding="utf-8")

    result = scan_rfc_text(
        filename=filename,
        content=content,
        loaded_model=loaded,
    )

    assert result["candidate_count"] == 0
    assert result["normative_count"] == 0
    assert result["scored_count"] == 0
    assert result["ambiguous_count"] == 0
    assert classifier.matrix_shape is None


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
        "decision": "YES",
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


def test_qwen_explanations_reject_no_predictions() -> None:
    store = ScanStore()
    row = {
        "clause_id": "RFC9460:1:1",
        "target_clause": "The server sends the response.",
        "previous_clause": "",
        "next_clause": "",
        "decision": "NO",
    }
    scan_id = store.add({"rows": [row]})

    with pytest.raises(ValueError, match="only available for YES"):
        explain_selected(
            value={"scan_id": scan_id, "clause_ids": [row["clause_id"]]},
            config={},
            client=None,  # type: ignore[arg-type]
            scan_store=store,
        )


def test_qwen_explanations_reject_unconfirmed_normativity() -> None:
    store = ScanStore()
    row = {
        "clause_id": "RFC9460:1:2",
        "target_clause": "A client may send the request.",
        "previous_clause": "",
        "next_clause": "",
        "decision": "REVIEW_NORMATIVITY",
    }
    scan_id = store.add({"rows": [row]})

    with pytest.raises(ValueError, match="only available for YES"):
        explain_selected(
            value={"scan_id": scan_id, "clause_ids": [row["clause_id"]]},
            config={},
            client=None,  # type: ignore[arg-type]
            scan_store=store,
        )


def test_qwen_explanation_retries_when_response_exceeds_character_limit() -> None:
    row = {
        "clause_id": "RFC9460:1:1",
        "target_clause": "A sender MUST choose an appropriate value.",
        "previous_clause": "",
        "next_clause": "",
        "decision": "YES",
        "qwen_justification": "",
    }
    store = ScanStore()
    scan_id = store.add({"rows": [row]})

    class RetryingOllamaClient:
        def __init__(self, responses: list[str]) -> None:
            self.responses = responses
            self.payloads: list[dict[str, Any]] = []

        def tags(self) -> dict[str, dict[str, str]]:
            return {"qwen3:32b": {"digest": "expected"}}

        def chat(self, payload: dict[str, Any]) -> dict[str, Any]:
            self.payloads.append(payload)
            return {"message": {"content": self.responses.pop(0)}}

    client = RetryingOllamaClient(
        [
            "x" * 241,
            "The word appropriate has no defined criterion.",
        ]
    )
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

    assert result["explanations"][0]["qwen_justification"] == (
        "The word appropriate has no defined criterion."
    )
    assert row["qwen_justification"] == result["explanations"][0]["qwen_justification"]
    assert len(client.payloads) == 2
    assert client.payloads[1]["messages"][-2] == {
        "role": "assistant",
        "content": "x" * 241,
    }
    assert client.payloads[1]["options"]["num_predict"] == 60

    failed_client = RetryingOllamaClient(["x" * 241, "y" * 241])
    with pytest.raises(OllamaError, match="exceeds 240 characters"):
        explain_selected(
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
            client=failed_client,  # type: ignore[arg-type]
            scan_store=store,
        )
    assert len(failed_client.payloads) == 2
