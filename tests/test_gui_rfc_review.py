from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from gui import rfc_scanner
from gui.app import (
    ScanStore,
    explain_selected,
    validate_explanation_input,
)
from gui.rfc_scanner import (
    FoldClassifier,
    LoadedClassifier,
    _build_fold_calibration,
    _score_candidate_rows,
    scan_rfc_text,
)
from models.llm.ollama_client import OllamaError

ROOT = Path(__file__).resolve().parents[1]


def _fake_loaded_classifier(thresholds: list[float] | None = None) -> LoadedClassifier:
    fold_thresholds = thresholds or [0.5] * 5
    folds = [
        FoldClassifier(
            fold=index,
            focal_model=(index, "focal"),
            partial4_model=(index, "partial4"),
            focal_reference=np.array([0.1, 0.5, 0.9]),
            partial4_reference=np.array([0.1, 0.5, 0.9]),
            threshold=threshold,
        )
        for index, threshold in enumerate(fold_thresholds, start=1)
    ]
    return LoadedClassifier(
        components={"folds": folds, "tokenizer": object(), "device": "cpu"},
        metadata={"vote_threshold": 3, "vote_count": 5, "prediction_threshold": 0.6},
    )


def test_rfc_scan_ranks_only_normative_clauses(monkeypatch: pytest.MonkeyPatch) -> None:
    inference_calls: list[tuple[Any, ...]] = []

    def predict(
        model: tuple[int, str], _tokenizer: Any, texts: list[str], _device: Any
    ) -> np.ndarray:
        assert _tokenizer is not None
        assert _device == "cpu"
        inference_calls.append(model)
        return np.linspace(0.1, 0.9, len(texts))

    monkeypatch.setattr(rfc_scanner, "_predict_model_probabilities", predict)
    loaded = _fake_loaded_classifier()
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
    assert len(inference_calls) == 20
    ambiguous_rows = result["rows"]
    assert ambiguous_rows
    assert result["ambiguous_count"] == len(ambiguous_rows)
    assert all(row["decision"] == "YES" for row in ambiguous_rows)
    assert ambiguous_rows[0]["relative_score"] > 0.85
    assert all(row["fold_votes"] >= 3 for row in ambiguous_rows)
    assert all(row["prediction_threshold"] == 0.6 for row in ambiguous_rows)
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
    filename: str,
) -> None:
    content = (ROOT / "corpus/raw/rfcs" / filename).read_text(encoding="utf-8")

    result = scan_rfc_text(
        filename=filename,
        content=content,
        loaded_model=_fake_loaded_classifier(),
    )

    assert result["candidate_count"] == 0
    assert result["normative_count"] == 0
    assert result["scored_count"] == 0
    assert result["ambiguous_count"] == 0


def test_fold_vote_uses_each_fold_ecdf_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_scores = {
        (1, "focal"): np.array([0.9, 0.1, 0.1]),
        (2, "focal"): np.array([0.9, 0.1, 0.1]),
        (3, "focal"): np.array([0.9, 0.5, 0.1]),
        (4, "focal"): np.array([0.5, 0.9, 0.1]),
        (5, "focal"): np.array([0.9, 0.9, 0.1]),
    }
    model_scores.update(
        {
            (fold, "partial4"): scores
            for (fold, _), scores in list(model_scores.items())
        }
    )
    def predict(
        model: tuple[int, str],
        _tokenizer: Any,
        texts: list[str],
        _device: Any,
    ) -> np.ndarray:
        assert _tokenizer is not None
        assert len(texts) == 3
        assert _device == "cpu"
        return model_scores[model]

    monkeypatch.setattr(rfc_scanner, "_predict_model_probabilities", predict)
    loaded = _fake_loaded_classifier([0.8, 0.9, 0.6, 0.9, 0.9])
    candidates = [
        {"clause_id": "a", "target_clause": "candidate a"},
        {"clause_id": "b", "target_clause": "candidate b"},
        {"clause_id": "c", "target_clause": "candidate c"},
    ]

    scored = _score_candidate_rows(candidates, loaded)

    assert [(row["clause_id"], row["fold_votes"], row["decision"]) for row in scored] == [
        ("a", 4, "YES"),
        ("b", 3, "YES"),
        ("c", 0, "NO"),
    ]
    assert [row["relative_score"] for row in scored] == pytest.approx(
        [14 / 15, 2 / 3, 1 / 3]
    )


def test_fold_calibration_reuses_inner_oof_ecdf_and_precision_target() -> None:
    focal = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "probability_yes": [0.1, 0.9, 0.2, 0.8],
        }
    )
    partial4 = pd.DataFrame(
        {
            "record_id": ["a", "b", "c", "d"],
            "label": [0, 1, 0, 1],
            "probability_yes": [0.2, 0.8, 0.1, 0.9],
        }
    )

    focal_reference, partial4_reference, threshold = _build_fold_calibration(
        focal, partial4
    )

    assert focal_reference.tolist() == [0.1, 0.9, 0.2, 0.8]
    assert partial4_reference.tolist() == [0.2, 0.8, 0.1, 0.9]
    paired_scores = (
        0.5 * rfc_scanner.empirical_cdf(focal_reference, focal_reference)
        + 0.5
        * rfc_scanner.empirical_cdf(partial4_reference, partial4_reference)
    )
    selected = rfc_scanner.select_precision_threshold(
        focal["label"], paired_scores, 0.30
    )
    assert threshold == pytest.approx(selected["threshold"])


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
