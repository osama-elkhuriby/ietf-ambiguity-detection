from __future__ import annotations

import json
from pathlib import Path

import pytest

from models.annotation_v2.aggregation import (
    conservative_aggregate,
    v1_compatible_comparison,
)
from models.annotation_v2.constants import OUTPUT_SCHEMA
from models.annotation_v2.data import (
    ContextBuilder,
    load_v1_votes_for_allowed_rows,
    render_user_payload,
)
from models.annotation_v2.ollama import (
    ModelAdapter,
    cache_identity,
    classify_envelope,
    execute_cell,
    post_probe_block_reasons,
)
from models.annotation_v2.schema import validate_annotation


def _annotation(label: str = "CLEAR") -> dict[str, str]:
    if label == "AMBIGUOUS":
        return {
            "label": label,
            "ambiguity_subtype": "U-1",
            "evidence_span": "term",
            "interpretation_a": "reading A",
            "interpretation_b": "reading B",
            "missing_information": "",
            "justification": "The two supplied-context readings change behavior.",
            "context_issue": "none",
        }
    if label == "INSUFFICIENT_CONTEXT":
        return {
            "label": label,
            "ambiguity_subtype": "none",
            "evidence_span": "",
            "interpretation_a": "",
            "interpretation_b": "",
            "missing_information": "The referenced definition is needed.",
            "justification": "The bundle does not contain potentially decisive context.",
            "context_issue": "missing_definition",
        }
    return {
        "label": label,
        "ambiguity_subtype": "none",
        "evidence_span": "",
        "interpretation_a": "",
        "interpretation_b": "",
        "missing_information": "",
        "justification": "The two choices are explicitly permitted.",
        "context_issue": "none",
    }


def _judgment(label: str | None, status: str = "VALID") -> dict[str, str | None]:
    return {"semantic_label": label, "operational_status": status}


def test_schema_enforces_label_dependent_rules_and_field_limits() -> None:
    assert validate_annotation(_annotation("CLEAR"), "term") == []
    assert validate_annotation(_annotation("AMBIGUOUS"), "a term appears") == []
    assert validate_annotation(_annotation("INSUFFICIENT_CONTEXT"), "term") == []
    oversized = _annotation("CLEAR")
    oversized["justification"] = "x" * 1001
    assert "justification exceeds 1000 characters" in validate_annotation(oversized, "term")
    bad_clear = _annotation("CLEAR")
    bad_clear["missing_information"] = "hidden definition"
    assert any("CLEAR requires empty" in error for error in validate_annotation(bad_clear, "term"))


def test_failure_and_semantic_abstention_are_distinct() -> None:
    status, annotation, errors = classify_envelope(
        {"done_reason": "stop", "message": {"content": json.dumps(_annotation("INSUFFICIENT_CONTEXT"))}},
        "term",
    )
    assert (status, annotation["label"], errors) == ("VALID", "INSUFFICIENT_CONTEXT", [])
    status, annotation, _ = classify_envelope(
        {"done_reason": "length", "message": {"content": "{}"}}, "term"
    )
    assert status == "TOKEN_EXHAUSTED"
    assert annotation is None
    status, annotation, _ = classify_envelope(
        {"done_reason": "stop", "message": {"content": "", "refusal": "no"}}, "term"
    )
    assert status == "REFUSAL"
    assert annotation is None
    assert classify_envelope({"message": {"content": "not json"}}, "term")[0] == "MALFORMED_OUTPUT"


def test_conservative_aggregation_requires_complete_three_model_panel() -> None:
    unanimous = conservative_aggregate([
        _judgment("CLEAR"), _judgment("CLEAR"), _judgment("CLEAR")
    ])
    assert unanimous["aggregation_status"] == "UNANIMOUS"
    assert unanimous["semantic_label"] == "CLEAR"
    disputed = conservative_aggregate([
        _judgment("CLEAR"), _judgment("CLEAR"), _judgment("AMBIGUOUS")
    ])
    assert disputed["aggregation_status"] == "MAJORITY_DISPUTED"
    abstained = conservative_aggregate([
        _judgment("CLEAR"), _judgment("CLEAR"), _judgment("INSUFFICIENT_CONTEXT")
    ])
    assert abstained["aggregation_status"] == "REVIEW_REQUIRED"
    incomplete = conservative_aggregate([
        _judgment("CLEAR"), _judgment("CLEAR"), _judgment(None, "TIMEOUT")
    ])
    assert incomplete["aggregation_status"] == "REVIEW_REQUIRED_INCOMPLETE_PANEL"
    assert incomplete["semantic_label"] is None
    assert v1_compatible_comparison([
        _judgment("AMBIGUOUS"), _judgment("AMBIGUOUS"), _judgment(None, "TIMEOUT")
    ])["label"] == "yes"


def test_cache_invalidates_every_contract_dimension() -> None:
    adapter = ModelAdapter("model:1", False, 0.0, 1, 4096, 256, 30)
    kwargs = {
        "model_digest": "digest-a",
        "adapter": adapter,
        "context_variant": "local",
        "user_payload": {"target_clause": "term"},
        "retrieval_manifest_hash": "retrieval-a",
        "template_hash": "template-a",
        "ollama_version": "1",
    }
    base = cache_identity(**kwargs)["cache_key"]
    identity = cache_identity(**kwargs)
    assert identity["system_prompt_sha256"]
    assert identity["output_schema_sha256"]
    assert identity["runner_version"]
    mutations = [
        {"model_digest": "digest-b"},
        {"context_variant": "resolved"},
        {"user_payload": {"target_clause": "changed"}},
        {"retrieval_manifest_hash": "retrieval-b"},
        {"template_hash": "template-b"},
        {"ollama_version": "2"},
        {"adapter": ModelAdapter("model:1", True, 0.0, 1, 4096, 256, 30)},
    ]
    for mutation in mutations:
        changed = {**kwargs, **mutation}
        assert cache_identity(**changed)["cache_key"] != base


def _write_document(root: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    family = "family-allowed"
    document = "draft-example-01"
    path = root / family / "run-v1" / "inputs" / f"{document}.json"
    path.parent.mkdir(parents=True)
    clauses = [
        {
            "index": 1,
            "clause_id": "c1",
            "document_id": document,
            "revision_id": "01",
            "section_id": "2",
            "section_heading": "Definitions",
            "block_id": "b1",
            "raw_text": "Choice: one of two explicitly permitted options.",
            "source_location": {"line_start": 10, "line_end": 10},
            "reference_targets": [],
        },
        {
            "index": 2,
            "clause_id": "target",
            "document_id": document,
            "revision_id": "01",
            "section_id": "3",
            "section_heading": "Operation",
            "block_id": "b2",
            "raw_text": "An endpoint MAY select either Choice.",
            "source_location": {"line_start": 20, "line_end": 20},
            "reference_targets": ["EXTERNAL"],
        },
        {
            "index": 3,
            "clause_id": "c3",
            "document_id": document,
            "revision_id": "01",
            "section_id": "3",
            "section_heading": "Operation",
            "block_id": "b2",
            "raw_text": "The selection is recorded.",
            "source_location": {"line_start": 21, "line_end": 21},
            "reference_targets": [],
        },
    ]
    path.write_text(json.dumps({
        "family_id": family,
        "document_id": document,
        "source_sha256": "source-hash",
        "clauses": clauses,
    }), encoding="utf-8")
    row = {
        "record_id": "record-1",
        "family_id": family,
        "document_id": document,
        "revision": "01",
        "section_ids": "3",
        "clause_ids": "target",
        "clause_text": "An endpoint MAY select either Choice.",
    }
    return row, (family,)


def test_context_is_same_version_traceable_and_external_reference_unavailable(tmp_path: Path) -> None:
    row, allowed = _write_document(tmp_path)
    bundle = ContextBuilder(tmp_path, allowed).build(row, "resolved")
    assert all(item["document_id"] == row["document_id"] for item in bundle["items"])
    assert all(item["revision"] == "01" for item in bundle["items"])
    assert all(item["content_sha256"] for item in bundle["items"])
    assert bundle["coverage_claim"] == "bounded_same_document_same_revision_context_only"
    assert bundle["retrieval"] == [{
        "reference": "EXTERNAL",
        "status": "unavailable_source",
        "reason": "external_reference_not_resolved_without_exact_permitted_version",
    }]
    with pytest.raises(PermissionError):
        ContextBuilder(tmp_path, allowed).build({**row, "family_id": "family-blocked"}, "local")


class _FakeClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def request(self, _path: str, payload: dict[str, object], _timeout: int) -> dict[str, object]:
        self.requests.append(payload)
        if len(self.requests) == 1:
            return {"done_reason": "stop", "message": {"content": "not-json"}}
        return {
            "done_reason": "stop",
            "message": {"content": json.dumps(_annotation("CLEAR")), "thinking": "separate"},
            "eval_count": 20,
        }


def test_retry_is_fresh_bounded_append_only_and_resumable(tmp_path: Path) -> None:
    row, allowed = _write_document(tmp_path / "alignments")
    bundle = ContextBuilder(tmp_path / "alignments", allowed).build(row, "local")
    client = _FakeClient()
    adapter = ModelAdapter("model:1", False, 0.0, 1, 4096, 256, 30)
    kwargs = {
        "client": client,
        "adapter": adapter,
        "model_digest": "digest",
        "model_metadata": {"template": "template"},
        "bundle": bundle,
        "output_root": tmp_path / "output",
        "experiment_id": "experiment",
        "ollama_version": "version",
        "max_retries": 1,
    }
    result = execute_cell(**kwargs)
    assert result["operational_status"] == "VALID"
    assert result["attempt_count"] == 2
    assert result["accepted_attempt"].endswith("attempt-002.json")
    assert len(client.requests) == 2
    second_messages = client.requests[1]["messages"]
    assert all(message["role"] != "assistant" for message in second_messages)
    assert "not-json" not in json.dumps(second_messages)
    attempts = list((tmp_path / "output").glob("**/attempt-*.json"))
    assert len(attempts) == 2
    resumed = execute_cell(**kwargs)
    assert resumed["cache_identity"]["cache_key"] == result["cache_identity"]["cache_key"]
    assert len(client.requests) == 2


def test_output_schema_has_finite_string_limits() -> None:
    for name, spec in OUTPUT_SCHEMA["properties"].items():
        if spec.get("type") == "string" and name not in {"label", "ambiguity_subtype", "context_issue"}:
            assert "maxLength" in spec


def test_post_probe_gate_enforces_ram_reserve_and_paging() -> None:
    gib = 1024**3
    limits = {
        "ram_reserve_gib": 8,
        "heavy_paging_delta_gib": 1,
        "projected_pilot_runtime_ceiling_seconds": 21600,
    }
    before = {"ram": {"available_bytes": 20 * gib}, "paging": {"used_bytes": 0}}
    after = {"ram": {"available_bytes": 4 * gib}, "paging": {"used_bytes": 2 * gib}}
    reasons = post_probe_block_reasons(before, after, 100, limits)
    assert "post-probe available RAM fell below frozen reserve" in reasons
    assert "heavy paging threshold exceeded" in reasons


def test_rendered_payload_excludes_votes_labels_and_predictions(tmp_path: Path) -> None:
    row, allowed = _write_document(tmp_path)
    payload = render_user_payload(ContextBuilder(tmp_path, allowed).build(row, "local"))
    serialized = json.dumps(payload).casefold()
    for forbidden in ("v1_pattern", "model_vote", "classifier_prediction", "confidence_tier"):
        assert forbidden not in serialized


def test_v1_join_rejects_cross_record_or_family_response(tmp_path: Path) -> None:
    row = {"record_id": "record-1", "family_id": "family-allowed"}
    for directory in ("gemma3__27b", "gpt-oss__20b", "qwen3__32b"):
        path = tmp_path / directory / "record-1.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "record_id": "wrong-record",
                "family_id": "family-allowed",
                "status": "valid",
                "annotation": {"is_ambiguous": "no"},
            }),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        load_v1_votes_for_allowed_rows([row], tmp_path)
