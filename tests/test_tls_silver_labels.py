from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from alignment.version_alignment.models import Clause, Document
from corpus.build_tls_silver_labels import (
    is_clear_control_eligible,
    match_erratum_to_rfc,
    normalize_evidence_text,
    normalize_section_id,
    path_reference,
)


def clause(clause_id: str, index: int, text: str, section: str = "4.2") -> Clause:
    return Clause(
        clause_id=clause_id,
        document_id="RFC8446",
        revision_id=None,
        index=index,
        block_id=f"RFC8446:{section}:b{index:04d}",
        block_type="paragraph",
        section_id=section,
        section_heading="Example",
        section_path=[section],
        raw_text=text,
        classification="CONTEXTUAL",
        source_location={"line_start": index, "line_end": index},
        text_start=0,
        text_end=len(text),
        normalized_text=text,
    )


def document(clauses: list[Clause]) -> Document:
    return Document(
        family_id="draft-ietf-tls-tls13",
        document_id="RFC8446",
        revision=None,
        publication_date=date(2018, 8, 1),
        available_at=date(2018, 8, 1),
        date_precision="month",
        source_path="RFC8446.txt",
        source_sha256="a" * 64,
        input_sha256="b" * 64,
        input_path="RFC8446.json",
        clauses=clauses,
    )


def test_evidence_normalization_preserves_protocol_content() -> None:
    assert normalize_evidence_text("  client\r\n MUST  send  0x01 ") == (
        "client MUST send 0x01"
    )
    assert normalize_section_id(" 4.6.1. ") == "4.6.1"


def test_path_reference_supports_repository_and_external_runs(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    assert path_reference(root / "run", root) == "run"
    assert path_reference(tmp_path / "external", root) == (
        tmp_path / "external"
    ).as_posix()


def test_erratum_maps_to_unique_contiguous_clause_group() -> None:
    rfc = document(
        [
            clause("c1", 1, "Unrelated text."),
            clause("c2", 2, "The client MUST send"),
            clause("c3", 3, "an alert."),
        ]
    )
    result = match_erratum_to_rfc(
        {
            "errata_id": "1",
            "section": "4.2.",
            "orig_text": "The client MUST send an alert.",
        },
        rfc,
    )
    assert result["status"] == "ACCEPTED"
    assert result["best_match"]["clause_ids"] == ["c2", "c3"]
    assert result["best_match"]["text_ratio"] == 1.0


def test_erratum_does_not_force_a_weak_match() -> None:
    rfc = document([clause("c1", 1, "The server sends a response.")])
    result = match_erratum_to_rfc(
        {
            "errata_id": "2",
            "section": "4.2",
            "orig_text": "Completely different.",
        },
        rfc,
    )
    assert result["status"] == "UNCERTAIN"
    assert result["decision_reason"] == "no_unique_high_similarity_section_match"


def test_clear_control_filter_rejects_draft_notes_and_questions() -> None:
    good = clause(
        "good",
        1,
        "The client sends this value only after it has validated the server response.",
    )
    good.classification = "NORMATIVE"
    assert is_clear_control_eligible(good)

    contextual = clause(
        "contextual",
        2,
        "The client sends this value only after it has validated the server response.",
    )
    assert not is_clear_control_eligible(contextual)

    todo = clause("todo", 2, "See the TODO above about early_data and PSK handling.??]")
    assert not is_clear_control_eligible(todo)

    question = clause(
        "question",
        3,
        "Should the server send this response before it validates the client?",
    )
    assert not is_clear_control_eligible(question)

    change_log = clause(
        "log",
        4,
        "The client now sends the extension after processing the response data.",
    )
    change_log.section_heading = "Change Log"
    assert not is_clear_control_eligible(change_log)


def test_saved_tls_candidate_set_is_review_only_and_traceable() -> None:
    root = Path(__file__).resolve().parents[1]
    family_dir = (
        root / "corpus/processed/ground_truth/draft-ietf-tls-tls13"
    )
    output = family_dir / "silver"
    envelope = json.loads(
        (output / "tls13_silver_ambiguity_candidates.json").read_text()
    )
    report = json.loads((output / "tls13_silver_label_report.json").read_text())
    records = envelope["records"]
    with (output / "tls13_silver_ambiguity_candidates.csv").open(newline="") as handle:
        csv_records = list(csv.DictReader(handle))
    with (family_dir / "review/tls13_ground_truth_review_template.csv").open(
        newline=""
    ) as handle:
        review_records = list(csv.DictReader(handle))

    input_dir = (
        root
        / "corpus/processed/alignments/draft-ietf-tls-tls13"
        / "pilot-semantic-v5/inputs"
    )
    known_clause_ids = {
        item["clause_id"]
        for path in input_dir.glob("*.json")
        for item in json.loads(path.read_text())["clauses"]
    }

    assert len(records) == report["candidate_records"] == 242
    assert len({row["record_id"] for row in records}) == len(records)
    assert {row["record_id"] for row in csv_records} == {
        row["record_id"] for row in records
    }
    assert {row["record_id"] for row in review_records} == {
        row["record_id"] for row in records
    }
    assert all(row["human_is_ambiguous"] == "" for row in review_records)
    assert all(row["adjudication_status"] == "PENDING" for row in review_records)
    assert all(row["ambiguity_label"] == "UNCERTAIN" for row in records)
    assert all(row["ground_truth"] is False for row in records)
    assert all(row["requires_human_review"] is True for row in records)
    assert all(set(row["clause_ids"]) <= known_clause_ids for row in records)
    assert report["accepted_errata_without_draft_candidates"] == ["8799"]
    assert report["uncertain_errata_ids"] == ["6142"]
