from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignment.clause_segmentation import (
    classify_clause,
    segment_document,
    split_sentences,
)
from alignment.clause_segmentation.cli import main as clause_main
from alignment.segmentation import (
    Block,
    Document,
    Reference,
    Section,
    SourceLocation,
    parse_document,
)


def _document(blocks: list[Block], heading: str = "Introduction") -> Document:
    return Document(
        document_id="draft-example-07",
        title="Clause Example",
        source_path="draft-example-07.txt",
        source_format="text",
        document_type="internet-draft",
        source_sha256="0" * 64,
        sections=[
            Section(
                section_id="1",
                title=heading,
                path=["1"],
                source_location=SourceLocation(line_start=10, line_end=30),
                blocks=blocks,
            )
        ],
    )


def _block(
    text: str,
    block_type: str = "paragraph",
    references: list[Reference] | None = None,
) -> Block:
    return Block(
        block_id="draft-example-07:1:b0001",
        index=1,
        block_type=block_type,  # type: ignore[arg-type]
        text=text,
        source_location=SourceLocation(line_start=12, line_end=14),
        references=references or [],
    )


def test_sentence_splitter_protects_rfc_punctuation_and_conditions() -> None:
    text = (
        "Version 1.2 follows e.g. RFC 9110. "
        "If X occurs, the client MUST send Y. It then waits."
    )

    spans = split_sentences(text)

    assert [span.text for span in spans] == [
        "Version 1.2 follows e.g. RFC 9110.",
        "If X occurs, the client MUST send Y.",
        "It then waits.",
    ]
    assert text[spans[1].start : spans[1].end] == spans[1].text


@pytest.mark.parametrize(
    ("text", "heading", "expected_label", "expected_reason"),
    [
        (
            "The client MUST send an ACK.",
            "Protocol Operation",
            "NORMATIVE",
            "explicit_bcp14_keyword",
        ),
        (
            "This section provides background information.",
            "Introduction",
            "CONTEXTUAL",
            "contextual_section_prior",
        ),
        (
            "The endpoint may discard the value.",
            "Protocol Operation",
            "MANUAL_REVIEW",
            "lowercase_normative_looking_language",
        ),
        (
            "The endpoint validates the field.",
            "Processing Rules",
            "MANUAL_REVIEW",
            "implicit_requirement_candidate",
        ),
        (
            'The key words "MUST" and "SHOULD" are to be interpreted as BCP 14 terms.',
            "Terminology",
            "CONTEXTUAL",
            "bcp14_terminology_definition",
        ),
        (
            'RFC 9110 says clients "MUST send the field".',
            "Protocol Operation",
            "EXCLUDED",
            "quoted_normative_text",
        ),
    ],
)
def test_evidence_first_classification(
    text: str,
    heading: str,
    expected_label: str,
    expected_reason: str,
) -> None:
    decision = classify_clause(text, "paragraph", heading, "body", ["RFC9110"])

    assert decision.classification == expected_label
    assert decision.reason == expected_reason


def test_referenced_normative_text_precedes_keyword_classification() -> None:
    decision = classify_clause(
        "RFC 9110 states that a client MUST send the field.",
        "paragraph",
        "Protocol Operation",
        "body",
        ["RFC9110"],
    )

    assert decision.classification == "EXCLUDED"
    assert decision.reason == "referenced_normative_text"
    assert "reference:RFC9110" in decision.evidence


def test_internal_reference_does_not_exclude_current_document_requirement() -> None:
    block = _block(
        "Section 2 states that the client MUST send the field.",
        references=[
            Reference(
                target="draft-example-07",
                target_section="2",
                kind="text",
                display_text="Section 2",
            )
        ],
    )

    result = segment_document(_document([block], heading="Protocol Operation"))
    clause = result.clauses[0]

    assert clause.classification == "NORMATIVE"
    assert clause.classification_reason == "explicit_bcp14_keyword"
    assert clause.reference_targets == ["draft-example-07"]


@pytest.mark.parametrize("block_type", ["artwork", "sourcecode", "table", "reference"])
def test_non_prose_blocks_are_retained_as_excluded(block_type: str) -> None:
    decision = classify_clause(
        "message = header payload",
        block_type,  # type: ignore[arg-type]
        "Message Format",
        "body",
    )

    assert decision.classification == "EXCLUDED"
    assert decision.reason == "formal_or_non_prose_block"


def test_segment_document_emits_stable_traceable_clauses() -> None:
    block = _block(
        "This section gives an overview. "
        "For interoperability, a client MUST send an ACK."
    )
    document = _document([block])

    first = segment_document(document)
    second = segment_document(document)

    assert first.to_dict() == second.to_dict()
    assert first.revision_id == "07"
    assert [clause.clause_id for clause in first.clauses] == [
        "draft-example-07:1:b0001:c01",
        "draft-example-07:1:b0001:c02",
    ]
    assert [clause.classification for clause in first.clauses] == [
        "CONTEXTUAL",
        "NORMATIVE",
    ]
    normative = first.clauses[1]
    assert normative.bcp14_keywords == ["MUST"]
    assert normative.mixed_clause is True
    assert normative.source_location.line_start == 12
    assert normative.raw_text == block.text[normative.text_start : normative.text_end]


def test_list_item_is_split_into_traceable_sentence_clauses() -> None:
    result = segment_document(
        _document(
            [
                _block(
                    "1. If X occurs, the client MUST send Y. It then closes.",
                    "list_item",
                )
            ]
        )
    )

    assert [clause.raw_text for clause in result.clauses] == [
        "1. If X occurs, the client MUST send Y.",
        "It then closes.",
    ]
    assert [clause.classification for clause in result.clauses] == [
        "NORMATIVE",
        "CONTEXTUAL",
    ]
    assert [clause.sentence_or_item_index for clause in result.clauses] == [1, 2]
    assert result.clauses[0].text_start == 0
    assert result.clauses[1].text_start == 40


def test_back_matter_is_retained_without_sentence_splitting_and_excluded() -> None:
    document = _document(
        [_block("Roy T. Fielding edited this document. Another sentence follows.")]
    )
    document.sections[0].analysis_scope = "back"

    result = segment_document(document)

    assert len(result.clauses) == 1
    assert result.clauses[0].classification == "EXCLUDED"
    assert result.clauses[0].classification_reason == "back_matter"


def test_reference_sections_are_excluded_without_dropping_text() -> None:
    document = _document([_block('[HTTP] "HTTP Semantics", RFC 9110.')])
    document.sections[0].title = "Normative References"
    document.sections[0].analysis_scope = "references"

    result = segment_document(document)

    assert result.clauses[0].classification == "EXCLUDED"
    assert result.clauses[0].classification_reason == "reference_material"
    assert result.clauses[0].raw_text.startswith("[HTTP]")


def test_contiguous_plaintext_list_items_become_independent_clauses(
    tmp_path: Path,
) -> None:
    source = tmp_path / "draft-list-example-00.txt"
    source.write_text(
        "Internet-Draft                                      January 2026\n\n"
        "List Example\n"
        "draft-list-example-00\n\n"
        "1.  Requirements\n\n"
        "   1.  The client MUST send A.\n"
        "   2.  The server MUST send B.\n",
        encoding="utf-8",
    )

    parsed = parse_document(source)
    result = segment_document(parsed)

    assert [block.block_type for block in parsed.sections[0].blocks] == [
        "list_item",
        "list_item",
    ]
    assert [clause.classification for clause in result.clauses] == [
        "NORMATIVE",
        "NORMATIVE",
    ]


def test_clause_cli_writes_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "draft-cli-example-00.txt"
    source.write_text(
        "Internet-Draft                                      January 2026\n\n"
        "CLI Example\n"
        "draft-cli-example-00\n\n"
        "1.  Requirements\n\n"
        "   A client MUST validate the field.\n",
        encoding="utf-8",
    )
    output = tmp_path / "clauses.json"

    result = clause_main([str(source), "--output", str(output)])
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert result == 0
    assert payload["clauses"][0]["classification"] == "NORMATIVE"
    assert payload["clauses"][0]["revision_id"] == "00"
    assert "Classified draft-cli-example-00" in capsys.readouterr().err


def test_real_rfc_clause_smoke_validation() -> None:
    result = segment_document(parse_document(Path("corpus/raw/rfcs/RFC9110.txt")))
    summary = result.summary()

    assert result.warnings == []
    assert summary["clauses"] > 2_000
    assert summary["normative"] > 100
    assert summary["contextual"] > 20
    assert summary["excluded"] > 50
    assert len({clause.clause_id for clause in result.clauses}) == summary["clauses"]


def test_real_xml_draft_clauses_retain_xpath_provenance() -> None:
    source = Path(
        "corpus/raw/drafts/draft-ietf-dnsop-svcb-https/"
        "draft-ietf-dnsop-svcb-https-12.xml"
    )

    result = segment_document(parse_document(source))

    assert result.revision_id == "12"
    assert result.warnings == []
    assert len(result.clauses) > 500
    assert all(clause.source_location.xpath for clause in result.clauses)
    assert all(
        clause.manual_review == (clause.classification == "MANUAL_REVIEW")
        for clause in result.clauses
    )
