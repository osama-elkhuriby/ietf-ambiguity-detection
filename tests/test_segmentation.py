from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignment.segmentation import parse_document
from alignment.segmentation.cli import main as segmentation_main
from alignment.segmentation.normalize import join_wrapped_lines

TEXT_DRAFT = """Network Working Group                                      A. Author
Internet-Draft                                             January 2026

                    A Small Protocol Example
                       draft-example-00

Abstract

   This document defines a small example protocol.

Status of This Memo

   This boilerplate is not part of the abstract.

Table of Contents

   1.  Introduction  . . . . . . . . . . . . . . . . . . . . . .  2
     1.1.  Requirements . . . . . . . . . . . . . . . . . . . .  2
   2.  References  . . . . . . . . . . . . . . . . . . . . . .  3

Author                  Expires July 2026                    [Page 1]
\f
Internet-Draft                  Example                    January 2026

1.  Introduction

   The client uses the behavior defined in Section 4.2 of [HTTP].

   The client retries as described in Section 1.1.
   The Section field identifies the selected protocol section.

   1.  This numbered sentence is a list item, not a new section.

   2.  It remains in the Introduction section.

      vector[sender] = {
        state: ready
      }

1.1.  Requirements

   A client MUST send a request as described by RFC 2119.

2.  References

2.1.  Normative References

   [HTTP]  Fielding, R., "HTTP Semantics", RFC 9110,
           https://www.rfc-editor.org/rfc/rfc9110.
"""


XML_DRAFT = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE rfc [<!ENTITY nbsp "&#160;">]>
<rfc docName="draft-example-01" version="3">
  <front>
    <title>Small XML Example</title>
    <abstract><t>An XML&nbsp;abstract.</t></abstract>
  </front>
  <middle>
    <section anchor="intro">
      <name>Introduction</name>
      <t>A client uses <xref target="HTTP" section="4.2"/> for its request.</t>
      <ul><li>The client MUST validate the response.</li></ul>
      <section anchor="wire"><name>Wire Format</name>
        <sourcecode type="abnf"><![CDATA[message = header body]]></sourcecode>
      </section>
    </section>
  </middle>
  <back>
    <references><name>Normative References</name>
      <reference anchor="HTTP" target="https://www.rfc-editor.org/info/rfc9110">
        <front><title>HTTP Semantics</title><seriesInfo name="RFC" value="9110"/></front>
      </reference>
    </references>
    <section anchor="examples"><name>Examples</name><t>One example.</t></section>
  </back>
</rfc>
"""


XML_V2_DRAFT = """<?xml version="1.0" encoding="utf-8"?>
<rfc docName="draft-example-v2-00">
  <front><title>Small XML2RFC v2 Example</title></front>
  <middle>
    <section title="Legacy Lists" anchor="legacy">
      <t>Available modes:
        <list style="hanging">
          <t hangText="safe">A client follows <xref target="RULES"/>.</t>
          <t hangText="fast">A client skips optional work.</t>
        </list>
      </t>
      <aside><t>An explanatory aside.</t></aside>
      <texttable>
        <ttcol>Mode</ttcol><ttcol>Meaning</ttcol>
        <c>safe</c><c>Validate input</c>
        <c>fast</c><c>Skip optional work</c>
      </texttable>
    </section>
  </middle>
  <back/>
</rfc>
"""


XML_EXTERNAL_REFERENCE_DRAFT = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE rfc [
  <!ENTITY RFC2119 SYSTEM "https://bib.ietf.org/public/rfc/bibxml/reference.RFC.2119.xml">
]>
<rfc docName="draft-external-reference-00">
  <front><title>External Reference Example</title></front>
  <middle>
    <section title="Requirements">
      <t>The key words are defined by <xref target="RFC2119"/>.</t>
    </section>
  </middle>
  <back>
    <references title="Normative References">&RFC2119;</references>
  </back>
</rfc>
"""


XML_STRUCTURE_EDGE_DRAFT = """<?xml version="1.0" encoding="utf-8"?>
<rfc docName="draft-structure-edge-00" version="3">
  <front><title>Structure Edge Cases</title></front>
  <middle>
    <section><name>Structures</name>
      <figure>
        <preamble>Before the diagram.</preamble>
        <artwork><![CDATA[client --> server]]></artwork>
        <postamble>After the diagram.</postamble>
      </figure>
      <ul>
        <li>Outer item.<ul><li>Inner item.</li></ul></li>
      </ul>
      <table>
        <thead><tr><th>Field</th><th>Meaning</th></tr></thead>
        <tbody><tr><td>TYPE</td><td>Message type</td></tr></tbody>
      </table>
    </section>
  </middle>
</rfc>
"""


TEXT_STRUCTURE_EDGE_DRAFT = """Internet-Draft                         January 2026

Plaintext Structure Edge Cases
draft-structure-edge-00

1.  URI References

   It is RECOMMENDED that implementations support long URIs.  This is
   still body text.

   +=======+=============+
   | Value | Description |
   +=======+=============+
   | one   | First value |
   +-------+-------------+

                         Table 1

   field-name = token
   token      = 1*tchar

   |  *Note:* A client MAY retain this value.  It can discard the value
   |  later.

2.  Requirements

   1.  A client MUST send A.  It then sends B.
   2.  A server SHOULD acknowledge both values.
"""


RFC_BACK_MATTER_EDGE = """Request for Comments: 9999
Category: Standards Track                              January 2026

Back Matter Edge Cases

1.  Introduction

   Ordinary body text.

Acknowledgements

   Contributors include Daniel W. Connolly, Phillip M. Hallam-Baker,
   Jeffery L. Hostetler, and Scott D. Lawrence.
"""


def _write(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_text_parser_builds_hierarchy_and_ignores_rendering_noise(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "draft-example-00.txt", TEXT_DRAFT)

    document = parse_document(source)

    assert document.source_format == "text"
    assert document.document_type == "internet-draft"
    assert document.document_id == "draft-example-00"
    assert [section.section_id for section in document.sections] == [
        "abstract",
        "1",
        "2",
    ]
    assert document.sections[1].children[0].section_id == "1.1"
    assert document.sections[2].children[0].analysis_scope == "references"
    assert "boilerplate" not in document.sections[0].blocks[0].text
    all_text = "\n".join(block.text for block in document.iter_blocks())
    assert "[Page 1]" not in all_text
    assert "This numbered sentence is a list item" in all_text
    assert len({block.block_id for block in document.iter_blocks()}) == len(
        list(document.iter_blocks())
    )


def test_text_parser_extracts_references_and_definitions(tmp_path: Path) -> None:
    source = _write(tmp_path / "draft-example-00.txt", TEXT_DRAFT)

    document = parse_document(source)
    introduction = document.sections[1]
    reference = introduction.blocks[0].references[0]

    assert reference.target == "HTTP"
    assert reference.target_section == "4.2"
    internal_reference = introduction.blocks[1].references[0]
    assert internal_reference.target == "draft-example-00"
    assert internal_reference.target_section == "1.1"
    assert len(introduction.blocks[1].references) == 1
    assert document.reference_definitions[0].tag == "HTTP"
    assert document.reference_definitions[0].document_id == "RFC9110"
    assert document.sections[2].children[0].blocks[0].block_type == "reference"


def test_text_artwork_does_not_treat_protocol_syntax_as_a_citation(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "draft-example-00.txt", TEXT_DRAFT)

    document = parse_document(source)
    artwork = next(
        block for block in document.sections[1].blocks if block.block_type == "artwork"
    )

    assert "vector[sender]" in artwork.text
    assert artwork.references == []


def test_plaintext_parser_distinguishes_tables_abnf_notes_and_body_references(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "draft-structure-edge-00.txt", TEXT_STRUCTURE_EDGE_DRAFT)

    document = parse_document(source)
    uri_references = document.sections[0]

    assert uri_references.title == "URI References"
    assert uri_references.analysis_scope == "body"
    assert [block.block_type for block in uri_references.blocks] == [
        "paragraph",
        "table",
        "table",
        "sourcecode",
        "paragraph",
    ]
    assert uri_references.blocks[-1].text == (
        "*Note:* A client MAY retain this value. It can discard the value later."
    )


def test_plaintext_parser_does_not_promote_wrapped_initials_to_list_items(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "RFC9999.txt", RFC_BACK_MATTER_EDGE)

    document = parse_document(source)
    acknowledgements = document.sections[-1]

    assert acknowledgements.analysis_scope == "back"
    assert len(acknowledgements.blocks) == 1
    assert acknowledgements.blocks[0].block_type == "paragraph"
    assert "Daniel W. Connolly" in acknowledgements.blocks[0].text
    assert "Jeffery L. Hostetler" in acknowledgements.blocks[0].text


def test_join_wrapped_lines_repairs_uppercase_hyphenation_but_not_suspension() -> None:
    assert join_wrapped_lines(["Accept-", "Encoding is supported."]) == (
        "Accept-Encoding is supported."
    )
    assert join_wrapped_lines(["a transport-", "or session-layer connection"]) == (
        "a transport- or session-layer connection"
    )


def test_xml_parser_preserves_structure_and_xref_metadata(tmp_path: Path) -> None:
    source = _write(tmp_path / "draft-example-01.xml", XML_DRAFT)

    document = parse_document(source)

    assert document.source_format == "xml"
    assert document.title == "Small XML Example"
    assert [section.section_id for section in document.sections] == [
        "abstract",
        "1",
        "references",
        "A",
    ]
    introduction = document.sections[1]
    assert [block.block_type for block in introduction.blocks] == [
        "paragraph",
        "list_item",
    ]
    assert introduction.children[0].blocks[0].block_type == "sourcecode"
    reference = introduction.blocks[0].references[0]
    assert (reference.target, reference.target_section, reference.kind) == (
        "HTTP",
        "4.2",
        "xref",
    )
    assert reference.display_text == "Section 4.2 of [HTTP]"
    assert "Section 4.2 of [HTTP]" in introduction.blocks[0].text
    assert document.reference_definitions[0].document_id == "RFC9110"
    assert document.sections[-1].analysis_scope == "appendix"
    assert all(block.source_location.xpath for block in document.iter_blocks())


def test_xml2rfc_v2_nested_lists_and_asides_are_preserved(tmp_path: Path) -> None:
    source = _write(tmp_path / "draft-example-v2-00.xml", XML_V2_DRAFT)

    document = parse_document(source)
    blocks = document.sections[0].blocks

    assert [block.block_type for block in blocks] == [
        "paragraph",
        "list_item",
        "list_item",
        "paragraph",
        "table",
    ]
    assert [block.text for block in blocks] == [
        "Available modes:",
        "safe: A client follows [RULES].",
        "fast: A client skips optional work.",
        "An explanatory aside.",
        "Mode | Meaning\nsafe | Validate input\nfast | Skip optional work",
    ]
    assert blocks[1].references[0].target == "RULES"


def test_xml2rfc_v2_external_reference_entities_are_preserved(tmp_path: Path) -> None:
    source = _write(
        tmp_path / "draft-external-reference-00.xml", XML_EXTERNAL_REFERENCE_DRAFT
    )

    document = parse_document(source)

    assert document.sections[-1].title == "Normative References"
    assert document.sections[-1].blocks[0].text == "[RFC2119]"
    assert document.reference_definitions[0].tag == "RFC2119"
    assert document.reference_definitions[0].document_id == "RFC2119"
    target_url = document.reference_definitions[0].target_url
    assert target_url is not None
    assert target_url.endswith("reference.RFC.2119.xml")
    assert "[RFC2119]" in document.sections[0].blocks[0].text


def test_xml_figures_nested_lists_and_tables_preserve_reading_order(
    tmp_path: Path,
) -> None:
    source = _write(tmp_path / "draft-structure-edge-00.xml", XML_STRUCTURE_EDGE_DRAFT)

    document = parse_document(source)
    blocks = document.sections[0].blocks

    assert [(block.block_type, block.text) for block in blocks] == [
        ("paragraph", "Before the diagram."),
        ("artwork", "client --> server"),
        ("paragraph", "After the diagram."),
        ("list_item", "Outer item."),
        ("list_item", "Inner item."),
        ("table", "Field | Meaning\nTYPE | Message type"),
    ]


def test_parse_is_deterministic(tmp_path: Path) -> None:
    source = _write(tmp_path / "draft-example-01.xml", XML_DRAFT)

    first = parse_document(source).to_dict()
    second = parse_document(source).to_dict()

    assert first == second


def test_cli_writes_canonical_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "draft-example-01.xml", XML_DRAFT)
    output = tmp_path / "parsed" / "document.json"

    result = segmentation_main([str(source), "--output", str(output)])

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["document_id"] == "draft-example-01"
    assert payload["sections"][1]["title"] == "Introduction"
    assert "Parsed draft-example-01" in capsys.readouterr().err


def test_cli_reports_malformed_xml_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _write(tmp_path / "draft-malformed-00.xml", "<rfc><middle></rfc>")

    result = segmentation_main([str(source)])

    assert result == 2
    assert "error:" in capsys.readouterr().err


def test_parser_rejects_unsupported_input(tmp_path: Path) -> None:
    source = _write(tmp_path / "draft-example.md", "# not an Internet-Draft")

    with pytest.raises(ValueError, match="Unsupported"):
        parse_document(source)


def test_real_xml_pilot_parses_without_warnings() -> None:
    source = Path(
        "corpus/raw/drafts/draft-ietf-dnsop-svcb-https/"
        "draft-ietf-dnsop-svcb-https-12.xml"
    )

    document = parse_document(source)

    assert document.document_id == "draft-ietf-dnsop-svcb-https-12"
    assert document.warnings == []
    assert document.summary()["sections"] > 50
    assert document.summary()["references"] > 100


def test_real_text_draft_does_not_promote_numbered_lists_to_sections() -> None:
    source = Path("corpus/raw/drafts/draft-ietf-tls-tls13/draft-ietf-tls-tls13-28.txt")

    document = parse_document(source)

    assert document.warnings == []
    assert [section.section_id for section in document.sections] == [
        "abstract",
        *[str(number) for number in range(1, 13)],
        *list("ABCDEFG"),
    ]
    assert all(
        "This data is not forward secret" not in section.title
        for section in document.iter_sections()
    )


def test_legacy_rfc_preserves_indented_subsections_and_references() -> None:
    document = parse_document(Path("corpus/raw/rfcs/RFC2328.txt"))
    sections = {section.section_id: section for section in document.iter_sections()}

    assert document.document_type == "rfc"
    assert document.document_id == "RFC2328"
    assert document.rfc_number == 2328
    assert document.title == "OSPF Version 2"
    assert document.publication_date == "1998-04-01"
    assert document.category == "Standards Track"
    assert sections["1.2"].path == ["1", "1.2"]
    assert sections["references"].analysis_scope == "references"
    assert len(document.reference_definitions) > 20
    assert document.warnings == []


def test_rfc_unnumbered_reference_groups_are_segmented() -> None:
    document = parse_document(Path("corpus/raw/rfcs/RFC4271.txt"))
    sections = {section.section_id: section for section in document.iter_sections()}

    assert document.title == "A Border Gateway Protocol 4 (BGP-4)"
    assert document.category == "Standards Track"
    assert sections["references.normative"].analysis_scope == "references"
    assert sections["references.informative"].analysis_scope == "references"
    assert len(document.reference_definitions) > 20


def test_all_collected_rfcs_match_metadata_and_have_source_locations() -> None:
    metadata = json.loads(Path("corpus/metadata/rfcs.json").read_text(encoding="utf-8"))

    assert len(metadata) == 12
    for document_id, record in metadata.items():
        document = parse_document(Path(record["local_txt_path"]))
        blocks = list(document.iter_blocks())
        section_paths = [tuple(section.path) for section in document.iter_sections()]

        assert document.document_type == "rfc", document_id
        assert document.document_id == document_id
        assert document.rfc_number == int(document_id.removeprefix("RFC"))
        assert document.title == record["title"], document_id
        assert document.publication_date == record["date"][:10], document_id
        assert document.warnings == [], document_id
        assert blocks, document_id
        assert all(block.source_location.line_start is not None for block in blocks)
        assert len({block.block_id for block in blocks}) == len(blocks)
        assert len(set(section_paths)) == len(section_paths), document_id


def test_rfc_toc_back_matter_is_not_promoted_to_real_sections() -> None:
    document = parse_document(Path("corpus/raw/rfcs/RFC9000.txt"))
    authors_sections = [
        section
        for section in document.iter_sections()
        if section.section_id == "authors-addresses"
    ]

    assert len(authors_sections) == 1
    assert authors_sections[0].source_location.line_start == 8474
    assert authors_sections[0].analysis_scope == "back"


def test_rfc_profile_rejects_xml_input(tmp_path: Path) -> None:
    source = _write(tmp_path / "RFC9999.xml", XML_DRAFT)

    with pytest.raises(ValueError, match="rendered .txt input only"):
        parse_document(source, document_type="rfc")
