from __future__ import annotations

import json
from pathlib import Path

import pytest

from alignment.segmentation import parse_document
from alignment.segmentation.cli import main as segmentation_main

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

   1.  This numbered sentence is a list item, not a new section.

   2.  It remains in the Introduction section.

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
    </section>
  </middle>
  <back/>
</rfc>
"""


def _write(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_text_parser_builds_hierarchy_and_ignores_rendering_noise(tmp_path: Path) -> None:
    source = _write(tmp_path / "draft-example-00.txt", TEXT_DRAFT)

    document = parse_document(source)

    assert document.source_format == "text"
    assert document.document_type == "internet-draft"
    assert document.document_id == "draft-example-00"
    assert [section.section_id for section in document.sections] == ["abstract", "1", "2"]
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
    assert document.reference_definitions[0].tag == "HTTP"
    assert document.reference_definitions[0].document_id == "RFC9110"


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
    assert [block.block_type for block in introduction.blocks] == ["paragraph", "list_item"]
    assert introduction.children[0].blocks[0].block_type == "sourcecode"
    reference = introduction.blocks[0].references[0]
    assert (reference.target, reference.target_section, reference.kind) == ("HTTP", "4.2", "xref")
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
    ]
    assert [block.text for block in blocks] == [
        "Available modes:",
        "safe: A client follows .",
        "fast: A client skips optional work.",
        "An explanatory aside.",
    ]
    assert blocks[1].references[0].target == "RULES"


def test_parse_is_deterministic(tmp_path: Path) -> None:
    source = _write(tmp_path / "draft-example-01.xml", XML_DRAFT)

    first = parse_document(source).to_dict()
    second = parse_document(source).to_dict()

    assert first == second


def test_cli_writes_canonical_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = _write(tmp_path / "draft-example-01.xml", XML_DRAFT)
    output = tmp_path / "parsed" / "document.json"

    result = segmentation_main([str(source), "--output", str(output)])

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["document_id"] == "draft-example-01"
    assert payload["sections"][1]["title"] == "Introduction"
    assert "Parsed draft-example-01" in capsys.readouterr().err


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
    source = Path(
        "corpus/raw/drafts/draft-ietf-tls-tls13/"
        "draft-ietf-tls-tls13-28.txt"
    )

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

        assert document.document_type == "rfc", document_id
        assert document.document_id == document_id
        assert document.rfc_number == int(document_id.removeprefix("RFC"))
        assert document.title == record["title"], document_id
        assert document.publication_date == record["date"][:10], document_id
        assert document.warnings == [], document_id
        assert blocks, document_id
        assert all(block.source_location.line_start is not None for block in blocks)
        assert len({block.block_id for block in blocks}) == len(blocks)


def test_rfc_profile_rejects_xml_input(tmp_path: Path) -> None:
    source = _write(tmp_path / "RFC9999.xml", XML_DRAFT)

    with pytest.raises(ValueError, match="rendered .txt input only"):
        parse_document(source, document_type="rfc")
