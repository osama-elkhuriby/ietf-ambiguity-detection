"""Canonical representation shared by parsed Internet-Drafts and RFCs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SourceFormat = Literal["text", "xml"]
DocumentType = Literal["internet-draft", "rfc"]
BlockType = Literal[
    "paragraph",
    "list_item",
    "definition",
    "artwork",
    "sourcecode",
    "table",
    "reference",
]


@dataclass(slots=True)
class SourceLocation:
    """Location of a parsed object in the original source."""

    line_start: int | None = None
    line_end: int | None = None
    xpath: str | None = None


@dataclass(slots=True)
class Reference:
    """A reference made by a content block."""

    target: str
    kind: Literal["xref", "relref", "eref", "text"]
    target_section: str | None = None
    display_text: str | None = None


@dataclass(slots=True)
class ReferenceDefinition:
    """Definition of a symbolic reference tag from the document back matter."""

    tag: str
    title: str | None = None
    document_id: str | None = None
    target_url: str | None = None


@dataclass(slots=True)
class Block:
    """Smallest structure-preserving unit emitted by the format parsers."""

    block_id: str
    index: int
    block_type: BlockType
    text: str
    source_location: SourceLocation
    references: list[Reference] = field(default_factory=list)


@dataclass(slots=True)
class Section:
    """A node in an Internet-Draft or RFC's natural section hierarchy."""

    section_id: str
    title: str
    path: list[str]
    source_location: SourceLocation
    blocks: list[Block] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)
    anchor: str | None = None
    analysis_scope: Literal["body", "front", "references", "appendix"] = "body"

    def iter_sections(self) -> Iterator[Section]:
        yield self
        for child in self.children:
            yield from child.iter_sections()

    def iter_blocks(self) -> Iterator[Block]:
        yield from self.blocks
        for child in self.children:
            yield from child.iter_blocks()


@dataclass(slots=True)
class Document:
    """Canonical parsed document, independent of whether the input was TXT or XML."""

    document_id: str
    title: str | None
    source_path: str
    source_format: SourceFormat
    document_type: DocumentType
    source_sha256: str
    sections: list[Section]
    rfc_number: int | None = None
    publication_date: str | None = None
    category: str | None = None
    reference_definitions: list[ReferenceDefinition] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def iter_sections(self) -> Iterator[Section]:
        for section in self.sections:
            yield from section.iter_sections()

    def iter_blocks(self) -> Iterator[Block]:
        for section in self.sections:
            yield from section.iter_blocks()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> dict[str, int]:
        sections = list(self.iter_sections())
        blocks = list(self.iter_blocks())
        return {
            "sections": len(sections),
            "blocks": len(blocks),
            "references": sum(len(block.references) for block in blocks),
            "reference_definitions": len(self.reference_definitions),
            "warnings": len(self.warnings),
        }
