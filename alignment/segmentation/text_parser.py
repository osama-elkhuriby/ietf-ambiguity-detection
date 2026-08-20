"""Parser for paginated Internet-Draft and RFC plaintext renderings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .models import Block, Document, ReferenceDefinition, Section, SourceLocation
from .normalize import (
    block_identifier,
    file_sha256,
    join_wrapped_lines,
    normalize_multiline,
)
from .references import extract_text_references, parse_text_reference_definition

TextDocumentType = Literal["auto", "internet-draft", "rfc"]

SECTION_RE = re.compile(
    r"^(?P<indent>\s*)(?:(?P<appendix>Appendix)\s+)?"
    r"(?P<identifier>(?:\d+(?:\.\d+)*)|(?:[A-Z](?:\.\d+)*))\.\s{1,}"
    r"(?P<title>\S.*)$"
)
PAGE_FOOTER_RE = re.compile(r"\[Page\s+\d+\]\s*$", re.IGNORECASE)
PAGE_HEADER_RE = re.compile(
    r"^\s*(?:Internet-Draft|RFC\s+\d+)\s{2,}.+\s{2,}"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}\s*$",
    re.IGNORECASE,
)
TOC_DOTS_RE = re.compile(r"(?:\.\s+){3,}\d+\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*o]|\d+[.)]|[a-zA-Z][.)])\s+")
ABSTRACT_RE = re.compile(r"^\s*Abstract\s*$", re.IGNORECASE)
FRONT_BOUNDARY_RE = re.compile(
    r"^\s*(?:Status of This Memo|Copyright Notice|Table of Contents)\s*$", re.IGNORECASE
)
TITLE_BOUNDARY_RE = re.compile(
    r"^\s*(?:Abstract|Status of This Memo|Table of Contents)\s*$", re.IGNORECASE
)
RFC_NUMBER_RE = re.compile(r"(?:Request for Comments:\s*|\bRFC\s+)(\d{3,5})\b", re.IGNORECASE)
RFC_CATEGORY_RE = re.compile(r"^\s*Category:\s*(.+?)\s*$", re.IGNORECASE)
PUBLICATION_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
MONTH_NUMBERS = {
    month.casefold(): index
    for index, month in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        start=1,
    )
}
UNNUMBERED_SECTIONS: dict[str, tuple[str, str]] = {
    "references": ("references", "references"),
    "normative references": ("references.normative", "references"),
    "informative references": ("references.informative", "references"),
    "acknowledgements": ("acknowledgements", "body"),
    "acknowledgments": ("acknowledgements", "body"),
    "author's address": ("authors-addresses", "body"),
    "authors' addresses": ("authors-addresses", "body"),
    "authors’ addresses": ("authors-addresses", "body"),
    "security considerations": ("security-considerations", "body"),
    "iana considerations": ("iana-considerations", "body"),
}


@dataclass(slots=True)
class _SectionBuilder:
    identifier: str
    title: str
    line_start: int
    appendix: bool = False
    analysis_scope: str | None = None
    content: list[tuple[int, str]] = field(default_factory=list)
    children: list[_SectionBuilder] = field(default_factory=list)


@dataclass(slots=True)
class _Heading:
    identifier: str
    title: str
    appendix: bool = False
    analysis_scope: str | None = None
    synthetic: bool = False


def _clean_lines(text: str) -> list[tuple[int, str]]:
    cleaned: list[tuple[int, str]] = []
    for line_number, line in enumerate(
        text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), start=1
    ):
        line = line.replace("\f", "")
        if PAGE_FOOTER_RE.search(line) or PAGE_HEADER_RE.match(line):
            continue
        cleaned.append((line_number, line.rstrip()))
    return cleaned


def _section_heading(
    line: str,
    document_type: Literal["internet-draft", "rfc"],
    known_section_ids: set[str],
) -> _Heading | None:
    unnumbered = UNNUMBERED_SECTIONS.get(line.strip().casefold())
    if document_type == "rfc" and unnumbered is not None:
        identifier, scope = unnumbered
        return _Heading(identifier, line.strip(), analysis_scope=scope, synthetic=True)

    match = SECTION_RE.match(line)
    if not match:
        return None
    title = match.group("title").strip()
    if TOC_DOTS_RE.search(title):
        return None
    identifier = match.group("identifier")
    indent = match.group("indent")
    if indent:
        if document_type != "rfc" or _section_depth(identifier) < 2:
            return None
        parent_identifier = identifier.rsplit(".", maxsplit=1)[0]
        if parent_identifier not in known_section_ids:
            return None
    return _Heading(identifier, title, bool(match.group("appendix")))


def _section_depth(identifier: str) -> int:
    return identifier.count(".") + 1


def _build_section_tree(
    lines: list[tuple[int, str]], document_type: Literal["internet-draft", "rfc"]
) -> list[_SectionBuilder]:
    roots: list[_SectionBuilder] = []
    stack: list[_SectionBuilder] = []
    current: _SectionBuilder | None = None
    known_section_ids: set[str] = set()

    for line_number, line in lines:
        heading = _section_heading(line, document_type, known_section_ids)
        if heading is None:
            if current is not None:
                current.content.append((line_number, line))
            continue

        builder = _SectionBuilder(
            heading.identifier,
            heading.title,
            line_number,
            heading.appendix,
            heading.analysis_scope,
        )
        if heading.synthetic:
            stack.clear()
            roots.append(builder)
        else:
            depth = _section_depth(heading.identifier)
            while stack and _section_depth(stack[-1].identifier) >= depth:
                stack.pop()
            if stack:
                stack[-1].children.append(builder)
            else:
                roots.append(builder)
        stack.append(builder)
        current = builder
        known_section_ids.add(heading.identifier)
    return roots


def _extract_abstract(lines: list[tuple[int, str]]) -> _SectionBuilder | None:
    start_index = next(
        (index for index, (_, line) in enumerate(lines) if ABSTRACT_RE.match(line)), None
    )
    if start_index is None:
        return None
    content: list[tuple[int, str]] = []
    for line_number, line in lines[start_index + 1 :]:
        if FRONT_BOUNDARY_RE.match(line):
            break
        content.append((line_number, line))
    if not any(line.strip() for _, line in content):
        return None
    return _SectionBuilder("abstract", "Abstract", lines[start_index][0], content=content)


def _paragraphs(lines: list[tuple[int, str]]) -> list[list[tuple[int, str]]]:
    paragraphs: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for item in lines:
        if item[1].strip():
            current.append(item)
        elif current:
            paragraphs.append(current)
            current = []
    if current:
        paragraphs.append(current)
    return paragraphs


def _looks_structural(lines: list[str]) -> bool:
    if len(lines) < 2:
        return False
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents or min(indents) < 6:
        return False
    symbolic = sum(
        bool(re.search(r"(?:^\s*[|+{}<>]|::=|-->?|\bstruct\s*\{|\bopaque\s+\w+<)", line))
        for line in lines
    )
    return symbolic >= max(1, len(lines) // 3)


def _paragraph_text(lines: list[str], structural: bool) -> str:
    return normalize_multiline("\n".join(lines)) if structural else join_wrapped_lines(lines)


def _convert_section(
    builder: _SectionBuilder,
    document_id: str,
    counter: list[int],
    parent_path: list[str],
    analysis_scope: str | None = None,
) -> Section:
    scope = analysis_scope or builder.analysis_scope or (
        "appendix"
        if builder.appendix or builder.identifier[:1].isalpha()
        else "references"
        if "reference" in builder.title.casefold()
        else "body"
    )
    section = Section(
        section_id=builder.identifier,
        title=builder.title,
        path=[*parent_path, builder.identifier],
        analysis_scope=scope,  # type: ignore[arg-type]
        source_location=SourceLocation(line_start=builder.line_start, line_end=builder.line_start),
    )
    for paragraph in _paragraphs(builder.content):
        raw_lines = [line for _, line in paragraph]
        structural = _looks_structural(raw_lines)
        text = _paragraph_text(raw_lines, structural)
        if not text:
            continue
        counter[0] += 1
        section.blocks.append(
            Block(
                block_id=block_identifier(document_id, builder.identifier, counter[0]),
                index=counter[0],
                block_type=(
                    "artwork"
                    if structural
                    else "list_item"
                    if LIST_ITEM_RE.match(raw_lines[0])
                    else "paragraph"
                ),
                text=text,
                source_location=SourceLocation(
                    line_start=paragraph[0][0], line_end=paragraph[-1][0]
                ),
                references=extract_text_references(text),
            )
        )
    if builder.content:
        section.source_location.line_end = builder.content[-1][0]
    section.children = [
        _convert_section(child, document_id, counter, section.path)
        for child in builder.children
    ]
    return section


def detect_text_document_type(path: Path, text: str) -> Literal["internet-draft", "rfc"]:
    """Identify a rendered Internet-Draft or RFC from its name and header."""
    header = "\n".join(text.splitlines()[:80])
    if path.stem.casefold().startswith("draft-") or re.search(
        r"^\s*Internet-Draft\b", header, re.IGNORECASE | re.MULTILINE
    ):
        return "internet-draft"
    if re.fullmatch(r"RFC\d+", path.stem, re.IGNORECASE) or re.search(
        r"^\s*Request for Comments:\s*\d+\b", header, re.IGNORECASE | re.MULTILINE
    ):
        return "rfc"
    raise ValueError(f"Cannot determine whether {path} is an Internet-Draft or RFC")


def parse_text_document(
    path: Path, document_type: TextDocumentType = "auto"
) -> Document:
    """Parse a plaintext Internet-Draft or RFC into a section/block tree."""
    text = path.read_text(encoding="utf-8", errors="replace")
    resolved_type = (
        detect_text_document_type(path, text) if document_type == "auto" else document_type
    )
    lines = _clean_lines(text)
    builders = _build_section_tree(lines, resolved_type)
    abstract = _extract_abstract(lines)
    if abstract is not None:
        builders.insert(0, abstract)

    rfc_number = _extract_rfc_number(lines, path) if resolved_type == "rfc" else None
    document_id = f"RFC{rfc_number}" if rfc_number is not None else path.stem
    counter = [0]
    sections = [
        _convert_section(
            builder,
            document_id,
            counter,
            [],
            "front" if builder.identifier == "abstract" else None,
        )
        for builder in builders
    ]

    definitions: list[ReferenceDefinition] = []
    for section in sections:
        for nested in section.iter_sections():
            if nested.analysis_scope != "references":
                continue
            for block in nested.blocks:
                definition = parse_text_reference_definition(block.text)
                if definition is not None:
                    definitions.append(definition)

    title = (
        _extract_rfc_title(lines)
        if resolved_type == "rfc"
        else _extract_draft_title(lines, document_id)
    )
    warnings: list[str] = []
    if not any(section.analysis_scope == "body" for section in sections):
        warnings.append("No numbered body sections were parsed")
    if not sections:
        warnings.append("No content sections were parsed")
    if resolved_type == "rfc" and rfc_number is None:
        warnings.append("RFC number could not be extracted")
    if resolved_type == "rfc" and title is None:
        warnings.append("RFC title could not be extracted")

    return Document(
        document_id=document_id,
        title=title,
        source_path=path.as_posix(),
        source_format="text",
        document_type=resolved_type,
        source_sha256=file_sha256(path),
        sections=sections,
        rfc_number=rfc_number,
        publication_date=_extract_publication_date(lines) if resolved_type == "rfc" else None,
        category=_extract_rfc_category(lines) if resolved_type == "rfc" else None,
        reference_definitions=definitions,
        warnings=warnings,
    )


def parse_internet_draft_text(path: Path) -> Document:
    return parse_text_document(path, "internet-draft")


def parse_rfc_text(path: Path) -> Document:
    return parse_text_document(path, "rfc")


def _extract_draft_title(lines: list[tuple[int, str]], document_id: str) -> str | None:
    stem_index = next(
        (index for index, (_, line) in enumerate(lines) if document_id in line), None
    )
    if stem_index is None:
        return None
    title_lines: list[str] = []
    for _, line in reversed(lines[max(0, stem_index - 4) : stem_index]):
        stripped = line.strip()
        if not stripped:
            if title_lines:
                break
            continue
        title_lines.append(stripped)
    return " ".join(reversed(title_lines)) or None


def _extract_rfc_title(lines: list[tuple[int, str]]) -> str | None:
    """Return the final blank-line-delimited block before Abstract/Status."""
    header_lines: list[str] = []
    for _, line in lines[:160]:
        if TITLE_BOUNDARY_RE.match(line):
            break
        header_lines.append(line)

    blocks: list[list[str]] = []
    current: list[str] = []
    for line in header_lines:
        if line.strip():
            current.append(line.strip())
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    if not blocks:
        return None
    return " ".join(blocks[-1])


def _extract_rfc_number(lines: list[tuple[int, str]], path: Path) -> int | None:
    filename_match = re.fullmatch(r"RFC(?P<number>\d+)", path.stem, re.IGNORECASE)
    if filename_match:
        return int(filename_match.group("number"))
    for _, line in lines[:80]:
        match = RFC_NUMBER_RE.search(line)
        if match:
            return int(match.group(1))
    return None


def _extract_publication_date(lines: list[tuple[int, str]]) -> str | None:
    for _, line in lines[:80]:
        match = PUBLICATION_DATE_RE.search(line)
        if match:
            month = MONTH_NUMBERS[match.group(1).casefold()]
            return f"{match.group(2)}-{month:02d}-01"
    return None


def _extract_rfc_category(lines: list[tuple[int, str]]) -> str | None:
    for _, line in lines[:80]:
        match = RFC_CATEGORY_RE.match(line)
        if match:
            # RFC title pages are two-column layouts. In rendered text, the
            # right-hand author or date can share the Category line.
            return re.split(r"\s{2,}", match.group(1).strip(), maxsplit=1)[0]
    return None
