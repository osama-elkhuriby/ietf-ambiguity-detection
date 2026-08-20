"""Deterministic reference extraction shared by text and XML parsers."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import Reference, ReferenceDefinition
from .normalize import normalize_prose

EXPLICIT_TAG_RE = re.compile(
    r"\bSections?\s+(?P<section>[A-Za-z0-9][A-Za-z0-9.()/-]*)\s+"
    r"(?:of|in)\s+\[(?P<tag>[A-Za-z0-9._-]+)\]",
    re.IGNORECASE,
)
EXPLICIT_RFC_RE = re.compile(
    r"\bSections?\s+(?P<section>[A-Za-z0-9][A-Za-z0-9.()/-]*)\s+"
    r"(?:of|in)\s+RFC\s*(?P<number>\d{3,5})\b",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"\[(?P<tag>[A-Za-z0-9._-]+)\]")
RFC_RE = re.compile(r"\bRFC\s*(?P<number>\d{3,5})\b", re.IGNORECASE)
REFERENCE_DEFINITION_RE = re.compile(r"^\[(?P<tag>[^\]]+)\]\s+(?P<body>.+)$", re.DOTALL)


def _deduplicate(references: Iterable[Reference]) -> list[Reference]:
    seen: set[tuple[str, str | None, str]] = set()
    result: list[Reference] = []
    for reference in references:
        key = (reference.target.casefold(), reference.target_section, reference.kind)
        if key not in seen:
            seen.add(key)
            result.append(reference)
    return result


def extract_text_references(text: str) -> list[Reference]:
    """Extract explicit section references and symbolic/RFC tags from rendered text."""
    references: list[Reference] = []
    explicit_spans: list[tuple[int, int]] = []

    for match in EXPLICIT_TAG_RE.finditer(text):
        references.append(
            Reference(
                target=match.group("tag"),
                target_section=match.group("section").rstrip("."),
                kind="text",
                display_text=match.group(0),
            )
        )
        explicit_spans.append(match.span())

    for match in EXPLICIT_RFC_RE.finditer(text):
        references.append(
            Reference(
                target=f"RFC{match.group('number')}",
                target_section=match.group("section").rstrip("."),
                kind="text",
                display_text=match.group(0),
            )
        )
        explicit_spans.append(match.span())

    def inside_explicit_span(start: int) -> bool:
        return any(span_start <= start < span_end for span_start, span_end in explicit_spans)

    for match in TAG_RE.finditer(text):
        if not inside_explicit_span(match.start()):
            references.append(
                Reference(target=match.group("tag"), kind="text", display_text=match.group(0))
            )

    for match in RFC_RE.finditer(text):
        if not inside_explicit_span(match.start()):
            references.append(
                Reference(
                    target=f"RFC{match.group('number')}",
                    kind="text",
                    display_text=match.group(0),
                )
            )

    return _deduplicate(references)


def parse_text_reference_definition(text: str) -> ReferenceDefinition | None:
    match = REFERENCE_DEFINITION_RE.match(normalize_prose(text))
    if not match:
        return None
    tag = match.group("tag")
    body = match.group("body")
    rfc_match = RFC_RE.search(body)
    url_match = re.search(r"https?://[^\s<>]+", body)
    quoted_titles = re.findall(r'"([^"]+)"', body)
    return ReferenceDefinition(
        tag=tag,
        title=max(quoted_titles, key=len) if quoted_titles else None,
        document_id=f"RFC{rfc_match.group('number')}" if rfc_match else None,
        target_url=url_match.group(0).rstrip(".,") if url_match else None,
    )

