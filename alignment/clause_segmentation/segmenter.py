"""Conservative clause refinement over canonical section/block documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from alignment.segmentation.models import Block, Document, Section

from .classifier import EXCLUDED_BLOCK_TYPES, classify_clause
from .models import Clause, ClauseDocument

PROTECTED_DOT = "\ue000"
ABBREVIATION_RE = re.compile(
    r"\b(?:e\.g\.|i\.e\.|etc\.|vs\.|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Fig\.|Eq\.|"
    r"Sec\.|No\.|Inc\.|Ltd\.|U\.S\.|a\.m\.|p\.m\.)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"(?:https?://|mailto:)[^\s<>]+", re.IGNORECASE)
DECIMAL_DOT_RE = re.compile(r"(?<=\d)\.(?=\d)")
SENTENCE_BOUNDARY_RE = re.compile(r"[.!?][\"'”’)]*\s+(?=(?:[\"'“‘(\[]*\s*)?[A-Z0-9])")
REVISION_RE = re.compile(r"-(?P<revision>\d{2,3})$")


@dataclass(slots=True)
class TextSpan:
    text: str
    start: int
    end: int


def _protect_match_dots(
    value: str, match: re.Match[str], protect_terminal: bool = True
) -> str:
    start, end = match.span()
    fragment = value[start:end]
    if not protect_terminal and fragment.endswith("."):
        fragment = fragment[:-1].replace(".", PROTECTED_DOT) + "."
    else:
        fragment = fragment.replace(".", PROTECTED_DOT)
    return value[:start] + fragment + value[end:]


def _protected_text(text: str) -> str:
    protected = text
    for pattern, protect_terminal in (
        (URL_RE, False),
        (ABBREVIATION_RE, True),
    ):
        matches = list(pattern.finditer(protected))
        for match in reversed(matches):
            protected = _protect_match_dots(protected, match, protect_terminal)
    return DECIMAL_DOT_RE.sub(PROTECTED_DOT, protected)


def split_sentences(text: str) -> list[TextSpan]:
    """Split prose conservatively while protecting common RFC punctuation."""
    if not text.strip():
        return []
    protected = _protected_text(text)
    spans: list[TextSpan] = []
    start = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(protected):
        end = match.start() + len(match.group(0).rstrip())
        _append_span(spans, text, start, end)
        start = match.end()
    _append_span(spans, text, start, len(text))
    return spans


def _append_span(spans: list[TextSpan], text: str, start: int, end: int) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append(TextSpan(text[start:end], start, end))


def _revision_id(document: Document) -> str | None:
    if document.document_type != "internet-draft":
        return None
    match = REVISION_RE.search(document.document_id)
    return match.group("revision") if match else None


def _reference_targets(block: Block, text: str, segment_count: int) -> list[str]:
    targets: list[str] = []
    folded = text.casefold()
    for reference in block.references:
        candidates = [reference.target, reference.display_text or ""]
        visible = segment_count == 1 or any(
            candidate and candidate.casefold() in folded for candidate in candidates
        )
        if visible and reference.target not in targets:
            targets.append(reference.target)
    return targets


def _block_spans(block: Block, analysis_scope: str) -> list[TextSpan]:
    if block.block_type in EXCLUDED_BLOCK_TYPES or analysis_scope == "references":
        return [TextSpan(block.text, 0, len(block.text))] if block.text else []
    if block.block_type == "list_item":
        return [TextSpan(block.text, 0, len(block.text))] if block.text else []
    return split_sentences(block.text)


def _section_clauses(
    document: Document,
    section: Section,
    revision_id: str | None,
    counter: list[int],
) -> list[Clause]:
    clauses: list[Clause] = []
    for paragraph_index, block in enumerate(section.blocks, start=1):
        spans = _block_spans(block, section.analysis_scope)
        for local_index, span in enumerate(spans, start=1):
            counter[0] += 1
            targets = _reference_targets(block, span.text, len(spans))
            classification_targets = [
                target for target in targets if target != document.document_id
            ]
            decision = classify_clause(
                span.text,
                block.block_type,
                section.title,
                section.analysis_scope,
                classification_targets,
            )
            clauses.append(
                Clause(
                    clause_id=f"{block.block_id}:c{local_index:02d}",
                    index=counter[0],
                    document_id=document.document_id,
                    revision_id=revision_id,
                    block_id=block.block_id,
                    section_id=section.section_id,
                    section_heading=section.title,
                    section_path=list(section.path),
                    paragraph_index=paragraph_index,
                    sentence_or_item_index=local_index,
                    raw_text=span.text,
                    block_type=block.block_type,
                    classification=decision.classification,
                    classification_reason=decision.reason,
                    section_prior=decision.section_prior,
                    source_location=block.source_location,
                    text_start=span.start,
                    text_end=span.end,
                    bcp14_keywords=decision.bcp14_keywords,
                    reference_targets=targets,
                    evidence=decision.evidence,
                    mixed_clause=decision.mixed_clause,
                    manual_review=decision.classification == "MANUAL_REVIEW",
                )
            )
    for child in section.children:
        clauses.extend(_section_clauses(document, child, revision_id, counter))
    return clauses


def segment_document(document: Document) -> ClauseDocument:
    """Refine one parsed document into traceable, classified clause units."""
    revision_id = _revision_id(document)
    counter = [0]
    clauses: list[Clause] = []
    for section in document.sections:
        clauses.extend(_section_clauses(document, section, revision_id, counter))
    warnings = [] if clauses else ["No clauses were emitted"]
    return ClauseDocument(
        document_id=document.document_id,
        revision_id=revision_id,
        source_path=document.source_path,
        source_sha256=document.source_sha256,
        clauses=clauses,
        warnings=warnings,
    )
