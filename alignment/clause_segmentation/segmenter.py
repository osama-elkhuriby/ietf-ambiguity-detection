"""Conservative clause refinement over canonical section/block documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from alignment.segmentation.models import Block, Document, Section

from .classifier import (
    EXCLUDED_BLOCK_TYPES,
    classify_clause,
    extract_bcp14_keywords,
)
from .models import Clause, ClauseDocument


# ============================================================
# Sentence segmentation
# ============================================================

PROTECTED_DOT = "\ue000"

ABBREVIATION_RE = re.compile(
    r"\b(?:e\.g\.|i\.e\.|etc\.|vs\.|Dr\.|Mr\.|Mrs\.|Ms\.|Prof\.|Fig\.|Eq\.|"
    r"Sec\.|No\.|Inc\.|Ltd\.|U\.S\.|a\.m\.|p\.m\.)",
    re.IGNORECASE,
)

URL_RE = re.compile(
    r"(?:https?://|mailto:)[^\s<>]+",
    re.IGNORECASE,
)

DECIMAL_DOT_RE = re.compile(
    r"(?<=\d)\.(?=\d)"
)

SENTENCE_BOUNDARY_RE = re.compile(
    r"[.!?][\"'”’)]*\s+"
    r"(?=(?:[\"'“‘]\s*[A-Za-z0-9]|[(\[]\s*[A-Z0-9]|[A-Z0-9]))"
)

REVISION_RE = re.compile(
    r"-(?P<revision>\d{2,3})$"
)

LIST_PREFIX_RE = re.compile(
    r"^\s*(?:[-*o]|\d+[.)]|[A-Za-z][.)])\s+"
)

CONDITION_LEAD_RE = re.compile(
    r"^(?:if|when|whenever|unless|until|while|once|"
    r"provided(?:\s+that)?|where)\b",
    re.IGNORECASE,
)


# ============================================================
# Formal protocol syntax detection
# ============================================================

FORMAL_SYNTAX_BLOCK_TYPE = "formal_syntax"


# Example:
#
# PskIdentity identities<7..2^16-1>;
#
TLS_VECTOR_DECL_RE = re.compile(
    r"""
    ^\s*
    [A-Za-z_][A-Za-z0-9_.-]*          # type
    \s+
    [A-Za-z_][A-Za-z0-9_.-]*          # field
    \s*
    <
        [^<>\n]+
    >
    \s*
    $
    """,
    re.VERBOSE,
)


# Example:
#
# uint16 length
# opaque data<0..2^16-1>
#
FIELD_DECL_RE = re.compile(
    r"""
    ^\s*
    [A-Za-z_][A-Za-z0-9_.-]*
    \s+
    [A-Za-z_][A-Za-z0-9_.-]*
    (?:
        \s*<[^<>\n]+>
        |
        \s*\[[^\[\]\n]+\]
    )?
    \s*
    $
    """,
    re.VERBOSE,
)


# Example:
#
# 7..2^16-1
# 0..255
#
RANGE_NOTATION_RE = re.compile(
    r"""
    \b
    (?:\d+|2\^\d+)
    \s*\.\.\s*
    (?:\d+|2\^\d+(?:\s*-\s*\d+)?)
    \b
    """,
    re.VERBOSE,
)


STRUCT_START_RE = re.compile(
    r"""
    ^\s*
    (?:
        struct
        |
        enum
        |
        select
        |
        union
    )
    \b
    .*?
    \{
    """,
    re.IGNORECASE | re.VERBOSE,
)


STRUCT_END_RE = re.compile(
    r"""
    ^\s*
    \}
    \s*
    [A-Za-z_][A-Za-z0-9_.-]*
    \s*
    ;?
    \s*$
    """,
    re.VERBOSE,
)


# Useful for things such as:
#
# opaque Random[32];
#
ARRAY_DECL_RE = re.compile(
    r"""
    ^\s*
    [A-Za-z_][A-Za-z0-9_.-]*
    \s+
    [A-Za-z_][A-Za-z0-9_.-]*
    \s*
    \[
        [^\[\]\n]+
    \]
    \s*
    $
    """,
    re.VERBOSE,
)


# Very conservative ABNF-like assignment detector.
#
# token = 1*DIGIT
#
ABNF_ASSIGNMENT_RE = re.compile(
    r"""
    ^\s*
    [A-Za-z][A-Za-z0-9-]*
    \s*=\s*
    (?:
        [0-9*%"/(\[]
        |
        [A-Za-z][A-Za-z0-9-]*\s*/\s*
    )
    """,
    re.VERBOSE,
)


@dataclass(slots=True)
class TextSpan:
    text: str
    start: int
    end: int


@dataclass(slots=True)
class FormalSyntaxDetection:
    detected: bool
    subtype: str | None
    score: int
    evidence: list[str]


# ============================================================
# Formal syntax helpers
# ============================================================

def _statement_fragments(text: str) -> list[str]:
    """
    Return semicolon-separated fragments while preserving only
    meaningful content.

    Example:

        PskIdentity identities<...>;
        PskBinderEntry binders<...>;

    becomes two declaration candidates.
    """

    fragments: list[str] = []

    for part in text.split(";"):
        cleaned = part.strip()

        if cleaned:
            fragments.append(cleaned)

    return fragments


def _syntax_punctuation_ratio(text: str) -> float:
    """
    Estimate how syntax-heavy a piece of text is.

    This is only a supporting signal; it must not independently
    classify natural-language prose as formal syntax.
    """

    compact = "".join(
        character
        for character in text
        if not character.isspace()
    )

    if not compact:
        return 0.0

    syntax_characters = set(
        "{}[]<>;=|^*:"
    )

    syntax_count = sum(
        1
        for character in compact
        if character in syntax_characters
    )

    return syntax_count / len(compact)


def detect_formal_syntax(
    text: str,
) -> FormalSyntaxDetection:
    """
    Conservatively identify non-prose protocol grammar or structure syntax.

    Detection deliberately requires multiple structural signals for generic
    cases so that normal prose containing punctuation is not excluded.
    """

    stripped = text.strip()

    if not stripped:
        return FormalSyntaxDetection(
            detected=False,
            subtype=None,
            score=0,
            evidence=[],
        )

    score = 0
    evidence: list[str] = []
    subtype: str | None = None

    lines = [
        line.strip()
        for line in stripped.splitlines()
        if line.strip()
    ]

    fragments = _statement_fragments(
        stripped
    )

    # --------------------------------------------------------
    # Strong signal: TLS-style vectors
    # --------------------------------------------------------

    vector_declarations = sum(
        1
        for fragment in fragments
        if TLS_VECTOR_DECL_RE.fullmatch(fragment)
    )

    if vector_declarations:

        score += 4

        evidence.append(
            f"tls_vector_declarations:{vector_declarations}"
        )

        subtype = "tls_presentation_language"

    # --------------------------------------------------------
    # Strong signal: numeric vector range notation
    # --------------------------------------------------------

    if RANGE_NOTATION_RE.search(stripped):

        score += 3

        evidence.append(
            "range_notation"
        )

        if subtype is None:
            subtype = "formal_protocol_syntax"

    # --------------------------------------------------------
    # Structure start
    # --------------------------------------------------------

    if any(
        STRUCT_START_RE.search(line)
        for line in lines
    ):

        score += 4

        evidence.append(
            "structure_start"
        )

        if subtype is None:
            subtype = "formal_structure"

    # --------------------------------------------------------
    # Structure end
    # --------------------------------------------------------

    if any(
        STRUCT_END_RE.fullmatch(line)
        for line in lines
    ):

        score += 3

        evidence.append(
            "structure_end"
        )

        if subtype is None:
            subtype = "formal_structure"

    # --------------------------------------------------------
    # Field declarations
    # --------------------------------------------------------

    field_declarations = sum(
        1
        for fragment in fragments
        if FIELD_DECL_RE.fullmatch(fragment)
    )

    if field_declarations >= 2:

        score += 3

        evidence.append(
            f"field_declarations:{field_declarations}"
        )

        if subtype is None:
            subtype = "formal_structure"

    elif (
        field_declarations == 1
        and (
            RANGE_NOTATION_RE.search(stripped)
            or "<" in stripped
            or "[" in stripped
        )
    ):

        score += 2

        evidence.append(
            "single_typed_declaration"
        )

    # --------------------------------------------------------
    # Array declaration
    # --------------------------------------------------------

    if any(
        ARRAY_DECL_RE.fullmatch(fragment)
        for fragment in fragments
    ):

        score += 2

        evidence.append(
            "array_declaration"
        )

        if subtype is None:
            subtype = "formal_structure"

    # --------------------------------------------------------
    # ABNF
    # --------------------------------------------------------

    abnf_lines = sum(
        1
        for line in lines
        if ABNF_ASSIGNMENT_RE.match(line)
    )

    if abnf_lines >= 1:

        score += 4

        evidence.append(
            f"abnf_assignments:{abnf_lines}"
        )

        subtype = "abnf"

    # --------------------------------------------------------
    # Semicolon density
    # --------------------------------------------------------

    semicolon_count = stripped.count(";")

    if semicolon_count >= 2:

        score += 2

        evidence.append(
            f"semicolons:{semicolon_count}"
        )

    # --------------------------------------------------------
    # Syntax punctuation density
    # --------------------------------------------------------

    punctuation_ratio = _syntax_punctuation_ratio(
        stripped
    )

    if punctuation_ratio >= 0.12:

        score += 1

        evidence.append(
            f"syntax_punctuation_ratio:{punctuation_ratio:.3f}"
        )

    # --------------------------------------------------------
    # Final conservative threshold
    # --------------------------------------------------------

    detected = score >= 4

    if not detected:
        subtype = None

    return FormalSyntaxDetection(
        detected=detected,
        subtype=subtype,
        score=score,
        evidence=evidence,
    )


def _effective_block_type(
    block: Block,
) -> tuple[str, FormalSyntaxDetection]:
    """
    Return the block type used for classification.

    Existing parser classifications always win. Formal-syntax
    detection is used only as a fallback for blocks that otherwise
    look like prose/list content.
    """

    detection = FormalSyntaxDetection(
        detected=False,
        subtype=None,
        score=0,
        evidence=[],
    )

    # Parser already knows this is excluded.
    if block.block_type in EXCLUDED_BLOCK_TYPES:

        return block.block_type, detection

    detection = detect_formal_syntax(
        block.text
    )

    if detection.detected:

        return FORMAL_SYNTAX_BLOCK_TYPE, detection

    return block.block_type, detection


# ============================================================
# Sentence protection
# ============================================================

def _protect_match_dots(
    value: str,
    match: re.Match[str],
    protect_terminal: bool = True,
) -> str:

    start, end = match.span()

    fragment = value[start:end]

    if (
        not protect_terminal
        and fragment.endswith(".")
    ):

        fragment = (
            fragment[:-1].replace(
                ".",
                PROTECTED_DOT,
            )
            + "."
        )

    else:

        fragment = fragment.replace(
            ".",
            PROTECTED_DOT,
        )

    return (
        value[:start]
        + fragment
        + value[end:]
    )


def _protected_text(
    text: str,
) -> str:

    protected = text

    for pattern, protect_terminal in (
        (URL_RE, False),
        (ABBREVIATION_RE, True),
    ):

        matches = list(
            pattern.finditer(protected)
        )

        for match in reversed(matches):

            protected = _protect_match_dots(
                protected,
                match,
                protect_terminal,
            )

    return DECIMAL_DOT_RE.sub(
        PROTECTED_DOT,
        protected,
    )


# ============================================================
# Sentence splitting
# ============================================================

def split_sentences(
    text: str,
) -> list[TextSpan]:
    """
    Split prose conservatively while protecting common RFC punctuation.
    """

    if not text.strip():
        return []

    protected = _protected_text(
        text
    )

    spans: list[TextSpan] = []

    start = 0

    for match in SENTENCE_BOUNDARY_RE.finditer(
        protected
    ):

        end = (
            match.start()
            + len(
                match.group(0).rstrip()
            )
        )

        _append_span(
            spans,
            text,
            start,
            end,
        )

        start = match.end()

    _append_span(
        spans,
        text,
        start,
        len(text),
    )

    return _merge_dependent_conditions(
        text,
        spans,
    )


def _merge_dependent_conditions(
    text: str,
    spans: list[TextSpan],
) -> list[TextSpan]:
    """
    Keep a condition sentence with an immediately following explicit
    requirement.
    """

    merged: list[TextSpan] = []

    index = 0

    while index < len(spans):

        current = spans[index]

        if (
            index + 1 < len(spans)
            and CONDITION_LEAD_RE.match(
                current.text
            )
            and not extract_bcp14_keywords(
                current.text
            )
            and extract_bcp14_keywords(
                spans[index + 1].text
            )
        ):

            following = spans[
                index + 1
            ]

            merged.append(
                TextSpan(
                    text[
                        current.start:
                        following.end
                    ],
                    current.start,
                    following.end,
                )
            )

            index += 2

            continue

        merged.append(
            current
        )

        index += 1

    return merged


def _append_span(
    spans: list[TextSpan],
    text: str,
    start: int,
    end: int,
) -> None:

    while (
        start < end
        and text[start].isspace()
    ):

        start += 1

    while (
        end > start
        and text[end - 1].isspace()
    ):

        end -= 1

    if start < end:

        spans.append(
            TextSpan(
                text[start:end],
                start,
                end,
            )
        )


# ============================================================
# Document metadata
# ============================================================

def _revision_id(
    document: Document,
) -> str | None:

    if document.document_type != "internet-draft":
        return None

    match = REVISION_RE.search(
        document.document_id
    )

    return (
        match.group("revision")
        if match
        else None
    )


def _reference_targets(
    block: Block,
    text: str,
    segment_count: int,
) -> list[str]:

    targets: list[str] = []

    folded = text.casefold()

    for reference in block.references:

        candidates = [
            reference.target,
            reference.display_text or "",
        ]

        visible = (
            segment_count == 1
            or any(
                candidate
                and candidate.casefold()
                in folded
                for candidate
                in candidates
            )
        )

        if (
            visible
            and reference.target
            not in targets
        ):

            targets.append(
                reference.target
            )

    return targets


# ============================================================
# Block segmentation
# ============================================================

def _block_spans(
    block: Block,
    analysis_scope: str,
) -> list[TextSpan]:
    """
    Produce clause candidate spans from a parser block.

    Formal protocol syntax is intentionally kept intact rather than
    sentence-split.
    """

    effective_block_type, _ = (
        _effective_block_type(block)
    )

    if (
        effective_block_type
        in EXCLUDED_BLOCK_TYPES
        or effective_block_type
        == FORMAL_SYNTAX_BLOCK_TYPE
        or analysis_scope
        in {
            "references",
            "back",
        }
    ):

        return (
            [
                TextSpan(
                    block.text,
                    0,
                    len(block.text),
                )
            ]
            if block.text
            else []
        )

    if block.block_type == "list_item":

        prefix = LIST_PREFIX_RE.match(
            block.text
        )

        payload_start = (
            prefix.end()
            if prefix is not None
            else 0
        )

        payload_spans = split_sentences(
            block.text[
                payload_start:
            ]
        )

        spans: list[TextSpan] = []

        for index, span in enumerate(
            payload_spans
        ):

            start = (
                0
                if index == 0
                else payload_start
                + span.start
            )

            end = (
                payload_start
                + span.end
            )

            spans.append(
                TextSpan(
                    block.text[
                        start:end
                    ],
                    start,
                    end,
                )
            )

        return (
            spans
            or (
                [
                    TextSpan(
                        block.text,
                        0,
                        len(block.text),
                    )
                ]
                if block.text
                else []
            )
        )

    return split_sentences(
        block.text
    )


# ============================================================
# Clause emission
# ============================================================

def _section_clauses(
    document: Document,
    section: Section,
    revision_id: str | None,
    counter: list[int],
) -> list[Clause]:

    clauses: list[Clause] = []

    for paragraph_index, block in enumerate(
        section.blocks,
        start=1,
    ):

        effective_block_type, formal_detection = (
            _effective_block_type(
                block
            )
        )

        spans = _block_spans(
            block,
            section.analysis_scope,
        )

        for local_index, span in enumerate(
            spans,
            start=1,
        ):

            counter[0] += 1

            targets = _reference_targets(
                block,
                span.text,
                len(spans),
            )

            classification_targets = [
                target
                for target in targets
                if target
                != document.document_id
            ]

            decision = classify_clause(
                span.text,
                effective_block_type,
                section.title,
                section.analysis_scope,
                classification_targets,
            )

            evidence = list(
                decision.evidence
            )

            if formal_detection.detected:

                evidence.append(
                    "formal_syntax_detected"
                )

                evidence.append(
                    f"formal_syntax_score:"
                    f"{formal_detection.score}"
                )

                if formal_detection.subtype:

                    evidence.append(
                        f"formal_syntax_subtype:"
                        f"{formal_detection.subtype}"
                    )

                evidence.extend(
                    f"formal_syntax_signal:{item}"
                    for item
                    in formal_detection.evidence
                )

            clauses.append(
                Clause(
                    clause_id=(
                        f"{block.block_id}:"
                        f"c{local_index:02d}"
                    ),
                    index=counter[0],
                    document_id=document.document_id,
                    revision_id=revision_id,
                    block_id=block.block_id,
                    section_id=section.section_id,
                    section_heading=section.title,
                    section_path=list(
                        section.path
                    ),
                    paragraph_index=paragraph_index,
                    sentence_or_item_index=local_index,
                    raw_text=span.text,

                    # Important:
                    # emit the corrected type, not the original parser type.
                    block_type=effective_block_type,

                    classification=decision.classification,
                    classification_reason=decision.reason,
                    section_prior=decision.section_prior,
                    source_location=block.source_location,
                    text_start=span.start,
                    text_end=span.end,
                    bcp14_keywords=decision.bcp14_keywords,
                    reference_targets=targets,
                    evidence=evidence,
                    mixed_clause=decision.mixed_clause,
                    manual_review=(
                        decision.classification
                        == "MANUAL_REVIEW"
                    ),
                )
            )

    for child in section.children:

        clauses.extend(
            _section_clauses(
                document,
                child,
                revision_id,
                counter,
            )
        )

    return clauses


# ============================================================
# Public API
# ============================================================

def segment_document(
    document: Document,
) -> ClauseDocument:
    """
    Refine one parsed document into traceable, classified clause units.
    """

    revision_id = _revision_id(
        document
    )

    counter = [0]

    clauses: list[Clause] = []

    for section in document.sections:

        clauses.extend(
            _section_clauses(
                document,
                section,
                revision_id,
                counter,
            )
        )

    warnings = (
        []
        if clauses
        else [
            "No clauses were emitted"
        ]
    )

    return ClauseDocument(
        document_id=document.document_id,
        revision_id=revision_id,
        source_path=document.source_path,
        source_sha256=document.source_sha256,
        clauses=clauses,
        warnings=warnings,
    )