"""Explainable evidence-first classification of RFC clauses."""

from __future__ import annotations

import re
from dataclasses import dataclass

from alignment.segmentation.models import BlockType

from .models import ClauseLabel, SectionPrior

BCP14_TERMS = (
    "NOT RECOMMENDED",
    "SHOULD NOT",
    "SHALL NOT",
    "MUST NOT",
    "RECOMMENDED",
    "REQUIRED",
    "OPTIONAL",
    "SHOULD",
    "SHALL",
    "MUST",
    "MAY",
)
BCP14_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    + "|".join(re.escape(term) for term in BCP14_TERMS)
    + r")(?![A-Za-z])"
)
NORMATIVE_LOOKING_RE = re.compile(
    r"\b(?:must|should|shall|may|required|recommended|optional)\b", re.IGNORECASE
)
QUOTE_RE = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’')
REFERENCE_REPORTING_RE = re.compile(
    r"\b(?:according to|defined by|defined in|requires?|specifies?|states?|says?|"
    r"quoted from|as described in)\b",
    re.IGNORECASE,
)
TERMINOLOGY_META_RE = re.compile(
    r"(?:\b(?:key\s+words?|terms?|words?)\b.{0,160}"
    r"\b(?:interpreted|defined|used|described)\b|"
    r"\b(?:interpreted|defined|used|described)\b.{0,160}"
    r"\b(?:key\s+words?|terms?|words?)\b)",
    re.IGNORECASE,
)
MIXED_EVIDENCE_RE = re.compile(
    r"\b(?:because|therefore|however|so that|in order to|for example|note that|"
    r"for interoperability|for compatibility)\b",
    re.IGNORECASE,
)

EXCLUDED_BLOCK_TYPES: set[BlockType] = {"artwork", "sourcecode", "table", "reference"}

CONTEXTUAL_HEADINGS = (
    "abstract",
    "introduction",
    "background",
    "motivation",
    "overview",
    "terminology",
    "conventions",
)
NORMATIVE_HEADINGS = (
    "requirements",
    "protocol operation",
    "processing rules",
    "message format",
    "packet format",
    "frame format",
)
MIXED_HEADINGS = ("security considerations", "iana considerations")


@dataclass(slots=True)
class ClassificationDecision:
    classification: ClauseLabel
    reason: str
    section_prior: SectionPrior
    bcp14_keywords: list[str]
    evidence: list[str]
    mixed_clause: bool = False


def section_prior(heading: str) -> SectionPrior:
    """Return a weak semantic prior derived from a section heading."""
    normalized = re.sub(r"[^a-z0-9]+", " ", heading.casefold()).strip()
    if any(name in normalized for name in MIXED_HEADINGS):
        return "mixed"
    if any(name in normalized for name in CONTEXTUAL_HEADINGS):
        return "contextual"
    if any(name in normalized for name in NORMATIVE_HEADINGS):
        return "normative"
    return "neutral"


def extract_bcp14_keywords(text: str) -> list[str]:
    """Extract intended uppercase RFC 2119/RFC 8174 terms without duplicates."""
    result: list[str] = []
    for match in BCP14_RE.finditer(text):
        keyword = match.group(0)
        if keyword not in result:
            result.append(keyword)
    return result


def _has_quoted_keyword(text: str) -> bool:
    return any(BCP14_RE.search(match.group(0)) for match in QUOTE_RE.finditer(text))


def _has_lowercase_normative_language(text: str) -> bool:
    return any(
        match.group(0) != match.group(0).upper()
        for match in NORMATIVE_LOOKING_RE.finditer(text)
    )


def classify_clause(
    text: str,
    block_type: BlockType,
    section_heading: str,
    analysis_scope: str,
    reference_targets: list[str] | None = None,
) -> ClassificationDecision:
    """Classify one candidate using the report's conservative evidence order."""
    targets = reference_targets or []
    prior = section_prior(section_heading)
    keywords = extract_bcp14_keywords(text)

    if block_type in EXCLUDED_BLOCK_TYPES:
        return ClassificationDecision(
            "EXCLUDED",
            "formal_or_non_prose_block",
            prior,
            keywords,
            [f"block_type:{block_type}"],
        )
    if analysis_scope == "references":
        return ClassificationDecision(
            "EXCLUDED",
            "reference_material",
            prior,
            keywords,
            ["analysis_scope:references"],
        )

    if keywords and TERMINOLOGY_META_RE.search(text):
        return ClassificationDecision(
            "CONTEXTUAL",
            "bcp14_terminology_definition",
            prior,
            keywords,
            ["meta_language_about_normative_terms"],
        )

    if keywords and _has_quoted_keyword(text):
        return ClassificationDecision(
            "EXCLUDED",
            "quoted_normative_text",
            prior,
            keywords,
            ["bcp14_keyword_inside_quotation"],
        )
    if keywords and targets and REFERENCE_REPORTING_RE.search(text):
        return ClassificationDecision(
            "EXCLUDED",
            "referenced_normative_text",
            prior,
            keywords,
            [
                "reports_external_requirement",
                *[f"reference:{target}" for target in targets],
            ],
        )

    if keywords:
        mixed = bool(MIXED_EVIDENCE_RE.search(text))
        evidence = [*[f"bcp14:{keyword}" for keyword in keywords]]
        if mixed:
            evidence.append("explanatory_and_normative_language")
        return ClassificationDecision(
            "NORMATIVE",
            "explicit_bcp14_keyword",
            prior,
            keywords,
            evidence,
            mixed,
        )

    if _has_lowercase_normative_language(text):
        return ClassificationDecision(
            "MANUAL_REVIEW",
            "lowercase_normative_looking_language",
            prior,
            [],
            ["ambiguous_lowercase_modal"],
        )

    if prior == "contextual":
        return ClassificationDecision(
            "CONTEXTUAL",
            "contextual_section_prior",
            prior,
            [],
            [f"section_prior:{prior}"],
        )
    if prior == "normative":
        return ClassificationDecision(
            "MANUAL_REVIEW",
            "implicit_requirement_candidate",
            prior,
            [],
            [f"section_prior:{prior}", "no_explicit_bcp14_keyword"],
        )
    return ClassificationDecision(
        "MANUAL_REVIEW",
        "insufficient_classification_evidence",
        prior,
        [],
        [f"section_prior:{prior}"],
    )
