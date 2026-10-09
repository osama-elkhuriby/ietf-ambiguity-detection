"""Deterministic, label-free features for IETF clause text.

The public API is intentionally small:

``extract_clause_features`` handles one clause and
``extract_dataframe_features`` handles a pandas DataFrame.  Feature columns
are fixed by ``FEATURE_COLUMNS`` so training and inference cannot silently
drift when a text or context value is missing.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import pandas as pd

from alignment.clause_segmentation import split_sentences
from alignment.clause_segmentation.classifier import BCP14_RE, BCP14_TERMS

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


def _phrase_pattern(phrases: Iterable[str]) -> re.Pattern[str]:
    alternatives = []
    for phrase in sorted(set(phrases), key=lambda value: (-len(value), value)):
        alternatives.append(re.escape(phrase).replace(r"\ ", r"\s+"))
    return re.compile(
        r"(?<![A-Za-z0-9])(?:" + "|".join(alternatives) + r")(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


VAGUE_QUANTIFIER_RE = _phrase_pattern(
    (
        "a few",
        "a number of",
        "almost all",
        "many",
        "most",
        "several",
        "some",
        "various",
    )
)
VAGUE_QUALIFIER_RE = _phrase_pattern(
    (
        "adequate",
        "appropriate",
        "as appropriate",
        "as needed",
        "considerable",
        "enough",
        "reasonable",
        "relevant",
        "significant",
        "sufficient",
    )
)
VAGUE_FREQUENCY_RE = _phrase_pattern(
    (
        "commonly",
        "frequently",
        "generally",
        "normally",
        "occasionally",
        "often",
        "rarely",
        "sometimes",
        "typically",
        "usually",
    )
)
VAGUE_MODAL_RE = _phrase_pattern(("can", "could", "might", "possibly", "potentially"))

ANAPHORA_RE = _phrase_pattern(
    (
        "above",
        "aforementioned",
        "below",
        "former",
        "it",
        "its",
        "latter",
        "such",
        "that",
        "these",
        "they",
        "this",
        "those",
        "their",
    )
)
REFERENCE_RE = re.compile(
    r"(?:\b(?:Section|Sections|Appendix|Figure|Table)\s+"
    r"[A-Za-z0-9][A-Za-z0-9.()\-]*(?:\s*(?:and|,|-)\s*"
    r"[A-Za-z0-9][A-Za-z0-9.()\-]*)*|"
    r"\bRFC\s*\d{3,5}\b|\[[A-Za-z0-9._-]+\])",
    re.IGNORECASE,
)

CONDITIONAL_RE = _phrase_pattern(
    (
        "as long as",
        "assuming that",
        "if",
        "in case",
        "only if",
        "provided that",
        "unless",
        "when",
        "whenever",
        "where",
        "whether",
    )
)
EXCEPTION_RE = _phrase_pattern(
    (
        "apart from",
        "except",
        "except for",
        "except when",
        "other than",
        "otherwise",
        "unless",
        "with the exception of",
    )
)

TEMPORAL_RE = _phrase_pattern(
    (
        "after",
        "as soon as",
        "before",
        "during",
        "eventually",
        "immediately",
        "later",
        "once",
        "prior to",
        "simultaneously",
        "subsequently",
        "until",
        "while",
    )
)
ORDERING_RE = _phrase_pattern(
    (
        "afterwards",
        "finally",
        "first",
        "following",
        "in order",
        "last",
        "next",
        "previous",
        "second",
        "then",
    )
)

SUBORDINATE_DEPENDENCIES = frozenset({"acl", "advcl", "ccomp", "relcl", "xcomp"})

BCP14_FEATURE_NAMES = {
    term: "normative_" + term.casefold().replace(" ", "_") + "_count"
    for term in BCP14_TERMS
}

BASIC_FEATURE_COLUMNS = (
    "basic_word_count",
    "basic_character_count",
    "basic_sentence_count",
    "basic_average_word_length",
    "basic_average_sentence_words",
    "basic_average_sentence_characters",
)
VAGUE_FEATURE_COLUMNS = (
    "vague_term_count",
    "vague_term_density",
    "vague_term_present",
    "vague_quantifier_count",
    "vague_qualifier_count",
    "vague_frequency_count",
    "vague_modal_count",
)
REFERENCE_FEATURE_COLUMNS = (
    "anaphora_count",
    "anaphora_density",
    "anaphora_present",
    "cross_reference_count",
    "cross_reference_density",
    "cross_reference_present",
)
CONDITION_FEATURE_COLUMNS = (
    "conditional_count",
    "conditional_density",
    "conditional_present",
    "exception_count",
    "exception_density",
    "exception_present",
)
TEMPORAL_FEATURE_COLUMNS = (
    "temporal_count",
    "temporal_density",
    "temporal_present",
    "ordering_count",
    "ordering_density",
    "ordering_present",
)
NORMATIVE_FEATURE_COLUMNS = (
    "normative_bcp14_count",
    "normative_bcp14_unique_count",
    "normative_bcp14_present",
    "normative_obligation_count",
    "normative_recommendation_count",
    "normative_permission_count",
    *(BCP14_FEATURE_NAMES[term] for term in BCP14_TERMS),
)
SYNTACTIC_FEATURE_COLUMNS = (
    "syntax_available",
    "syntax_mean_dependency_distance",
    "syntax_max_dependency_depth",
    "syntax_subordinate_clause_count",
    "syntax_coordinate_conjunction_count",
)
CONTEXT_FEATURE_COLUMNS = (
    "context_previous_present",
    "context_previous_word_count",
    "context_previous_character_count",
    "context_previous_sentence_count",
    "context_next_present",
    "context_next_word_count",
    "context_next_character_count",
    "context_next_sentence_count",
    "context_section_depth",
)
FEATURE_COLUMNS = (
    *BASIC_FEATURE_COLUMNS,
    *VAGUE_FEATURE_COLUMNS,
    *REFERENCE_FEATURE_COLUMNS,
    *CONDITION_FEATURE_COLUMNS,
    *TEMPORAL_FEATURE_COLUMNS,
    *NORMATIVE_FEATURE_COLUMNS,
    *SYNTACTIC_FEATURE_COLUMNS,
    *CONTEXT_FEATURE_COLUMNS,
)


def safe_text(value: Any) -> str:
    """Return normalized text for strings and an empty string for missing values."""
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _word_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def _sentence_texts(text: str) -> list[str]:
    return [span.text.strip() for span in split_sentences(text) if span.text.strip()]


def _density(count: int, word_count: int) -> float:
    return count / word_count if word_count else 0.0


def _count(pattern: re.Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))


def _basic_features(text: str) -> dict[str, float | int]:
    words = _word_tokens(text)
    sentences = _sentence_texts(text)
    return {
        "basic_word_count": len(words),
        "basic_character_count": len(text),
        "basic_sentence_count": len(sentences),
        "basic_average_word_length": (
            sum(len(word) for word in words) / len(words) if words else 0.0
        ),
        "basic_average_sentence_words": (
            sum(len(_word_tokens(sentence)) for sentence in sentences) / len(sentences)
            if sentences
            else 0.0
        ),
        "basic_average_sentence_characters": (
            sum(len(sentence) for sentence in sentences) / len(sentences)
            if sentences
            else 0.0
        ),
    }


def _pattern_group_features(
    text: str,
    word_count: int,
    prefix: str,
    pattern: re.Pattern[str],
) -> dict[str, float | int]:
    count = _count(pattern, text)
    return {
        f"{prefix}_count": count,
        f"{prefix}_density": _density(count, word_count),
        f"{prefix}_present": int(count > 0),
    }


def _vague_features(text: str, word_count: int) -> dict[str, float | int]:
    counts = {
        "vague_quantifier_count": _count(VAGUE_QUANTIFIER_RE, text),
        "vague_qualifier_count": _count(VAGUE_QUALIFIER_RE, text),
        "vague_frequency_count": _count(VAGUE_FREQUENCY_RE, text),
        "vague_modal_count": _count(VAGUE_MODAL_RE, text),
    }
    total = sum(counts.values())
    return {
        "vague_term_count": total,
        "vague_term_density": _density(total, word_count),
        "vague_term_present": int(total > 0),
        **counts,
    }


def _normative_features(text: str) -> dict[str, float | int]:
    keywords = [match.group(0) for match in BCP14_RE.finditer(text)]
    counts = {term: keywords.count(term) for term in BCP14_TERMS}
    obligation = sum(
        counts[term] for term in ("MUST", "MUST NOT", "SHALL", "SHALL NOT", "REQUIRED")
    )
    recommendation = sum(
        counts[term]
        for term in ("SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED")
    )
    permission = counts["MAY"] + counts["OPTIONAL"]
    return {
        "normative_bcp14_count": len(keywords),
        "normative_bcp14_unique_count": len(set(keywords)),
        "normative_bcp14_present": int(bool(keywords)),
        "normative_obligation_count": obligation,
        "normative_recommendation_count": recommendation,
        "normative_permission_count": permission,
        **{BCP14_FEATURE_NAMES[term]: counts[term] for term in BCP14_TERMS},
    }


def _token_depth(token: Any, seen: set[int] | None = None) -> int:
    seen = set() if seen is None else seen
    marker = id(token)
    if marker in seen:
        return 0
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(_token_depth(child, seen | {marker}) for child in children)


def _syntactic_features(
    text: str,
    nlp: Callable[[str], Any] | None,
) -> dict[str, float | int]:
    empty = {
        "syntax_available": 0,
        "syntax_mean_dependency_distance": 0.0,
        "syntax_max_dependency_depth": 0,
        "syntax_subordinate_clause_count": 0,
        "syntax_coordinate_conjunction_count": 0,
    }
    if not text or nlp is None:
        return empty
    doc = nlp(text)
    tokens = [token for token in doc if not getattr(token, "is_space", False)]
    if not tokens or not all(
        hasattr(token, "dep_") and hasattr(token, "head") for token in tokens
    ):
        return empty
    distances = [
        abs(int(token.i) - int(token.head.i))
        for token in tokens
        if token.head is not token
    ]
    roots = [token for token in tokens if token.head is token or token.dep_ == "ROOT"]
    return {
        "syntax_available": 1,
        "syntax_mean_dependency_distance": (
            sum(distances) / len(distances) if distances else 0.0
        ),
        "syntax_max_dependency_depth": max(
            (_token_depth(root) for root in roots), default=0
        ),
        "syntax_subordinate_clause_count": sum(
            token.dep_ in SUBORDINATE_DEPENDENCIES for token in tokens
        ),
        "syntax_coordinate_conjunction_count": sum(
            token.dep_ == "cc" for token in tokens
        ),
    }


def section_depth(section_id: Any) -> int:
    """Return a conservative hierarchy depth for numeric or appendix sections."""
    value = safe_text(section_id)
    if not value:
        return 0
    depths: list[int] = []
    for candidate in value.split("|"):
        normalized = re.sub(r"^appendix\s+", "", candidate.strip(), flags=re.IGNORECASE)
        parts = [part for part in normalized.split(".") if part]
        depths.append(len(parts) if parts else 0)
    return max(depths, default=0)


def _context_features(
    previous_text: str,
    next_text: str,
    section_id: Any,
) -> dict[str, float | int]:
    previous_basic = _basic_features(previous_text)
    next_basic = _basic_features(next_text)
    return {
        "context_previous_present": int(bool(previous_text)),
        "context_previous_word_count": previous_basic["basic_word_count"],
        "context_previous_character_count": previous_basic["basic_character_count"],
        "context_previous_sentence_count": previous_basic["basic_sentence_count"],
        "context_next_present": int(bool(next_text)),
        "context_next_word_count": next_basic["basic_word_count"],
        "context_next_character_count": next_basic["basic_character_count"],
        "context_next_sentence_count": next_basic["basic_sentence_count"],
        "context_section_depth": section_depth(section_id),
    }


def extract_clause_features(
    text: Any,
    *,
    previous_text: Any = "",
    next_text: Any = "",
    section_id: Any = "",
    nlp: Callable[[str], Any] | None = None,
) -> dict[str, float | int]:
    """Extract the stable numerical feature schema for one target clause."""
    target = safe_text(text)
    previous = safe_text(previous_text)
    following = safe_text(next_text)
    basic = _basic_features(target)
    word_count = int(basic["basic_word_count"])

    features: dict[str, float | int] = {
        **basic,
        **_vague_features(target, word_count),
        **_pattern_group_features(target, word_count, "anaphora", ANAPHORA_RE),
        **_pattern_group_features(target, word_count, "cross_reference", REFERENCE_RE),
        **_pattern_group_features(target, word_count, "conditional", CONDITIONAL_RE),
        **_pattern_group_features(target, word_count, "exception", EXCEPTION_RE),
        **_pattern_group_features(target, word_count, "temporal", TEMPORAL_RE),
        **_pattern_group_features(target, word_count, "ordering", ORDERING_RE),
        **_normative_features(target),
        **_syntactic_features(target, nlp),
        **_context_features(previous, following, section_id),
    }
    return {name: features[name] for name in FEATURE_COLUMNS}


def extract_dataframe_features(
    frame: pd.DataFrame,
    *,
    text_column: str = "clause_text",
    previous_column: str = "previous_clause",
    next_column: str = "next_clause",
    section_column: str = "section_ids",
    nlp: Callable[[str], Any] | None = None,
) -> pd.DataFrame:
    """Extract features row-wise without reading any label or metadata column."""
    if text_column not in frame.columns:
        raise ValueError(f"Missing text column: {text_column}")

    def column_value(row: Mapping[str, Any], column: str) -> Any:
        return row[column] if column in frame.columns else ""

    rows = []
    for row in frame.to_dict(orient="records"):
        rows.append(
            extract_clause_features(
                row[text_column],
                previous_text=column_value(row, previous_column),
                next_text=column_value(row, next_column),
                section_id=column_value(row, section_column),
                nlp=nlp,
            )
        )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, index=frame.index)
