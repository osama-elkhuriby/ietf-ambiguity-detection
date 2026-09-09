"""Explainable correspondence scores and conservative surface change signatures."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from rapidfuzz.fuzz import ratio

from alignment.clause_segmentation.classifier import extract_bcp14_keywords

from .config import Config
from .models import Clause

TOKEN = re.compile(r"0x[0-9A-Fa-f]+|[A-Za-z_][\w.-]*|\d+(?:\.\d+)*|[^\w\s]")
STOP = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "be",
    "to",
    "of",
    "and",
    "or",
    "in",
    "on",
    "for",
    "with",
    "this",
    "that",
    "it",
    "as",
    "by",
    "from",
    "at",
    "can",
    "will",
    "then",
}
ACTORS = {
    "client",
    "server",
    "sender",
    "receiver",
    "endpoint",
    "peer",
    "implementation",
}


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text)


def content_tokens(text: str) -> set[str]:
    return {
        t.casefold() for t in tokens(text) if len(t) > 1 and t.casefold() not in STOP
    }


def signature(text: str) -> dict[str, list[str]]:
    ts = tokens(text)
    lower = [t.casefold() for t in ts]
    return {
        "actor": sorted(set(lower) & ACTORS),
        "modal": extract_bcp14_keywords(text),
        "negation": sorted(set(lower) & {"not", "never", "neither", "without"}),
        "number": sorted(
            set(re.findall(r"\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)*)\b", text))
        ),
        "identifier": sorted(
            {t for t in ts if "_" in t or re.search(r"[a-z][A-Z]", t)}
        ),
        "condition": sorted(
            set(lower)
            & {"if", "unless", "when", "until", "except", "provided", "otherwise"}
        ),
    }


def change_summary(a: str, b: str) -> dict:
    left, right = signature(a), signature(b)
    return {
        key: {
            "removed": sorted(set(left[key]) - set(right[key])),
            "added": sorted(set(right[key]) - set(left[key])),
        }
        for key in left
        if left[key] != right[key]
    }


def lexical(a: str, b: str) -> float:
    x, y = (
        Counter(t.casefold() for t in tokens(a)),
        Counter(t.casefold() for t in tokens(b)),
    )
    overlap = 2 * sum((x & y).values()) / max(1, sum(x.values()) + sum(y.values()))
    return 0.6 * overlap + 0.4 * ratio(a, b) / 100


def similarity_set(a: list[str], b: list[str]) -> float:
    return len(set(a) & set(b)) / len(set(a) | set(b)) if a or b else 1.0


def block_compatibility(a: Clause, b: Clause) -> float:
    if a.block_type == b.block_type:
        return 1.0
    if {a.block_type, b.block_type} <= {"paragraph", "list_item", "definition"}:
        return 0.8
    return 0.0


def score(
    a: Clause, b: Clause, position: float, semantic: float | None, config: Config
) -> dict[str, float | None]:
    sa, sb = signature(a.normalized_text), signature(b.normalized_text)
    result: dict[str, float | None] = {
        "semantic": semantic,
        "lexical": lexical(a.normalized_text, b.normalized_text),
        "structural": ratio(a.section_heading.casefold(), b.section_heading.casefold())
        / 100,
        "position": position,
        "block_compatibility": block_compatibility(a, b),
        **{f"{key}_consistency": similarity_set(sa[key], sb[key]) for key in sa},
        "reference_consistency": similarity_set(
            a.reference_targets, b.reference_targets
        ),
    }
    protected = float(
        np.mean([v for k, v in result.items() if k.endswith("_consistency")])
    )
    weighted = [
        (result["lexical"], config.lexical_weight),
        (result["structural"], config.structural_weight),
        (position, config.position_weight),
        (result["block_compatibility"], config.block_weight),
        (protected, config.protected_weight),
    ]
    if semantic is not None:
        weighted.append((semantic, config.semantic_weight))
    result["combined"] = sum(float(v) * w for v, w in weighted) / sum(
        w for _, w in weighted
    )
    return result
