"""Aggregation policies that keep semantic and operational states distinct."""

from __future__ import annotations

from collections import Counter
from typing import Any


def conservative_aggregate(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [item for item in judgments if item.get("operational_status") == "VALID"]
    labels = [item.get("semantic_label") for item in valid]
    counts = Counter(labels)
    binary = [label for label in labels if label in {"AMBIGUOUS", "CLEAR"}]
    base = {
        "valid_semantic_count": len(valid),
        "positive_count": counts["AMBIGUOUS"],
        "negative_count": counts["CLEAR"],
        "insufficient_context_count": counts["INSUFFICIENT_CONTEXT"],
        "failed_count": len(judgments) - len(valid),
        "binary_denominator": len(binary),
        "positive_vote_fraction": (
            counts["AMBIGUOUS"] / len(binary) if binary else None
        ),
        "semantic_label": None,
    }
    if len(valid) < 3:
        return {**base, "aggregation_status": "REVIEW_REQUIRED_INCOMPLETE_PANEL"}
    if counts["INSUFFICIENT_CONTEXT"]:
        return {**base, "aggregation_status": "REVIEW_REQUIRED"}
    if len(set(binary)) == 1:
        return {
            **base,
            "aggregation_status": "UNANIMOUS",
            "semantic_label": binary[0],
        }
    winner = counts.most_common(1)[0][0]
    return {
        **base,
        "aggregation_status": "MAJORITY_DISPUTED",
        "semantic_label": winner,
    }


def v1_compatible_comparison(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    """Comparison view only: IC abstains; operational failures never vote."""
    valid = [item for item in judgments if item.get("operational_status") == "VALID"]
    labels = [item.get("semantic_label") for item in valid]
    counts = Counter(labels)
    if counts["AMBIGUOUS"] >= 2:
        label = "yes"
    elif counts["CLEAR"] >= 2:
        label = "no"
    else:
        label = "uncertain"
    return {
        "policy": "v1-compatible-comparison-v2",
        "label": label,
        "yes": counts["AMBIGUOUS"],
        "no": counts["CLEAR"],
        "uncertain": counts["INSUFFICIENT_CONTEXT"],
        "failed": len(judgments) - len(valid),
    }
