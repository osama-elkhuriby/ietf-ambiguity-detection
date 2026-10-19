"""Strict response validation with label-dependent invariants."""

from __future__ import annotations

from typing import Any

from .constants import (
    AMBIGUITY_SUBTYPES,
    CONTEXT_ISSUES,
    FIELD_LIMITS,
    OUTPUT_SCHEMA,
    SEMANTIC_LABELS,
)


def validate_annotation(value: Any, target: str) -> list[str]:
    if not isinstance(value, dict):
        return ["response is not a JSON object"]
    required = set(OUTPUT_SCHEMA["required"])
    fields = set(value)
    errors: list[str] = []
    if required - fields:
        errors.append(f"missing fields: {sorted(required - fields)}")
    if fields - required:
        errors.append(f"unexpected fields: {sorted(fields - required)}")
    if errors:
        return errors
    if any(not isinstance(value[field], str) for field in required):
        return ["every field must be a string"]

    for field, limit in FIELD_LIMITS.items():
        if len(value[field]) > limit:
            errors.append(f"{field} exceeds {limit} characters")

    label = value["label"]
    subtype = value["ambiguity_subtype"]
    issue = value["context_issue"]
    evidence = value["evidence_span"]
    first = value["interpretation_a"].strip()
    second = value["interpretation_b"].strip()
    missing = value["missing_information"].strip()

    if label not in SEMANTIC_LABELS:
        errors.append(f"invalid label: {label!r}")
    if subtype not in AMBIGUITY_SUBTYPES:
        errors.append(f"invalid ambiguity_subtype: {subtype!r}")
    if issue not in CONTEXT_ISSUES:
        errors.append(f"invalid context_issue: {issue!r}")
    if not value["justification"].strip():
        errors.append("justification is empty")

    if label == "AMBIGUOUS":
        if subtype == "none":
            errors.append("AMBIGUOUS requires a non-none subtype")
        if not evidence:
            errors.append("AMBIGUOUS requires evidence_span")
        elif evidence not in target:
            errors.append("evidence_span is not an exact target substring")
        has_distinct_readings = bool(first and second and first.casefold() != second.casefold())
        if not has_distinct_readings and not (subtype == "U-2" and missing):
            errors.append("AMBIGUOUS requires two distinct readings or a U-2 constraint")
    elif label == "CLEAR":
        if subtype != "none" or issue != "none":
            errors.append("CLEAR requires subtype and context_issue none")
        if evidence or first or second or missing:
            errors.append("CLEAR requires empty evidence, interpretations, and missing_information")
    elif label == "INSUFFICIENT_CONTEXT":
        if subtype != "none":
            errors.append("INSUFFICIENT_CONTEXT requires subtype none")
        if issue == "none":
            errors.append("INSUFFICIENT_CONTEXT requires a context_issue")
        if not missing:
            errors.append("INSUFFICIENT_CONTEXT requires missing_information")
        if first or second:
            errors.append("INSUFFICIENT_CONTEXT requires empty interpretations")
        if evidence and evidence not in target:
            errors.append("evidence_span is not an exact target substring")
    return errors
