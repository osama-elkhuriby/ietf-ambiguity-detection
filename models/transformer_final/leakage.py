"""Label-blind, Dataset-A-specific leakage audits and fold exclusions."""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import pandas as pd

from .common import sha256_text

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")
BCP14_RE = re.compile(
    r"\b(?:MUST(?:\s+NOT)?|SHOULD(?:\s+NOT)?|SHALL(?:\s+NOT)?|MAY|REQUIRED|"
    r"RECOMMENDED|NOT\s+RECOMMENDED|NOT\s+REQUIRED|OPTIONAL)\b",
    flags=re.IGNORECASE,
)
CONTEXT_FIELDS = ("section_heading", "previous_clause", "next_clause")


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())


def token_ngrams(value: Any, size: int = 3) -> frozenset[tuple[str, ...]]:
    tokens = TOKEN_RE.findall(normalize_text(value))
    if not tokens:
        return frozenset()
    width = min(size, len(tokens))
    return frozenset(
        tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)
    )


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def lineage_atoms(value: Any) -> tuple[str, ...]:
    return tuple(
        sorted({part.strip() for part in str(value or "").split("|") if part.strip()})
    )


def _pair_payload(frame: pd.DataFrame, left: int, right: int) -> dict[str, Any]:
    a, b = frame.iloc[left], frame.iloc[right]
    return {
        "left_record_id": str(a.record_id),
        "right_record_id": str(b.record_id),
        "left_family": str(a.family_id),
        "right_family": str(b.family_id),
        "left_fold": int(a.fold),
        "right_fold": int(b.fold),
    }


def _groups(values: Iterable[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        if value:
            result[value].append(index)
    return result


def audit_leakage(frame: pd.DataFrame, *, near_threshold: float) -> dict[str, Any]:
    """Audit targets and canonical contexts without consulting any labels."""
    if not 0 < near_threshold <= 1:
        raise ValueError("near_threshold must be in (0, 1]")
    required = {
        "record_id",
        "family_id",
        "document_id",
        "revision",
        "lineage_ids",
        "target_clause",
        "section_heading",
        "previous_clause",
        "next_clause",
        "fold",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Leakage frame is missing columns: {sorted(missing)}")

    targets = [normalize_text(value) for value in frame["target_clause"]]
    grams = [token_ngrams(value) for value in frame["target_clause"]]
    fold_values = frame["fold"].astype(int).to_numpy()
    exact: list[dict[str, Any]] = []
    for indices in _groups(targets).values():
        for left, right in itertools.combinations(indices, 2):
            item = _pair_payload(frame, left, right)
            item["cross_fold"] = item["left_fold"] != item["right_fold"]
            item["normalized_text_sha256"] = sha256_text(targets[left])
            exact.append(item)

    near: list[dict[str, Any]] = []
    for left, right in itertools.combinations(range(len(frame)), 2):
        if fold_values[left] == fold_values[right]:
            continue
        if targets[left] == targets[right] or not grams[left] or not grams[right]:
            continue
        upper = min(len(grams[left]), len(grams[right])) / max(
            len(grams[left]), len(grams[right])
        )
        if upper < near_threshold:
            continue
        score = jaccard(grams[left], grams[right])
        if score >= near_threshold:
            item = _pair_payload(frame, left, right)
            item["token_3gram_jaccard"] = round(score, 8)
            near.append(item)

    lineage_index: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(frame["lineage_ids"]):
        for atom in lineage_atoms(value):
            lineage_index[atom].append(index)
    lineage: list[dict[str, Any]] = []
    seen_lineage_pairs: set[tuple[int, int]] = set()
    for atom, indices in lineage_index.items():
        for left, right in itertools.combinations(indices, 2):
            if fold_values[left] == fold_values[right]:
                continue
            key = (left, right)
            if key in seen_lineage_pairs:
                continue
            seen_lineage_pairs.add(key)
            item = _pair_payload(frame, left, right)
            item["lineage_id_sha256"] = sha256_text(atom)
            lineage.append(item)

    skeletons = [
        normalize_text(BCP14_RE.sub("<bcp14>", str(value)))
        if BCP14_RE.search(str(value))
        else ""
        for value in frame["target_clause"]
    ]
    bcp14: list[dict[str, Any]] = []
    for skeleton, indices in _groups(skeletons).items():
        for left, right in itertools.combinations(indices, 2):
            if fold_values[left] == fold_values[right]:
                continue
            item = _pair_payload(frame, left, right)
            item["skeleton_sha256"] = sha256_text(skeleton)
            bcp14.append(item)

    target_index = _groups(targets)
    context_overlap: list[dict[str, Any]] = []
    for context_index, row in frame.iterrows():
        for field in CONTEXT_FIELDS:
            normalized = normalize_text(row[field])
            for target_index_value in target_index.get(normalized, []):
                if target_index_value == context_index:
                    continue
                target_row = frame.iloc[target_index_value]
                if int(target_row.fold) == int(row.fold):
                    continue
                context_overlap.append(
                    {
                        "context_record_id": str(row.record_id),
                        "context_family": str(row.family_id),
                        "context_fold": int(row.fold),
                        "context_field": field,
                        "target_record_id": str(target_row.record_id),
                        "target_family": str(target_row.family_id),
                        "target_fold": int(target_row.fold),
                        "normalized_text_sha256": sha256_text(normalized),
                    }
                )

    document_overlap: list[dict[str, Any]] = []
    for document, indices in _groups(frame["document_id"].astype(str)).items():
        folds = sorted({int(frame.iloc[index].fold) for index in indices})
        if len(folds) > 1:
            document_overlap.append({"document_id": document, "folds": folds})
    family_folds = frame.groupby("family_id")["fold"].nunique()
    family_overlap = sorted(family_folds[family_folds > 1].index.astype(str))

    for values in (exact, near, lineage, bcp14):
        values.sort(key=lambda item: (item["left_record_id"], item["right_record_id"]))
    context_overlap.sort(
        key=lambda item: (
            item["context_record_id"],
            item["context_field"],
            item["target_record_id"],
        )
    )
    blocking = bool(document_overlap or family_overlap)
    return {
        "schema_version": 1,
        "label_blind": True,
        "normalization": "Unicode NFC, case-folded, whitespace-collapsed",
        "near_duplicate_method": "word 3-gram Jaccard; shorter targets use all tokens",
        "near_duplicate_threshold": near_threshold,
        "exact_target_duplicates": exact,
        "near_cross_fold_targets": near,
        "cross_fold_lineage_conflicts": lineage,
        "cross_fold_bcp14_skeleton_conflicts": bcp14,
        "cross_fold_target_context_overlaps": context_overlap,
        "cross_fold_document_overlaps": document_overlap,
        "families_assigned_to_multiple_folds": family_overlap,
        "counts": {
            "exact_target_duplicates_all": len(exact),
            "exact_target_duplicates_cross_fold": sum(
                bool(item["cross_fold"]) for item in exact
            ),
            "near_cross_fold_targets": len(near),
            "cross_fold_lineage_conflicts": len(lineage),
            "cross_fold_bcp14_skeleton_conflicts": len(bcp14),
            "cross_fold_target_context_overlaps": len(context_overlap),
            "targets_with_bcp14_keyword": sum(
                bool(BCP14_RE.search(str(value))) for value in frame["target_clause"]
            ),
            "cross_fold_document_overlaps": len(document_overlap),
            "families_assigned_to_multiple_folds": len(family_overlap),
        },
        "training_blocked": blocking,
    }


def build_fold_manifests(
    frame: pd.DataFrame, audit: dict[str, Any]
) -> list[dict[str, Any]]:
    """Create label-independent fitting exclusions while retaining every scoring row."""
    folds = sorted(frame["fold"].astype(int).unique())
    result: list[dict[str, Any]] = []
    pair_sources = (
        ("exact_target_duplicates", "exact_target_of_evaluation"),
        ("near_cross_fold_targets", "near_target_of_evaluation"),
        ("cross_fold_lineage_conflicts", "shared_lineage_with_evaluation"),
        ("cross_fold_bcp14_skeleton_conflicts", "bcp14_skeleton_of_evaluation"),
    )
    for fold_value in folds:
        fold = int(fold_value)
        evaluation = frame.loc[frame["fold"].astype(int) == fold]
        fitting = frame.loc[frame["fold"].astype(int) != fold]
        excluded: dict[str, set[str]] = defaultdict(set)
        for source, reason in pair_sources:
            for pair in audit[source]:
                left_fold, right_fold = int(pair["left_fold"]), int(pair["right_fold"])
                if left_fold == fold and right_fold != fold:
                    excluded[str(pair["right_record_id"])].add(
                        f"{reason}:{pair['left_record_id']}"
                    )
                if right_fold == fold and left_fold != fold:
                    excluded[str(pair["left_record_id"])].add(
                        f"{reason}:{pair['right_record_id']}"
                    )

        sanitizations: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for item in audit["cross_fold_target_context_overlaps"]:
            context_fold, target_fold = (
                int(item["context_fold"]),
                int(item["target_fold"]),
            )
            if target_fold == fold:
                key = (
                    str(item["context_record_id"]),
                    str(item["context_field"]),
                    "fitting",
                )
                sanitizations[key].add(
                    f"matches_evaluation_target:{item['target_record_id']}"
                )
            if context_fold == fold:
                key = (
                    str(item["context_record_id"]),
                    str(item["context_field"]),
                    "evaluation",
                )
                sanitizations[key].add(
                    f"matches_fitting_target:{item['target_record_id']}"
                )

        fitting_ids = set(fitting["record_id"].astype(str))
        unknown = sorted(set(excluded) - fitting_ids)
        if unknown:
            raise RuntimeError(
                f"Fold {fold} exclusions are not fitting records: {unknown}"
            )
        kept = fitting.loc[~fitting["record_id"].astype(str).isin(excluded)]
        kept_fitting_ids = set(kept["record_id"].astype(str))
        effective_sanitizations = {
            key: reasons
            for key, reasons in sanitizations.items()
            if key[2] != "fitting" or key[0] in kept_fitting_ids
        }
        manifest = {
            "schema_version": 1,
            "fold": fold,
            "held_out_families": sorted(evaluation["family_id"].astype(str).unique()),
            "fitting_families": sorted(fitting["family_id"].astype(str).unique()),
            "candidate_fitting_record_ids": sorted(fitting_ids),
            "fitting_record_ids": sorted(kept["record_id"].astype(str)),
            "scoring_record_ids": sorted(evaluation["record_id"].astype(str)),
            "fitting_exclusions": [
                {"record_id": record_id, "reasons": sorted(reasons)}
                for record_id, reasons in sorted(excluded.items())
            ],
            "context_sanitizations": [
                {
                    "record_id": key[0],
                    "field": key[1],
                    "role": key[2],
                    "reasons": sorted(reasons),
                }
                for key, reasons in sorted(effective_sanitizations.items())
            ],
            "counts": {
                "candidate_fitting_rows": len(fitting),
                "excluded_fitting_rows": len(excluded),
                "fitting_rows": len(kept),
                "fitting_positive_labels": int(kept["label"].sum()),
                "held_out_rows": len(evaluation),
                "held_out_positive_labels": int(evaluation["label"].sum()),
                "scoring_rows": len(evaluation),
                "context_fields_sanitized": len(effective_sanitizations),
            },
            "selection_policy": {
                "epochs": "fixed_by_configuration",
                "checkpoint_selection": "none_final_epoch_only",
                "outer_labels_used_for_selection": False,
                "threshold_selected_on_outer_predictions": False,
            },
        }
        result.append(manifest)
    return result


def apply_context_sanitization(
    frame: pd.DataFrame, manifest: dict[str, Any], *, role: str
) -> pd.DataFrame:
    if role not in {"fitting", "evaluation"}:
        raise ValueError("role must be fitting or evaluation")
    result = frame.copy()
    index = {
        str(record_id): position
        for position, record_id in enumerate(result["record_id"].astype(str))
    }
    for item in manifest.get("context_sanitizations", []):
        if item["role"] != role:
            continue
        if item["field"] not in CONTEXT_FIELDS:
            raise RuntimeError(f"Unsafe context field in manifest: {item['field']}")
        if item["record_id"] not in index:
            raise RuntimeError(
                f"Context policy references a missing {role} record: {item['record_id']}"
            )
        result.at[result.index[index[item["record_id"]]], item["field"]] = ""
    return result
