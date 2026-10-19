"""Governed development data, sampling, and auditable same-revision context."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .constants import CONTEXT_BUILDER_VERSION

V1_MODELS = {
    "gemma": "gemma3__27b",
    "gpt_oss": "gpt-oss__20b",
    "qwen": "qwen3__32b",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_allowed_families(folds_path: Path) -> tuple[str, ...]:
    payload = json.loads(folds_path.read_text(encoding="utf-8"))
    families = tuple(sorted(payload["family_to_fold"]))
    if not families:
        raise ValueError("frozen fold mapping contains no development families")
    return families


def assert_governed_family(family_id: str, allowed_families: Iterable[str]) -> None:
    if family_id not in frozenset(allowed_families):
        raise PermissionError(f"family is outside the frozen development allowlist: {family_id}")


def load_development_rows(
    ground_truth_root: Path, allowed_families: Iterable[str]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for family in sorted(allowed_families):
        assert_governed_family(family, allowed_families)
        path = ground_truth_root / family / "review" / "all_candidates_v1" / "reviewer_a.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing governed review source: {path}")
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if row.get("family_id") != family:
                    raise ValueError(f"family mismatch in {path}")
                record_id = row["record_id"]
                if record_id in seen:
                    raise ValueError(f"duplicate development record_id: {record_id}")
                seen.add(record_id)
                rows.append(dict(row))
        sources.append({"path": path.as_posix(), "sha256": file_sha256(path)})
    return sorted(rows, key=lambda row: (row["family_id"], row["record_id"])), sources


def load_v1_votes_for_allowed_rows(
    rows: Iterable[dict[str, str]], response_root: Path
) -> dict[str, dict[str, dict[str, Any]]]:
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        record_id = row["record_id"]
        model_votes: dict[str, dict[str, Any]] = {}
        for alias, directory in V1_MODELS.items():
            path = response_root / directory / f"{record_id}.json"
            if not path.is_file():
                model_votes[alias] = {"status": "missing", "label": None}
                continue
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("record_id") != record_id or saved.get("family_id") != row["family_id"]:
                raise ValueError(f"v1 response identity mismatch: {path}")
            annotation = saved.get("annotation") or {}
            model_votes[alias] = {
                "status": saved.get("status"),
                "label": annotation.get("is_ambiguous"),
            }
        output[record_id] = model_votes
    return output


def v1_pattern(votes: dict[str, dict[str, Any]]) -> str:
    labels = {name: item.get("label") if item.get("status") == "valid" else None for name, item in votes.items()}
    binary = [value for value in labels.values() if value in {"yes", "no"}]
    if len(binary) == 3 and all(value == "yes" for value in binary):
        return "unanimous_positive"
    if len(binary) == 3 and all(value == "no" for value in binary):
        return "unanimous_negative"
    if labels == {"gemma": "yes", "gpt_oss": "no", "qwen": "no"}:
        return "gemma_only_positive"
    if labels == {"gemma": "no", "gpt_oss": "yes", "qwen": "no"}:
        return "gpt_oss_only_positive"
    if len(binary) == 3 and binary.count("yes") == 2:
        return "two_positive_disagreement"
    if len(binary) < 3 or any(value == "uncertain" for value in labels.values()):
        return "consensus_uncertain_or_abstention"
    return "other"


REFERENCE_RE = re.compile(
    r"(?:\bsection\s+[A-Za-z0-9][A-Za-z0-9.\-]*|\[[A-Za-z0-9_.\-]+\])",
    re.IGNORECASE,
)


def looks_structural_or_fragment(row: dict[str, str]) -> bool:
    text = row["clause_text"].strip()
    lower = text.casefold()
    return bool(
        len(text.split()) < 7
        or text.endswith(":")
        or lower.startswith(("and ", "or ", "unless ", "except ", "when ", "where "))
        or lower.endswith((" following", " as follows", " one of"))
        or not re.search(r"[.!?;:]$", text)
    )


TARGET_QUOTAS = (
    ("unanimous_positive", 8),
    ("unanimous_negative", 8),
    ("gemma_only_positive", 8),
    ("gpt_oss_only_positive", 8),
    ("two_positive_disagreement", 8),
    ("consensus_uncertain_or_abstention", 8),
    ("explicit_reference", 6),
    ("segmentation_or_structural_context", 6),
)


def select_review_sample(
    rows: list[dict[str, str]],
    votes: dict[str, dict[str, dict[str, Any]]],
    seed: int,
) -> dict[str, Any]:
    if len(rows) < 100:
        raise ValueError(f"eligible cohort has only {len(rows)} rows; 100 required")
    ordered = sorted(rows, key=lambda row: row["record_id"])
    rng = random.Random(seed)
    random_pool = list(ordered)
    rng.shuffle(random_pool)
    random_rows = random_pool[:40]
    selected_ids = {row["record_id"] for row in random_rows}

    strata: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ordered:
        pattern = v1_pattern(votes[row["record_id"]])
        strata[pattern].append(row)
        if REFERENCE_RE.search(row["clause_text"]):
            strata["explicit_reference"].append(row)
        if looks_structural_or_fragment(row):
            strata["segmentation_or_structural_context"].append(row)
    for values in strata.values():
        rng.shuffle(values)

    targeted: list[dict[str, str]] = []
    requested: dict[str, int] = {}
    achieved: Counter[str] = Counter()
    shortages: dict[str, int] = {}
    for name, quota in TARGET_QUOTAS:
        requested[name] = quota
        choices = [row for row in strata[name] if row["record_id"] not in selected_ids]
        take = choices[:quota]
        for row in take:
            selected_ids.add(row["record_id"])
            targeted.append(row)
            achieved[name] += 1
        if len(take) < quota:
            shortages[name] = quota - len(take)

    # Deterministic top-up: first any unselected member of a requested stratum,
    # then the remaining governed cohort. This never fabricates a vote pattern.
    targeted_union = sorted(
        {
            row["record_id"]: row
            for name, _ in TARGET_QUOTAS
            for row in strata[name]
            if row["record_id"] not in selected_ids
        }.values(),
        key=lambda row: sha256_text(f"{seed}:targeted-topup:{row['record_id']}"),
    )
    general = sorted(
        (row for row in ordered if row["record_id"] not in selected_ids),
        key=lambda row: sha256_text(f"{seed}:general-topup:{row['record_id']}"),
    )
    for row in [*targeted_union, *general]:
        if len(targeted) >= 60:
            break
        if row["record_id"] not in selected_ids:
            selected_ids.add(row["record_id"])
            targeted.append(row)
    if len(targeted) != 60:
        raise ValueError(f"could select only {len(targeted)} targeted/top-up rows")

    sample = random_rows + targeted
    pilot = random_rows[:4] + targeted[:8]
    return {
        "seed": seed,
        "random_count": len(random_rows),
        "targeted_count": len(targeted),
        "requested_quotas": requested,
        "achieved_quotas": dict(achieved),
        "shortages_before_topup": shortages,
        "topup_count": 60 - sum(achieved.values()),
        "ordered_sample_ids": [row["record_id"] for row in sample],
        "ordered_pilot_ids": [row["record_id"] for row in pilot],
        "private_selection": [
            {
                "record_id": row["record_id"],
                "component": "random" if index < 40 else "targeted",
                "v1_pattern": v1_pattern(votes[row["record_id"]]),
                "explicit_reference": bool(REFERENCE_RE.search(row["clause_text"])),
                "structural_or_fragment": looks_structural_or_fragment(row),
            }
            for index, row in enumerate(sample)
        ],
    }


class ContextBuilder:
    """Build local and conservative resolved bundles from governed alignment inputs."""

    def __init__(
        self,
        alignment_root: Path,
        allowed_families: Iterable[str],
        max_context_chars: int = 24000,
    ) -> None:
        self.alignment_root = alignment_root
        self.allowed_families = frozenset(allowed_families)
        self.max_context_chars = max_context_chars
        self._documents: dict[tuple[str, str], tuple[list[dict[str, Any]], Path, str]] = {}

    def _document(self, row: dict[str, str]) -> tuple[list[dict[str, Any]], Path, str]:
        family = row["family_id"]
        document_id = row["document_id"]
        assert_governed_family(family, self.allowed_families)
        key = (family, document_id)
        if key in self._documents:
            return self._documents[key]
        candidates = sorted((self.alignment_root / family).glob(f"*/inputs/{document_id}.json"))
        wanted = set(filter(None, row["clause_ids"].split("|")))
        best: tuple[int, list[dict[str, Any]], Path, str] | None = None
        for path in candidates:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("family_id", family) != family or payload.get("document_id") != document_id:
                continue
            clauses = payload.get("clauses")
            if not isinstance(clauses, list):
                continue
            score = len(wanted & {str(item.get("clause_id", "")) for item in clauses})
            candidate = (score, clauses, path, str(payload.get("source_sha256", "")))
            if best is None or score > best[0]:
                best = candidate
        if best is None or best[0] == 0:
            raise FileNotFoundError(f"no same-version parsed source for {document_id}")
        self._documents[key] = (best[1], best[2], best[3])
        return self._documents[key]

    @staticmethod
    def _item(kind: str, clause: dict[str, Any], source_hash: str, reason: str) -> dict[str, Any]:
        content = str(clause.get("raw_text", ""))
        return {
            "kind": kind,
            "document_id": str(clause.get("document_id", "")),
            "revision": str(clause.get("revision_id", "")),
            "section_id": str(clause.get("section_id", "")),
            "section_heading": str(clause.get("section_heading", "")),
            "clause_id": str(clause.get("clause_id", "")),
            "block_id": str(clause.get("block_id", "")),
            "source_span": clause.get("source_location", {}),
            "retrieval_reason": reason,
            "source_sha256": source_hash,
            "content_sha256": sha256_text(content),
            "content": content,
        }

    def build(self, row: dict[str, str], variant: str) -> dict[str, Any]:
        if variant not in {"local", "resolved"}:
            raise ValueError(f"unknown context variant: {variant}")
        clauses, source_path, source_hash = self._document(row)
        ordered = sorted(clauses, key=lambda item: int(item.get("index", 0)))
        wanted = set(filter(None, row["clause_ids"].split("|")))
        positions = [i for i, item in enumerate(ordered) if item.get("clause_id") in wanted]
        if not positions:
            raise ValueError(f"target clauses absent from source: {row['record_id']}")
        first, last = min(positions), max(positions)
        target_items = [ordered[i] for i in positions]
        section_id = str(target_items[0].get("section_id", ""))
        items: list[dict[str, Any]] = []
        if first > 0 and str(ordered[first - 1].get("section_id", "")) == section_id:
            items.append(self._item("previous", ordered[first - 1], source_hash, "local_previous_clause"))
        for target in target_items:
            items.append(self._item("target", target, source_hash, "frozen_target_clause"))
        if last + 1 < len(ordered) and str(ordered[last + 1].get("section_id", "")) == section_id:
            items.append(self._item("next", ordered[last + 1], source_hash, "local_next_clause"))

        retrieval: list[dict[str, Any]] = []
        if variant == "resolved":
            seen = {item["clause_id"] for item in items}
            target_blocks = {str(item.get("block_id", "")) for item in target_items}
            for clause in ordered:
                cid = str(clause.get("clause_id", ""))
                if cid not in seen and str(clause.get("block_id", "")) in target_blocks:
                    items.append(self._item("parent_structure", clause, source_hash, "same_parent_block"))
                    seen.add(cid)

            internal_sections = sorted(
                set(re.findall(r"\b[Ss]ection\s+([A-Za-z0-9][A-Za-z0-9.\-]*)", row["clause_text"]))
            )
            for target_section in internal_sections:
                matches = [
                    clause for clause in ordered if str(clause.get("section_id", "")).rstrip(".") == target_section.rstrip(".")
                ]
                if not matches:
                    retrieval.append({
                        "reference": f"Section {target_section}",
                        "status": "not_found_in_scoped_location",
                        "reason": "explicit_internal_section_reference",
                    })
                    continue
                for clause in matches:
                    cid = str(clause.get("clause_id", ""))
                    if cid not in seen:
                        items.append(self._item("reference_body", clause, source_hash, f"explicit_section:{target_section}"))
                        seen.add(cid)
                retrieval.append({
                    "reference": f"Section {target_section}",
                    "status": "retrieved",
                    "reason": "explicit_internal_section_reference",
                    "matched_clauses": len(matches),
                })

            citation_targets = sorted({
                str(ref)
                for target in target_items
                for ref in target.get("reference_targets", [])
                if ref
            })
            for reference in citation_targets:
                retrieval.append({
                    "reference": reference,
                    "status": "unavailable_source",
                    "reason": "external_reference_not_resolved_without_exact_permitted_version",
                })

            # Deterministic definition candidates in the same document/revision.
            target_terms = {
                token.casefold()
                for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", row["clause_text"])
                if token.casefold() not in {"must", "should", "shall", "endpoint", "sender", "receiver", "this", "that", "with", "from"}
            }
            definition_candidates: list[dict[str, Any]] = []
            for clause in ordered:
                text = str(clause.get("raw_text", ""))
                heading = str(clause.get("section_heading", ""))
                if not ({"definition", "terminology", "convention"} & set(re.findall(r"[a-z]+", heading.casefold()))):
                    continue
                lowered = text.casefold()
                if any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in target_terms):
                    definition_candidates.append(clause)
            for clause in definition_candidates[:8]:
                cid = str(clause.get("clause_id", ""))
                if cid not in seen:
                    items.append(self._item("definition", clause, source_hash, "same_revision_definition_candidate"))
                    seen.add(cid)

        # Priority is target, local neighbours, parent structure, explicit bodies, definitions.
        priority = {"target": 0, "previous": 1, "next": 1, "parent_structure": 2, "reference_body": 3, "definition": 4}
        items = sorted(items, key=lambda item: (priority[item["kind"]], item["section_id"], item["clause_id"]))
        kept: list[dict[str, Any]] = []
        used = 0
        omitted: list[dict[str, Any]] = []
        for item in items:
            cost = len(item["content"])
            if item["kind"] == "target" or used + cost <= self.max_context_chars:
                kept.append(item)
                used += cost
            else:
                omitted.append({
                    "clause_id": item["clause_id"],
                    "kind": item["kind"],
                    "status": "omitted_for_budget",
                    "content_sha256": item["content_sha256"],
                })
        return {
            "context_builder_version": CONTEXT_BUILDER_VERSION,
            "variant": variant,
            "record_id": row["record_id"],
            "family_id": row["family_id"],
            "document_id": row["document_id"],
            "revision": row["revision"],
            "section_id": row["section_ids"],
            "section_heading": str(target_items[0].get("section_heading", "")),
            "target_clause": row["clause_text"],
            "source_artifact": source_path.as_posix(),
            "source_sha256": source_hash or file_sha256(source_path),
            "items": kept,
            "retrieval": retrieval,
            "omitted": omitted,
            "context_char_limit": self.max_context_chars,
            "context_chars": used,
            "coverage_claim": "bounded_same_document_same_revision_context_only",
        }


def render_user_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_scope": "Judge TARGET CLAUSE using only this supplied same-version context bundle.",
        "document_id": bundle["document_id"],
        "revision": bundle["revision"],
        "section_id": bundle["section_id"],
        "section_heading": bundle["section_heading"],
        "context_variant": bundle["variant"],
        "coverage_warning": bundle["coverage_claim"],
        "retrieval_status": bundle["retrieval"],
        "omitted_for_budget": bundle["omitted"],
        "context_items": [
            {
                "kind": item["kind"],
                "section_id": item["section_id"],
                "section_heading": item["section_heading"],
                "content": item["content"],
            }
            for item in bundle["items"]
            if item["kind"] != "target"
        ],
        "target_clause": bundle["target_clause"],
    }
