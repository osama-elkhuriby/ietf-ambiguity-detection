"""Audit sealed-family lineage and copied normative text using frozen corpus files."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from corpus.paths import REPO_ROOT
from models.classical.feature_pipeline import sha256_file

DEFAULT_SPLIT_CONFIG = REPO_ROOT / "experiments/transformer/configs/split_v2.json"
DEFAULT_MANIFEST = REPO_ROOT / "corpus/metadata/final_rfc_manifest.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/transformer/governance/dns_lineage_audit.json"
)
RELATION_COLUMNS = ("obsoletes", "updates", "obsoleted_by", "updated_by")
NORMATIVE = re.compile(r"\b(?:MUST|MUST NOT|SHOULD|SHOULD NOT|MAY|REQUIRED|SHALL|RECOMMENDED|OPTIONAL)\b")
RFC_REFERENCE = re.compile(r"\bRFC[ -]?(\d{3,5})\b", re.IGNORECASE)
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
BOILERPLATE_MARKERS = (
    "bcp 14",
    "key words must must not required shall",
    "code components extracted from this document must include",
    "provided without warranty as described",
)


def _relations(value: Any) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {
        item.strip().upper().replace(" ", "")
        for item in str(value).split(";")
        if item.strip()
    }


def _connected_components(frame: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        rfc = str(row.rfc_number).upper().replace(" ", "")
        graph.setdefault(rfc, set())
        for column in RELATION_COLUMNS:
            for related in _relations(getattr(row, column)):
                graph[rfc].add(related)
                graph[related].add(rfc)
    components: dict[str, set[str]] = {}
    for start in graph:
        seen: set[str] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(graph[current] - seen)
        components[start] = seen
    return components


def _normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def _normative_paragraphs(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").replace("\f", "\n\n")
    paragraphs: list[dict[str, Any]] = []
    for raw in re.split(r"\n\s*\n", text):
        joined = " ".join(line.strip() for line in raw.splitlines() if line.strip())
        if not NORMATIVE.search(joined):
            continue
        normalized = _normalize(joined)
        tokens = TOKEN.findall(normalized)
        if len(tokens) < 12:
            continue
        paragraphs.append(
            {
                "text": joined,
                "normalized": normalized,
                "tokens": tokens,
                "shingles": {
                    tuple(tokens[index : index + 5])
                    for index in range(max(1, len(tokens) - 4))
                },
            }
        )
    return paragraphs


def _is_boilerplate(normalized: str) -> bool:
    return any(marker in normalized for marker in BOILERPLATE_MARKERS)


def _jaccard(first: set[tuple[str, ...]], second: set[tuple[str, ...]]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 0.0


def _text_pairs(
    test_row: pd.Series,
    training_rows: pd.DataFrame,
    *,
    near_threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    test_path = REPO_ROOT / str(test_row["rfc_text_filename"])
    if not test_path.is_file():
        test_path = REPO_ROOT / "corpus/raw/rfcs" / str(test_row["rfc_text_filename"])
    test_paragraphs = _normative_paragraphs(test_path)
    copied: list[dict[str, Any]] = []
    boilerplate: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for training_row in training_rows.to_dict(orient="records"):
        training_path = REPO_ROOT / "corpus/raw/rfcs" / str(
            training_row["rfc_text_filename"]
        )
        training_paragraphs = _normative_paragraphs(training_path)
        for test_paragraph in test_paragraphs:
            for training_paragraph in training_paragraphs:
                first_tokens = test_paragraph["tokens"]
                second_tokens = training_paragraph["tokens"]
                length_ratio = len(first_tokens) / len(second_tokens)
                if not 0.70 <= length_ratio <= 1.43:
                    continue
                exact = test_paragraph["normalized"] == training_paragraph["normalized"]
                similarity = 1.0 if exact else _jaccard(
                    test_paragraph["shingles"], training_paragraph["shingles"]
                )
                if similarity < near_threshold:
                    continue
                key = (
                    str(training_row["draft_family"]),
                    test_paragraph["normalized"],
                    training_paragraph["normalized"],
                )
                if key in seen:
                    continue
                seen.add(key)
                item = {
                    "test_family": str(test_row["draft_family"]),
                    "test_rfc": str(test_row["rfc_number"]),
                    "training_family": str(training_row["draft_family"]),
                    "training_rfc": str(training_row["rfc_number"]),
                    "match": "exact" if exact else "near",
                    "word_5gram_jaccard": round(similarity, 6),
                    "test_excerpt": test_paragraph["text"][:500],
                    "training_excerpt": training_paragraph["text"][:500],
                }
                if _is_boilerplate(test_paragraph["normalized"]):
                    boilerplate.append(item)
                else:
                    copied.append(item)
    copied.sort(key=lambda item: (-item["word_5gram_jaccard"], item["training_family"]))
    boilerplate.sort(
        key=lambda item: (-item["word_5gram_jaccard"], item["training_family"])
    )
    return copied, boilerplate


def audit_lineage(
    *,
    split_config: Path = DEFAULT_SPLIT_CONFIG,
    manifest_path: Path = DEFAULT_MANIFEST,
    near_threshold: float = 0.85,
) -> dict[str, Any]:
    policy = json.loads(split_config.read_text(encoding="utf-8"))
    training_families = set(policy["assignments"]["train"])
    test_families = policy["assignments"]["test"]
    if len(test_families) != 1:
        raise ValueError("Lineage audit requires exactly one sealed test family")
    test_family = test_families[0]
    manifest = pd.read_csv(manifest_path).fillna("")
    indexed = manifest.set_index("draft_family", drop=False)
    unknown = (training_families | {test_family}) - set(indexed.index)
    if unknown:
        raise ValueError(f"Families missing from RFC manifest: {sorted(unknown)}")
    test_row = indexed.loc[test_family]
    training_rows = indexed.loc[sorted(training_families)].reset_index(drop=True)
    components = _connected_components(manifest)
    test_rfc = str(test_row["rfc_number"]).upper().replace(" ", "")

    lineage_pairs: list[dict[str, Any]] = []
    shared_draft_pairs: list[dict[str, Any]] = []
    for training_row in training_rows.to_dict(orient="records"):
        training_rfc = str(training_row["rfc_number"]).upper().replace(" ", "")
        shared_chain = sorted(
            components.get(test_rfc, {test_rfc})
            & components.get(training_rfc, {training_rfc})
        )
        if shared_chain:
            lineage_pairs.append(
                {
                    "test_family": test_family,
                    "training_family": training_row["draft_family"],
                    "shared_rfc_chain_members": shared_chain,
                }
            )
        if str(test_row["final_draft_revision"]).rsplit("-", 1)[0] == str(
            training_row["final_draft_revision"]
        ).rsplit("-", 1)[0]:
            shared_draft_pairs.append(
                {
                    "test_family": test_family,
                    "training_family": training_row["draft_family"],
                }
            )

    copied, boilerplate = _text_pairs(
        test_row, training_rows, near_threshold=near_threshold
    )
    test_text_path = REPO_ROOT / "corpus/raw/rfcs" / str(
        test_row["rfc_text_filename"]
    )
    referenced_rfcs = {
        f"RFC{number}" for number in RFC_REFERENCE.findall(
            test_text_path.read_text(encoding="utf-8", errors="replace")
        )
    }
    cross_references = [
        {
            "test_family": test_family,
            "training_family": row["draft_family"],
            "referenced_rfc": str(row["rfc_number"]),
        }
        for row in training_rows.to_dict(orient="records")
        if str(row["rfc_number"]) in referenced_rfcs
    ]
    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "test_family": test_family,
            "training_families": sorted(training_families),
            "metadata_relations": list(RELATION_COLUMNS),
            "text_source": "final RFC plaintext",
            "normative_filter": "uppercase RFC 2119/8174 keywords",
            "near_match": "word 5-gram Jaccard",
            "near_threshold": near_threshold,
        },
        "source_hashes": {
            "split_config": sha256_file(split_config),
            "rfc_manifest": sha256_file(manifest_path),
            "test_rfc_text": sha256_file(test_text_path),
        },
        "rfc_lineage_pairs": lineage_pairs,
        "shared_draft_pairs": shared_draft_pairs,
        "copied_normative_text_pairs": copied,
        "shared_boilerplate_pairs": boilerplate,
        "cross_reference_pairs_not_lineage": cross_references,
        "counts": {
            "rfc_lineage_pairs": len(lineage_pairs),
            "shared_draft_pairs": len(shared_draft_pairs),
            "copied_normative_text_pairs": len(copied),
            "shared_boilerplate_pairs": len(boilerplate),
            "cross_reference_pairs_not_lineage": len(cross_references),
        },
        "lineage_overlap_detected": bool(lineage_pairs or shared_draft_pairs or copied),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-config", type=Path, default=DEFAULT_SPLIT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--near-threshold", type=float, default=0.85)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_lineage(
        split_config=args.split_config,
        manifest_path=args.manifest,
        near_threshold=args.near_threshold,
    )
    if args.output.exists():
        raise FileExistsError(f"Audit output is immutable: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["counts"], indent=2, sort_keys=True))
    return 2 if report["lineage_overlap_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
