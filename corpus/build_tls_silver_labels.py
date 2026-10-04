"""Build conservative TLS 1.3 silver ambiguity candidates from RFC errata.

This module deliberately produces weak labels, not human ground truth. It uses the
frozen TLS alignment snapshots and their retained semantic-vector cache to recover
the missing correspondence decisions without invoking the embedding model.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from rapidfuzz.fuzz import partial_ratio, ratio

from alignment.version_alignment.candidates import candidate_graph
from alignment.version_alignment.config import Config
from alignment.version_alignment.io import canonical, digest, file_hash
from alignment.version_alignment.lineage import build_lineage
from alignment.version_alignment.matching import select
from alignment.version_alignment.models import Alignment, Clause, Document
from alignment.version_alignment.review import records as alignment_records

FAMILY_ID = "draft-ietf-tls-tls13"
SILVER_SCHEMA_VERSION = "1.0"
SILVER_ALGORITHM_VERSION = "tls-errata-distant-supervision-v1"
RFC_ID = "RFC8446"

ACCEPTED_RELATIONS = {"UNCHANGED", "MODIFIED", "MOVED", "SPLIT", "MERGED"}
CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
CONTROL_EXCLUSION_RE = re.compile(
    r"\b(?:TODO|TBD|TBA|FIXME|XXX)\b|[?？]", re.IGNORECASE
)
CONTROL_EXCLUDED_HEADINGS = ("change log", "open issues", "todo")
HUMAN_REVIEW_FIELDS = [
    "human_is_ambiguous",
    "human_label_tier",
    "human_revision_reason",
    "human_ambiguity_subtype",
    "human_evidence_span",
    "annotator",
    "date_annotated",
    "adjudication_status",
    "human_notes",
    "human_positive_evidence",
]


def normalize_evidence_text(text: str) -> str:
    """Normalize rendering whitespace without deleting protocol information."""
    return re.sub(
        r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n")
    ).strip()


def normalize_section_id(section: str) -> str:
    return section.strip().rstrip(".")


def path_reference(path: Path, root: Path) -> str:
    """Prefer a portable repository path but allow an external CLI input path."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _contiguous_groups(
    clauses: list[Clause], max_group: int
) -> Iterable[list[Clause]]:
    for start in range(len(clauses)):
        group: list[Clause] = []
        for clause in clauses[start : start + max_group]:
            if group and (
                clause.index != group[-1].index + 1
                or clause.section_id != group[0].section_id
            ):
                break
            group.append(clause)
            yield list(group)


def match_erratum_to_rfc(
    erratum: dict[str, Any], rfc: Document, *, max_group: int = 8
) -> dict[str, Any]:
    """Return a transparent unique-match decision for one erratum passage."""
    original = normalize_evidence_text(str(erratum.get("orig_text") or ""))
    section = normalize_section_id(str(erratum.get("section") or ""))
    eligible = [
        clause
        for clause in rfc.clauses
        if clause.section_id == section or clause.section_id.startswith(section + ".")
    ]
    candidates: list[dict[str, Any]] = []
    for group in _contiguous_groups(eligible, max_group):
        candidate_text = normalize_evidence_text(" ".join(c.raw_text for c in group))
        candidates.append(
            {
                "clause_ids": [c.clause_id for c in group],
                "section_id": group[0].section_id,
                "text": candidate_text,
                "text_ratio": ratio(original, candidate_text) / 100,
                "partial_ratio": partial_ratio(original, candidate_text) / 100,
            }
        )
    candidates.sort(
        key=lambda item: (
            -item["text_ratio"],
            -item["partial_ratio"],
            abs(len(item["text"]) - len(original)),
            item["clause_ids"],
        )
    )
    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None
    exact_count = sum(item["text_ratio"] == 1.0 for item in candidates)
    margin = (
        best["text_ratio"] - second["text_ratio"]
        if best is not None and second is not None
        else best["text_ratio"]
        if best is not None
        else 0.0
    )
    accepted = bool(
        original
        and best
        and (
            (best["text_ratio"] == 1.0 and exact_count == 1)
            or (
                best["text_ratio"] >= 0.90
                and best["partial_ratio"] >= 0.95
                and margin >= 0.05
            )
        )
    )
    return {
        "errata_id": str(erratum.get("errata_id")),
        "section": section,
        "source_url": erratum.get("source_url"),
        "orig_text": erratum.get("orig_text", ""),
        "correct_text": erratum.get("correct_text", ""),
        "status": "ACCEPTED" if accepted else "UNCERTAIN",
        "decision_reason": (
            "unique_exact_section_match"
            if accepted and best and best["text_ratio"] == 1.0
            else "unique_high_similarity_section_match"
            if accepted
            else "no_unique_high_similarity_section_match"
        ),
        "best_match": best,
        "score_margin": margin,
        "alternatives": candidates[1:4],
    }


def _timeline_documents(run_dir: Path) -> list[Document]:
    documents = [
        Document.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (run_dir / "inputs").glob("*.json")
    ]

    def order(document: Document) -> tuple[int, int]:
        return (
            (0, int(document.revision))
            if document.revision is not None
            else (1, 0)
        )

    documents.sort(key=order)
    expected = [f"{FAMILY_ID}-{revision:02d}" for revision in range(29)] + [RFC_ID]
    if [document.document_id for document in documents] != expected:
        raise ValueError(
            "TLS frozen inputs must contain revisions 00-28 followed by RFC8446"
        )
    return documents


def _cached_vectors(
    document: Document, model: dict[str, Any], cache_dir: Path
) -> np.ndarray:
    key = digest(
        {
            "texts": [clause.normalized_text for clause in document.clauses],
            "model": model,
        }
    )
    vector_path = cache_dir / f"{key}.npy"
    metadata_path = cache_dir / f"{key}.json"
    if not vector_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing retained embedding cache for {document.document_id}: {key}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("key") != key
        or metadata.get("artifact_sha256") != file_hash(vector_path)
    ):
        raise ValueError(f"Invalid embedding cache for {document.document_id}: {key}")
    vectors = np.load(vector_path, allow_pickle=False)
    if vectors.shape[0] != len(document.clauses) or not np.isfinite(vectors).all():
        raise ValueError(f"Embedding shape/content mismatch for {document.document_id}")
    return vectors


def reconstruct_alignments(
    run_dir: Path,
    cache_dir: Path,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[Document], list[Alignment], list[dict[str, Any]]]:
    """Reconstruct relationship records using frozen inputs and retained vectors."""
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    model = json.loads((run_dir / "model.json").read_text(encoding="utf-8"))
    config = Config.model_validate(request["config"])
    documents = _timeline_documents(run_dir)
    vectors = {
        document.document_id: _cached_vectors(document, model, cache_dir)
        for document in documents
    }
    all_rows: list[Alignment] = []
    audits: list[dict[str, Any]] = []
    model_id = digest(model)
    for pair_index, (source, target) in enumerate(
        zip(documents, documents[1:]), start=1
    ):
        candidates = candidate_graph(
            source,
            target,
            config,
            vectors[source.document_id],
            vectors[target.document_id],
        )
        selected, diagnostics = select(candidates, config)
        rows = alignment_records(
            source,
            target,
            candidates,
            selected,
            config,
            request["created_at"],
            model_id,
        )
        expected_path = (
            run_dir
            / "pairs"
            / f"{source.document_id}__{target.document_id}"
            / "report.json"
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        actual_counts = dict(Counter(row.relation for row in rows))
        if (
            len(rows) != expected["relationships"]
            or actual_counts != expected["relationship_counts"]
        ):
            raise ValueError(
                "Recovered relationships disagree with retained report for "
                f"{source.document_id} -> {target.document_id}"
            )
        audits.append(
            {
                "from_document_id": source.document_id,
                "to_document_id": target.document_id,
                "candidates_reconstructed": len(candidates),
                "candidates_reported": expected["candidate_count"],
                "candidate_count_delta": len(candidates)
                - expected["candidate_count"],
                "relationships": len(rows),
                "relationship_counts": actual_counts,
                "relationship_reproduction": "EXACT_COUNTS_AND_DISTRIBUTION",
                "solver": diagnostics,
            }
        )
        all_rows.extend(rows)
        if progress:
            progress(
                f"Aligned pair {pair_index:02d}/29: {source.document_id} -> "
                f"{target.document_id} ({len(rows)} relationships)"
            )
    return documents, all_rows, audits


def _strict_predecessor(
    target_ids: list[str], rows_by_target: dict[str, Alignment]
) -> tuple[list[str], list[Alignment]] | None:
    used: dict[str, Alignment] = {}
    for clause_id in target_ids:
        row = rows_by_target.get(clause_id)
        if row is None:
            return None
        used[row.record_id] = row
    relationships = list(used.values())
    if any(
        row.relation not in ACCEPTED_RELATIONS
        or row.confidence == "LOW"
        or (
            row.relation in {"MOVED", "SPLIT", "MERGED"}
            and row.confidence != "HIGH"
        )
        for row in relationships
    ):
        return None
    covered = {
        clause_id for row in relationships for clause_id in row.target_clause_ids
    }
    if not set(target_ids) <= covered:
        return None
    source_ids = list(
        dict.fromkeys(
            clause_id
            for row in relationships
            for clause_id in row.source_clause_ids
        )
    )
    return source_ids, relationships


def _clause_lookup(documents: list[Document]) -> dict[str, Clause]:
    return {
        clause.clause_id: clause
        for document in documents
        for clause in document.clauses
    }


def build_positive_candidates(
    matches: list[dict[str, Any]],
    documents: list[Document],
    rows: list[Alignment],
    lineage_by_clause: dict[str, str],
) -> list[dict[str, Any]]:
    clauses = _clause_lookup(documents)
    pair_rows: dict[tuple[str, str], dict[str, Alignment]] = defaultdict(dict)
    for row in rows:
        for target_id in row.target_clause_ids:
            pair_rows[(row.from_document_id, row.to_document_id)][target_id] = row
    output: list[dict[str, Any]] = []
    for match in matches:
        if match["status"] != "ACCEPTED":
            continue
        current_ids = list(match["best_match"]["clause_ids"])
        target_document = documents[-1]
        for source_document in reversed(documents[:-1]):
            predecessor = _strict_predecessor(
                current_ids,
                pair_rows[(source_document.document_id, target_document.document_id)],
            )
            if predecessor is None:
                break
            source_ids, relationships = predecessor
            current_ids = source_ids
            candidate_clauses = [clauses[clause_id] for clause_id in source_ids]
            if any(
                clause.classification != "NORMATIVE"
                for clause in candidate_clauses
            ):
                target_document = source_document
                continue
            confidence = min(
                (row.confidence for row in relationships),
                key=CONFIDENCE_RANK.get,
            )
            record_id = digest(
                [
                    SILVER_ALGORITHM_VERSION,
                    match["errata_id"],
                    source_document.document_id,
                    source_ids,
                ]
            )
            output.append(
                {
                    "record_id": record_id,
                    "schema_version": SILVER_SCHEMA_VERSION,
                    "family_id": FAMILY_ID,
                    "document_id": source_document.document_id,
                    "revision": source_document.revision,
                    "clause_ids": source_ids,
                    "clause_text": " ".join(
                        clause.raw_text for clause in candidate_clauses
                    ),
                    "section_ids": list(
                        dict.fromkeys(
                            clause.section_id for clause in candidate_clauses
                        )
                    ),
                    "classification": "NORMATIVE",
                    "clause_classifications": [
                        clause.classification for clause in candidate_clauses
                    ],
                    "section_headings": list(
                        dict.fromkeys(
                            clause.section_heading for clause in candidate_clauses
                        )
                    ),
                    "weak_label": "SILVER_POSITIVE_CANDIDATE",
                    "ambiguity_label": "UNCERTAIN",
                    "label_tier": "SILVER",
                    "ground_truth": False,
                    "evidence_type": "VERIFIED_TECHNICAL_ERRATUM",
                    "evidence_role": "LABEL_SUPPORT",
                    "errata_id": match["errata_id"],
                    "errata_url": match["source_url"],
                    "rfc_clause_ids": match["best_match"]["clause_ids"],
                    "lineage_ids": sorted(
                        {lineage_by_clause[clause_id] for clause_id in source_ids}
                    ),
                    "alignment_record_ids": [
                        row.record_id for row in relationships
                    ],
                    "alignment_confidence": confidence,
                    "rfc_text_match": match["best_match"]["text_ratio"],
                    "requires_human_review": True,
                    "notes": (
                        "Verified technical erratum mapped through conservative "
                        "automatic lineage; technical correction is not automatically "
                        "ambiguity."
                    ),
                }
            )
            target_document = source_document
    return sorted(
        output, key=lambda item: (int(item["revision"]), item["errata_id"])
    )


def is_clear_control_eligible(clause: Clause) -> bool:
    """Reject draft machinery and malformed/underspecified control text."""
    heading = (clause.section_heading or "").casefold()
    word_count = len(re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", clause.raw_text))
    return bool(
        clause.classification == "NORMATIVE"
        and clause.block_type in {"paragraph", "list_item", "definition"}
        and 40 <= len(clause.raw_text) <= 500
        and word_count >= 8
        and not CONTROL_EXCLUSION_RE.search(clause.raw_text)
        and not any(excluded in heading for excluded in CONTROL_EXCLUDED_HEADINGS)
    )


def build_clear_controls(
    positives: list[dict[str, Any]],
    documents: list[Document],
    lineage_by_clause: dict[str, str],
) -> list[dict[str, Any]]:
    """Select matched stable controls; these remain weak negative candidates."""
    clauses = _clause_lookup(documents)
    positive_ids = {
        clause_id for row in positives for clause_id in row["clause_ids"]
    }
    positive_lineages = {
        lineage for row in positives for lineage in row["lineage_ids"]
    }
    occurrences: dict[tuple[str, str], list[str]] = defaultdict(list)
    for document in documents[:-1]:
        for clause in document.clauses:
            lineage = lineage_by_clause[clause.clause_id]
            occurrences[(lineage, clause.normalized_text)].append(clause.clause_id)
    stable = {
        clause_id
        for ids in occurrences.values()
        if len({clauses[clause_id].document_id for clause_id in ids}) >= 3
        for clause_id in ids
    }
    by_document_section: dict[tuple[str, str], list[Clause]] = defaultdict(list)
    for document in documents[:-1]:
        for clause in document.clauses:
            if (
                clause.clause_id in stable
                and clause.clause_id not in positive_ids
                and lineage_by_clause[clause.clause_id] not in positive_lineages
                and is_clear_control_eligible(clause)
            ):
                by_document_section[
                    (document.document_id, clause.section_id)
                ].append(clause)
    for choices in by_document_section.values():
        choices.sort(key=lambda clause: digest(["clear-control", clause.clause_id]))
    output: list[dict[str, Any]] = []
    used: set[str] = set()
    for positive in positives:
        choice = None
        for section in positive["section_ids"]:
            choice = next(
                (
                    candidate
                    for candidate in by_document_section[
                        (positive["document_id"], section)
                    ]
                    if candidate.clause_id not in used
                ),
                None,
            )
            if choice:
                break
        if choice is None:
            candidates = [
                clause
                for (document_id, _), options in by_document_section.items()
                if document_id == positive["document_id"]
                for clause in options
                if clause.clause_id not in used
            ]
            candidates.sort(
                key=lambda clause: digest(["clear-fallback", clause.clause_id])
            )
            choice = candidates[0] if candidates else None
        if choice is None:
            continue
        used.add(choice.clause_id)
        lineage = lineage_by_clause[choice.clause_id]
        output.append(
            {
                "record_id": digest(
                    [SILVER_ALGORITHM_VERSION, "clear-control", choice.clause_id]
                ),
                "schema_version": SILVER_SCHEMA_VERSION,
                "family_id": FAMILY_ID,
                "document_id": choice.document_id,
                "revision": choice.revision_id,
                "clause_ids": [choice.clause_id],
                "clause_text": choice.raw_text,
                "section_ids": [choice.section_id],
                "classification": choice.classification,
                "clause_classifications": [choice.classification],
                "section_headings": [choice.section_heading],
                "weak_label": "SILVER_CLEAR_CANDIDATE",
                "ambiguity_label": "UNCERTAIN",
                "label_tier": "SILVER",
                "ground_truth": False,
                "evidence_type": "STABLE_LINEAGE_CONTROL",
                "evidence_role": "LABEL_SUPPORT",
                "errata_id": None,
                "errata_url": None,
                "rfc_clause_ids": [],
                "lineage_ids": [lineage],
                "alignment_record_ids": [],
                "alignment_confidence": "UNMEASURED",
                "rfc_text_match": None,
                "requires_human_review": True,
                "notes": (
                    "Stable for at least three document versions and unrelated to "
                    "mapped errata; absence of errata is not proof of clarity."
                ),
            }
        )
    return sorted(
        output, key=lambda item: (int(item["revision"]), item["record_id"])
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(canonical(payload) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    include_human_review: bool = False,
) -> None:
    fields = [
        "record_id",
        "family_id",
        "document_id",
        "revision",
        "clause_ids",
        "clause_text",
        "section_ids",
        "classification",
        "clause_classifications",
        "weak_label",
        "ambiguity_label",
        "label_tier",
        "ground_truth",
        "evidence_type",
        "errata_id",
        "errata_url",
        "rfc_clause_ids",
        "lineage_ids",
        "alignment_confidence",
        "rfc_text_match",
        "requires_human_review",
        "notes",
    ]
    if include_human_review:
        fields.extend(HUMAN_REVIEW_FIELDS)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            if include_human_review:
                serialized.update(
                    {
                        field: "PENDING" if field == "adjudication_status" else ""
                        for field in HUMAN_REVIEW_FIELDS
                    }
                )
            for field in (
                "clause_ids",
                "section_ids",
                "clause_classifications",
                "rfc_clause_ids",
                "lineage_ids",
            ):
                serialized[field] = "|".join(
                    str(value) for value in serialized[field]
                )
            writer.writerow(serialized)
    temporary.replace(path)


def build(
    root: Path,
    run_dir: Path,
    family_dir: Path,
    *,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    errata_payload = json.loads(
        (root / "corpus/metadata/verified_technical_errata.json").read_text(
            encoding="utf-8"
        )
    )
    errata = errata_payload["by_rfc"]["rfc8446"]
    cache_dir = root / "corpus/processed/alignments/_embedding_cache"
    documents, rows, audits = reconstruct_alignments(
        run_dir, cache_dir, progress=progress
    )
    rfc = documents[-1]
    matches = [match_erratum_to_rfc(item, rfc) for item in errata]
    lineage = build_lineage(documents, rows)
    lineage_by_clause = {
        node["clause_id"]: node["lineage_id"] for node in lineage
    }
    positives = build_positive_candidates(matches, documents, rows, lineage_by_clause)
    controls = build_clear_controls(positives, documents, lineage_by_clause)
    positive_counts = Counter(row["errata_id"] for row in positives)
    accepted_errata = {
        match["errata_id"] for match in matches if match["status"] == "ACCEPTED"
    }
    candidates = positives + controls
    candidates.sort(
        key=lambda item: (
            int(item["revision"]),
            item["weak_label"],
            item["record_id"],
        )
    )
    created_at = datetime.now(UTC).isoformat()
    run_manifest = run_dir / "run_manifest.json"
    report = {
        "schema_version": SILVER_SCHEMA_VERSION,
        "algorithm_version": SILVER_ALGORITHM_VERSION,
        "created_at": created_at,
        "family_id": FAMILY_ID,
        "ground_truth": False,
        "label_status": "AUTOMATIC_SILVER_CANDIDATES_REQUIRING_HUMAN_REVIEW",
        "classification_policy": (
            "All candidate clauses must carry the NORMATIVE label from the "
            "frozen first-stage clause classifier."
        ),
        "source_alignment_run": path_reference(run_dir, root),
        "source_alignment_manifest_sha256": file_hash(run_manifest),
        "documents": len(documents),
        "document_pairs": len(audits),
        "relationships_reconstructed": len(rows),
        "relationship_reproduction": (
            "EXACT_RELATIONSHIP_COUNTS_AND_DISTRIBUTIONS_FOR_ALL_PAIRS"
        ),
        "relationship_endpoint_comparison": (
            "UNAVAILABLE_BECAUSE_HISTORICAL_ALIGNMENT_JSONL_IS_NOT_PRESENT"
        ),
        "candidate_count_deltas": {
            "total": sum(audit["candidate_count_delta"] for audit in audits),
            "pairs_with_delta": sum(
                audit["candidate_count_delta"] != 0 for audit in audits
            ),
        },
        "verified_technical_errata": len(errata),
        "errata_matches_accepted": sum(
            match["status"] == "ACCEPTED" for match in matches
        ),
        "errata_matches_uncertain": sum(
            match["status"] == "UNCERTAIN" for match in matches
        ),
        "silver_positive_candidates": len(positives),
        "silver_clear_candidates": len(controls),
        "candidate_records": len(candidates),
        "positive_candidates_by_erratum": dict(positive_counts),
        "accepted_errata_without_draft_candidates": sorted(
            accepted_errata - positive_counts.keys(), key=int
        ),
        "uncertain_errata_ids": sorted(
            (
                match["errata_id"]
                for match in matches
                if match["status"] == "UNCERTAIN"
            ),
            key=int,
        ),
        "disclaimer": (
            "Verified technical errata and stable lineages provide distant-supervision "
            "evidence, not human-adjudicated ambiguity ground truth."
        ),
        "reconstruction_audit": audits,
    }
    envelope = {
        "metadata": {
            key: value
            for key, value in report.items()
            if key != "reconstruction_audit"
        },
        "records": candidates,
    }
    silver_dir = family_dir / "silver"
    review_dir = family_dir / "review"
    silver_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "gold").mkdir(parents=True, exist_ok=True)
    _write_json(silver_dir / "tls13_errata_clause_map.json", matches)
    _write_json(silver_dir / "tls13_silver_ambiguity_candidates.json", envelope)
    _write_csv(silver_dir / "tls13_silver_ambiguity_candidates.csv", candidates)
    _write_csv(
        review_dir / "tls13_ground_truth_review_template.csv",
        candidates,
        include_human_review=True,
    )
    _write_json(silver_dir / "tls13_silver_label_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build TLS 1.3 automatic silver ambiguity candidates (never gold)."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Frozen TLS alignment run (default: pilot-semantic-v5).",
    )
    parser.add_argument(
        "--family-dir",
        type=Path,
        help=(
            "Family label root (default: corpus/processed/ground_truth/"
            "draft-ietf-tls-tls13)."
        ),
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    run_dir = args.run_dir or (
        root
        / "corpus/processed/alignments/draft-ietf-tls-tls13/pilot-semantic-v5"
    )
    family_dir = (
        args.family_dir
        or root
        / "corpus/processed/ground_truth/draft-ietf-tls-tls13"
    )
    try:
        report = build(root, run_dir.resolve(), family_dir.resolve())
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Created {report['candidate_records']} TLS silver candidates: "
        f"{report['silver_positive_candidates']} positive and "
        f"{report['silver_clear_candidates']} clear controls -> {family_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
