"""Build normative silver ambiguity candidates from a completed alignment run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alignment.version_alignment.io import file_hash, read_jsonl
from alignment.version_alignment.models import Alignment, Document
from corpus.build_tls_silver_labels import (
    SILVER_ALGORITHM_VERSION,
    SILVER_SCHEMA_VERSION,
    _write_csv,
    _write_json,
    build_positive_candidates,
    digest,
    is_clear_control_eligible,
    match_erratum_to_rfc,
    path_reference,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def build_revision_change_candidates(
    documents: list[Document],
    alignments: list[Alignment],
    lineage_by_clause: dict[str, str],
    family_id: str,
) -> list[dict[str, Any]]:
    """Select one latest normative source occurrence per changed lineage.

    A later edit is only retrospective evidence for human review.  It is never
    converted into an automatic ambiguity-positive label.
    """
    clauses = {
        clause.clause_id: clause
        for document in documents
        for clause in document.clauses
    }
    candidates: list[dict[str, Any]] = []
    for alignment in alignments:
        if (
            alignment.relation not in {"MODIFIED", "SPLIT", "MERGED"}
            or not alignment.text_changed
            or alignment.confidence == "LOW"
            or (
                alignment.relation in {"SPLIT", "MERGED"}
                and alignment.confidence != "HIGH"
            )
            or not alignment.source_clause_ids
        ):
            continue
        source_clauses = [clauses[clause_id] for clause_id in alignment.source_clause_ids]
        if any(clause.classification != "NORMATIVE" for clause in source_clauses):
            continue
        revision = source_clauses[0].revision_id
        if revision is None:
            continue
        lineage_ids = tuple(
            sorted({lineage_by_clause[clause.clause_id] for clause in source_clauses})
        )
        row = {
            "record_id": digest(
                [
                    SILVER_ALGORITHM_VERSION,
                    "normative-revision-change",
                    alignment.record_id,
                ]
            ),
            "schema_version": SILVER_SCHEMA_VERSION,
            "family_id": family_id,
            "document_id": alignment.from_document_id,
            "revision": revision,
            "clause_ids": alignment.source_clause_ids,
            "clause_text": " ".join(clause.raw_text for clause in source_clauses),
            "section_ids": list(
                dict.fromkeys(clause.section_id for clause in source_clauses)
            ),
            "classification": "NORMATIVE",
            "clause_classifications": [
                clause.classification for clause in source_clauses
            ],
            "section_headings": list(
                dict.fromkeys(clause.section_heading for clause in source_clauses)
            ),
            "weak_label": "SILVER_CHANGE_CANDIDATE",
            "ambiguity_label": "UNCERTAIN",
            "label_tier": "SILVER",
            "ground_truth": False,
            "evidence_type": "SUBSTANTIVE_REVISION_CHANGE",
            "evidence_role": "LABEL_SUPPORT",
            "errata_id": None,
            "errata_url": None,
            "rfc_clause_ids": (
                alignment.target_clause_ids
                if alignment.to_document_id.upper().startswith("RFC")
                else []
            ),
            "lineage_ids": list(lineage_ids),
            "alignment_record_ids": [alignment.record_id],
            "alignment_confidence": alignment.confidence,
            "rfc_text_match": None,
            "requires_human_review": True,
            "notes": (
                f"Normative clause later had a {alignment.relation} alignment; "
                "the edit is a review signal, not proof of ambiguity."
            ),
        }
        candidates.append(row)
    selected: list[dict[str, Any]] = []
    used_lineages: set[str] = set()
    for row in sorted(
        candidates,
        key=lambda item: (int(item["revision"]), item["record_id"]),
        reverse=True,
    ):
        row_lineages = set(row["lineage_ids"])
        if row_lineages & used_lineages:
            continue
        selected.append(row)
        used_lineages.update(row_lineages)
    return sorted(
        selected,
        key=lambda item: (int(item["revision"]), item["record_id"]),
    )


def build_stable_normative_controls(
    signal_candidates: list[dict[str, Any]],
    documents: list[Document],
    lineage_by_clause: dict[str, str],
    family_id: str,
) -> list[dict[str, Any]]:
    """Choose one latest occurrence from stable normative lineages as controls."""
    clauses = {
        clause.clause_id: clause
        for document in documents
        for clause in document.clauses
    }
    excluded_lineages = {
        lineage_id
        for row in signal_candidates
        for lineage_id in row["lineage_ids"]
    }
    occurrences: dict[tuple[str, str], list[str]] = defaultdict(list)
    for document in documents[:-1]:
        for clause in document.clauses:
            occurrences[(lineage_by_clause[clause.clause_id], clause.normalized_text)].append(
                clause.clause_id
            )

    choices_by_lineage: dict[str, Any] = {}
    for (lineage_id, _), clause_ids in occurrences.items():
        if lineage_id in excluded_lineages:
            continue
        if len({clauses[clause_id].document_id for clause_id in clause_ids}) < 3:
            continue
        eligible = [
            clauses[clause_id]
            for clause_id in clause_ids
            if is_clear_control_eligible(clauses[clause_id])
        ]
        if eligible:
            choice = max(
                eligible,
                key=lambda clause: (int(clause.revision_id), clause.clause_id),
            )
            previous = choices_by_lineage.get(lineage_id)
            if previous is None or (
                int(choice.revision_id),
                choice.clause_id,
            ) > (int(previous.revision_id), previous.clause_id):
                choices_by_lineage[lineage_id] = choice
    choices = list(choices_by_lineage.values())
    choices.sort(key=lambda clause: digest(["stable-normative-control", clause.clause_id]))
    choices = choices[: len(signal_candidates)]
    controls: list[dict[str, Any]] = []
    for clause in choices:
        lineage_id = lineage_by_clause[clause.clause_id]
        controls.append(
            {
                "record_id": digest(
                    [SILVER_ALGORITHM_VERSION, "clear-control", clause.clause_id]
                ),
                "schema_version": SILVER_SCHEMA_VERSION,
                "family_id": family_id,
                "document_id": clause.document_id,
                "revision": clause.revision_id,
                "clause_ids": [clause.clause_id],
                "clause_text": clause.raw_text,
                "section_ids": [clause.section_id],
                "classification": "NORMATIVE",
                "clause_classifications": [clause.classification],
                "section_headings": [clause.section_heading],
                "weak_label": "SILVER_CLEAR_CANDIDATE",
                "ambiguity_label": "UNCERTAIN",
                "label_tier": "SILVER",
                "ground_truth": False,
                "evidence_type": "STABLE_LINEAGE_CONTROL",
                "evidence_role": "LABEL_SUPPORT",
                "errata_id": None,
                "errata_url": None,
                "rfc_clause_ids": [],
                "lineage_ids": [lineage_id],
                "alignment_record_ids": [],
                "alignment_confidence": "UNMEASURED",
                "rfc_text_match": None,
                "requires_human_review": True,
                "notes": (
                    "Normative text remained identical in at least three document "
                    "versions; stability is not proof of clarity."
                ),
            }
        )
    return sorted(
        controls, key=lambda item: (int(item["revision"]), item["record_id"])
    )


def load_alignment_run(
    run_dir: Path, family_id: str, rfc_id: str
) -> tuple[list[Document], list[Alignment], dict[str, str], dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise ValueError(f"Alignment run is not complete: {run_dir}")
    if manifest.get("family_id") != family_id:
        raise ValueError(
            f"Alignment family {manifest.get('family_id')} does not match {family_id}"
        )

    documents = [
        Document.model_validate_json(path.read_text(encoding="utf-8"))
        for path in (run_dir / "inputs").glob("*.json")
    ]
    documents.sort(
        key=lambda document: (
            0 if document.revision is not None else 1,
            int(document.revision) if document.revision is not None else 0,
        )
    )
    if not documents or documents[-1].document_id != rfc_id:
        raise ValueError(f"Alignment timeline does not end in {rfc_id}")
    if any(document.family_id != family_id for document in documents):
        raise ValueError("Alignment inputs contain another document family")

    alignments = [
        Alignment.model_validate(row) for row in read_jsonl(run_dir / "alignments.jsonl")
    ]
    lineage_by_clause = {
        row["clause_id"]: row["lineage_id"]
        for row in read_jsonl(run_dir / "lineage.jsonl")
    }
    clause_ids = {
        clause.clause_id for document in documents for clause in document.clauses
    }
    if set(lineage_by_clause) != clause_ids:
        raise ValueError("Lineage output does not cover every aligned clause exactly once")
    return documents, alignments, lineage_by_clause, manifest


def build(
    root: Path,
    family_id: str,
    rfc_id: str,
    run_dir: Path,
    family_dir: Path,
) -> dict[str, Any]:
    documents, alignments, lineage_by_clause, manifest = load_alignment_run(
        run_dir, family_id, rfc_id
    )
    errata_path = root / "corpus/metadata/verified_technical_errata.json"
    errata_payload = json.loads(errata_path.read_text(encoding="utf-8"))
    errata = errata_payload["by_rfc"].get(rfc_id.casefold(), [])
    rfc = documents[-1]
    matches = [match_erratum_to_rfc(item, rfc) for item in errata]
    positives = build_positive_candidates(
        matches,
        documents,
        alignments,
        lineage_by_clause,
        family_id,
    )
    change_candidates = build_revision_change_candidates(
        documents,
        alignments,
        lineage_by_clause,
        family_id,
    )
    signal_candidates = [*positives, *change_candidates]
    controls = build_stable_normative_controls(
        signal_candidates, documents, lineage_by_clause, family_id
    )
    candidates = sorted(
        [*signal_candidates, *controls],
        key=lambda item: (
            int(item["revision"]),
            item["weak_label"],
            item["record_id"],
        ),
    )
    accepted_errata = {
        match["errata_id"] for match in matches if match["status"] == "ACCEPTED"
    }
    positive_counts = Counter(row["errata_id"] for row in positives)
    created_at = datetime.now(UTC).isoformat()
    report = {
        "schema_version": SILVER_SCHEMA_VERSION,
        "algorithm_version": SILVER_ALGORITHM_VERSION,
        "created_at": created_at,
        "family_id": family_id,
        "rfc_id": rfc_id,
        "ground_truth": False,
        "label_status": "AUTOMATIC_SILVER_CANDIDATES_REQUIRING_HUMAN_REVIEW",
        "classification_policy": (
            "All candidate clauses carry the NORMATIVE label from the explicit "
            "first-stage clause snapshot source recorded by the alignment run."
        ),
        "source_alignment_run": path_reference(run_dir, root),
        "source_alignment_manifest_sha256": file_hash(run_dir / "run_manifest.json"),
        "source_clause_stage": manifest["input_info"]["clause_source"],
        "verified_technical_errata": len(errata),
        "errata_matches_accepted": sum(
            match["status"] == "ACCEPTED" for match in matches
        ),
        "errata_matches_uncertain": sum(
            match["status"] == "UNCERTAIN" for match in matches
        ),
        "silver_positive_candidates": len(positives),
        "silver_change_candidates": len(change_candidates),
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
            "evidence. Revision changes provide review signals only. None of these "
            "records is human-adjudicated ambiguity ground truth."
        ),
    }
    envelope = {"metadata": report, "records": candidates}
    silver_dir = family_dir / "silver"
    review_dir = family_dir / "review"
    silver_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    (family_dir / "gold").mkdir(parents=True, exist_ok=True)
    _write_json(silver_dir / "errata_clause_map.json", matches)
    _write_json(silver_dir / "silver_ambiguity_candidates.json", envelope)
    _write_csv(silver_dir / "silver_ambiguity_candidates.csv", candidates)
    _write_csv(
        review_dir / "ground_truth_review_template.csv",
        candidates,
        include_human_review=True,
    )
    _write_json(silver_dir / "silver_label_report.json", report)
    _write_text(
        family_dir / "README.md",
        f"""# {family_id} label lifecycle

This directory contains automatically prepared review candidates for
`{family_id}` through `{rfc_id}`.

- `silver/` contains reproducible automatic evidence and candidates.
- `review/` contains pending independent-review CSVs.
- `gold/` is reserved for completed, adjudicated records.

Only clauses classified `NORMATIVE` by the frozen first-stage classifier are
eligible. `SILVER_CHANGE_CANDIDATE` means that a normative clause was later
substantively revised; the edit is a review signal, not proof of ambiguity.
`SILVER_CLEAR_CANDIDATE` means identical normative text persisted for at least
three versions; stability is not proof of clarity. Every automatic record keeps
`ambiguity_label=UNCERTAIN`, `ground_truth=false`, and requires human review.
""",
    )
    _write_text(
        family_dir / "gold/README.md",
        """# Gold labels

Reserved for records that completed independent review and adjudication.
Automatic silver candidates must never be copied here without that process.
""",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True)
    parser.add_argument("--rfc", required=True)
    parser.add_argument("--alignment-run", type=Path, required=True)
    parser.add_argument("--family-dir", type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    run_dir = args.alignment_run
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    family_dir = args.family_dir or (
        root / "corpus/processed/ground_truth" / args.family
    )
    if not family_dir.is_absolute():
        family_dir = root / family_dir
    try:
        report = build(
            root,
            args.family,
            args.rfc,
            run_dir.resolve(),
            family_dir.resolve(),
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"Created {report['candidate_records']} {args.family} silver candidates: "
        f"{report['silver_positive_candidates']} errata-positive, "
        f"{report['silver_change_candidates']} change-review, and "
        f"{report['silver_clear_candidates']} clear controls -> {family_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
