"""Create a lineage-deduplicated human-review batch.

This never changes automatic silver artifacts.  It copies selected review rows
into reviewer-specific CSVs and an empty adjudication sheet. TLS paths remain
the defaults for backward compatibility; command-line paths support any family.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY_DIR = ROOT / "corpus/processed/ground_truth/draft-ietf-tls-tls13"
SOURCE = FAMILY_DIR / "review/tls13_ground_truth_review_template.csv"
OUTPUT_DIR = FAMILY_DIR / "review/pilot_v1"
CLASSIFICATION_INPUT_DIR = (
    ROOT
    / "corpus/processed/alignments/draft-ietf-tls-tls13"
    / "pilot-semantic-v5/inputs"
)
CLASSIFICATIONS = ("NORMATIVE", "CONTEXTUAL", "EXCLUDED", "MANUAL_REVIEW")
HUMAN_FIELDS = (
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
)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_reference(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def choose_one_per_lineage(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the latest revision for every lineage, with stable tie-breaking."""
    chosen: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: (int(r["revision"]), r["record_id"])):
        chosen[row["lineage_ids"]] = row
    return sorted(chosen.values(), key=lambda r: (r["weak_label"], r["record_id"]))


def load_first_stage_classifications(input_dir: Path) -> dict[str, str]:
    """Load the classifications already assigned to frozen clause inputs."""
    classifications: dict[str, str] = {}
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No first-stage classification inputs found in {input_dir}")
    for path in paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        for clause in document["clauses"]:
            clause_id = clause["clause_id"]
            classification = clause["classification"]
            if classification not in CLASSIFICATIONS:
                raise ValueError(
                    f"Unknown classification {classification!r} for {clause_id}"
                )
            previous = classifications.setdefault(clause_id, classification)
            if previous != classification:
                raise ValueError(f"Conflicting classifications for {clause_id}")
    return classifications


def attach_first_stage_classifications(
    rows: list[dict[str, str]], classifications: dict[str, str]
) -> list[dict[str, str]]:
    """Attach original per-clause labels without reclassifying clause text."""
    enriched: list[dict[str, str]] = []
    for row in rows:
        clause_ids = row["clause_ids"].split("|")
        try:
            clause_classifications = [
                classifications[clause_id] for clause_id in clause_ids
            ]
        except KeyError as exc:
            raise ValueError(
                f"Missing first-stage classification for {exc.args[0]}"
            ) from exc
        result = dict(row)
        result["classification"] = "|".join(
            dict.fromkeys(clause_classifications)
        )
        result["clause_classifications"] = "|".join(clause_classifications)
        enriched.append(result)
    return enriched


def classified_entirely_as(row: dict[str, str], classification: str) -> bool:
    """Require every constituent clause to have the requested original label."""
    return all(
        value == classification
        for value in row["clause_classifications"].split("|")
    )


def blank_human_fields(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    for field in HUMAN_FIELDS:
        result[field] = "PENDING" if field == "adjudication_status" else ""
    return result


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create family CSVs for independent human review."
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Include every silver candidate rather than one record per lineage.",
    )
    parser.add_argument(
        "--all-lineages",
        action="store_true",
        help=(
            "Include one latest-revision record for every eligible lineage, "
            "without pilot control downsampling."
        ),
    )
    parser.add_argument(
        "--classification",
        choices=CLASSIFICATIONS,
        help=(
            "Keep only records whose constituent clauses all have this label "
            "from the frozen first-stage classifier."
        ),
    )
    parser.add_argument(
        "--classification-input-dir",
        type=Path,
        default=CLASSIFICATION_INPUT_DIR,
        help="Directory containing frozen clause JSON with first-stage labels.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE,
        help="Silver review-template CSV to select from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for reviewer_a.csv, reviewer_b.csv, and adjudication.csv.",
    )
    args = parser.parse_args()
    if args.all_candidates and args.all_lineages:
        parser.error("--all-candidates and --all-lineages are mutually exclusive")

    source = args.source
    if not source.is_absolute():
        source = ROOT / source
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Missing header in {source}")
        fields = list(reader.fieldnames)
        for field in HUMAN_FIELDS:
            if field not in fields:
                fields.append(field)
        source_rows = list(reader)

    classification_input_dir = args.classification_input_dir
    if not classification_input_dir.is_absolute():
        classification_input_dir = ROOT / classification_input_dir
    classifications = load_first_stage_classifications(classification_input_dir)
    source_rows = attach_first_stage_classifications(source_rows, classifications)
    for field in ("classification", "clause_classifications"):
        if field not in fields:
            fields.insert(fields.index("weak_label"), field)

    rows = source_rows
    filters: list[str] = []
    if args.classification:
        rows = [
            row
            for row in rows
            if classified_entirely_as(row, args.classification)
        ]
        filters.append(
            "all constituent clauses classified as "
            f"{args.classification} by the frozen first-stage classifier"
        )

    all_positives = [
        row for row in rows if row["weak_label"] == "SILVER_POSITIVE_CANDIDATE"
    ]
    all_changes = [
        row for row in rows if row["weak_label"] == "SILVER_CHANGE_CANDIDATE"
    ]
    all_controls = [
        row for row in rows if row["weak_label"] == "SILVER_CLEAR_CANDIDATE"
    ]
    if args.all_candidates:
        positives, changes, controls = all_positives, all_changes, all_controls
        selection = "all eligible silver candidate records"
    elif args.all_lineages:
        positives = choose_one_per_lineage(all_positives)
        changes = choose_one_per_lineage(all_changes)
        controls = choose_one_per_lineage(all_controls)
        selection = "one latest-revision record per eligible lineage"
    else:
        positives = choose_one_per_lineage(all_positives)
        changes = choose_one_per_lineage(all_changes)
        controls = choose_one_per_lineage(all_controls)[
            : (len(positives) + len(changes)) * 2
        ]
        selection = (
            "one latest-revision record per signal lineage and two clear-control "
            "lineages per signal lineage"
        )
    batch = sorted(
        [*positives, *changes, *controls],
        key=lambda r: (r["weak_label"], r["record_id"]),
    )
    batch = [blank_human_fields(row) for row in batch]

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reviewer_a.csv", "reviewer_b.csv", "adjudication.csv"):
        write_rows(output_dir / name, fields, batch)
    manifest = {
        "purpose": "Independent human review before promotion to gold",
        "source": path_reference(source),
        "source_sha256": source_hash(source),
        "classification_source": path_reference(classification_input_dir),
        "classification_records": len(classifications),
        "selection": {
            "method": selection,
            "filters": filters,
            "grouping_key": None if args.all_candidates else "lineage_ids",
            "representative": None if args.all_candidates else "latest_revision",
        },
        "source_records": len(source_rows),
        "eligible_records": len(rows),
        "records": len(batch),
        "positive_records": len(positives),
        "change_candidate_records": len(changes),
        "clear_control_records": len(controls),
        "unique_lineages": len({row["lineage_ids"] for row in batch}),
        "record_ids": [row["record_id"] for row in batch],
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)
    family_id = (
        batch[0]["family_id"]
        if batch
        else source.parent.parent.name
    )
    readme_path = output_dir / "README.md"
    temporary_readme = readme_path.with_name(f".{readme_path.name}.tmp")
    temporary_readme.write_text(
        f"""# {family_id} independent review batch

This package contains {len(batch)} lineage-deduplicated, normative records:
{len(positives)} errata-supported candidates, {len(changes)} revision-change
candidates, and {len(controls)} stable-lineage controls.

`reviewer_a.csv` and `reviewer_b.csv` are identical starting copies for
independent annotation. Record decisions in the `human_*`, `annotator`, date,
status, and notes fields. Use `adjudication.csv` only to resolve disagreements.

All records were filtered with the frozen first-stage classifier. Automatic
labels remain silver and `UNCERTAIN`: a later edit is not proof of ambiguity,
and stable wording is not proof of clarity. Promote a record to `gold/` only
after completed independent review and adjudication.
""",
        encoding="utf-8",
    )
    temporary_readme.replace(readme_path)
    print(
        f"Created {len(batch)} records ({len(positives)} positive candidates, "
        f"{len(changes)} change candidates, {len(controls)} clear controls) "
        f"in {output_dir}"
    )


if __name__ == "__main__":
    main()
