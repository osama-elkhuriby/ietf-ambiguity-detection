"""Create a small, lineage-deduplicated TLS human-review batch.

This never changes automatic silver artifacts.  It copies selected review rows
into reviewer-specific CSVs and an empty adjudication sheet.
"""

from __future__ import annotations

import csv
import hashlib
import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FAMILY_DIR = ROOT / "corpus/processed/ground_truth/draft-ietf-tls-tls13"
SOURCE = FAMILY_DIR / "review/tls13_ground_truth_review_template.csv"
OUTPUT_DIR = FAMILY_DIR / "review/pilot_v1"
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
)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_one_per_lineage(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep the latest revision for every lineage, with stable tie-breaking."""
    chosen: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: (int(r["revision"]), r["record_id"])):
        chosen[row["lineage_ids"]] = row
    return sorted(chosen.values(), key=lambda r: (r["weak_label"], r["record_id"]))


def blank_human_fields(row: dict[str, str]) -> dict[str, str]:
    result = dict(row)
    for field in HUMAN_FIELDS:
        result[field] = "PENDING" if field == "adjudication_status" else ""
    return result


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create TLS CSVs for independent human review."
    )
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help="Include every silver candidate rather than one record per lineage.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for reviewer_a.csv, reviewer_b.csv, and adjudication.csv.",
    )
    args = parser.parse_args()
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields:
            raise ValueError(f"Missing header in {SOURCE}")
        rows = list(reader)

    all_positives = [
        row for row in rows if row["weak_label"] == "SILVER_POSITIVE_CANDIDATE"
    ]
    all_controls = [
        row for row in rows if row["weak_label"] == "SILVER_CLEAR_CANDIDATE"
    ]
    if args.all_candidates:
        positives, controls = all_positives, all_controls
        selection = "all generated silver candidate records"
    else:
        positives = choose_one_per_lineage(all_positives)
        controls = choose_one_per_lineage(all_controls)[: len(positives) * 2]
        selection = (
            "one latest-revision record per positive lineage and two clear-control "
            "lineages per positive lineage"
        )
    batch = sorted(
        [*positives, *controls], key=lambda r: (r["weak_label"], r["record_id"])
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
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_hash(SOURCE),
        "selection": {
            "method": selection,
        },
        "records": len(batch),
        "positive_records": len(positives),
        "clear_control_records": len(controls),
        "record_ids": [row["record_id"] for row in batch],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Created {len(batch)} records ({len(positives)} positive candidates, "
        f"{len(controls)} clear controls) in {output_dir}"
    )


if __name__ == "__main__":
    main()
