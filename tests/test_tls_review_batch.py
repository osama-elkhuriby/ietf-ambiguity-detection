from __future__ import annotations

import csv
import json
from pathlib import Path

from corpus.create_tls_review_batch import (
    attach_first_stage_classifications,
    choose_one_per_lineage,
    classified_entirely_as,
)


def review_row(
    record_id: str,
    revision: str,
    lineage_id: str,
    text: str = "The client MUST continue.",
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "revision": revision,
        "lineage_ids": lineage_id,
        "weak_label": "SILVER_POSITIVE_CANDIDATE",
        "clause_text": text,
    }


def test_choose_one_per_lineage_keeps_latest_revision() -> None:
    rows = [
        review_row("older", "07", "lineage:a"),
        review_row("other", "12", "lineage:b"),
        review_row("latest", "28", "lineage:a"),
    ]

    chosen = choose_one_per_lineage(rows)

    assert {row["record_id"] for row in chosen} == {"latest", "other"}
    assert len({row["lineage_ids"] for row in chosen}) == len(chosen)


def test_review_filter_reuses_first_stage_classifications() -> None:
    rows = [
        {**review_row("normative", "01", "lineage:a"), "clause_ids": "c1"},
        {**review_row("mixed", "01", "lineage:b"), "clause_ids": "c1|c2"},
    ]

    enriched = attach_first_stage_classifications(
        rows, {"c1": "NORMATIVE", "c2": "CONTEXTUAL"}
    )

    assert enriched[0]["classification"] == "NORMATIVE"
    assert enriched[1]["classification"] == "NORMATIVE|CONTEXTUAL"
    assert classified_entirely_as(enriched[0], "NORMATIVE")
    assert not classified_entirely_as(enriched[1], "NORMATIVE")


def test_saved_normative_review_batch_has_one_record_per_lineage() -> None:
    root = Path(__file__).resolve().parents[1]
    review_dir = (
        root
        / "corpus/processed/ground_truth/draft-ietf-tls-tls13/review"
        / "all_candidates_v1"
    )
    reviewer_path = review_dir / "reviewer_a.csv"
    with reviewer_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((review_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == manifest["records"]
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert len({row["lineage_ids"] for row in rows}) == len(rows)
    assert all(row["classification"] == "NORMATIVE" for row in rows)
    assert all(
        classified_entirely_as(row, "NORMATIVE")
        for row in rows
    )
    assert manifest["selection"]["grouping_key"] == "lineage_ids"
    assert "first-stage classifier" in manifest["selection"]["filters"][0]
    assert manifest["unique_lineages"] == len(rows)
