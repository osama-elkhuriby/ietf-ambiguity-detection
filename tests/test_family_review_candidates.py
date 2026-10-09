from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from corpus.create_tls_review_batch import classified_entirely_as

FAMILY_COUNTS = {
    "draft-ietf-6man-rfc2460bis": 1,
    "draft-ietf-dnsop-svcb-https": 163,
    "draft-ietf-httpbis-cache-header": 5,
    "draft-ietf-httpbis-http2bis": 266,
    "draft-ietf-httpbis-semantics": 420,
    "draft-ietf-quic-http": 351,
    "draft-ietf-quic-recovery": 172,
    "draft-ietf-quic-tls": 200,
    "draft-ietf-quic-transport": 842,
    "draft-ietf-tcpm-rfc793bis": 143,
}


@pytest.mark.parametrize(("family_id", "expected"), FAMILY_COUNTS.items())
def test_generated_family_review_package_is_normative_and_deduplicated(
    family_id: str, expected: int
) -> None:
    root = Path(__file__).resolve().parents[1]
    review_dir = (
        root
        / "corpus/processed/ground_truth"
        / family_id
        / "review/all_candidates_v1"
    )
    with (review_dir / "reviewer_a.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads((review_dir / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == manifest["records"] == expected
    assert {row["weak_label"] for row in rows} <= {
        "SILVER_CHANGE_CANDIDATE",
        "SILVER_CLEAR_CANDIDATE",
    }
    assert all(row["classification"] == "NORMATIVE" for row in rows)
    assert all(classified_entirely_as(row, "NORMATIVE") for row in rows)
    assert all(row["ambiguity_label"] == "UNCERTAIN" for row in rows)
    assert all(row["ground_truth"] == "False" for row in rows)
    assert all(row["adjudication_status"] == "PENDING" for row in rows)

    lineage_ids = [
        lineage_id
        for row in rows
        for lineage_id in row["lineage_ids"].split("|")
    ]
    assert len(lineage_ids) == len(set(lineage_ids))
    assert len({row["record_id"] for row in rows}) == len(rows)
    assert manifest["classification_records"] > 0
    assert manifest["change_candidate_records"] + manifest["clear_control_records"] == expected

