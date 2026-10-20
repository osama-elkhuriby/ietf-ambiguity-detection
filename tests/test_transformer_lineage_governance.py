from __future__ import annotations

import json
from pathlib import Path

from models.classical.feature_pipeline import sha256_file


def test_frozen_dns_lineage_audit_has_no_training_lineage_overlap() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (root / "results/transformer/governance/dns_lineage_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["scope"]["test_family"] == "draft-ietf-dnsop-svcb-https"
    assert report["rfc_lineage_pairs"] == []
    assert report["shared_draft_pairs"] == []
    assert report["copied_normative_text_pairs"] == []
    assert report["lineage_overlap_detected"] is False
    assert report["counts"]["shared_boilerplate_pairs"] == 8
    assert {
        (item["training_family"], item["referenced_rfc"])
        for item in report["cross_reference_pairs_not_lineage"]
    } == {
        ("draft-ietf-httpbis-semantics", "RFC9110"),
        ("draft-ietf-quic-http", "RFC9114"),
        ("draft-ietf-tls-tls13", "RFC8446"),
    }
    assert report["source_hashes"]["split_config"] == sha256_file(
        root / "experiments/transformer/configs/split_v2.json"
    )
    assert report["source_hashes"]["rfc_manifest"] == sha256_file(
        root / "corpus/metadata/final_rfc_manifest.csv"
    )


def test_official_rfc_crosscheck_records_no_dns_chain_relationships() -> None:
    root = Path(__file__).resolve().parents[1]
    crosscheck = json.loads(
        (
            root
            / "results/transformer/governance/dns_official_rfc_crosscheck.json"
        ).read_text(encoding="utf-8")
    )
    assert crosscheck["rfc"] == "RFC9460"
    assert crosscheck["source_draft"] == "draft-ietf-dnsop-svcb-https-12"
    for relation in ("obsoletes", "updates", "obsoleted_by", "updated_by"):
        assert crosscheck[relation] == []


def test_governance_plan_freezes_dns_disclosure_and_lr_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = (root / "experiments/transformer/EXPERIMENT_PLAN.md").read_text(
        encoding="utf-8"
    )
    assert "## Selection rule and deviations" in plan
    assert "before any model was evaluated on DNS" in plan
    assert "researcher degree of freedom" in plan
    assert "sample_SD(differences) / sqrt(5)" in plan
    assert "seed 2718" in plan
    assert "TCP is outside CV" in plan
    assert "eight exact BCP 14 boilerplate pairs" in plan
    assert "ordinary exact-duplicate rule still excludes it" in plan
    assert "related protocol clusters" in plan
    assert "RFC 9460" in plan
    assert "154 rows and 34 positives" in plan
