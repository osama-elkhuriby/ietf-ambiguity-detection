from __future__ import annotations

import fetch_errata
import link_draft_rfc
import rfc_ids
from paths import REPO_ROOT, repo_path


def test_repo_path_points_at_project_root() -> None:
    assert (REPO_ROOT / "corpus" / "paths.py").is_file()
    assert repo_path("DECISIONS.md").is_file()


def test_build_links_high_confidence() -> None:
    drafts = {
        "draft-ietf-dnsop-svcb-https": {"rfc_number": 9460},
    }
    rfcs = {"rfc9460": {"draft": "draft-ietf-dnsop-svcb-https"}}
    links = link_draft_rfc.build_links(drafts, rfcs)
    assert len(links) == 1
    assert links[0]["confidence"] == "high"
    assert links[0]["rfc"] == "rfc9460"


def test_build_links_conflict() -> None:
    drafts = {"draft-a": {"rfc_number": 9460}}
    rfcs = {"rfc8259": {"draft": "draft-a"}}
    links = link_draft_rfc.build_links(drafts, rfcs)
    assert links[0]["confidence"] == "conflict"
    assert links[0]["rfc"] is None


def test_build_links_single_source_datatracker() -> None:
    drafts = {"draft-a": {"rfc_number": 9460}}
    links = link_draft_rfc.build_links(drafts, {})
    assert len(links) == 1
    assert links[0]["confidence"] == "single-source"
    assert links[0]["rfc"] == "rfc9460"


def test_build_links_single_source_rfc_index_normalized_name() -> None:
    drafts = {"draft-ietf-dnsop-svcb-https": {}}
    rfcs = {"rfc9460": {"draft": "draft-ietf-dnsop-svcb-https-12"}}
    links = link_draft_rfc.build_links(drafts, rfcs)
    assert len(links) == 1
    assert links[0]["confidence"] == "single-source"
    assert links[0]["rfc"] == "rfc9460"


def test_filter_and_group_errata() -> None:
    bulk = [
        {"doc-id": "RFC9460", "errata_id": 1, "errata_status_code": "Verified", "errata_type_code": "Technical"},
        {"doc-id": "RFC8259", "errata_id": 2, "errata_status_code": "Reported", "errata_type_code": "Editorial"},
        {"doc-id": "draft-ietf-foo", "errata_id": 3},
    ]
    grouped = fetch_errata.filter_and_group(bulk, {9460})
    assert len(grouped["rfc9460"]) == 1
    assert grouped["rfc9460"][0]["errata_id"] == 1
    assert grouped["rfc9460"][0]["status"] == "Verified"


def test_normalize_rfc_id() -> None:
    assert rfc_ids.normalize_rfc_id("9460") == "rfc9460"
    assert rfc_ids.normalize_rfc_id("RFC9460") == "rfc9460"
