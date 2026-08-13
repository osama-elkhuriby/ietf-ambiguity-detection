from __future__ import annotations

from pathlib import Path

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
    assert links[0]["rfc"] == "RFC9460"


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
    assert links[0]["rfc"] == "RFC9460"


def test_build_links_single_source_rfc_index_normalized_name() -> None:
    drafts = {"draft-ietf-dnsop-svcb-https": {}}
    rfcs = {"rfc9460": {"draft": "draft-ietf-dnsop-svcb-https-12"}}
    links = link_draft_rfc.build_links(drafts, rfcs)
    assert len(links) == 1
    assert links[0]["confidence"] == "single-source"
    assert links[0]["rfc"] == "RFC9460"


def test_filter_and_group_errata() -> None:
    bulk = [
        {
            "doc-id": "RFC9460",
            "errata_id": 1,
            "errata_status_code": "Verified",
            "errata_type_code": "Technical",
        },
        {
            "doc-id": "RFC8259",
            "errata_id": 2,
            "errata_status_code": "Reported",
            "errata_type_code": "Editorial",
        },
        {"doc-id": "draft-ietf-foo", "errata_id": 3},
    ]
    grouped = fetch_errata.filter_and_group(bulk, {9460})
    assert len(grouped["rfc9460"]) == 1
    assert grouped["rfc9460"][0]["errata_id"] == 1
    assert grouped["rfc9460"][0]["status"] == "Verified"


def test_filter_verified_technical_errata() -> None:
    bulk = [
        {
            "doc-id": "RFC9460",
            "errata_id": "20",
            "errata_status_code": "Verified",
            "errata_type_code": "Technical",
            "section": "3",
            "orig_text": "ambiguous",
            "correct_text": "clarified",
            "verifier_id": 7,
        },
        {
            "doc-id": "rfc9460",
            "errata_id": "3",
            "errata_status_code": " verified ",
            "errata_type_code": "technical",
        },
        {
            "doc-id": "RFC9460",
            "errata_id": "4",
            "errata_status_code": "Reported",
            "errata_type_code": "Technical",
        },
        {
            "doc-id": "RFC9460",
            "errata_id": "5",
            "errata_status_code": "Verified",
            "errata_type_code": "Editorial",
        },
        {
            "doc-id": "RFC9000",
            "errata_id": "6",
            "errata_status_code": "Verified",
            "errata_type_code": "Technical",
        },
    ]

    grouped = fetch_errata.filter_verified_technical(bulk, {9460, 9113})

    assert list(grouped) == ["rfc9113", "rfc9460"]
    assert grouped["rfc9113"] == []
    assert [entry["errata_id"] for entry in grouped["rfc9460"]] == ["3", "20"]
    assert grouped["rfc9460"][1]["doc_id"] == "RFC9460"
    assert grouped["rfc9460"][1]["verifier_id"] == 7
    assert grouped["rfc9460"][1]["source_url"].endswith("eid20")


def test_rfc_numbers_from_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "draft_family,rfc_number\ndraft-a,RFC9460\ndraft-b,9000\ndraft-c,RFC9460\n",
        encoding="utf-8",
    )

    assert fetch_errata.rfc_numbers_from_manifest(manifest) == [9000, 9460]


def test_verified_technical_payload_counts_empty_rfcs() -> None:
    payload = fetch_errata.build_verified_technical_payload(
        {"rfc9000": [{"errata_id": 1}], "rfc9001": []},
        "manifest.csv",
    )

    assert payload["filters"] == {"status": "Verified", "type": "Technical"}
    assert payload["rfc_count"] == 2
    assert payload["errata_count"] == 1
    assert payload["source_url"] == fetch_errata.ERRATA_URL


def test_normalize_rfc_id() -> None:
    assert rfc_ids.normalize_rfc_id("9460") == "RFC9460"
    assert rfc_ids.normalize_rfc_id("RFC9460") == "RFC9460"


def test_to_repo_relative_roundtrip(tmp_path: Path, monkeypatch) -> None:
    import storage

    monkeypatch.setattr(storage, "REPO_ROOT", tmp_path)
    file_path = tmp_path / "corpus" / "raw" / "drafts" / "draft-a" / "draft-a-00.txt"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("hello", encoding="utf-8")

    relative = storage.to_repo_relative(file_path)
    assert relative == "corpus/raw/drafts/draft-a/draft-a-00.txt"
    assert storage.resolve_local_path(relative) == file_path.resolve()


def test_audit_draft_counts_present_revisions(tmp_path: Path, monkeypatch) -> None:
    import validate_corpus

    draft = "draft-ietf-dnsop-svcb-https"
    draft_dir = tmp_path / "corpus" / "raw" / "drafts" / draft
    draft_dir.mkdir(parents=True)
    (draft_dir / f"{draft}-00.txt").write_text("rev00", encoding="utf-8")
    (draft_dir / f"{draft}-01.txt").write_text("rev01", encoding="utf-8")

    monkeypatch.setattr(validate_corpus, "expected_draft_dir", lambda name: draft_dir if name == draft else tmp_path / name)

    row = {
        "draft_family": draft,
        "draft_revision_count": "2",
        "last_available_revision": "01",
        "rfc_number": "RFC9460",
    }
    audit = validate_corpus.audit_draft(row, drafts={}, links_by_draft={})

    assert audit.revisions_present == 2
    assert audit.revisions_missing == 0
    assert audit.final_rev_present is True


def test_fetch_rfc_txt_format_fallback(monkeypatch, tmp_path: Path) -> None:
    import fetch_rfc

    class FakeEntry:
        doc_id = "RFC9460"
        title = "SVCB"
        stream = "IETF"
        wg = "dnsop"
        area = "ops"
        curr_status = "PROPOSED STANDARD"
        formats = ["HTML", "TXT", "PDF", "XML"]
        draft = "draft-ietf-dnsop-svcb-https-12"
        errata_url = None

        def date(self):
            from datetime import datetime

            return datetime(2023, 11, 1)

        def content_url(self, required_format: str) -> str | None:
            return None

    class FakeResponse:
        status_code = 200
        text = "RFC 9460 plaintext"

    monkeypatch.setattr(fetch_rfc, "RAW_RFC_DIR", tmp_path / "corpus" / "raw" / "rfcs")
    monkeypatch.setattr(fetch_rfc, "to_repo_relative", lambda path: f"corpus/raw/rfcs/{Path(path).name}")
    monkeypatch.setattr(fetch_rfc.requests, "get", lambda url, timeout=30: FakeResponse())

    record = fetch_rfc.fetch_rfc(
        "RFC9460",
        index=type("Idx", (), {"rfc": lambda self, rfc_id: FakeEntry()})(),  # type: ignore[arg-type]
    )
    assert record["local_txt_path"] == "corpus/raw/rfcs/RFC9460.txt"


def test_repair_metadata_paths_clears_stale_entries(tmp_path: Path, monkeypatch) -> None:
    import storage
    import validate_corpus

    monkeypatch.setattr(storage, "REPO_ROOT", tmp_path)
    drafts = {
        "draft-a": {
            "versions": [
                {"rev": "00", "local_txt_path": "/old/machine/draft-a-00.txt"},
            ]
        }
    }
    rfcs = {"RFC99999": {"local_txt_path": "/old/machine/RFC99999.txt"}}

    draft_updates, rfc_updates = validate_corpus.repair_metadata_paths(drafts, rfcs)

    assert draft_updates == 1
    assert rfc_updates == 1
    assert drafts["draft-a"]["versions"][0]["local_txt_path"] is None
    assert rfcs["RFC99999"]["local_txt_path"] is None


def test_backfill_rfc_numbers_from_rfc_index() -> None:
    import sync_metadata

    drafts = {
        "draft-ietf-dnsop-svcb-https": {"name": "draft-ietf-dnsop-svcb-https", "rfc_number": None},
    }
    rfcs = {
        "RFC9460": {"draft": "draft-ietf-dnsop-svcb-https-12"},
    }

    count, families = sync_metadata.backfill_rfc_numbers(drafts, rfcs)

    assert count == 1
    assert drafts["draft-ietf-dnsop-svcb-https"]["rfc_number"] == 9460
    assert "draft-ietf-dnsop-svcb-https" in families


def test_rfc_only_draft_not_listed_as_missing_gap() -> None:
    import validate_corpus

    row = {
        "draft_family": "draft-ietf-ospf-vers2",
        "draft_revision_count": "0",
        "rfc_number": "RFC2328",
    }
    audit = validate_corpus.audit_draft(row, drafts={"draft-ietf-ospf-vers2": {}}, links_by_draft={})
    assert audit.collection_mode == "rfc-only"
    assert audit.revisions_missing == 0
