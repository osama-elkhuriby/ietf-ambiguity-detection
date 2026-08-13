"""
Metadata synchronizer — normalize manifests after fetch/build steps.

Runs offline over local metadata and plaintext. Typical post-build workflow:

    python corpus/sync_metadata.py
    python corpus/sync_metadata.py --refresh-manifest --write-audit

Steps:
  1. Backfill ``rfc_number`` in drafts.json from rfcs.json (RFC index cross-reference)
  2. Rebuild draft_rfc_links.json (expect high-confidence joins after backfill)
  3. Optionally refresh final_rfc_manifest.csv from on-disk state
  4. Optionally write corpus_audit.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import link_draft_rfc
from paths import REPO_ROOT
from rfc_ids import normalize_rfc_id, parse_rfc_number
from storage import (
    DRAFTS_METADATA_PATH,
    FINAL_RFC_MANIFEST_PATH,
    LINKS_METADATA_PATH,
    RFCS_METADATA_PATH,
    RFC_ONLY_DRAFT_FAMILIES,
    draft_collection_mode,
    expected_draft_dir,
    expected_rfc_path,
    resolve_local_path,
    to_repo_relative,
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_draft_family(draft_name: str) -> str:
    return re.sub(r"-\d+$", "", draft_name)


def family_to_rfc_from_index(rfcs: dict[str, Any]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for rfc_id, record in rfcs.items():
        draft_name = record.get("draft")
        if not draft_name:
            continue
        number = parse_rfc_number(rfc_id)
        if number is None:
            continue
        family = normalize_draft_family(str(draft_name))
        mapping[family] = number
    return mapping


def backfill_rfc_numbers(
    drafts: dict[str, Any], rfcs: dict[str, Any]
) -> tuple[int, list[str]]:
    """Set ``rfc_number`` on draft records when the RFC index provides it."""
    index_map = family_to_rfc_from_index(rfcs)
    updated: list[str] = []

    for family, record in drafts.items():
        if record.get("rfc_number") is not None:
            continue
        number = index_map.get(family)
        if number is None:
            continue
        record["rfc_number"] = number
        updated.append(family)

    return len(updated), updated


def rebuild_links(drafts: dict[str, Any], rfcs: dict[str, Any]) -> list[dict[str, Any]]:
    links = link_draft_rfc.build_links(drafts, rfcs)
    save_json(LINKS_METADATA_PATH, links)
    return links


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_final_revision_file(draft_family: str, final_rev: str | None) -> Path | None:
    if not final_rev:
        return None
    draft_dir = expected_draft_dir(draft_family)
    for suffix in (".txt", ".xml", ".html"):
        candidate = draft_dir / f"{draft_family}-{final_rev}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def count_revision_files(draft_family: str) -> tuple[int, str | None, str | None]:
    draft_dir = expected_draft_dir(draft_family)
    if not draft_dir.is_dir():
        return 0, None, None
    files = sorted(
        path
        for path in draft_dir.iterdir()
        if path.is_file() and path.stat().st_size > 0
    )
    if not files:
        return 0, None, None
    latest = files[-1]
    rev = latest.stem.rsplit("-", 1)[-1]
    return len(files), rev, latest.name


def refresh_manifest_csv(
    manifest_path: Path = FINAL_RFC_MANIFEST_PATH,
    drafts: dict[str, Any] | None = None,
    rfcs: dict[str, Any] | None = None,
    links: list[dict[str, Any]] | None = None,
) -> int:
    """Update manifest provenance columns from the current on-disk corpus."""
    drafts = drafts if drafts is not None else load_json(DRAFTS_METADATA_PATH)
    rfcs = rfcs if rfcs is not None else load_json(RFCS_METADATA_PATH)
    links = links if links is not None else json.loads(LINKS_METADATA_PATH.read_text(encoding="utf-8"))
    links_by_draft = {entry["draft"]: entry for entry in links if entry.get("draft")}

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{manifest_path} is missing a header row.")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    updated_rows: list[dict[str, str]] = []
    for row in rows:
        family = row["draft_family"]
        mode = draft_collection_mode(family)
        count, last_rev, final_file = count_revision_files(family)
        draft_record = drafts.get(family, {})

        if mode == "full-history":
            row["draft_revision_count"] = str(count or len(draft_record.get("versions") or []))
            if count:
                revs = sorted(
                    version.get("rev")
                    for version in draft_record.get("versions") or []
                    if version.get("rev") is not None
                )
                row["first_revision"] = revs[0] if revs else "00"
                row["last_available_revision"] = last_rev or (revs[-1] if revs else "")
                row["final_draft_present"] = "True" if final_file else "False"
                row["final_draft_files"] = final_file or ""
            else:
                row["final_draft_present"] = "False"
                row["final_draft_files"] = ""

        rfc_id = normalize_rfc_id(row["rfc_number"])
        rfc_record = rfcs.get(rfc_id) or {}
        rfc_path = resolve_local_path(rfc_record.get("local_txt_path")) or expected_rfc_path(rfc_id)
        if rfc_path.is_file():
            row["rfc_text_present"] = "True"
            row["rfc_text_filename"] = rfc_path.name
            row["rfc_text_size_bytes"] = str(rfc_path.stat().st_size)
            row["rfc_text_sha256"] = sha256_file(rfc_path)
        else:
            row["rfc_text_present"] = "False"
            row["rfc_text_filename"] = ""
            row["rfc_text_size_bytes"] = ""
            row["rfc_text_sha256"] = ""

        link = links_by_draft.get(family)
        row["match_verified"] = "True" if link and link.get("confidence") == "high" else "False"
        updated_rows.append(row)

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    return len(updated_rows)


def sync_all(
    refresh_manifest: bool = False,
    write_audit: bool = False,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    drafts = load_json(DRAFTS_METADATA_PATH)
    rfcs = load_json(RFCS_METADATA_PATH)

    if not drafts:
        raise SystemExit(f"{DRAFTS_METADATA_PATH} is empty — run a fetch step first.")
    if not rfcs:
        raise SystemExit(f"{RFCS_METADATA_PATH} is empty — run fetch_rfc.py first.")

    backfilled, families = backfill_rfc_numbers(drafts, rfcs)
    if backfilled:
        save_json(DRAFTS_METADATA_PATH, drafts)

    links = rebuild_links(drafts, rfcs)
    manifest_rows = 0
    if refresh_manifest:
        manifest_rows = refresh_manifest_csv(drafts=drafts, rfcs=rfcs, links=links)

    audit_written = None
    if write_audit:
        import validate_corpus

        audit_path = audit_path or REPO_ROOT / "corpus" / "metadata" / "corpus_audit.json"
        report = validate_corpus.build_report()
        validate_corpus.save_json(audit_path, report)
        audit_written = str(audit_path)

    return {
        "backfilled_rfc_numbers": backfilled,
        "backfilled_families": families,
        "links_total": len(links),
        "links_high": sum(1 for link in links if link.get("confidence") == "high"),
        "links_single_source": sum(1 for link in links if link.get("confidence") == "single-source"),
        "links_conflict": sum(1 for link in links if link.get("confidence") == "conflict"),
        "manifest_rows_updated": manifest_rows,
        "audit_written": audit_written,
        "rfc_only_families": sorted(RFC_ONLY_DRAFT_FAMILIES),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synchronize corpus metadata after fetch/build (offline)."
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Rewrite provenance columns in final_rfc_manifest.csv from on-disk state.",
    )
    parser.add_argument(
        "--write-audit",
        action="store_true",
        help="Write corpus/metadata/corpus_audit.json after synchronization.",
    )
    parser.add_argument(
        "--audit-path",
        type=Path,
        default=REPO_ROOT / "corpus" / "metadata" / "corpus_audit.json",
        help="Output path used with --write-audit.",
    )
    args = parser.parse_args()

    result = sync_all(
        refresh_manifest=args.refresh_manifest,
        write_audit=args.write_audit,
        audit_path=args.audit_path,
    )

    print("=== Metadata sync complete ===")
    print(f"RFC numbers backfilled: {result['backfilled_rfc_numbers']}")
    if result["backfilled_families"]:
        print(f"  families: {', '.join(result['backfilled_families'])}")
    print(
        f"Draft↔RFC links: {result['links_total']} total "
        f"(high={result['links_high']}, single-source={result['links_single_source']}, "
        f"conflict={result['links_conflict']})"
    )
    if result["manifest_rows_updated"]:
        print(f"Manifest rows updated: {result['manifest_rows_updated']}")
    if result["audit_written"]:
        print(f"Audit report: {result['audit_written']}")
    print(f"RFC-only families (by design): {', '.join(result['rfc_only_families'])}")


if __name__ == "__main__":
    main()
