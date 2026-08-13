"""
Corpus validator — audit structured storage, download completeness, and protocol coverage.

Checks metadata manifests against corpus/metadata/final_rfc_manifest.csv, verifies that
referenced plaintext files exist on disk, and reports gaps by protocol area.

Usage:
    python corpus/validate_corpus.py
    python corpus/validate_corpus.py --json corpus/metadata/corpus_audit.json
    python corpus/validate_corpus.py --repair-paths
    python corpus/validate_corpus.py --relocate-cache
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from paths import REPO_ROOT
from rfc_ids import normalize_rfc_id, parse_rfc_number
from storage import (
    DRAFTS_METADATA_PATH,
    ERRATA_METADATA_PATH,
    FINAL_RFC_MANIFEST_PATH,
    LINKS_METADATA_PATH,
    PROTOCOL_AREA,
    RAW_DRAFTS_DIR,
    RAW_ERRATA_DIR,
    RAW_RFCS_DIR,
    RFCS_METADATA_PATH,
    RFC_ONLY_DRAFT_FAMILIES,
    VERIFIED_TECHNICAL_ERRATA_PATH,
    draft_collection_mode,
    expected_draft_dir,
    expected_rfc_path,
    relocate_misplaced_rfc_index_cache,
    resolve_local_path,
    to_repo_relative,
)

STALE_PATH_PREFIXES = (
    "/media/yahya/",
    "/media/osama/",
)


@dataclass
class RevisionStatus:
    rev: str
    expected_path: str | None
    metadata_path: str | None
    exists: bool
    stale_path: bool = False


@dataclass
class DraftAudit:
    draft: str
    protocol_area: str
    collection_mode: str
    in_metadata: bool
    revision_count_expected: int | None
    revision_count_metadata: int
    revisions_present: int
    revisions_missing: int
    final_rev_expected: str | None
    final_rev_present: bool
    rfc_number: int | None
    link_confidence: str | None
    revisions: list[RevisionStatus] = field(default_factory=list)


@dataclass
class RfcAudit:
    rfc_id: str
    protocol_area: str
    in_metadata: bool
    metadata_path: str | None
    expected_path: str
    exists: bool
    stale_path: bool = False


@dataclass
class ProtocolCoverage:
    protocol_area: str
    families: int
    drafts_in_metadata: int
    drafts_complete: int
    rfcs_in_metadata: int
    rfcs_complete: int


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(path: Path = FINAL_RFC_MANIFEST_PATH) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def normalize_draft_name(name: str) -> str:
    return re.sub(r"-\d+$", "", name)


def manifest_protocol_area(row: dict[str, str]) -> str:
    family = row["draft_family"]
    return PROTOCOL_AREA.get(family, row.get("area") or "unknown")


def find_revision_file(draft_dir: Path, draft_name: str, rev: str) -> Path | None:
    if not draft_dir.is_dir():
        return None
    for suffix in (".txt", ".xml", ".html"):
        candidate = draft_dir / f"{draft_name}-{rev}{suffix}"
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def is_stale_path(path: str | None) -> bool:
    if not path:
        return False
    if path.startswith(STALE_PATH_PREFIXES):
        return True
    resolved = resolve_local_path(path)
    return resolved is not None and not resolved.exists()


def audit_draft(
    row: dict[str, str],
    drafts: dict[str, Any],
    links_by_draft: dict[str, dict[str, Any]],
) -> DraftAudit:
    draft = row["draft_family"]
    protocol_area = manifest_protocol_area(row)
    record = drafts.get(draft, {})
    versions = record.get("versions") or []

    expected_count_raw = (row.get("draft_revision_count") or "").strip()
    expected_count = int(expected_count_raw) if expected_count_raw.isdigit() else None
    collection_mode = draft_collection_mode(draft, expected_count)
    final_rev = (row.get("last_available_revision") or row.get("final_draft_revision") or "").strip()
    if final_rev:
        final_rev = final_rev.split("-")[-1]

    draft_dir = expected_draft_dir(draft)
    revision_statuses: list[RevisionStatus] = []
    present = 0
    missing = 0

    if expected_count is not None and expected_count > 0:
        rev_width = len(final_rev) if final_rev else 2
        for index in range(expected_count):
            rev = f"{index:0{rev_width}d}"
            found = find_revision_file(draft_dir, draft, rev)
            metadata_rev = next((v for v in versions if v.get("rev") == rev), None)
            metadata_path = metadata_rev.get("local_txt_path") if metadata_rev else None
            exists = found is not None
            if exists:
                present += 1
            else:
                missing += 1
            revision_statuses.append(
                RevisionStatus(
                    rev=rev,
                    expected_path=to_repo_relative(found) if found else None,
                    metadata_path=metadata_path,
                    exists=exists,
                    stale_path=is_stale_path(metadata_path),
                )
            )
    elif versions:
        for version in versions:
            rev = version.get("rev", "?")
            metadata_path = version.get("local_txt_path")
            resolved = resolve_local_path(metadata_path)
            found = find_revision_file(draft_dir, draft, rev)
            exists = (resolved is not None and resolved.is_file()) or found is not None
            if exists:
                present += 1
            else:
                missing += 1
            revision_statuses.append(
                RevisionStatus(
                    rev=rev,
                    expected_path=to_repo_relative(found or resolved),
                    metadata_path=metadata_path,
                    exists=exists,
                    stale_path=is_stale_path(metadata_path),
                )
            )
    else:
        if collection_mode == "rfc-only":
            missing = 0
        else:
            missing = 1

    final_present = False
    if final_rev:
        final_present = find_revision_file(draft_dir, draft, final_rev) is not None

    link = links_by_draft.get(draft)
    rfc_number = parse_rfc_number(row.get("rfc_number") or "")
    if rfc_number is None and link and link.get("rfc"):
        rfc_number = parse_rfc_number(str(link["rfc"]))

    return DraftAudit(
        draft=draft,
        protocol_area=protocol_area,
        collection_mode=collection_mode,
        in_metadata=draft in drafts,
        revision_count_expected=expected_count,
        revision_count_metadata=len(versions),
        revisions_present=present,
        revisions_missing=missing,
        final_rev_expected=final_rev or None,
        final_rev_present=final_present,
        rfc_number=rfc_number,
        link_confidence=link.get("confidence") if link else None,
        revisions=revision_statuses,
    )


def audit_rfc(row: dict[str, str], rfcs: dict[str, Any]) -> RfcAudit:
    draft = row["draft_family"]
    protocol_area = manifest_protocol_area(row)
    rfc_number = normalize_rfc_id(row["rfc_number"])
    record = rfcs.get(rfc_number) or rfcs.get(rfc_number.lower()) or {}
    metadata_path = record.get("local_txt_path")
    expected = expected_rfc_path(rfc_number)
    resolved = resolve_local_path(metadata_path)
    exists = expected.is_file() or (resolved is not None and resolved.is_file())

    return RfcAudit(
        rfc_id=rfc_number,
        protocol_area=protocol_area,
        in_metadata=bool(record),
        metadata_path=metadata_path,
        expected_path=to_repo_relative(expected) or expected.as_posix(),
        exists=exists,
        stale_path=is_stale_path(metadata_path),
    )


def draft_family_complete(d: DraftAudit, rfc_exists: bool) -> bool:
    if d.collection_mode == "rfc-only":
        return d.in_metadata and rfc_exists
    return d.in_metadata and d.revisions_missing == 0 and d.revisions_present > 0


def build_protocol_coverage(
    draft_audits: list[DraftAudit], rfc_audits: list[RfcAudit]
) -> list[ProtocolCoverage]:
    areas = sorted(set(PROTOCOL_AREA.values()))
    coverage: list[ProtocolCoverage] = []
    rfc_by_id = {r.rfc_id: r for r in rfc_audits}

    for area in areas:
        area_drafts = [d for d in draft_audits if d.protocol_area == area]
        area_rfcs = [r for r in rfc_audits if r.protocol_area == area]
        drafts_complete = 0
        for draft_audit in area_drafts:
            rfc_id = (
                normalize_rfc_id(str(draft_audit.rfc_number))
                if draft_audit.rfc_number is not None
                else None
            )
            rfc_exists = bool(rfc_id and rfc_by_id.get(rfc_id) and rfc_by_id[rfc_id].exists)
            if draft_family_complete(draft_audit, rfc_exists=rfc_exists):
                drafts_complete += 1
        rfcs_complete = sum(1 for r in area_rfcs if r.exists)
        coverage.append(
            ProtocolCoverage(
                protocol_area=area,
                families=len(area_drafts),
                drafts_in_metadata=sum(1 for d in area_drafts if d.in_metadata),
                drafts_complete=drafts_complete,
                rfcs_in_metadata=sum(1 for r in area_rfcs if r.in_metadata),
                rfcs_complete=rfcs_complete,
            )
        )
    return coverage


def repair_metadata_paths(drafts: dict[str, Any], rfcs: dict[str, Any]) -> tuple[int, int]:
    draft_updates = 0
    rfc_updates = 0

    for draft_name, record in drafts.items():
        draft_dir = expected_draft_dir(draft_name)
        for version in record.get("versions") or []:
            rev = version.get("rev")
            if not rev:
                continue
            found = find_revision_file(draft_dir, draft_name, rev)
            new_path = to_repo_relative(found) if found else None
            if new_path and version.get("local_txt_path") != new_path:
                version["local_txt_path"] = new_path
                draft_updates += 1
            elif not found and version.get("local_txt_path"):
                resolved = resolve_local_path(version["local_txt_path"])
                if resolved is None or not resolved.exists():
                    version["local_txt_path"] = None
                    draft_updates += 1

    for rfc_id, record in rfcs.items():
        expected = expected_rfc_path(rfc_id)
        new_path = to_repo_relative(expected) if expected.is_file() else None
        if new_path and record.get("local_txt_path") != new_path:
            record["local_txt_path"] = new_path
            rfc_updates += 1
        elif not expected.is_file() and record.get("local_txt_path"):
            resolved = resolve_local_path(record["local_txt_path"])
            if resolved is None or not resolved.exists():
                record["local_txt_path"] = None
                rfc_updates += 1

    return draft_updates, rfc_updates


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print("=== Corpus audit summary ===")
    print(f"Manifest families:     {summary['manifest_families']}")
    print(f"Draft revisions:       {summary['draft_revisions_present']}/{summary['draft_revisions_expected']} present")
    print(f"RFC plaintext:         {summary['rfcs_present']}/{summary['rfcs_expected']} present")
    print(f"Metadata stale paths:  {summary['stale_metadata_paths']}")
    print(f"Link confidence:       high={summary['links_high']} single-source={summary['links_single_source']} conflict={summary['links_conflict']}")
    print(f"Verified technical:    {summary['verified_technical_errata']} errata across {summary['verified_technical_rfcs']} RFCs")
    if summary.get("rfc_only_families"):
        print(f"RFC-only families:     {summary['rfc_only_families']} (no Datatracker revision history)")
    print()
    print("=== Protocol coverage ===")
    for row in report["protocol_coverage"]:
        print(
            f"  {row['protocol_area']:12}  "
            f"drafts {row['drafts_complete']}/{row['families']} complete,  "
            f"RFCs {row['rfcs_complete']}/{row['families']} present"
        )

    gaps = report["gaps"]
    if gaps["missing_drafts"]:
        print()
        print("=== Draft families needing catch-up ===")
        for item in gaps["missing_drafts"]:
            print(
                f"  {item['draft']}: missing {item['revisions_missing']} revisions "
                f"({item['protocol_area']})"
            )
    if gaps["missing_rfcs"]:
        print()
        print("=== RFC plaintext missing ===")
        for item in gaps["missing_rfcs"]:
            print(f"  {item['rfc_id']} ({item['protocol_area']})")


def build_report(repair_paths: bool = False, relocate_cache: bool = False) -> dict[str, Any]:
    if relocate_cache:
        relocated = relocate_misplaced_rfc_index_cache()
    else:
        relocated = False

    manifest_rows = load_manifest()
    drafts = load_json(DRAFTS_METADATA_PATH) or {}
    rfcs = load_json(RFCS_METADATA_PATH) or {}
    links = load_json(LINKS_METADATA_PATH) or []
    errata = load_json(ERRATA_METADATA_PATH) or {}
    verified = load_json(VERIFIED_TECHNICAL_ERRATA_PATH) or {}

    links_by_draft = {entry["draft"]: entry for entry in links if entry.get("draft")}

    if repair_paths and (drafts or rfcs):
        draft_updates, rfc_updates = repair_metadata_paths(drafts, rfcs)
        if draft_updates or rfc_updates:
            save_json(DRAFTS_METADATA_PATH, drafts)
            save_json(RFCS_METADATA_PATH, rfcs)
    else:
        draft_updates, rfc_updates = 0, 0

    draft_audits = [audit_draft(row, drafts, links_by_draft) for row in manifest_rows]
    rfc_audits = [audit_rfc(row, rfcs) for row in manifest_rows]
    protocol_coverage = build_protocol_coverage(draft_audits, rfc_audits)

    draft_revisions_expected = sum(
        d.revision_count_expected or d.revision_count_metadata for d in draft_audits
    )
    draft_revisions_present = sum(d.revisions_present for d in draft_audits)
    stale_paths = sum(1 for d in draft_audits for r in d.revisions if r.stale_path)
    stale_paths += sum(1 for r in rfc_audits if r.stale_path)

    verified_by_rfc = verified.get("by_rfc") if isinstance(verified, dict) else None
    if verified_by_rfc is None and isinstance(verified, dict) and "rfc2328" in verified:
        verified_by_rfc = verified

    report = {
        "repo_root": str(REPO_ROOT),
        "storage_layout": {
            "drafts": to_repo_relative(RAW_DRAFTS_DIR),
            "rfcs": to_repo_relative(RAW_RFCS_DIR),
            "errata_cache": to_repo_relative(RAW_ERRATA_DIR / "errata_full.json"),
        },
        "summary": {
            "manifest_families": len(manifest_rows),
            "draft_revisions_expected": draft_revisions_expected,
            "draft_revisions_present": draft_revisions_present,
            "rfcs_expected": len(rfc_audits),
            "rfcs_present": sum(1 for r in rfc_audits if r.exists),
            "stale_metadata_paths": stale_paths,
            "links_high": sum(1 for link in links if link.get("confidence") == "high"),
            "links_single_source": sum(1 for link in links if link.get("confidence") == "single-source"),
            "links_conflict": sum(1 for link in links if link.get("confidence") == "conflict"),
            "verified_technical_errata": verified.get("errata_count", 0) if isinstance(verified, dict) else 0,
            "verified_technical_rfcs": verified.get("rfc_count", 0) if isinstance(verified, dict) else 0,
            "metadata_path_repairs": draft_updates + rfc_updates,
            "relocated_rfc_index_cache": relocated,
            "errata_rfc_keys": len(errata) if isinstance(errata, dict) else 0,
            "rfc_only_families": len(RFC_ONLY_DRAFT_FAMILIES),
        },
        "protocol_coverage": [asdict(row) for row in protocol_coverage],
        "drafts": [asdict(row) for row in draft_audits],
        "rfcs": [asdict(row) for row in rfc_audits],
        "gaps": {
            "missing_drafts": [
                {
                    "draft": d.draft,
                    "protocol_area": d.protocol_area,
                    "collection_mode": d.collection_mode,
                    "revisions_missing": d.revisions_missing,
                    "final_rev_present": d.final_rev_present,
                }
                for d in draft_audits
                if d.revisions_missing > 0
                or (not d.in_metadata and d.collection_mode != "rfc-only")
            ],
            "missing_rfcs": [
                {"rfc_id": r.rfc_id, "protocol_area": r.protocol_area}
                for r in rfc_audits
                if not r.exists
            ],
            "single_source_links": [
                {"draft": link["draft"], "rfc": link.get("rfc"), "note": link.get("note")}
                for link in links
                if link.get("confidence") == "single-source"
            ],
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit corpus storage, completeness, and coverage.")
    parser.add_argument("--json", type=Path, help="Write the full audit report to this JSON file.")
    parser.add_argument(
        "--repair-paths",
        action="store_true",
        help="Rewrite stale local_txt_path entries to repo-relative paths when files exist.",
    )
    parser.add_argument(
        "--relocate-cache",
        action="store_true",
        help="Move legacy corpus/corpus/raw/rfcs/.rfcindex-cache into the canonical tree.",
    )
    args = parser.parse_args()

    report = build_report(repair_paths=args.repair_paths, relocate_cache=args.relocate_cache)
    print_summary(report)

    if args.json:
        save_json(args.json, report)
        print(f"\nWrote audit report to {args.json}")


if __name__ == "__main__":
    main()
