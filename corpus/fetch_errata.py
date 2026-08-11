"""
Errata fetcher — downloads the RFC Editor's full errata database once (bulk JSON,
cached locally), filters it to the RFCs already in corpus/metadata/rfcs.json, and writes
the result to corpus/metadata/errata.json.

Endpoint and schema verified by fetching https://www.rfc-editor.org/errata.json directly
(see ../DECISIONS.md, 2026-08-08) and cross-checked against the official IETF process
description in draft-rpc-errata-process. Confirmed fields per entry:
    errata_id, doc-id, errata_status_code, errata_type_code, section,
    orig_text, correct_text, notes, submit_date, submitter_name,
    verifier_id, verifier_name, update_date
errata_status_code is one of: Reported / Verified / Held for Document Update / Rejected.
errata_type_code is one of: Technical / Editorial.

Usage:
    python corpus/fetch_errata.py                # filter to RFCs already in metadata/rfcs.json
    python corpus/fetch_errata.py 9460 8259       # filter to specific RFC numbers instead
    python corpus/fetch_errata.py --refresh       # force re-download of the bulk file
    python corpus/fetch_errata.py --verified-technical
        # Verified Technical only; target RFCs come from final_rfc_manifest.csv

Note: the bulk file is large (the full history of RFC errata) and is cached at
corpus/raw/errata/errata_full.json for 1 day, similar to how ietfdata caches rfc-index.xml
— re-run with --refresh to force an update sooner. Needs normal internet access to run.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import requests
from paths import REPO_ROOT, repo_path
from requests.adapters import HTTPAdapter
from rfc_ids import normalize_rfc_number, parse_rfc_number
from urllib3.util.retry import Retry

ERRATA_URL = "https://www.rfc-editor.org/errata.json"
RAW_CACHE_PATH = repo_path("corpus", "raw", "errata", "errata_full.json")
RFCS_METADATA_PATH = repo_path("corpus", "metadata", "rfcs.json")
ERRATA_METADATA_PATH = repo_path("corpus", "metadata", "errata.json")
FINAL_RFC_MANIFEST_PATH = repo_path("corpus", "metadata", "final_rfc_manifest.csv")
VERIFIED_TECHNICAL_ERRATA_PATH = repo_path(
    "corpus", "metadata", "verified_technical_errata.json"
)
CACHE_MAX_AGE_SECONDS = (
    24 * 60 * 60
)  # 1 day, matching ietfdata's own rfc-index.xml cache policy


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON deterministically without leaving a partial target file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def validate_bulk_errata(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list) or any(not isinstance(entry, dict) for entry in data):
        raise ValueError("RFC Editor errata export must be a JSON array of objects")
    return data


def request_session() -> requests.Session:
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        status=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session = requests.Session()
    session.mount("https://", adapter)
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "ietf-ambiguity-detection/1.0 (research corpus fetcher)",
        }
    )
    return session


def download_bulk_errata(force: bool = False) -> list[dict[str, Any]]:
    if not force and RAW_CACHE_PATH.exists():
        age = time.time() - RAW_CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            print(f"Using cached {RAW_CACHE_PATH} ({age / 3600:.1f}h old)")
            cached = json.loads(RAW_CACHE_PATH.read_text(encoding="utf-8"))
            return validate_bulk_errata(cached)

    print(f"Downloading {ERRATA_URL} ...")
    with request_session() as session:
        response = session.get(ERRATA_URL, timeout=(15, 120))
        response.raise_for_status()
        all_errata = validate_bulk_errata(response.json())

    write_json_atomic(RAW_CACHE_PATH, all_errata)
    print(f"Cached {len(all_errata)} total errata records to {RAW_CACHE_PATH}")
    return all_errata


def rfc_numbers_from_rfcs_metadata() -> list[int]:
    if not RFCS_METADATA_PATH.exists():
        raise SystemExit(
            f"{RFCS_METADATA_PATH} doesn't exist yet — run fetch_rfc.py first, "
            f"or pass explicit RFC numbers."
        )
    rfcs = json.loads(RFCS_METADATA_PATH.read_text())
    numbers = []
    for rfc_id in rfcs:
        numbers.append(normalize_rfc_number(rfc_id))
    return numbers


def rfc_numbers_from_manifest(path: Path = FINAL_RFC_MANIFEST_PATH) -> list[int]:
    if not path.exists():
        raise SystemExit(f"{path} does not exist; pass RFC numbers explicitly instead.")

    numbers: list[int] = []
    with path.open(newline="", encoding="utf-8-sig") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames is None or "rfc_number" not in reader.fieldnames:
            raise SystemExit(f"{path} must contain an 'rfc_number' column.")
        for row in reader:
            value = (row.get("rfc_number") or "").strip()
            if value:
                numbers.append(normalize_rfc_number(value))

    return sorted(set(numbers))


def filter_and_group(
    all_errata: list[dict[str, Any]], rfc_numbers: set[int]
) -> dict[str, list[dict[str, Any]]]:
    by_rfc: dict[str, list[dict[str, Any]]] = {f"rfc{n}": [] for n in rfc_numbers}

    for entry in all_errata:
        doc_id = entry.get("doc-id", "")
        if not doc_id.upper().startswith("RFC"):
            continue  # errata.json also covers some non-RFC document types; skip those
        num = parse_rfc_number(doc_id)
        if num is None:
            continue
        if num in rfc_numbers:
            by_rfc[f"rfc{num}"].append(
                {
                    "errata_id": entry.get("errata_id"),
                    "status": entry.get("errata_status_code"),
                    "type": entry.get("errata_type_code"),
                    "section": entry.get("section"),
                    "orig_text": entry.get("orig_text"),
                    "correct_text": entry.get("correct_text"),
                    "notes": entry.get("notes"),
                    "submit_date": entry.get("submit_date"),
                    "submitter_name": entry.get("submitter_name"),
                    "verifier_name": entry.get("verifier_name"),
                    "update_date": entry.get("update_date"),
                }
            )

    return by_rfc


def filter_verified_technical(
    all_errata: list[dict[str, Any]], rfc_numbers: set[int]
) -> dict[str, list[dict[str, Any]]]:
    """Return only RFC Editor records whose status is Verified and type is Technical."""
    by_rfc: dict[str, list[dict[str, Any]]] = {
        f"rfc{number}": [] for number in sorted(rfc_numbers)
    }

    for entry in all_errata:
        status = str(entry.get("errata_status_code") or "").strip()
        errata_type = str(entry.get("errata_type_code") or "").strip()
        if status.casefold() != "verified" or errata_type.casefold() != "technical":
            continue

        doc_id = str(entry.get("doc-id") or "").strip()
        if not doc_id.upper().startswith("RFC"):
            continue
        number = parse_rfc_number(doc_id)
        if number is None or number not in rfc_numbers:
            continue

        errata_id = entry.get("errata_id")
        by_rfc[f"rfc{number}"].append(
            {
                "correct_text": entry.get("correct_text"),
                "doc_id": f"RFC{number}",
                "errata_id": errata_id,
                "notes": entry.get("notes"),
                "orig_text": entry.get("orig_text"),
                "section": entry.get("section"),
                "source_url": (
                    f"https://www.rfc-editor.org/errata/eid{errata_id}"
                    if errata_id is not None
                    else None
                ),
                "status": status,
                "submit_date": entry.get("submit_date"),
                "submitter_name": entry.get("submitter_name"),
                "type": errata_type,
                "update_date": entry.get("update_date"),
                "verifier_id": entry.get("verifier_id"),
                "verifier_name": entry.get("verifier_name"),
            }
        )

    def sort_key(record: dict[str, Any]) -> tuple[int, str]:
        value = record.get("errata_id")
        try:
            return int(value), str(value)
        except (TypeError, ValueError):
            return 2**63 - 1, str(value)

    for records in by_rfc.values():
        records.sort(key=sort_key)
    return by_rfc


def build_verified_technical_payload(
    by_rfc: dict[str, list[dict[str, Any]]], target_source: str
) -> dict[str, Any]:
    return {
        "by_rfc": by_rfc,
        "errata_count": sum(len(records) for records in by_rfc.values()),
        "filters": {"status": "Verified", "type": "Technical"},
        "rfc_count": len(by_rfc),
        "source_url": ERRATA_URL,
        "target_source": target_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and filter RFC errata from the RFC Editor's bulk export."
    )
    parser.add_argument(
        "rfcs",
        nargs="*",
        help="RFC number(s) to filter to, e.g. 9460. Default: everything in metadata/rfcs.json",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of the bulk errata file",
    )
    parser.add_argument(
        "--verified-technical",
        action="store_true",
        help=(
            "Write only Verified Technical errata. Without positional RFCs, "
            "targets come from corpus/metadata/final_rfc_manifest.csv."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FINAL_RFC_MANIFEST_PATH,
        help="Manifest used by --verified-technical when no RFCs are passed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the output JSON path.",
    )
    args = parser.parse_args()

    if args.verified_technical:
        if args.rfcs:
            rfc_numbers = {normalize_rfc_number(rfc) for rfc in args.rfcs}
            target_source = "command-line RFC arguments"
        else:
            rfc_numbers = set(rfc_numbers_from_manifest(args.manifest))
            try:
                target_source = str(args.manifest.resolve().relative_to(REPO_ROOT))
            except ValueError:
                target_source = str(args.manifest.resolve())
    else:
        rfc_numbers = (
            {normalize_rfc_number(rfc) for rfc in args.rfcs}
            if args.rfcs
            else set(rfc_numbers_from_rfcs_metadata())
        )

    if not rfc_numbers:
        print("No target RFCs were found. Nothing to do.")
        return

    all_errata = download_bulk_errata(force=args.refresh)

    if args.verified_technical:
        by_rfc = filter_verified_technical(all_errata, rfc_numbers)
        payload = build_verified_technical_payload(by_rfc, target_source)
        output_path = args.output or VERIFIED_TECHNICAL_ERRATA_PATH
        write_json_atomic(output_path, payload)

        print(
            f"\n{payload['errata_count']} Verified Technical errata across "
            f"{payload['rfc_count']} target RFCs -> {output_path}"
        )
        for rfc_id, entries in by_rfc.items():
            print(f"  {rfc_id}: {len(entries)}")
        return

    by_rfc = filter_and_group(all_errata, rfc_numbers)
    output_path = args.output or ERRATA_METADATA_PATH
    write_json_atomic(output_path, by_rfc)

    total = sum(len(v) for v in by_rfc.values())
    verified = sum(1 for v in by_rfc.values() for e in v if e["status"] == "Verified")
    print(
        f"\n{total} errata across {len(rfc_numbers)} RFCs ({verified} Verified) -> {output_path}"
    )
    for rfc_id, entries in sorted(by_rfc.items()):
        if entries:
            print(f"  {rfc_id}: {len(entries)} errata")


if __name__ == "__main__":
    main()
