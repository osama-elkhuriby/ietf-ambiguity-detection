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

Note: the bulk file is large (the full history of RFC errata) and is cached at
corpus/raw/errata/errata_full.json for 1 day, similar to how ietfdata caches rfc-index.xml
— re-run with --refresh to force an update sooner. Needs normal internet access to run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from paths import repo_path
from rfc_ids import normalize_rfc_number, parse_rfc_number

ERRATA_URL = "https://www.rfc-editor.org/errata.json"
RAW_CACHE_PATH = repo_path("corpus", "raw", "errata", "errata_full.json")
RFCS_METADATA_PATH = repo_path("corpus", "metadata", "rfcs.json")
ERRATA_METADATA_PATH = repo_path("corpus", "metadata", "errata.json")
CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # 1 day, matching ietfdata's own rfc-index.xml cache policy


def download_bulk_errata(force: bool = False) -> list[dict[str, Any]]:
    if not force and RAW_CACHE_PATH.exists():
        age = time.time() - RAW_CACHE_PATH.stat().st_mtime
        if age < CACHE_MAX_AGE_SECONDS:
            print(f"Using cached {RAW_CACHE_PATH} ({age / 3600:.1f}h old)")
            return json.loads(RAW_CACHE_PATH.read_text())

    print(f"Downloading {ERRATA_URL} ...")
    resp = requests.get(ERRATA_URL, timeout=60)
    resp.raise_for_status()
    all_errata = resp.json()

    RAW_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_PATH.write_text(json.dumps(all_errata, indent=2))
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


def filter_and_group(all_errata: list[dict[str, Any]], rfc_numbers: set[int]) -> dict[str, list[dict[str, Any]]]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and filter RFC errata from the RFC Editor's bulk export.")
    parser.add_argument("rfcs", nargs="*", help="RFC number(s) to filter to, e.g. 9460. Default: everything in metadata/rfcs.json")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of the bulk errata file")
    args = parser.parse_args()

    rfc_numbers = {normalize_rfc_number(r) for r in args.rfcs} if args.rfcs else set(rfc_numbers_from_rfcs_metadata())
    if not rfc_numbers:
        print("No target RFCs (metadata/rfcs.json is empty and none were passed explicitly). Nothing to do.")
        return

    all_errata = download_bulk_errata(force=args.refresh)
    by_rfc = filter_and_group(all_errata, rfc_numbers)

    ERRATA_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERRATA_METADATA_PATH.write_text(json.dumps(by_rfc, indent=2, sort_keys=True))

    total = sum(len(v) for v in by_rfc.values())
    verified = sum(1 for v in by_rfc.values() for e in v if e["status"] == "Verified")
    print(f"\n{total} errata across {len(rfc_numbers)} RFCs ({verified} Verified) -> {ERRATA_METADATA_PATH}")
    for rfc_id, entries in sorted(by_rfc.items()):
        if entries:
            print(f"  {rfc_id}: {len(entries)} errata")


if __name__ == "__main__":
    main()