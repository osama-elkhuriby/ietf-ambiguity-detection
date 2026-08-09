"""
RFC fetcher — pulls RFC metadata (via ietfdata's RFCIndex, backed by the official
rfc-index.xml) and RFC plaintext (via rfc-editor.org) for one or more RFCs, and records
each in corpus/metadata/rfcs.json.

Verified against ietfdata==0.8.1 by inspecting rfcindex.py directly: RFCIndex() downloads
and caches https://www.rfc-editor.org/rfc-index.xml, and RfcEntry.content_url("TEXT")
returns https://www.rfc-editor.org/rfc/rfc<N>.txt — the same plaintext URL pattern
confirmed independently via IETF's own publishing documentation (see ../DECISIONS.md).

Usage:
    python corpus/fetch_rfc.py 9460 8259
    python corpus/fetch_rfc.py rfc9460
    python corpus/fetch_rfc.py --from-drafts     # fetch every RFC linked from
                                                  # corpus/metadata/drafts.json

For each RFC, this writes:
    corpus/raw/rfcs/rfc<N>.txt          -- raw plaintext
And upserts one entry per RFC into:
    corpus/metadata/rfcs.json           -- master RFC manifest

Note: this makes real network calls to rfc-editor.org, so it needs to run somewhere with
normal internet access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from ietfdata.rfcindex import RFCIndex

from paths import repo_path
from rfc_ids import normalize_rfc_id

DRAFTS_METADATA_PATH = repo_path("corpus", "metadata", "drafts.json")
RFCS_METADATA_PATH = repo_path("corpus", "metadata", "rfcs.json")
RAW_RFC_DIR = repo_path("corpus", "raw", "rfcs")
INDEX_CACHE_DIR = repo_path("corpus", "raw", "rfcs", ".rfcindex-cache")


def rfc_numbers_from_drafts_metadata() -> list[str]:
    if not DRAFTS_METADATA_PATH.exists():
        raise SystemExit(
            f"{DRAFTS_METADATA_PATH} doesn't exist yet — run fetch_datatracker.py "
            f"(or link_draft_rfc.py) first, or pass explicit RFC numbers."
        )
    drafts = json.loads(DRAFTS_METADATA_PATH.read_text())
    return [f"rfc{d['rfc_number']}" for d in drafts.values() if d.get("rfc_number")]


def load_metadata() -> dict[str, Any]:
    if RFCS_METADATA_PATH.exists():
        return json.loads(RFCS_METADATA_PATH.read_text())
    return {}


def save_metadata(data: dict[str, Any]) -> None:
    RFCS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    RFCS_METADATA_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def fetch_rfc(rfc_id: str, index: RFCIndex) -> dict[str, Any]:
    entry = index.rfc(rfc_id)
    if entry is None:
        raise SystemExit(f"'{rfc_id}' not found in the RFC index — check the number.")

    record: dict[str, Any] = {
        "doc_id": str(entry.doc_id),
        "title": entry.title,
        "stream": entry.stream,
        "wg": entry.wg,
        "area": entry.area,
        "status": entry.curr_status,
        "date": entry.date().isoformat(),
        "page_count": getattr(entry, "page_count", None),
        "draft": entry.draft,          # the final draft name before publication, per rfc-index.xml
        "errata_url": entry.errata_url,
        "local_txt_path": None,
    }

    text_url = None
    text_format = None
    for fmt in ("TEXT", "TXT", "ASCII"):
        text_url = entry.content_url(fmt)
        if text_url:
            text_format = fmt
            break

    if text_url:
        RAW_RFC_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(text_url, timeout=30)
        if resp.status_code == 200:
            txt_path = RAW_RFC_DIR / f"{rfc_id}.txt"
            txt_path.write_text(resp.text, encoding="utf-8", errors="replace")
            record["local_txt_path"] = str(txt_path)
            if text_format != "TEXT":
                print(f"  note: using {text_format} format for {rfc_id} (formats: {entry.formats})")
        else:
            print(f"  warning: {rfc_id} .txt fetch returned HTTP {resp.status_code}")
    else:
        print(f"  warning: no TEXT/TXT/ASCII format listed for {rfc_id} (formats: {entry.formats})")

    print(f"  fetched {rfc_id}  ({record['date']}, draft={record['draft']})")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull RFC metadata + plaintext via the RFC index.")
    parser.add_argument("rfcs", nargs="*", help="RFC number(s), e.g. 9460 or rfc9460")
    parser.add_argument(
        "--from-drafts",
        action="store_true",
        help="Ignore positional args; fetch every RFC linked from corpus/metadata/drafts.json",
    )
    args = parser.parse_args()

    if args.from_drafts:
        rfc_ids = rfc_numbers_from_drafts_metadata()
        if not rfc_ids:
            print("No drafts in corpus/metadata/drafts.json have a linked RFC yet. Nothing to do.")
            return
    elif args.rfcs:
        rfc_ids = [normalize_rfc_id(r) for r in args.rfcs]
    else:
        raise SystemExit("Pass RFC number(s), or --from-drafts. See --help.")

    INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = RFCIndex(cache_dir=str(INDEX_CACHE_DIR))  # caches rfc-index.xml for 1 day

    metadata = load_metadata()
    for rfc_id in rfc_ids:
        print(f"Fetching {rfc_id} ...")
        metadata[rfc_id] = fetch_rfc(rfc_id, index)

    save_metadata(metadata)
    print(f"\nUpdated {RFCS_METADATA_PATH} ({len(metadata)} RFCs total)")


if __name__ == "__main__":
    main()