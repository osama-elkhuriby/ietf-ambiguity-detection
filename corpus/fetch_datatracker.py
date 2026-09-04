"""
Draft fetcher — pulls the full submitted-revision history of one or more Internet-Drafts
from the IETF Datatracker, via the `ietfdata` library, and records each in
corpus/metadata/drafts.json.

Verified against ietfdata==0.8.1 (glasgow-ipl/ietfdata) by inspecting the installed
package directly (datatracker.py: Document, Submission, DataTracker classes) —
see ../DECISIONS.md, 2026-08-08 entries.

Usage:
    python corpus/fetch_datatracker.py draft-ietf-dnsop-svcb-https
    python corpus/fetch_datatracker.py draft-name-one draft-name-two

For each draft, this writes:
    corpus/raw/drafts/<draft-name>/<draft-name>-NN.txt   -- raw plaintext of each revision
And upserts one entry per draft into:
    corpus/metadata/drafts.json                          -- master draft manifest

If a draft's Document record already has an rfc_number (i.e. it was published), that's
recorded directly in metadata/drafts.json — link_draft_rfc.py reads it from there rather
than re-querying the Datatracker.

Note: this makes real network calls to datatracker.ietf.org and ietf.org, so it needs
to run somewhere with normal internet access.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests
from ietfdata.datatracker import DataTracker

from paths import REPO_ROOT, repo_path
from storage import to_repo_relative

try:
    from ietfdata.datatracker import DTBackendLive
except ImportError:
    DTBackendLive = None  # type: ignore[misc, assignment]


def make_datatracker() -> DataTracker:
    """Support both legacy (DTBackendLive) and current ietfdata DataTracker APIs."""
    if DTBackendLive is not None:
        return DataTracker(DTBackendLive())
    return DataTracker(cache_dir=str(REPO_ROOT))

METADATA_PATH = repo_path("corpus", "metadata", "drafts.json")
DEFAULT_RAW_DRAFTS_DIR = repo_path("corpus", "raw", "drafts")


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def save_metadata(data: dict[str, Any]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def fetch_draft_history(draft_name: str, out_dir: Path) -> dict[str, Any]:
    dt = make_datatracker()
    doc = dt.document_from_draft(draft_name)
    if doc is None:
        raise SystemExit(
            f"No document found for '{draft_name}'. "
            f"Check the exact name on https://datatracker.ietf.org/doc/{draft_name}/"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    versions: list[dict[str, Any]] = []

    for sub_uri in doc.submissions:
        sub = dt.submission(sub_uri)
        if sub is None:
            continue

        record: dict[str, Any] = {
            "rev": sub.rev,
            "submission_date": str(sub.submission_date),
            "document_date": str(sub.document_date) if sub.document_date else None,
            "pages": sub.pages,
            "words": sub.words,
            "local_txt_path": None,
        }

        raw_url = None
        raw_type = None
        for file_type, url in sub.urls():
            if file_type and file_type.lower() in {".txt", "txt", "text"}:
                raw_url = url
                raw_type = file_type.lower()
                break
        if raw_url is None:
            for file_type, url in sub.urls():
                if file_type and file_type.lower() == ".xml":
                    raw_url = url
                    raw_type = ".xml"
                    break
        if raw_url is None:
            for file_type, url in sub.urls():
                if file_type and file_type.lower() == ".html":
                    raw_url = url
                    raw_type = ".html"
                    break

        if raw_url is not None:
            try:
                resp = requests.get(raw_url, timeout=30)
                resp.raise_for_status()

                out_suffix = Path(raw_url).suffix or ".txt"
                raw_path = out_dir / f"{sub.name}-{sub.rev}{out_suffix}"
                raw_path.write_text(
                    resp.text,
                    encoding="utf-8",
                    errors="replace"
                )

                record["local_txt_path"] = to_repo_relative(raw_path)
                print(f"    saved: {raw_path}")
                if raw_type not in {".txt", "txt", "text"}:
                    print(f"    note: saved {raw_type} source for rev {sub.rev}")

            except requests.RequestException as e:
                print(f"  warning: rev {sub.rev} fetch failed: {e}")
        else:
            print(f"  warning: rev {sub.rev} has no TXT/XML/HTML source")

        versions.append(record)
        print(f"  fetched rev {sub.rev}  ({sub.submission_date}, {sub.words or '?'} words)")

    versions.sort(key=lambda v: v["rev"])

    draft_record = {
        "name": doc.name,
        "title": doc.title,
        "group": str(doc.group) if doc.group else None,
        "rfc": doc.rfc,
        "rfc_number": doc.rfc_number,
        "latest_rev": doc.rev,
        "num_versions": len(versions),
        "versions": versions,
    }

    became_rfc = f" Became RFC {doc.rfc_number}." if doc.rfc_number else " Not published as an RFC (yet)."
    print(f"-> {len(versions)} revisions." + became_rfc)
    return draft_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Internet-Draft version history from the IETF Datatracker.")
    parser.add_argument("drafts", nargs="+", help="Draft name(s), e.g. draft-ietf-dnsop-svcb-https")
    parser.add_argument(
        "--out-dir",
        "--output-dir",
        dest="out_dir",
        default=str(DEFAULT_RAW_DRAFTS_DIR),
        help="Base output directory; one subfolder per draft is created inside it (default: corpus/raw/drafts)",
    )
    args = parser.parse_args()

    metadata = load_metadata()
    base_out = Path(args.out_dir)

    for name in args.drafts:
        print(f"Fetching {name} ...")
        metadata[name] = fetch_draft_history(name, base_out / name)

    save_metadata(metadata)
    print(f"\nUpdated {METADATA_PATH} ({len(metadata)} drafts total)")


if __name__ == "__main__":
    main()                                              