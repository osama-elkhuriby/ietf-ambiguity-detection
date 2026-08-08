"""
Pilot corpus fetcher — pulls the full submitted-revision history of one or more
Internet-Drafts from the IETF Datatracker, via the `ietfdata` library.

Verified against ietfdata==0.8.1 (glasgow-ipl/ietfdata) by inspecting the installed
package directly (datatracker.py: Document, Submission, DataTracker classes) —
see ../DECISIONS.md, 2026-08-08 entry.

Usage:
    python corpus/fetch_datatracker.py draft-ietf-dnsop-svcb-https
    python corpus/fetch_datatracker.py draft-name-one draft-name-two

For each draft, this writes:
    corpus/raw/<draft-name>/versions.json        -- manifest: rev, dates, word/page counts
    corpus/raw/<draft-name>/<draft-name>-NN.txt   -- raw plaintext of each revision

Note: this makes real network calls to datatracker.ietf.org and ietf.org, so it needs
to run somewhere with normal internet access. (It will NOT run inside a locked-down
sandbox that only allows package-registry domains -- confirmed while building this.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import requests
from ietfdata.datatracker import DataTracker


def fetch_draft_history(draft_name: str, out_dir: Path) -> list[dict]:
    dt = DataTracker()
    doc = dt.document_from_draft(draft_name)
    if doc is None:
        raise SystemExit(
            f"No document found for '{draft_name}'. "
            f"Check the exact name on https://datatracker.ietf.org/doc/{draft_name}/"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    versions: list[dict] = []

    for sub_uri in doc.submissions:
        sub = dt.submission(sub_uri)
        if sub is None:
            continue

        record = {
            "name": sub.name,
            "rev": sub.rev,
            "title": sub.title,
            "submission_date": str(sub.submission_date),
            "document_date": str(sub.document_date) if sub.document_date else None,
            "pages": sub.pages,
            "words": sub.words,
            "local_txt_path": None,
        }

        # Grab the plaintext rendering of this specific revision, if available
        for file_type, url in sub.urls():
            if file_type == ".txt":
                resp = requests.get(url, timeout=30)
                if resp.status_code == 200:
                    txt_path = out_dir / f"{sub.name}-{sub.rev}.txt"
                    txt_path.write_text(resp.text, encoding="utf-8", errors="replace")
                    record["local_txt_path"] = str(txt_path)
                else:
                    print(f"  warning: rev {sub.rev} .txt fetch returned HTTP {resp.status_code}")
                break

        versions.append(record)
        print(f"  fetched rev {sub.rev}  ({sub.submission_date}, {sub.words or '?'} words)")

    versions.sort(key=lambda v: v["rev"])

    manifest_path = out_dir / "versions.json"
    manifest_path.write_text(json.dumps(versions, indent=2))
    print(f"-> {len(versions)} revisions. Manifest: {manifest_path}")
    return versions


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fetch_datatracker.py <draft-name> [<draft-name> ...]")
        print("Example: python fetch_datatracker.py draft-ietf-dnsop-svcb-https")
        sys.exit(1)

    for name in sys.argv[1:]:
        print(f"Fetching {name} ...")
        fetch_draft_history(name, Path("corpus/raw") / name)
