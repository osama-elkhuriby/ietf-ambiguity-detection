"""
Draft<->RFC linker — reads corpus/metadata/drafts.json and corpus/metadata/rfcs.json and
writes corpus/metadata/draft_rfc_links.json, joining each draft to the RFC it became (if
any).

Two independent sources are checked against each other (see ../DECISIONS.md, 2026-08-08):
  1. Primary: each draft's own `rfc_number` field (from the Datatracker, via
     fetch_datatracker.py) -- populated directly on the draft's Document record once
     published.
  2. Cross-check: each RFC's own `draft` field (from the RFC index, via fetch_rfc.py) --
     the final draft name before publication, independently sourced from rfc-index.xml.
If both are present and agree, confidence is "high". If only one source is available,
confidence is "single-source". If they're both present but disagree, it's flagged for
manual review rather than silently picking one.

Usage:
    python corpus/link_draft_rfc.py

Does not touch the network -- pure local join over already-fetched metadata. Run
fetch_datatracker.py and (optionally) fetch_rfc.py first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paths import repo_path

DRAFTS_METADATA_PATH = repo_path("corpus", "metadata", "drafts.json")
RFCS_METADATA_PATH = repo_path("corpus", "metadata", "rfcs.json")
LINKS_METADATA_PATH = repo_path("corpus", "metadata", "draft_rfc_links.json")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def normalize_draft_name(draft_name: str) -> str:
    return re.sub(r"-\d+$", "", draft_name)


def build_links(drafts: dict[str, Any], rfcs: dict[str, Any]) -> list[dict[str, Any]]:
    # Reverse index: draft name (from the RFC index's own 'draft' field) -> rfc_id
    draft_to_rfc_from_index: dict[str, str] = {}
    for rfc_id, rfc_record in rfcs.items():
        draft_name = rfc_record.get("draft")
        if not draft_name:
            continue

        draft_to_rfc_from_index[draft_name] = rfc_id
        normalized_name = normalize_draft_name(draft_name)
        if normalized_name != draft_name and normalized_name not in draft_to_rfc_from_index:
            draft_to_rfc_from_index[normalized_name] = rfc_id

    links: list[dict[str, Any]] = []

    for draft_name, draft_record in drafts.items():
        datatracker_rfc_number = draft_record.get("rfc_number")
        datatracker_rfc_id = f"rfc{datatracker_rfc_number}" if datatracker_rfc_number else None
        index_rfc_id = draft_to_rfc_from_index.get(draft_name)
        if index_rfc_id is None:
            index_rfc_id = draft_to_rfc_from_index.get(normalize_draft_name(draft_name))

        if datatracker_rfc_id is None and index_rfc_id is None:
            continue  # this draft hasn't been linked to an RFC by either source; not an error

        if datatracker_rfc_id and index_rfc_id:
            if datatracker_rfc_id == index_rfc_id:
                links.append({"draft": draft_name, "rfc": datatracker_rfc_id, "confidence": "high", "note": "datatracker rfc_number and rfc-index draft field agree"})
            else:
                links.append(
                    {
                        "draft": draft_name,
                        "rfc": None,
                        "confidence": "conflict",
                        "note": f"datatracker says {datatracker_rfc_id}, rfc-index says {index_rfc_id} -- needs manual review",
                    }
                )
        else:
            source = "datatracker" if datatracker_rfc_id else "rfc-index"
            links.append(
                {
                    "draft": draft_name,
                    "rfc": datatracker_rfc_id or index_rfc_id,
                    "confidence": "single-source",
                    "note": f"only found via {source}; run the other fetcher to cross-check",
                }
            )

    return links


def main() -> None:
    drafts = load_json(DRAFTS_METADATA_PATH)
    rfcs = load_json(RFCS_METADATA_PATH)

    if not drafts:
        raise SystemExit(f"{DRAFTS_METADATA_PATH} is empty or missing -- run fetch_datatracker.py first.")
    if not rfcs:
        print(f"Note: {RFCS_METADATA_PATH} is empty -- links will be single-source (datatracker only), with no cross-check.")

    links = build_links(drafts, rfcs)

    LINKS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    LINKS_METADATA_PATH.write_text(json.dumps(links, indent=2, sort_keys=True))

    conflicts = [l for l in links if l["confidence"] == "conflict"]
    print(f"{len(links)} draft-RFC links written to {LINKS_METADATA_PATH}")
    print(f"  high-confidence: {sum(1 for l in links if l['confidence'] == 'high')}")
    print(f"  single-source:   {sum(1 for l in links if l['confidence'] == 'single-source')}")
    print(f"  conflicts:       {len(conflicts)}")
    if conflicts:
        print("\n  CONFLICTS NEED MANUAL REVIEW:")
        for c in conflicts:
            print(f"    {c['draft']}: {c['note']}")


if __name__ == "__main__":
    main()