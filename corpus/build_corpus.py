"""
Corpus builder — the single entry point that runs the full Stage 1 pipeline for a list of
Internet-Drafts: fetch draft history -> link to any published RFC -> fetch that RFC ->
fetch its errata.

This is intentionally a thin orchestrator over the four scripts in this folder, not a
reimplementation of them -- each step's logic (and its own verification) lives in its own
file. Run steps individually instead if you only need one of them.

Usage:
    python corpus/build_corpus.py draft-ietf-dnsop-svcb-https draft-ietf-dnsop-something

Equivalent to running, in order:
    python corpus/fetch_datatracker.py <drafts...>
    python corpus/link_draft_rfc.py
    python corpus/fetch_rfc.py --from-drafts
    python corpus/fetch_errata.py

Note: this makes real network calls (Datatracker, RFC index, RFC Editor errata), so it
needs to run somewhere with normal internet access.
"""
from __future__ import annotations

import argparse
import json
import re

import fetch_datatracker
import fetch_errata
import fetch_rfc
import link_draft_rfc
from ietfdata.rfcindex import RFCIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Build (or extend) the corpus for one or more Internet-Drafts.")
    parser.add_argument("drafts", nargs="+", help="Draft name(s), e.g. draft-ietf-dnsop-svcb-https")
    args = parser.parse_args()

    print(f"=== Step 1/4: fetching {len(args.drafts)} draft(s) ===")
    metadata = fetch_datatracker.load_metadata()
    for name in args.drafts:
        print(f"Fetching {name} ...")
        metadata[name] = fetch_datatracker.fetch_draft_history(
            name, fetch_datatracker.DEFAULT_RAW_DRAFTS_DIR / name
        )
    fetch_datatracker.save_metadata(metadata)
    print(f"Updated {fetch_datatracker.METADATA_PATH} ({len(metadata)} drafts total)\n")

    print("=== Step 2/4: linking drafts to any published RFC ===")
    rfcs_before = link_draft_rfc.load_json(link_draft_rfc.RFCS_METADATA_PATH)
    requested_metadata = {name: metadata[name] for name in args.drafts}
    links = link_draft_rfc.build_links(requested_metadata, rfcs_before)
    link_draft_rfc.LINKS_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    link_draft_rfc.LINKS_METADATA_PATH.write_text(json.dumps(links, indent=2, sort_keys=True))
    linked_rfc_ids = sorted({link["rfc"] for link in links if link.get("rfc")})
    linked_draft_names = {link["draft"] for link in links if link.get("rfc")}
    missing_drafts = [name for name in args.drafts if name not in linked_draft_names]
    print(f"{len(linked_rfc_ids)} of {len(args.drafts)} requested drafts are published: {linked_rfc_ids}\n")

    if missing_drafts:
        print("=== Step 2b/4: scanning RFC index for requested drafts ===")
        fetch_rfc.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        index = RFCIndex(cache_dir=str(fetch_rfc.INDEX_CACHE_DIR))
        normalized_missing = {re.sub(r"-\d+$", "", name) for name in missing_drafts}
        candidate_rfc_ids: set[str] = set()
        for rfc_entry in index.rfcs():
            if not getattr(rfc_entry, "draft", None):
                continue
            draft_name = rfc_entry.draft
            if draft_name in missing_drafts or re.sub(r"-\d+$", "", draft_name) in normalized_missing:
                candidate_rfc_ids.add(str(rfc_entry.doc_id))

        if candidate_rfc_ids:
            linked_rfc_ids = sorted(set(linked_rfc_ids) | candidate_rfc_ids)
            print(f"{len(candidate_rfc_ids)} RFCs found in RFC index for missing requested drafts: {sorted(candidate_rfc_ids)}\n")
        else:
            print(f"No RFCs found in RFC index for missing requested drafts: {missing_drafts}\n")

    if linked_rfc_ids:
        print("=== Step 3/4: fetching linked RFC(s) ===")
        rfc_ids = linked_rfc_ids
        fetch_rfc.INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        index = RFCIndex(cache_dir=str(fetch_rfc.INDEX_CACHE_DIR))
        rfc_metadata = fetch_rfc.load_metadata()
        for rfc_id in rfc_ids:
            print(f"Fetching {rfc_id} ...")
            rfc_metadata[rfc_id] = fetch_rfc.fetch_rfc(rfc_id, index)
        fetch_rfc.save_metadata(rfc_metadata)
        print(f"Updated {fetch_rfc.RFCS_METADATA_PATH} ({len(rfc_metadata)} RFCs total)\n")

        print("=== Step 4/4: fetching errata for linked RFC(s) ===")
        rfc_numbers = {int(rfc_id[3:]) for rfc_id in linked_rfc_ids}
        all_errata = fetch_errata.download_bulk_errata()
        by_rfc = fetch_errata.filter_and_group(all_errata, rfc_numbers)
        fetch_errata.ERRATA_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        fetch_errata.ERRATA_METADATA_PATH.write_text(json.dumps(by_rfc, indent=2, sort_keys=True))
        total = sum(len(v) for v in by_rfc.values())
        print(f"{total} errata -> {fetch_errata.ERRATA_METADATA_PATH}\n")

        # Re-run the link step now that rfcs.json has the 'draft' field from the index too,
        # so the cross-check in link_draft_rfc.py actually has a second source to check against.
        print("=== Re-linking with the now-available RFC-index cross-check ===")
        rfcs_after = link_draft_rfc.load_json(link_draft_rfc.RFCS_METADATA_PATH)
        links = link_draft_rfc.build_links(metadata, rfcs_after)
        link_draft_rfc.LINKS_METADATA_PATH.write_text(json.dumps(links, indent=2, sort_keys=True))
        conflicts = [l for l in links if l["confidence"] == "conflict"]
        if conflicts:
            print(f"  {len(conflicts)} CONFLICT(S) need manual review -- see {link_draft_rfc.LINKS_METADATA_PATH}")
    else:
        print("None of the requested drafts are published as RFCs yet -- skipping RFC/errata steps.")

    print("\nDone.")

    import sync_metadata

    print("\n=== Post-build metadata sync ===")
    result = sync_metadata.sync_all(refresh_manifest=True, write_audit=True)
    print(
        f"Links: high={result['links_high']} single-source={result['links_single_source']} "
        f"conflict={result['links_conflict']}"
    )


if __name__ == "__main__":
    main()