# Decisions Log

Every methodological choice gets a dated entry: what we decided, why, and what we rejected.
This becomes the raw material for the thesis methodology chapter — don't skip entries because
a decision feels "obvious" at the time. It won't be obvious in week 14.

---

## 2026-08-08 — Pilot protocol family: DNS

**Decision:** Use a small DNS-related Internet-Draft family (e.g. one or two drafts under the
`dnsop` working group) for the week-1 end-to-end pilot, before scaling to the full multi-family
corpus in Phase 2.

**Why:** RFCScope's own case study found new logical ambiguities in 14 of the 20 most recent DNS
RFCs it examined. That gives us independently-discovered ground truth to sanity-check our
pipeline against, instead of validating on data with no external reference point.

**Rejected alternatives:** Starting directly with a larger, more central protocol family (e.g.
core TCP or TLS) — more representative long-term, but no cheap way to sanity-check a first-pass
pipeline against known findings.

**Owner:** Osama Elkhuribi

---

## 2026-08-08 — Reuse RFCScope's public dataset rather than re-mining errata from scratch

**Decision:** Build on RFCScope's published artifact (github.com/HIPREL-Group/RFCScope) —
their errata-derived dataset and seven-subtype taxonomy — as a starting point for positive-label
sourcing in proposal SS6.3, rather than independently re-mining a decade of RFC errata before
any modeling starts.

**Why:** Their dataset already covers Jan 2014–Jan 2025 verified technical errata, manually
sorted into seven ambiguity subtypes. Re-deriving this from scratch would consume weeks of
Phase 1/2 time for something already public and citable.

**Rejected alternatives:** Full independent errata mining — kept as a *stretch* extension for
later (e.g., extending coverage to Internet-Draft stage revisions specifically, which RFCScope
does not cover), not a Phase 1/2 blocker.

**Owner:** Osama Elkhuribi

---

## 2026-08-08 — Use `ietfdata` instead of a hand-rolled Datatracker scraper

**Decision:** Use the `ietfdata` Python package (glasgow-ipl, PyPI) for Stage 1 corpus pulls
instead of writing custom `requests`/`BeautifulSoup` scraping logic against the Datatracker.

**Why:** Verified against the installed package (v0.8.1): `DataTracker().document_from_draft(name)`
returns a `Document` with a `.submissions` list; each `Submission` exposes `.rev`,
`.submission_date`, and `.urls()` for the raw text of that revision. This covers most of what
SS6.3's "all archived revisions... in machine-readable text" requirement needs, out of the box.

**Rejected alternatives:** Custom scraper — more control, but duplicates a maintained library
for no clear benefit at this stage. Can revisit if `ietfdata` proves to be missing a field we need.

**Owner:** Osama Elkhuribi

---

## 2026-08-08 — RFCScope taxonomy as the subtype vocabulary

**Decision:** Use RFCScope's seven logical-ambiguity subtypes (I-1…I-3, U-1…U-4) for
`ambiguity_subtype` labels and for comparing detector output to reused errata data — not a
parallel taxonomy.

**Why:** The team is reusing RFCScope's public errata categorization (see entry above). A single
vocabulary keeps silver labels comparable and avoids double-annotation work.

**Rejected alternatives:** Merging RFCScope categories with informal "linguistic ambiguity" tags
only — useful as *features*, but not as the primary evaluation subtype axis.

**Owner:** Osama Elkhuribi

---

## 2026-08-08 — Pilot clause unit: section paragraph (plaintext)

**Decision:** For Phase 1 hand alignment and labeling, treat a **clause** as one paragraph within
a numbered section of the Datatracker `.txt` revision, with the exceptions documented in
`experiments/pilot/docs/annotation_guide_v0.md` §1.

**Why:** Matches how authors edit I-Ds in practice, aligns with iddiff hunks at paragraph
granularity, and is coarse enough for 15–20 clause pilot labels in week 1.

**Rejected alternatives:** Sentence-level units (too fragmented for cross-ref and ABNF); whole
sections (mixes many independent requirements and hurts class balance).

**Owner:** Osama Elkhuribi

---

## 2026-08-09 — Repo-root paths for corpus scripts

**Decision:** Resolve all `corpus/metadata/` and `corpus/raw/` paths from `corpus/paths.py`
(`REPO_ROOT`), not from the process working directory.

**Why:** Avoids silent writes to the wrong tree when someone runs a script from `corpus/` or
another directory; keeps pilot mirrors (`--out-dir`) independent of metadata location.

**Rejected alternatives:** Require "always cd to repo root" in docs only — easy to forget during
pilot crunch.

**Owner:** Osama Elkhuribi

---

## 2026-08-13 — Phase-2 curated corpus manifest (12 RFC families)

**Decision:** Use `corpus/metadata/final_rfc_manifest.csv` as the canonical target list
for Phase-2 corpus collection, spanning transport, routing, DNS, security, and application
families (QUIC/TCP, OSPF/BGP/IPv6, SVCB/HTTPS, TLS, HTTP).

**Why:** Gives reproducible coverage across protocol areas named in the proposal, with one
row per published RFC family, explicit revision counts, and on-disk provenance columns
refreshed by `sync_metadata.py`.

**Rejected alternatives:** Ad-hoc per-experiment draft lists — harder to audit and compare
across alignment/modeling stages.

**Owner:** Osama Elkhuribi

---

## 2026-08-13 — Structured corpus storage and repo-relative paths

**Decision:** Store raw plaintext under `corpus/raw/{drafts,rfcs,errata}/` and record
`local_txt_path` values repo-relative to the project root via `corpus/storage.py`.

**Why:** Metadata JSON is committed; raw blobs are gitignored and re-fetchable. Relative
paths survive machine moves and keep manifests portable across team members' checkouts.

**Rejected alternatives:** Absolute paths in metadata — caused silent breakage when the
repo moved between machines.

**Owner:** Osama Elkhuribi

---

## 2026-08-13 — RFC-only treatment for pre-Datatracker routing standards

**Decision:** Mark `draft-ietf-ospf-vers2` (RFC2328) and `draft-ietf-idr-bgp4` (RFC4271)
as **`rfc-only`** collection mode. Collect RFC plaintext and errata, but do not treat
missing Datatracker submission history as a corpus gap.

**Why:** The IETF Datatracker returns zero submissions for these pre-2006 documents.
Revision-level draft alignment is not available from the official archive; RFC-level
analysis and errata remain valid.

**Rejected alternatives:** Excluding legacy routing RFCs entirely — they are important
routing coverage and have rich errata histories.

**Owner:** Osama Elkhuribi

---

## 2026-08-13 — Backfill `rfc_number` from the RFC index when Datatracker omits it

**Decision:** After fetch, run `corpus/sync_metadata.py` to populate `drafts.json`
`rfc_number` from `rfcs.json` (RFC index `draft` field) when Datatracker leaves it null,
then rebuild `draft_rfc_links.json` for high-confidence joins.

**Why:** Published drafts fetched by family name often have `rfc_number=null` in Datatracker
even though the RFC index cross-reference is authoritative. Backfilling avoids false
`single-source` links without extra network calls.

**Rejected alternatives:** Accepting permanent `single-source` links — correct but weaker
provenance for thesis reproducibility reporting.

**Owner:** Osama Elkhuribi

---
