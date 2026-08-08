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
