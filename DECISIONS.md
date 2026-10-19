# Decisions Log

## 2026-09-09 — Approved TLS-only alignment MVP

Implement the approved milestone only for collected TLS 1.3 drafts and RFC8446.
Prefer existing `clauses_llm` snapshots, preparing missing TLS-22/RFC8446 snapshots
only inside new immutable run directories using deterministic segmentation. Do not
invoke LLM refinement or alter original inputs.

Use pinned MiniLM embeddings with direct token-ID windows, explicit special-token
budgets, deterministic boundaries and hashes. This replaces decode/retokenize window
construction after the preserved revision-08 failure. Keep the failed run for audit.
Use bipartite matching plus bounded split/merge optimization, provisional confidence,
review sampling and prefix-only feature guards. Do not assign ambiguity labels or
revision reasons. Human-reviewed accuracy, cross-family evaluation, calibration and
benchmark construction remain future work.

This entry records the user's approved scope and continuation requirements.

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

## 2026-10-18 — Isolate final-annotation context training from prior LLM experiments

**Decision:** Train the new Phi-4 LoRA experiment from the resolved binary labels in
`results/annotation/final_annotation.csv`, excluding unresolved placeholder
`item_id=2161`. Use the previous, target, and next clauses with the versioned
normal-engineer prompt. Preserve document-family grouped nested cross-validation,
reserve TCP, and keep DNS sealed. Treat the labels as an operational reference produced
by human annotation supported by two AI models, not as independently verified external
ground truth.

**Why:** The annotation source, positive prevalence, semantic rubric, and input context
differ materially from the earlier 2,253-row target-only consensus experiment. Reusing
old folds, adapters, calibration, or conclusions would mix incompatible experimental
contracts. Physical partitioning, pinned hashes, and a fresh hyperparameter screen keep
the new results reproducible and prevent validation/test leakage.

**Rejected alternatives:** Reusing the old adapters or calibration; treating the
placeholder unresolved label as negative; training on TCP/DNS; using annotation
provenance or rationales as model inputs; silently truncating local context.

**Owner:** Osama Elkhuribi

---

## 2026-10-18 — Establish zero/4/8-shot baselines before LoRA fine-tuning

**Decision:** Evaluate the pinned Phi-4 Mini base model in zero-shot, four-shot, and
eight-shot conditions before making any parameter updates. Reuse the frozen grouped
outer folds, select balanced and family-diverse demonstrations only from each fold's
fitting rows, tune each classification threshold only on that fold's independent inner
family, and rank shot conditions by mean fold PR-AUC with a paired one-standard-error
preference for fewer examples. Report precision, recall, F1, F2, PR-AUC, balanced
accuracy, accuracy, per-row scores, PR curves, and confusion matrices.

**Why:** The no-training baselines measure how much the task rubric and examples alone
provide, give LoRA a fair reference point, and expose prompt or label problems before an
expensive hyperparameter screen. Fold-local demonstrations and thresholds prevent
evaluation leakage. Accuracy cannot be primary because the development cohort is only
6.38% positive and an all-negative classifier is approximately 93.6% accurate.

**Rejected alternatives:** Reporting accuracy alone; choosing examples globally;
letting demonstrations overlap inner or outer targets; fixing 0.5 without validating a
decision threshold; selecting the number of examples from TCP or sealed DNS; using
free-form generation whose output parsing could change the evaluation.

**Owner:** Osama Elkhuribi

---

## 2026-10-18 — Version the ordinary-engineer rubric without leaking an example

**Decision:** Add `implementer_context_v2` as a new prompt contract rather than editing
v1 in place. Encode the requested ordinary-engineer test as fixed A1/A2/A3 criteria,
require previous/next context, preserve citation and local-pointer exceptions, and keep
the model output to exactly `yes` or `no`. Start new in-context and LoRA output versions.

**Why:** A v1 LoRA run was already active, so changing that contract would mix prompts
across folds. The proposed If-Range example is exactly development `item_id=120`; using
it as a labeled example in every system prompt would contaminate held-out evaluation.
The positive count and human/AI annotation history are provenance, not evidence about a
clause, and could bias the model toward the majority class. The dataset has no verified
rationale column, so rationale generation cannot be a supervised target here.

**Rejected alternatives:** Mutating v1 during an active run; including class prevalence
in the prompt; copying the labeled If-Range row into the prompt; inventing explanation
targets; mixing free-form rationales with one-token conditional scoring.

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
the pilot annotation guide used at that time. The unused pilot experiment templates and
guides were retired on 2026-10-13 after the production pipeline superseded them.

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

## 2026-08-20 — Format-independent section/block parser before RFCScope partitioning

**Decision:** Parse rendered Internet-Draft plaintext, XML2RFC source, and RFC Editor
plaintext into one canonical document → section → block tree before clause refinement
or token-aware partitioning. Use separate plaintext profiles so legacy indented RFC
subsections and unnumbered reference groups are preserved. Preserve source locations and
reference metadata, retain substantive back matter, and remove only rendering artifacts
at this stage.

**Why:** The collected revision histories mix `.txt` and `.xml` sources. RFCScope relies
on the natural RFC section hierarchy for coherent partitioning, while this project also
needs smaller provenance-bearing units for revision alignment. A shared loss-aware tree
supports both uses without tying downstream analysis to one source format.

**Rejected alternatives:** Fixed-size text chunks — they cross section boundaries and lose
logical structure. XML-only parsing — 79 archived revisions are available only as rendered
plaintext. Dropping References, IANA Considerations, or Security Considerations during
parsing — these sections contain dependency and sometimes normative evidence needed by the
ambiguity taxonomy.

**Owner:** Osama Elkhuribi

---

## 2026-08-24 — Separate clause refinement from structural document parsing

**Decision:** Keep TXT/XML/RFC parsing format-specific and loss-aware, then run a
separate deterministic clause layer over the canonical section/block tree. Treat
prose sentences and sentences within list items as traceable clauses while retaining
the parent list-item block; retain formal and reference blocks but classify them as
`EXCLUDED`. Apply classification evidence in
this order: exclusions, terminology/quoted/imported safeguards, explicit uppercase
RFC 2119/8174 terms, ambiguous lowercase modal language, and finally weak section
priors. Route unresolved cases to `MANUAL_REVIEW`. This refines the paragraph-sized
unit retained for the Phase-1 pilot; that earlier decision remains the historical
pilot annotation unit, while sentence-level clauses with list-item provenance are the
Phase-2 unit.

**Why:** Parsing structure and interpreting normative meaning are different decisions.
The separation gives every clause stable parent/source provenance, keeps the format
parsers deterministic, makes classification reasons auditable, and allows heuristics
to be recalibrated without reparsing the corpus. Conservative manual review avoids
turning section headings or lowercase modal verbs into unsupported labels.

**Rejected alternatives:** Embedding classification inside each format parser — this
would duplicate semantic rules and make TXT/XML results diverge. Classifying every
uppercase keyword as normative — quoted text and BCP 14 terminology boilerplate are
counterexamples. Treating all unmarked prose as contextual — this hides implicit
requirements. Dropping ABNF/code/tables/references — this loses provenance needed by
alignment and later context construction.

**Owner:** Osama Elkhuribi

---
