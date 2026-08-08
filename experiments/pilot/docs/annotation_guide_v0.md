# Annotation Guide — v0 (pilot draft)

Status: **draft, to be revised after the week-1 pilot.** Every open question below should get
resolved (or explicitly deferred with a reason) before Phase 2 annotation starts at scale.
Log any change to this guide as its own entry in `../../../DECISIONS.md`.

## 1. Unit of annotation

A single **clause** within one revision of one Internet-Draft.

**Pilot working definition (use unless the pilot proves it wrong):** one numbered-section
**paragraph** in the Datatracker plaintext (`.txt`) rendering — from blank line to blank line,
excluding RFC boilerplate (title page, copyright, TOC, references, IANA, acknowledgments).
Split a paragraph only when it contains multiple independent normative requirements (BCP 14
`MUST`/`SHOULD`/… each governing a distinct behavior); keep ABNF blocks attached to the
paragraph that introduces them.

Edge cases to log in §8: tables, indented examples, `NOTE:`/`TODO:` blocks, and text that moved
sections between revisions.

_Working definition (refine after pilot):_

## 2. Label tiers (gold / silver / uncertain)

Per the proposal abstract, every labeled clause gets both a **class** (ambiguous / not ambiguous)
and a **confidence tier**:

| Tier | Meaning | Typical source |
|---|---|---|
| **Gold** | High-confidence label | Verified technical errata; documented author clarification |
| **Silver** | Reasonable-confidence label | RFCScope-style ambiguity category match; clear revision pattern |
| **Uncertain** | Annotators disagree or evidence is thin | Flag for dual review — do not silently drop |

## 3. Positive labels (ambiguous)

Supported by (per proposal SS6.3), in descending order of confidence:

1. Verified technical errata on the **published RFC** mapped back to a draft revision via
   section/clause alignment (RFCScope errata IDs are a good index; RFC Editor errata for
   authoritative status)
2. Documented author clarification (working-group list, GitHub issue, revision `-00` changelog)
3. RFCScope-style ambiguity **subtype** match on the published text, projected to draft stage
4. Manual annotation: a later draft revision resolves multiple plausible readings of this clause

**Draft-stage caveat:** Errata formally apply to published RFCs, not Internet-Drafts. For
pre-publication claims, always record `evidence_source` and the RFC↔draft mapping you used.
Do not treat “no erratum yet” as negative evidence while the document is still an ID.

## 4. Negative labels (not ambiguous)

Stable, clear clauses from the same documents. **Important guardrail from the proposal:**
absence of an erratum is *not* proof of clarity — a clause with no reported errata may simply be
unread or unimplemented. Negative labels need active review, not just "nothing was reported."

## 5. Revision-reason labels

Every *changed* clause (regardless of ambiguity label) gets exactly one of these six tags,
distinguishing genuine ambiguity clarification from unrelated churn:

- `ambiguity_clarification`
- `editorial_change`
- `new_functionality`
- `terminology_change`
- `reference_correction`
- `other_technical_redesign`

This is the label that keeps H1/H4 honest — without it, any edit between versions could get
mistaken for evidence of ambiguity.

## 6. Ambiguity subtypes

Adopt **RFCScope's seven logical-ambiguity subtypes** (Pawagi et al., ASE 2025) for consistency
with reused errata data. Map each positive label where possible; use `other` only for clear
logical issues that fall outside this taxonomy (and note why in `notes`).

| ID | Category | Description (short) |
|----|----------|---------------------|
| **I-1** | Inconsistency | Direct inconsistency within or across specifications |
| **I-2** | Inconsistency | Indirect inconsistency within or across specifications |
| **I-3** | Inconsistency | Inconsistency with commonly accepted knowledge |
| **U-1** | Under-specification | Undefined terms |
| **U-2** | Under-specification | Incomplete constraints (often needs implementation feedback) |
| **U-3** | Under-specification | Indirect under-specification within or across specs |
| **U-4** | Under-specification | Incorrect or missing references |

Heuristic detectors (vague quantifiers, undefined terms, unresolved references) overlap **U-1**,
**U-4**, and parts of **I-1**/**I-2** — subtype labels help compare against RFCScope, not duplicate it.

Source artifact: https://github.com/HIPREL-Group/RFCScope

## 7. Worked examples from the pilot

_Fill in during Step 3 of the pilot — 3 to 5 real examples, one per revision-reason category
where possible. Each example should include: draft name + revision, clause text, label, tier,
revision-reason, and a one-line rationale. These examples are what actually make this guide
usable by someone who wasn't in the room when the decision was made._

| Draft / rev | Clause (short) | Label | Tier | Revision reason | Rationale |
|---|---|---|---|---|---|
| | | | | | |

## 8. Open questions from the pilot

_What did the pilot surface that this guide doesn't answer yet? List them here, then resolve
and fold the answer back into sections 1–6 before Phase 2._

-
