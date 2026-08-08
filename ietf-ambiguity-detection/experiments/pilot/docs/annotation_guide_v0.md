# Annotation Guide — v0 (pilot draft)

Status: **draft, to be revised after the week-1 pilot.** Every open question below should get
resolved (or explicitly deferred with a reason) before Phase 2 annotation starts at scale.
Log any change to this guide as its own entry in `../DECISIONS.md`.

## 1. Unit of annotation

A single **clause** within one revision of one Internet-Draft. Exact clause-boundary rules
(sentence? normative statement? paragraph?) are not yet fixed — pick a working definition during
the pilot, write it here, and note edge cases you hit.

_Working definition (fill in after pilot):_

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

1. Verified technical errata (e.g. from the RFC Editor's errata system, or RFCScope's dataset)
2. Documented author clarification (mailing list, revision notes)
3. RFCScope-style ambiguity category match
4. Manual annotation showing a later revision resolves multiple plausible interpretations

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

Placeholder — align with RFCScope's seven-subtype taxonomy once the team has reviewed their
public artifact (github.com/HIPREL-Group/RFCScope), rather than inventing a parallel scheme.
List the seven subtypes here once reviewed, with one worked example each from the pilot.

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
