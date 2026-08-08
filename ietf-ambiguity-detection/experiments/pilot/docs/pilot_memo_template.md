# Pilot Memo — Week 1–2

**Team:** Mostafa Elofy, Mohamed Ghanem, Mohamed Abdelkhalek, Osama Elkhuribi
**Date:**
**For:** Prof. Yuanzhu Chen, Dr. Ismail Anwar

Purpose: decide, with real (small-scale) evidence, whether the plan in the proposal survives
contact with real data — before committing Phase 2's four weeks to the full corpus build.

## 1. What we pulled

- Protocol family:
- Draft(s) and revision count:
- Source: `ietfdata` / manual download (circle one)
- Any gaps or surprises in what the Datatracker actually returned:

## 2. What we hand-labeled

- Clauses aligned across versions:
- Clauses labeled (gold / silver / uncertain breakdown):
- **Real annotator time per clause:** _(this number drives whether "several thousand
  clause-version instances" in SS6.3 is realistic on our timeline — be honest here)_
- Revision-reason label distribution:

## 3. What the baseline detectors did

- Heuristic detector: flagged ___ of ___ clauses. Agreement with hand labels:
- LLM prompt used (model, prompt version):
- **Real LLM cost per clause:** $___ — extrapolated to SS6.3's target corpus size: $___
- Cases where heuristic and hand-label disagreed — what caused it:

## 4. What surprised us

_The most useful section. Anything that didn't match the proposal's assumptions goes here,_
_even (especially) if it's inconvenient._

## 5. Reuse vs. novel

What we're building on directly vs. reimplementing vs. genuinely new:

| Component | Reusing | Building ourselves | Why |
|---|---|---|---|
| Errata-derived labels | RFCScope dataset | | |
| Datatracker access | `ietfdata` | | |
| Draft revision diffing | `iddiff` (if used) | | |
| Ambiguity taxonomy | RFCScope 7 subtypes (if adopted) | | |
| Pre-publication (draft-stage) framing | | Novel — RFCScope covers published RFCs only | |
| Hybrid ML→LLM routing | | Novel per proposal | |

## 6. Recommendation

- [ ] Proceed to Phase 2 as scoped
- [ ] Proceed to Phase 2 with these adjustments: _____
- [ ] Need advisor input before proceeding on: _____

## 7. Questions for Prof. Chen / Dr. Anwar

-
-
