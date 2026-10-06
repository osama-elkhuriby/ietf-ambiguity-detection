# draft-ietf-httpbis-semantics label lifecycle

This directory contains automatically prepared review candidates for
`draft-ietf-httpbis-semantics` through `RFC9110`.

- `silver/` contains reproducible automatic evidence and candidates.
- `review/` contains pending independent-review CSVs.
- `gold/` is reserved for completed, adjudicated records.

Only clauses classified `NORMATIVE` by the frozen first-stage classifier are
eligible. `SILVER_CHANGE_CANDIDATE` means that a normative clause was later
substantively revised; the edit is a review signal, not proof of ambiguity.
`SILVER_CLEAR_CANDIDATE` means identical normative text persisted for at least
three versions; stability is not proof of clarity. Every automatic record keeps
`ambiguity_label=UNCERTAIN`, `ground_truth=false`, and requires human review.
