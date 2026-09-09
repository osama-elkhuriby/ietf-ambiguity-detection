# TLS alignment review

Use existing Phase-2 sentence/list clauses and their block context. Review
correspondence independently of ambiguity and revision reasons.

Start with the generated 150-record `review_sample.jsonl`. In a separate annotation
copy record reviewer, date, proposed relation, corrected endpoint IDs and rationale.
Do not modify automatic outputs.

1. Inspect both texts, provenance, surrounding sections and alternative candidates.
2. A changed number/modal/actor can still be a correct MODIFIED link.
3. For splits/merges, verify all members contribute and inspect nearby omitted text.
4. Distinguish section renumbering from genuine movement.
5. Search the opposite document before accepting insertion/deletion.
6. Retain UNCERTAIN if correspondence cannot be established.

Synthetic examples: `client MUST send 10` → `client MUST send 20` may be MODIFIED,
never automatically ambiguity clarification. One clause containing two requirements
may SPLIT into two corresponding clauses. Identical text after section renumbering
is UNCHANGED unless structure shows a real move. Repeated generic text without
disambiguating context remains UNCERTAIN.

Error categories: missed/false link, correct link/wrong relation, incomplete group,
false move, duplicate boilerplate, parser boundaries, missing context, classifier
variability. Prefer independent second review and adjudication.

This relation/confidence-stratified sample overrepresents difficult cases. Its raw
precision is not corpus-weighted; isolated samples cannot measure full recall.
The MVP evaluator accepts complete reviewed document-pair scopes. Only those scopes
may use `scope_complete=true`; section-scoped evaluation is future work. Every link
in a final high-quality gold subset needs complete verification. Synthetic fixtures
and automatic classifications are not human-reviewed gold.
