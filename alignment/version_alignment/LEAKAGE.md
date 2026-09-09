# Prediction-time boundaries

Offline final-draft-to-RFC alignment can support later labels, but not earlier draft
features. `prediction_view` selects only the target draft and previous revisions
before extracting features. RFCs and outgoing future edges are excluded.

The recursive evidence guard requires `MODEL_FEATURE`, known availability no later
than the target, and admissible dependencies. `LABEL_SUPPORT`,
`ANNOTATION_METADATA`, undated evidence and future evidence raise actionable errors.
External evidence on the same day is excluded when intra-day order is unknown.
The target's own text remains admissible.

Validated numeric predecessor revisions in the same document family are also
admissible on the same submission date: TLS-00 and TLS-01 both have 2014-04-17.
The timeline establishes 00 before 01 without inventing timestamps. This exception
does not admit later revisions, same-day external documents, errata or discussions.
All target prefixes are checked before embedding or alignment work begins.

Draft availability uses submission dates. Month-only RFC dates retain precision and
use a conservative month-end bound; first-of-month placeholders are not treated as
exact publication times. Chronology conflicts require better evidence.

`resolve_reference` resolves exact identifiers or the latest admissible version of
an explicitly unversioned family from the supplied local catalog. It never fetches
today's latest document. Missing/symbolic references remain unresolved. External
discussion retrieval and symbolic bibliography expansion are future work.

Basic features include prior observed edits, uncertainty/incomplete-history flags,
observed revision count, elapsed age, previous lexical/semantic distance and surface
change signatures. Shared ancestral edit events are counted once across merges.
Future stability, final revision count, days to publication, later errata, discussions
and revision labels are excluded. Appending later revisions must leave earlier
features and lineage prefixes unchanged; regression tests enforce this.

Metadata such as current RFC status is not a feature column. IDs and hashes are join
and provenance fields; downstream models must use an explicit feature allowlist.
Rule/LLM classes remain available for alignment review but are not numeric ambiguity
features. Reviewed offline alignments cannot replace automatic prediction history
without an evidence policy; the current feature view rejects them.

The modern pinned encoder supports retrospective analysis, not a claim that its
weights existed at every historical target date or excluded RFCs from pretraining.
This limitation is separate from explicit source-evidence leakage. Lexical runs
provide a reproducible model-free comparison. No generalization or benchmark claim
is made for this single-family milestone.
