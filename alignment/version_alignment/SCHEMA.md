# Schema 1.0

`schema.json` describes the Pydantic `Document`, `Alignment`, and `Evidence`
contracts. A regression test checks synchronization with the code.

Document envelopes store family, document, revision, reported publication date,
conservative availability and precision, source/input paths and hashes, preparation
status, warnings and clauses. Clauses retain existing segmentation fields and
extensions plus normalized text and normalization steps. Original text/spans remain
unchanged. Family/date metadata is shared through the document envelope.

Alignments store endpoint IDs, dates, relation, textual/structural changes, component
scores, confidence, review reasons/status, alternatives, lineage and provenance.

| Relation | Source count | Target count |
|---|---:|---:|
| UNCHANGED / MODIFIED / MOVED | 1 | 1 |
| INSERTED | 0 | 1 |
| DELETED | 1 | 0 |
| SPLIT | 1 | 2 or more |
| MERGED | 2 or more | 1 |
| UNCERTAIN | Any unresolved nonempty endpoint set | |

UNCERTAIN is not an asserted Cartesian product of links. Alternatives can reference
assigned occurrences but are not selected relationships. Each occurrence receives
one disposition per document pair. Movement and textual change are separate; a
moved split retains SPLIT with movement metadata. Undefined scores are null;
nonfinite values are rejected. Confidence probabilities remain null in this MVP.
Raw alignment contracts reject ambiguity labels and revision-reason fields.

Lineage nodes retain occurrence IDs, incoming relationship, parents and source hash.
One-to-one continuations keep identity; splits/merges create descendants. Later
merges never rename earlier nodes. Uncertain ancestry does not manufacture links.

Review bundles contain endpoint text and empty human review fields. They are
annotation material, not model features. Proposed corrections belong in separate
files; automatic outputs are immutable. Correction application is future work.
Gold import requires reviewer identity and complete reviewed document-pair scopes.
Never set `scope_complete=true` for isolated samples.

`created_at` is execution metadata; stable IDs exclude it. Partial failures contain
`failure.json` and no successful manifest. Artifact checksums are in the final
manifest; prepared input snapshot hashes are distinct from original raw-source hashes.
