# TLS full-candidate human review

This package contains only automatically generated TLS candidates with uppercase
RFC 8174 requirement keywords (`MUST`, `SHOULD`, `MAY`, and related forms) for
independent human review. The automatic files under `silver/` remain unchanged.

Two reviewers independently complete `reviewer_a.csv` and `reviewer_b.csv`.
They fill only the `human_*` columns. Once both reviews are complete, record
the agreed final labels and rationale in `adjudication.csv`; set
`adjudication_status` to `ADJUDICATED` for every resolved row.

## How to label one row

1. Read `clause_text` and the linked erratum when `errata_url` is present.
2. Decide whether the wording permits two plausible technical readings that
   would produce different implementation behavior.
3. Complete the following columns. Leave all automatic silver columns
   unchanged, including `weak_label`, `ambiguity_label`, `label_tier`, and
   `ground_truth`.

| Column | Allowed value / required content |
|---|---|
| `human_is_ambiguous` | `yes`, `no`, or `uncertain` |
| `human_label_tier` | `gold` for a supported `yes` or `no`; `uncertain` otherwise |
| `human_revision_reason` | `ambiguity_clarification`, `editorial_change`, `new_functionality`, `terminology_change`, `reference_correction`, or `other_technical_redesign`; blank if no revision comparison is needed |
| `human_ambiguity_subtype` | `I-1`, `I-2`, `I-3`, `U-1`, `U-2`, `U-3`, `U-4` for a `yes` label; blank for `no` or `uncertain` |
| `human_evidence_span` | Exact relevant words from `clause_text` |
| `annotator` | Your name or initials |
| `date_annotated` | `YYYY-MM-DD` |
| `human_notes` | One or two sentences explaining the decision |
| `human_positive_evidence` | For a final `yes` only: `ERRATUM`, `DRAFT_CLARIFICATION`, or `BOTH`; blank for `no` and `uncertain` |

Keep `adjudication_status` as `PENDING` in reviewer files. Do not consult the
other reviewer’s file until both reviews are complete.

### Decision examples

- Choose `yes` when a phrase such as an undefined condition permits two
  technically reasonable behaviors. State both readings in `human_notes`.
- Choose `no` when the condition, actor, and required behavior are explicit.
  State why only one reasonable reading remains.
- Choose `uncertain` when deciding would require evidence that is unavailable;
  explain what evidence is missing.

For an `ERRATUM` label, verify that the erratum exposes two plausible readings;
an erratum can be a typo or technical repair rather than ambiguity. For
`DRAFT_CLARIFICATION`, compare the earlier and later draft wording. Use it only
when the newer wording removes one plausible behavior without changing the
intended protocol behavior. Do not infer ambiguity from an ordinary edit.

Review every row even when it is linked to a verified technical erratum. An
erratum can be a typo or a technical repair rather than an ambiguity. A gold
positive needs an evidence span and an explanation of two plausible technical
readings; a gold negative needs an explanation of why the clause has one
reasonable reading. Use `uncertain` if the evidence cannot support either.

These rows include the same clause lineage across several draft revisions.
They are useful to show how wording changed, but are not independent examples.
When training and testing, all rows sharing a `lineage_ids` value must be in
the same split.
