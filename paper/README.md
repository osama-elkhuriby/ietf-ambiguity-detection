# Master's project report source

`main.tex` is the single-cohort master's project report. It follows the required
title-page, front-matter, seven-chapter, IEEE-reference, and appendix structure. All
quantitative results in the report use the same frozen evaluation dataset: 2,143
clauses, 138 positive labels, 2,005 negative labels, and nine draft families. The
reported comparison is limited to focal-loss DeBERTa, partial fine-tuning, and their
equal-weight ECDF rank ensemble evaluated with the same family-isolated folds. The
report treats the labels as an operational reference and the metrics as development
evidence, not independent expert-gold or deployment results.

`references.bib` contains the bibliography and citations. The tracked `main.pdf`
predates the current report source and is stale; do not distribute it. Rebuild the
PDF after changing the source or generated figures and before distribution.

## Presentation

The editable 20-slide project talk and its approximately 27-minute speaker guide are in
[`presentation/`](presentation/). The PowerPoint deck is generated from editable
native text, shapes, and charts; see [`presentation/README.md`](presentation/README.md)
for editing and regeneration instructions.

## Build

Use a TeX distribution containing the standard `report` class and the
`IEEEtran.bst` bibliography style, then run:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

or:

```text
latexmk -pdf main.tex
```

Run these commands from this directory so BibTeX finds `references.bib`.

## Overleaf

Upload `main.tex` and `references.bib`, preserving their relative paths, and select
`main.tex` as the main document. The report uses tables rather than external figure
files. Private annotation records, row-level predictions, model weights, and complete
run directories are not part of the submission bundle, which limits independent
reproduction from a fresh clone.

## Author action required before submission

1. Confirm the School of Computing submission template, declaration requirements,
   cover-page wording, and any required signatures before final deposit.
2. Confirm author order, spelling, affiliations, acknowledgments, and consent.
3. Confirm that the frozen cohort count is reported as 2,143 clauses with 138 positive
   and 2,005 negative labels; these counts match the experiment manifest.
4. Do not present the OOF development metrics as independent test or deployment
   performance. The labels are an operational reference, not independently
   adjudicated expert gold.
5. Review author details, data-handling requirements, citations, and the rebuilt PDF
   before submission.

The report uses "average precision (AP)" for the ranking metric. Its model comparisons
share one dataset and one family-isolated evaluation protocol. Pooled OOF review-budget
counts are retrospective and should not be interpreted as calibrated deployment
precision or expected reviewer yield.
