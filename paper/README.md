# IEEE paper source

`main.tex` is the current IEEE conference-format manuscript, updated on 2026-10-20
against the classical/heuristic, earlier LOFO, latest hash-pinned transformer, and
separate Qwen/hybrid development reports. These results are intentionally reported
as separate cohorts, not a single head-to-head comparison.
`references.bib` contains the bibliography and citations. `main.pdf` is an older
2026-10-19 preview and does **not** reflect the current source; rebuild it before
distribution or submission. The manuscript references the PNG figures in `figures/`.

## Build

Use a TeX distribution containing `IEEEtran`, then run:

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

Upload `main.tex`, `references.bib`, and the complete `figures/` folder, preserving
the relative paths. Select `main.tex` as the main document. Do not upload or rely on
the stale `main.pdf` preview.

Figures included by the manuscript:

- `project_pipeline.png`
- `project_cohorts.png`
- `tls_alignment_counts.png`
- `ml_heuristic_ap.png`
- `classical_lofo.png`
- `language_model_ap.png`
- `deberta_ap.png`
- `transformer_review_budget.png`

## Author action required before submission

1. Replace `[corresponding-author email to be supplied]` in `main.tex` with the
   corresponding author's institutional email.
2. Select the actual IEEE venue and apply its page limit, copyright notice, anonymity,
   and artifact-availability rules. Use the anonymous author block if double-blind review
   is required.
3. Confirm author order, spelling, affiliations, acknowledgments, and consent.
4. Do not rewrite the development results as final-test performance. TCP and DNS were
   not opened in the reported experiment.
5. If those evaluations are later authorized and completed, update the abstract, Results,
   Threats to Validity, Conclusion, artifact commit, and all associated tables together.
6. Obtain an independently adjudicated expert sample before making accuracy or deployment
   claims. The current labels are an operational human/AI-assisted reference.

The paper deliberately calls the repository's `average_precision_score` output
"average precision (AP)" rather than using the less precise historical label "PR-AUC."
It also labels pooled out-of-fold review-budget counts as exploratory because the
scores arise from fold-specific models and are not a deployment calibration.
The classical/heuristic, older LOFO, Qwen/hybrid, and latest DeBERTa results have
different evaluation cohorts or protocols; the paper explicitly avoids direct model
ranking across these studies.
