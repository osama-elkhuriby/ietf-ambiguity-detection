# IEEE paper source

`main.tex` is the current IEEE conference-format manuscript. It aims to tell the
project-wide research story: literature review; corpus, parsing, and revision alignment;
annotation workflow, disagreement, and target sensitivity; the complete matched
classical/heuristic grid; earlier LOFO, embedding, context, and revision-signal studies;
Qwen V1--V4 prompting and hybrid results; the initial low-prevalence DeBERTa experiment;
later annotation-ready DeBERTa pilot and nested calibration; staged weighting/context
ablations; review-budget and routing feasibility; protocol corrections; and blocked or
incomplete work. Local-only and `.gitignore`-excluded results are included where
aggregate evidence is available. Missing results (for example, blocked annotation v2,
TCP, and sealed DNS) are identified as not run, not filled in or implied.

Evaluation tracks remain separate when snapshots, cohorts, labels, and protocols
differ. The paper reports negative findings and protocol limitations without turning
the results into one cross-cohort leaderboard.

`references.bib` contains the bibliography and citations. `main.pdf` is an older
preview and does **not** reflect the current source; rebuild it before distribution
or submission.

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
the stale `main.pdf` preview. The paper incorporates aggregated evidence from
local-only and ignored experiment outputs, but private annotations, row-level
predictions, model weights, and complete run directories are not embedded in the
submission bundle. Their absence limits independent reproduction from a fresh clone.

Figures included by the manuscript:

- `project_pipeline.png`
- `project_cohorts.png`
- `tls_alignment_counts.png`
- `ml_heuristic_ap.png`
- `ml_model_metric_comparison.jpg`
- `classical_pr_curves.png`
- `classical_lofo.png`
- `language_model_ap.png`
- `deberta_ap.png`
- `annotation_ready_pilot_pr.png`
- `hybrid_pr_curves.png`
- `hybrid_pr_curves.pdf`
- `hybrid_review_budget.png`
- `hybrid_review_budget.pdf`
- `transformer_review_budget.png`

The final V4/DeBERTa/classical rank-ensemble figures can be regenerated from the
saved, label-compatible OOF predictions with:

```powershell
python -m models.hybrid.plot_final_rank_ensemble
```

This writes PNG and PDF versions to `paper/figures/`. The chart cohort excludes
17 current records missing from the earlier classical OOF artifact and 28 shared
records whose labels conflict with the current annotation snapshot. The plots are
exploratory development summaries, not independent-test results.

## Author action required before submission

1. Replace `[corresponding-author email to be supplied]` in `main.tex` with the
   corresponding author's institutional email.
2. Select the actual IEEE venue and apply its page limit, copyright notice, anonymity,
   and artifact-availability rules. Use the anonymous author block if double-blind
   review is required.
3. Confirm author order, spelling, affiliations, acknowledgments, and consent.
4. Do not rewrite the development results as final-test performance. TCP and DNS were
   not opened in the reported experiment.
5. If those evaluations are later authorized and completed, update the abstract,
   Results, Threats to Validity, Conclusion, and all associated tables together.
6. Obtain an independently adjudicated expert sample before making accuracy or
   deployment claims. The current labels are an operational human/AI-assisted
   reference.
7. Reconcile the candidate-export discrepancy reported in the manuscript (five
   errata-derived rows versus a prior progress report's 121 positives and 242 TLS
   records) before making errata-coverage claims.
8. Review all source outputs and data versions before submission; this is a
   development-stage research report, not evidence of a validated production system.

The paper deliberately calls the repository's `average_precision_score` output
"average precision (AP)" rather than using the less precise historical label "PR-AUC."
Pooled out-of-fold review-budget counts are exploratory because the scores arise from
fold-specific models and are not a deployment calibration. The classical/heuristic,
earlier LOFO, earlier transformer-ablation, Qwen/hybrid, and final target-only DeBERTa
results have different evaluation cohorts, label definitions, or protocols; the paper
explicitly avoids direct model ranking across these studies.
