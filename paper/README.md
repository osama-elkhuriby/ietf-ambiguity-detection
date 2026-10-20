# Master's project report source

`main.tex` is the current single-column master's project report. It follows the
required title-page, front-matter, seven-chapter, IEEE-reference, and appendix
structure. It aims to tell the
project-wide research story: literature review; corpus, parsing, and revision alignment;
annotation workflow, disagreement, and target sensitivity; the complete matched
classical/heuristic grid; earlier LOFO, embedding, context, and revision-signal studies;
Qwen V1--V4 prompting and hybrid results; the initial low-prevalence DeBERTa experiment;
later annotation-ready DeBERTa pilot and nested calibration; staged weighting/context
ablations; review-budget and routing feasibility; protocol corrections; and blocked or
incomplete work. It now includes the updated revision-feature ML grid, the fixed
three-model rank ensemble, the implemented local RFC-review GUI, and the completed
analysis of the updated human-reviewed label source. The latest DeBERTa results compare
three methods on one common 2,143-row cohort; the 100 human-confirmed records were
selected from historical OOF errors and the results are explicitly qualified for that
selection bias. The repair analysis validates and reuses parent-run predictions rather
than retraining. Classical ML and earlier Qwen/hybrid studies remain separate
evaluations on their own label snapshots and protocols; their results are not presented
as a common-data leaderboard. The latest 2,425-row classical rerun includes TCP and
DNS, so those families are no longer untouched project-level holdouts even though the
transformer and Qwen development studies exclude them. Blocked annotation v2 remains
unreported. Local-only and `.gitignore`-excluded results are included where aggregate
evidence is available.

Evaluation tracks remain separate when snapshots, cohorts, labels, and protocols
differ. The paper reports negative findings and protocol limitations without turning
the results into one cross-cohort leaderboard.

`references.bib` contains the bibliography and citations. Rebuild `main.pdf`
after changing the source or generated figures and before distribution.

## Presentation

The editable 18-slide project talk and its 25-minute speaker guide are in
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

Upload `main.tex`, `references.bib`, and the complete `figures/` folder, preserving
the relative paths. Select `main.tex` as the main document. The tracked `main.pdf` is
the rebuilt report preview, but it should be rebuilt again after any source change.
The paper incorporates aggregated evidence from
local-only and ignored experiment outputs, but private annotations, row-level
predictions, model weights, and complete run directories are not embedded in the
submission bundle. Their absence limits independent reproduction from a fresh clone.

Figures included by the manuscript:

- `project_pipeline.png`
- `project_cohorts.png`
- `tls_alignment_counts.png`
- `classical_ap_summary.png`
- `ml_model_metric_comparison_current.png`
- `classical_pr_curves_current.png`
- `revision_feature_ablation.png`
- `classical_lofo.png`
- `language_model_ap.png`
- `reviewed_label_transformer_comparison.png`
- `annotation_ready_pilot_pr.png`
- `hybrid_pr_curves.png`
- `hybrid_pr_curves.pdf`
- `hybrid_review_budget.png`
- `hybrid_review_budget.pdf`
- `transformer_review_budget.png`

The reviewed-label transformer comparison figure is based on the aggregate AP, ROC-AUC,
and family-cluster intervals in the local
`final_annotation_human_reviewed_100_v1_analysis_fix_v1/outputs/` directory. It
compares focal loss, partial fine-tuning, and their rank ensemble on the same 2,143-row
cohort. The 100 confirmed records were error-selected; this figure is not an independent
test comparison. The analysis-repair run reuses parent predictions.

The final V4/DeBERTa/classical rank-ensemble figures can be regenerated from the
saved, label-compatible OOF predictions with:

```powershell
python -m models.hybrid.plot_final_rank_ensemble
```

This writes PNG and PDF versions to `paper/figures/`. The chart cohort excludes
17 current records missing from the earlier classical OOF artifact and 28 shared
records whose labels conflict with the current annotation snapshot. The plots are
exploratory development summaries, not independent-test results.

The report-specific revision-ablation figure is regenerated from committed aggregate
metrics with `python paper/generate_report_figures.py` from the repository root.

## Author action required before submission

1. Confirm the School of Computing submission template, declaration requirements,
   cover-page wording, and any required signatures before final deposit.
2. Confirm author order, spelling, affiliations, acknowledgments, and consent.
3. Do not rewrite the development results as final-test performance. The latest
   classical rerun has already consumed TCP and DNS; reserve new families or later
   time-sliced drafts for any future final evaluation.
4. If a new independent evaluation is completed, update the abstract, Results,
   Threats to Validity, Conclusion, and all associated tables together.
5. Obtain an independently adjudicated expert sample before making accuracy or
   deployment claims. The current labels are an operational human/AI-assisted
   reference.
6. Reconcile the candidate-export discrepancy reported in the manuscript (five
   errata-derived rows versus a prior progress report's 121 positives and 242 TLS
   records) before making errata-coverage claims.
7. Review all source outputs and data versions before submission; this is a
   development-stage research report, not evidence of a validated production system.

The paper deliberately calls the repository's `average_precision_score` output
"average precision (AP)" rather than using the less precise historical label "PR-AUC."
Pooled out-of-fold review-budget counts are exploratory because the scores arise from
fold-specific models and are not a deployment calibration. The classical/heuristic,
earlier LOFO, earlier transformer-ablation, Qwen/hybrid, and cross-model rank-ensemble
results have different evaluation cohorts, label definitions, or protocols; the paper
explicitly avoids direct model ranking across those studies.
The three methods in the updated reviewed-label DeBERTa comparison share the same
2,143-row cohort and folds, but this cohort includes 100 error-selected human-confirmed
labels and does not constitute an independent population sample.

The classical precision--recall grid and best-model metric summary were regenerated
from the current matched result artifacts. The classical table reports every model at
its recorded 0.5 threshold. Revision features slightly reduce AP for all three
classifiers in the latest rerun, while thresholded F2 is unchanged or improves. The
older `results/revision_features/summary.md` describes a superseded 2,406-row run and
must not be used as the current aggregate. The latest updated-label DeBERTa analysis
is complete, but the 100
human-confirmed records were selected from historical OOF errors. The analysis-repair
step validates and reuses parent experiment predictions, so it does not constitute an
independent rerun. Obtain a separately sampled expert-adjudicated test set before making
generalization or deployment claims.
