# Step 2b protocol-audit report

Status: complete. No weighting, label-regime, context, calibration, or final-test
experiment was started. The DNS sealed test was not opened, and no final lock exists.

## Protocol finding and correction

In `lr-screen-v1`, the held-out outer fold supplied the PR-AUC used for both early
stopping and best-checkpoint selection. The selected checkpoint was then reported on
that same outer fold. TF-IDF did not tune a hyperparameter on the held-out fold: its
vectorizer and fixed logistic-regression configuration were fitted only on each
fold's fitting rows.

Across the 15 v1 cells, selected-epoch PR-AUC exceeded final-epoch PR-AUC by 0.0314
on average, epoch-3 PR-AUC by 0.0385, and epoch-5 PR-AUC by 0.0195 where available.
The cell-level audit is in
`governance/step2b_protocol_audit_v1/v1_checkpoint_selection.csv`.

The corrected trainer removes one deterministic, hash-pinned family from every outer
training fold for inner checkpoint selection. Fitting, inner selection, and outer
evaluation record IDs are disjoint. The outer loader is materialized only after the
best inner-selected checkpoint is loaded, and each manifest records zero outer-fold
evaluations during training and one afterward. The assignment SHA-256 is
`7db817c91e962a6a921942e14bc2d41737bd1060a111a8c7fb798bb929054616`.

## Corrected learning-rate screen

All rows below are target-only, HIGH+MEDIUM, unweighted, fp32, seed 1729. Exact
cross-boundary duplicate rows are excluded consistently from both models' scores.

| LR | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean fold | Pooled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-5 | 0.3356 | 0.2816 | 0.1972 | 0.3384 | 0.2558 | 0.2817 | 0.2563 |
| 2e-5 | 0.3782 | 0.2961 | 0.2070 | 0.2369 | 0.2237 | 0.2683 | 0.2474 |
| 3e-5 | 0.3349 | 0.2689 | 0.1509 | 0.2620 | 0.2835 | 0.2600 | 0.2578 |
| TF-IDF | 0.3249 | 0.3156 | 0.2000 | 0.3135 | 0.2575 | 0.2823 | 0.2712 |
| No skill | 0.2076 | 0.1966 | 0.1437 | 0.1359 | 0.1476 | — | 0.1765 |

For selected 1e-5, the mean-fold difference from TF-IDF is -0.0006 and the pooled
difference is -0.0149. The pooled 95% paired-row interval is [-0.0461, 0.0165], and
the section-cluster interval is [-0.0448, 0.0168]. Detailed per-fold confidence
intervals and secondary threshold-0.5 metrics are in
`results/lr-screen-v2/analysis-primary-final/fold_metrics.csv`.

The top-two mean paired difference was 0.0134 with standard error 0.0251. Following
the unchanged frozen rule, 1e-5 was selected and the top two rates were rerun at seed
2718. Mean-fold PR-AUC was 0.3153 for 1e-5 and 0.2865 for 2e-5; their difference was
0.0288 with standard error 0.0131. The second seed therefore did not contradict the
fixed selection. Its selected-rate pooled PR-AUC was 0.2774 versus TF-IDF 0.2712;
the section-cluster difference interval was [-0.0235, 0.0362].

## Confidence tiers and planned weights

The full training set contains 713 HIGH rows with 17 positives and 1,540 MEDIUM rows
with 378 positives. Fold 3 has no HIGH positive, so its HIGH-only PR-AUC is undefined.
All per-fold tier counts and PR-AUC values for TF-IDF and all three transformer rates
are in `analysis-primary-final/tier_metrics.csv`.

| Planned weighting | Positive effective weight | Negative effective weight | Positive share |
| --- | ---: | ---: | ---: |
| Unweighted | 395.000 | 1,858.000 | 0.1753 |
| Class | 1,126.500 | 1,126.500 | 0.5000 |
| Confidence | 312.959 | 1,940.041 | 0.1389 |
| Combined | 972.010 | 1,280.990 | 0.4314 |

Confidence-only weighting decreases positive mass because positive labels are heavily
concentrated in MEDIUM-tier rows. Fold-local values are saved in
`governance/step2b_protocol_audit_v1/effective_weight_totals.csv`; this is disclosed
before any weighting result exists.

## Reproducibility and runtime

The corrected primary screen contains 15 cells and used 896.5 seconds total runtime.
The required second-seed check contains 10 cells and used 711.8 seconds. Peak allocated
GPU memory was 2,951.8 MiB and peak reserved memory was 3,768 MiB. Every prediction
file preserves `logit_0` and `logit_1`.

The authoritative analysis directories are `results/lr-screen-v2/analysis-primary-final`
and `results/lr-screen-v2/protocol-comparison-v3-final`. Earlier comparison directories
carry explicit `SUPERSEDED.md` notices and must not be cited.
