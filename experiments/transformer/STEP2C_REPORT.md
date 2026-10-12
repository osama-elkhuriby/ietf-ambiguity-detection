# Step 2c feasibility and noise audit

Status: complete using existing out-of-fold predictions only. No GPU training was run,
Step 3 was not started, and the sealed DNS test was not opened.

## Routing definition

The single-threshold router refers every row whose ambiguity score is greater than or
equal to the threshold. Rows below it are handled locally as non-ambiguous. Local
coverage is the fraction handled locally; local selective risk is the ambiguous-class
fraction among those locally handled rows. All boundary ties are referred.

The prevalence-matched no-skill model assigns the observed prevalence to every row, so
a deterministic single threshold must refer either every tied row or none; it therefore
requires 100% referral to achieve any positive recall target. The conceptual oracle
perfectly ranks positives and provides the lower referral bound. Pooled bootstrap
resampling preserves fold sizes. Every reported interval uses 2,000 replicates.

## Pooled referral feasibility

Entries are referral rate with 95% bootstrap interval.

| Model | Recall 80% | Recall 90% | Recall 95% |
| --- | ---: | ---: | ---: |
| TF-IDF unweighted | 0.611 [0.557, 0.655] | 0.755 [0.710, 0.780] | 0.834 [0.773, 0.881] |
| TF-IDF class | 0.611 [0.552, 0.659] | 0.751 [0.697, 0.796] | 0.846 [0.795, 0.885] |
| DeBERTa seed 1729 | 0.685 [0.644, 0.732] | 0.851 [0.800, 0.883] | 0.916 [0.884, 0.943] |
| DeBERTa seed 2718 | 0.702 [0.648, 0.745] | 0.841 [0.789, 0.859] | 0.885 [0.856, 0.922] |
| DeBERTa logit average | 0.638 [0.586, 0.706] | 0.819 [0.774, 0.866] | 0.891 [0.866, 0.935] |
| Prevalence-matched no skill | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| Oracle | 0.141 [0.128, 0.154] | 0.159 [0.145, 0.174] | 0.168 [0.153, 0.183] |

At 90% recall, TF-IDF avoids about 25% of LLM calls and the current DeBERTa ensemble
about 18%. These are meaningful but modest development-set savings. They are not yet a
deployable guarantee because thresholds are uncalibrated and transfer to TCP has not
been checked. At 95% recall, savings fall to about 15%-17% for TF-IDF and 11% for the
ensemble.

Per-fold 90%-recall referral rates demonstrate the variability:

| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF unweighted | 0.744 | 0.779 | 0.789 | 0.663 | 0.805 |
| TF-IDF class | 0.761 | 0.764 | 0.789 | 0.670 | 0.802 |
| DeBERTa seed 1729 | 0.781 | 0.740 | 0.826 | 0.628 | 0.766 |
| DeBERTa seed 2718 | 0.769 | 0.764 | 0.709 | 0.728 | 0.694 |
| DeBERTa logit average | 0.741 | 0.757 | 0.765 | 0.615 | 0.744 |

The complete pooled and per-fold table for all three recall targets, including every
bootstrap interval, is `governance/step2c_feasibility_v1/referral_operating_points.csv`.
The full empirical risk-coverage curves are in `risk_coverage_curves.csv`.

## Seed variance and ensemble

| Fold | Seed 1729 | Seed 2718 | Two-seed SD | Logit average | TF-IDF unweighted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3356 | 0.3432 | 0.0054 | 0.3424 | 0.3249 |
| 2 | 0.2816 | 0.3125 | 0.0219 | 0.3132 | 0.3156 |
| 3 | 0.1972 | 0.2477 | 0.0357 | 0.2401 | 0.2000 |
| 4 | 0.3384 | 0.3523 | 0.0098 | 0.3364 | 0.3135 |
| 5 | 0.2558 | 0.3207 | 0.0459 | 0.2985 | 0.2575 |

Mean two-seed SD across folds is 0.0237; the maximum is 0.0459. The logit-average
ensemble has mean-fold PR-AUC 0.3061 and pooled PR-AUC 0.2771. Against unweighted
TF-IDF, the mean-fold difference is 0.0238 and pooled difference is 0.0060. The pooled
95% paired-row interval is [-0.0214, 0.0346], and the section-cluster interval is
[-0.0221, 0.0360]. Against class-weighted TF-IDF, pooled difference is -0.0014 and both
intervals also cross zero. H1 therefore remains unsupported.

## Inner validation audit

| Outer fold | Inner family | Rows | Scored | Positives | Epochs: 1e-5 (1729/2718) | Epochs: 2e-5 (1729/2718) | Epoch: 3e-5 (1729) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | QUIC HTTP | 339 | 327 | 47 | 6 / 3 | 6 / 3 | 6 |
| 2 | QUIC Recovery | 164 | 162 | 34 | 1 / 7 | 1 / 1 | 1 |
| 3 | HTTP Semantics | 407 | 407 | 80 | 3 / 5 | 6 / 3 | 1 |
| 4 | QUIC Recovery | 164 | 162 | 34 | 5 / 4 | 1 / 6 | 1 |
| 5 | HTTP/2bis | 262 | 249 | 34 | 3 / 4 | 3 / 3 | 3 |

No inner family has fewer than 20 scored positives. The limitation is instead data
comparability: TF-IDF fits all outer-training families, while DeBERTa reserves the
listed family for checkpoint selection and therefore fits fewer rows and families.

## Step 3 governance frozen before training

Step 3 uses LR 1e-5, the same folds and inner assignment, and seeds 1729, 2718, and
31415. Existing unweighted cells for the first two seeds are reused; only 50 missing
cells may be trained. Selection uses seed-averaged per-fold PR-AUC and the paired-fold
one-SE rule, preferring simpler weighting on a tie. Required clean contrasts are
unweighted versus confidence, class versus combined, and class versus unweighted.
TF-IDF receives the same four weighting comparisons. HIGH-only is no longer a training
regime; a future versioned vote-fraction artifact enables soft-target sensitivity.

## Runtime and verification

The CPU-only audit, including all 2,000-replicate bootstraps, completed in approximately
92 seconds wall time. It performed no GPU training. Ruff passed on all three modified
Python/test files, and the full repository suite passed 203 tests.
