# Stage A — Family-Specific Recall Diagnostics

## Provenance and scope

- Baseline: Experiment 2, protocol SHA-256 `242f98c56d88a11dea3b355c4c3b14c693029e3d8a607fdd7a067f29af630b36`.
- Prediction artifact SHA-256: `56789213c4e5cf653f7c253f506d2df0ae43ad39e01d489eee5657f5c9a35c74`.
- Effective cohort SHA-256: `a25e99563563f71570e48f63b9515fbb51ff845762ff718763d98e0b3eefae2f`.
- Rows verified: 2142 unique records, one prediction per frozen cohort record.
- Read-only analysis; historical predictions and checkpoints were not modified.

## Per-fold ranking and class score distributions

| Fold | Rows | Yes | Prevalence | AP | ROC-AUC | Predicted Yes @ 0.5 | Positive scores (min / median / mean / max) | Negative scores (min / median / mean / max) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 793 | 46 | 0.0580 | 0.2054 | 0.6927 | 11 | 0.0041 / 0.0051 / 0.1261 / 0.5333 | 0.0040 / 0.0042 / 0.0198 / 0.5506 |
| 2 | 385 | 32 | 0.0831 | 0.2280 | 0.8011 | 0 | 0.0094 / 0.0128 / 0.0135 / 0.0205 | 0.0081 / 0.0098 / 0.0106 / 0.0272 |
| 3 | 314 | 10 | 0.0318 | 0.2811 | 0.7007 | 11 | 0.0020 / 0.0036 / 0.3967 / 0.9925 | 0.0020 / 0.0025 / 0.0243 / 0.9929 |
| 4 | 299 | 18 | 0.0602 | 0.3804 | 0.8108 | 12 | 0.0016 / 0.0028 / 0.3071 / 0.9959 | 0.0015 / 0.0017 / 0.0227 / 0.9956 |
| 5 | 351 | 20 | 0.0570 | 0.3078 | 0.7832 | 20 | 0.0025 / 0.0052 / 0.2976 / 0.9886 | 0.0023 / 0.0027 / 0.0457 / 0.9879 |

P90, P95, and P99 values for both classes are included in `stage_a_fold_diagnostics.json`; every requested threshold's TP/FP/TN/FN, precision, recall, F1, and F2 is in `stage_a_per_fold_thresholds.csv`. The score-distribution charts are in `plots/stage_a/`.

## Fold 2 diagnosis

Fold 2 contains 32 positives among 385 records (prevalence 8.31%), AP 0.2280, and ROC-AUC 0.8011. Its positive score median is 0.0128 and maximum is 0.0205; the frozen 0.5 rule predicts 0 positives.

Evidence: the reported AP/ROC-AUC and separated score ranks indicate useful ranking within this fold, while all positive scores below 0.5 explain why the global threshold yields zero recall there. This is threshold-induced zero recall on a fold with ranking signal; it does not by itself establish well-calibrated probabilities or prove a single global threshold transfers across families.

## Interpretation and limitations

- **Threshold versus ranking:** Fold-level ranking is materially better than chance in some families, while the 0.5 decision rule yields zero recall in Fold 2. The held-out threshold sweep is descriptive only and cannot choose a threshold for this study.
- **Family-dependent score behavior:** Variations in positive score summaries and fixed-threshold counts are consistent with family-dependent score scales; these observations do not prove calibration error without fitting-side reliability analysis.
- **Score compression:** Any concentration at low probabilities is directly visible in the per-fold histograms and score quantiles. Do not interpret low scores as evidence of absent ranking signal.
- **Weak ranking and minority learning:** AP and class-conditional score overlap quantify remaining discrimination limits. The small positive counts per family make these estimates variable.
- **Missing semantic context:** Target-only predictions cannot establish that context would help; that requires the frozen Stage C ablation.

## Overall retrieval baseline

The top-K table in `stage_a_top_k.csv` reports actual-positive retrieval at the fixed review budgets K=10, 25, 50, 100, and 200. No operating threshold was selected.
