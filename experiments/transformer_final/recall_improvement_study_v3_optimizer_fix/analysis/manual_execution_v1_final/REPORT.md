# Recall Improvement Study V3 — Completed-Run Analysis

Frozen V3 protocol SHA-256: `9502e659c1f246f433ce534fa7165959a6519c6637c71b3be0289f6638e72c51`.

Only completed five-fold stages are included in aggregate performance tables. Incomplete or failed stages remain listed as incomplete in the analysis manifest.

## Pooled results

| Model | Operating point | AP | ROC-AUC | Precision | Recall | F1 | F2 | TP | FP | TN | FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_fine_tuning | threshold_0.5 | 0.2332 | 0.7286 | 0.3704 | 0.1587 | 0.2222 | 0.1792 | 20 | 34 | 1982 | 106 |
| full_fine_tuning | nested_fold_threshold | 0.2332 | 0.7286 | 0.2609 | 0.2857 | 0.2727 | 0.2804 | 36 | 102 | 1914 | 90 |
| local_context | threshold_0.5 | 0.1592 | 0.7303 | 0.3333 | 0.0317 | 0.0580 | 0.0388 | 4 | 8 | 2008 | 122 |
| local_context | nested_fold_threshold | 0.1592 | 0.7303 | 0.1715 | 0.3730 | 0.2350 | 0.3021 | 47 | 227 | 1789 | 79 |
| partial_4 | threshold_0.5 | 0.3008 | 0.7950 | 0.4348 | 0.0794 | 0.1342 | 0.0949 | 10 | 13 | 2003 | 116 |
| partial_4 | nested_fold_threshold | 0.3008 | 0.7950 | 0.2071 | 0.6032 | 0.3083 | 0.4363 | 76 | 291 | 1725 | 50 |
| focal_loss | threshold_0.5 | 0.3295 | 0.7921 | 0.3462 | 0.3571 | 0.3516 | 0.3549 | 45 | 85 | 1931 | 81 |
| focal_loss | nested_fold_threshold | 0.3295 | 0.7921 | 0.0588 | 1.0000 | 0.1111 | 0.2381 | 126 | 2016 | 0 | 0 |
| partial_2 | threshold_0.5 | 0.1834 | 0.7489 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 2016 | 126 |
| partial_2 | nested_fold_threshold | 0.1834 | 0.7489 | 0.0816 | 0.7063 | 0.1463 | 0.2790 | 89 | 1002 | 1014 | 37 |

## Included outputs

- `pooled_metrics.csv`: pooled AP, ROC-AUC, precision, recall, F1, F2, and confusion counts.
- `mean_fold_metrics.csv`: mean-fold AP and the other fold-level metrics.
- `top_k_retrieval.csv`: frozen Top-K review budgets.
- `family_metrics.csv`: family-level ranking and classification performance.
- `threshold_calibration.csv` and `threshold_calibration_performance.csv`: fitting-side threshold selection and untouched outer-fold performance.
- `gpu_training_time_by_fold.csv` and `gpu_training_time_summary.csv`: recorded GPU runtimes.
- `planned_comparisons.csv`: full versus Partial-4 and Partial-2 deltas when complete.

## Stage status

- partial_4: COMPLETE
- focal_loss: COMPLETE
- partial_2: COMPLETE
