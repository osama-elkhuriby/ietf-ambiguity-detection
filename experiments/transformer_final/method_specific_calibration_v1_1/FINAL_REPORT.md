# Final Report — Method-Specific Calibration V1.1

## Dataset and governance

The unchanged development cohort contains 2142 clauses, 126 positive and 2016 negative. All outer/inner family assignments and leakage exclusions are inherited unchanged. TCP and DNS were not consumed.

## Compatibility recovery

The first Focal outer-1/inner-1 cell was reused by immutable reference after semantic validation of both hash fields, record identities, family isolation, leakage exclusions, predictions, configuration, and provenance. Its original V1 `FAILED_VALIDATION` evidence remains unchanged. It was not retrained.

## Training execution

Logical cells: 30; immutable V1 reuse: 1; new sequential V1.1 GPU cells: 29. Total recorded training time including the reused cell: 5881.5 seconds; new V1.1 time: 5736.4 seconds; peak allocated memory: 3976.3 MiB. Automatic retries: zero.

## Valid nested operating points

| Method | Fit precision target | AP | AUC | Precision | Recall | F1 | F2 | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| focal_loss | 0.20 | 0.3295 | 0.7921 | 0.1593 | 0.5714 | 0.2491 | 0.3766 | 72 | 380 | 54 |
| focal_loss | 0.30 | 0.3295 | 0.7921 | 0.2644 | 0.4365 | 0.3293 | 0.3862 | 55 | 153 | 71 |
| focal_loss | 0.50 | 0.3295 | 0.7921 | 0.5000 | 0.1984 | 0.2841 | 0.2256 | 25 | 25 | 101 |
| partial_4 | 0.20 | 0.3008 | 0.7950 | 0.2571 | 0.5000 | 0.3396 | 0.4206 | 63 | 182 | 63 |
| partial_4 | 0.30 | 0.3008 | 0.7950 | 0.4050 | 0.3889 | 0.3968 | 0.3920 | 49 | 72 | 77 |
| partial_4 | 0.50 | infeasible in at least one fold | | | | | | | | |
| focal_partial4_ecdf_equal | 0.20 | 0.3715 | 0.8125 | 0.2250 | 0.5714 | 0.3229 | 0.4369 | 72 | 248 | 54 |
| focal_partial4_ecdf_equal | 0.30 | 0.3715 | 0.8125 | 0.3273 | 0.4286 | 0.3711 | 0.4036 | 54 | 111 | 72 |
| focal_partial4_ecdf_equal | 0.50 | infeasible in at least one fold | | | | | | | | |

Precision targets are fitting-side objectives, not guarantees on unseen families. Infeasible objectives remain explicit and have no always-positive fallback.

## Ranking and ensemble

Frozen outer prediction AP: Focal 0.3295, Partial-4 0.3008. The pre-registered equal-weight fitting-side ECDF ensemble achieved AP 0.3715 and ROC-AUC 0.8125. No ensemble weight was fit on outer labels.

Top-K budgets in `TOP_K_COMPARISON.csv` are global pooled development OOF rankings with deterministic record-ID tie breaking.

## Cross-family robustness and uncertainty

Family metrics are reported in `FAMILY_ROBUSTNESS.csv`. Family-clustered bootstrap intervals are reported in `UNCERTAINTY_INTERVALS.csv`; only nine family clusters exist, and the intervals do not measure single-seed training variability.

## Annotation and deployment limitations

The labels are an operational AI-assisted reference, not an independent expert gold standard. Use the models as a ranked human-review assistant with an uncertain/review band; do not certify clauses automatically. A separately double-annotated expert sample and multi-seed confirmation remain necessary before TCP transfer validation and the one-time sealed DNS evaluation.
