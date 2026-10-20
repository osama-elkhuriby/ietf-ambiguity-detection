# Final report

| method   |   rows |   yes |   no |   prevalence |   average_precision |   roc_auc |
|:---------|-------:|------:|-----:|-------------:|--------------------:|----------:|
| focal    |   2143 |   138 | 2005 |    0.0643957 |            0.58231  |  0.881481 |
| partial4 |   2143 |   138 | 2005 |    0.0643957 |            0.576129 |  0.892385 |
| ensemble |   2143 |   138 | 2005 |    0.0643957 |            0.673573 |  0.911842 |

Operational metrics are in `FINAL_NESTED_METRICS.csv`; Top-K, family robustness, uncertainty, GPU telemetry, and error-review artifacts are separate. Raw model probabilities and ensemble ECDF scores are intentionally not pooled onto one calibration scale.
