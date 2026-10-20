# Final report

| method   |   rows |   yes |   no |   prevalence |   average_precision |   roc_auc |
|:---------|-------:|------:|-----:|-------------:|--------------------:|----------:|
| focal    |   2125 |   136 | 1989 |        0.064 |            0.305357 |  0.781966 |
| partial4 |   2125 |   136 | 1989 |        0.064 |            0.326174 |  0.828424 |
| ensemble |   2125 |   136 | 1989 |        0.064 |            0.371075 |  0.827428 |

Operational metrics are in `FINAL_NESTED_METRICS.csv`; Top-K, family robustness, uncertainty, GPU telemetry, and error-review artifacts are separate. Raw model probabilities and ensemble ECDF scores are intentionally not pooled onto one calibration scale.
