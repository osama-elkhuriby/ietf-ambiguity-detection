# Nested Fitting-Side Threshold Calibration

This analysis changes only each outer fold's decision threshold. Model weights, scores, and outer-fold labels were not used to choose thresholds.

| Outer fold | Inner OOF rows | Inner precision | Inner recall | Selected threshold | Selection rule |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1349 | 0.2192 | 0.4000 | 0.0100 | maximum_recall_at_precision_at_least_0.20 |
| 2 | 1757 | 0.2208 | 0.1809 | 0.0300 | maximum_recall_at_precision_at_least_0.20 |
| 3 | 1828 | 0.2769 | 0.3103 | 0.0200 | maximum_recall_at_precision_at_least_0.20 |
| 4 | 1843 | 0.2007 | 0.5370 | 0.0100 | maximum_recall_at_precision_at_least_0.20 |
| 5 | 1791 | 0.2865 | 0.4623 | 0.0100 | maximum_recall_at_precision_at_least_0.20 |

The outer-fold metrics are descriptive evaluation only. Per-fold thresholds are selected on the other four families' inner OOF predictions, after the frozen leakage exclusions.

Probability-based average precision, ROC-AUC, and Brier score are unchanged by threshold deployment. The pooled fold-specific precision/recall counts are computed using one selected cutoff per outer fold; there is no single pooled cutoff.
