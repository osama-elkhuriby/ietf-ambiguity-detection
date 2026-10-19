# Final Annotation-Ready DeBERTa Pilot — Exploratory Results

## Execution status and frozen protocol

- Protocol: `final_annotation_ready_weighted_ce_fp32_pilot_v1`; SHA-256 `242f98c56d88a11dea3b355c4c3b14c693029e3d8a607fdd7a067f29af630b36`.
- Dataset source SHA-256: `c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712`.
- Effective cohort SHA-256: `a25e99563563f71570e48f63b9515fbb51ff845762ff718763d98e0b3eefae2f`.
- Five sequential cells completed: folds 1–5, seed 2718, weighted cross-entropy.
- Every cell used a fresh local DeBERTa-v3-base initialization, FP32 parameters and forward pass, FP32 cross-entropy, five fixed epochs, and no outer-fold selection.
- No downloads or Experiment 1 outputs were used as training inputs.

## Dataset verification and labels

- Source: `D:\RFC\ietf-ambiguity-detection\results\annotation\final_annotation_ready.csv`; 2,425 rows; 178 document versions; 0 duplicate item IDs; 0 blank target clauses.
- `final_label` is the only binary Yes/No outcome column in the dataset schema; `final_source` and `final_status` are provenance/status fields.
- Resolved source labels: Yes 148; No 2,276.
- Unresolved: 1 record (item_id 2161); its placeholder No value was excluded under the explicit unresolved policy.
- Effective cohort: 2,142 clauses; 126 Yes; 2,016 No; prevalence 5.88%.
- The 121 candidate records absent from the approved annotation cohort remain excluded. TCP (131) and DNS (151) source rows were not used.
- Leakage audit: {"cross_fold_bcp14_skeleton_conflicts": 0, "cross_fold_document_overlaps": 0, "cross_fold_lineage_conflicts": 0, "cross_fold_target_context_overlaps": 13, "exact_target_duplicates_all": 1, "exact_target_duplicates_cross_fold": 0, "families_assigned_to_multiple_folds": 0, "near_cross_fold_targets": 4, "targets_with_bcp14_keyword": 2127}; cross-fold document/family blocking was false.
- Existing family-to-fold assignment was unchanged; fitting-side exact/near/lineage/BCP14 exclusions and context sanitization remain in force.

## Fold counts, class weights, and training

| Fold | Held Out Families | Fitting Rows | Fitting Positive Labels | Fitting Negative Labels | Held Out Rows | Held Out Positive Labels | Held Out Negative Labels | Fitting Exclusion Count | Class Weights | Final Epoch Training Loss | Runtime Seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | draft-ietf-quic-transport | 1348 | 79 | 1269 | 793 | 46 | 747 | 1 | {"0": 1.0, "1": 16.063291139240505} | 0.62907 | 246.3 |
| 2 | draft-ietf-httpbis-semantics | 1757 | 94 | 1663 | 385 | 32 | 353 | 0 | {"0": 1.0, "1": 17.69148936170213} | 1.01917 | 293.5 |
| 3 | draft-ietf-quic-http | 1825 | 116 | 1709 | 314 | 10 | 304 | 3 | {"0": 1.0, "1": 14.732758620689655} | 0.48781 | 286.6 |
| 4 | draft-ietf-6man-rfc2460bis\|draft-ietf-httpbis-cache-header\|draft-ietf-httpbis-http2bis\|draft-ietf-tls-tls13 | 1840 | 108 | 1732 | 299 | 18 | 281 | 3 | {"0": 1.0, "1": 16.037037037037038} | 0.35678 | 284.8 |
| 5 | draft-ietf-quic-recovery\|draft-ietf-quic-tls | 1790 | 105 | 1685 | 351 | 20 | 331 | 1 | {"0": 1.0, "1": 16.047619047619047} | 0.47987 | 277.7 |

- Python 3.11.16; PyTorch 2.12.0; CUDA 13.0; cuDNN 92501; GPU NVIDIA RTX 5000 Ada Generation; driver 596.59.

## Threshold-independent ranking and calibration diagnostics

- AP / PR-AUC: **0.2332**; new-cohort prevalence baseline: **0.0588**; AP/prevalence lift: **3.964**.
- ROC-AUC: **0.7286**; actual positives 126; actual negatives 2016.
- Brier score: 0.06034; prevalence-only descriptive baseline: 0.05536.
- Histogram overlap coefficient for Yes/No score distributions: 0.752. This is a descriptive 50-bin overlap measure, not a hypothesis test.
- AP above prevalence and ROC-AUC above 0.5 are evidence of ranking signal on this one-seed cohort, not proof of robust family-generalizable learning.

## Fixed-threshold held-out results (exploratory)

All thresholds below were applied to the pooled, unique 2,142-record OOF cohort with `probability_yes >= threshold`. These are exploratory outer-fold diagnostics, not an unbiased threshold-selection procedure.

| Threshold | Predicted Yes | Tp | Fp | Tn | Fn | Precision | Recall | Specificity | F1 | F2 | Balanced Accuracy | Mcc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.001 | 2142.0 | 126.0 | 2016.0 | 0.0 | 0.0 | 0.0588 | 1.0000 | 0.0000 | 0.1111 | 0.2381 | 0.5000 | 0.0000 |
| 0.002 | 1892.0 | 119.0 | 1773.0 | 243.0 | 7.0 | 0.0629 | 0.9444 | 0.1205 | 0.1179 | 0.2483 | 0.5325 | 0.0476 |
| 0.003 | 1408.0 | 108.0 | 1300.0 | 716.0 | 18.0 | 0.0767 | 0.8571 | 0.3552 | 0.1408 | 0.2824 | 0.6062 | 0.1053 |
| 0.005 | 597.0 | 77.0 | 520.0 | 1496.0 | 49.0 | 0.1290 | 0.6111 | 0.7421 | 0.2130 | 0.3497 | 0.6766 | 0.1853 |
| 0.01 | 326.0 | 65.0 | 261.0 | 1755.0 | 61.0 | 0.1994 | 0.5159 | 0.8705 | 0.2876 | 0.3916 | 0.6932 | 0.2531 |
| 0.02 | 120.0 | 35.0 | 85.0 | 1931.0 | 91.0 | 0.2917 | 0.2778 | 0.9578 | 0.2846 | 0.2804 | 0.6178 | 0.2411 |
| 0.03 | 104.0 | 32.0 | 72.0 | 1944.0 | 94.0 | 0.3077 | 0.2540 | 0.9643 | 0.2783 | 0.2632 | 0.6091 | 0.2389 |
| 0.05 | 97.0 | 31.0 | 66.0 | 1950.0 | 95.0 | 0.3196 | 0.2460 | 0.9673 | 0.2780 | 0.2579 | 0.6066 | 0.2414 |
| 0.075 | 91.0 | 29.0 | 62.0 | 1954.0 | 97.0 | 0.3187 | 0.2302 | 0.9692 | 0.2673 | 0.2437 | 0.5997 | 0.2326 |
| 0.1 | 85.0 | 28.0 | 57.0 | 1959.0 | 98.0 | 0.3294 | 0.2222 | 0.9717 | 0.2654 | 0.2377 | 0.5970 | 0.2338 |
| 0.15 | 82.0 | 28.0 | 54.0 | 1962.0 | 98.0 | 0.3415 | 0.2222 | 0.9732 | 0.2692 | 0.2389 | 0.5977 | 0.2397 |
| 0.2 | 82.0 | 28.0 | 54.0 | 1962.0 | 98.0 | 0.3415 | 0.2222 | 0.9732 | 0.2692 | 0.2389 | 0.5977 | 0.2397 |
| 0.3 | 74.0 | 26.0 | 48.0 | 1968.0 | 100.0 | 0.3514 | 0.2063 | 0.9762 | 0.2600 | 0.2249 | 0.5913 | 0.2352 |
| 0.4 | 69.0 | 26.0 | 43.0 | 1973.0 | 100.0 | 0.3768 | 0.2063 | 0.9787 | 0.2667 | 0.2269 | 0.5925 | 0.2466 |
| 0.5 | 54.0 | 20.0 | 34.0 | 1982.0 | 106.0 | 0.3704 | 0.1587 | 0.9831 | 0.2222 | 0.1792 | 0.5709 | 0.2129 |

- First requested threshold producing any predicted Yes: **0.001**, 2142 predictions, of which 126 were correct.
- Thresholds with at least one true positive: 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5.
- At reference threshold 0.5: 54 predicted Yes, 20 correct.
- Highest observed F1 is 0.2876 at 0.01; highest observed F2 is 0.3916 at 0.01. These are post-hoc exploratory optima and must not be treated as selected operating thresholds.

### Per-fold threshold-independent performance at the fixed 0.5 reference

| Fold | Rows | Positive Labels | Average Precision | Roc Auc | Predicted Yes | False Positives | False Negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 793.0 | 46.0 | 0.2054 | 0.6927 | 11.0 | 7.0 | 42.0 |
| 2.0 | 385.0 | 32.0 | 0.2280 | 0.8011 | 0.0 | 0.0 | 32.0 |
| 3.0 | 314.0 | 10.0 | 0.2811 | 0.7007 | 11.0 | 7.0 | 6.0 |
| 4.0 | 299.0 | 18.0 | 0.3804 | 0.8108 | 12.0 | 6.0 | 12.0 |
| 5.0 | 351.0 | 20.0 | 0.3078 | 0.7832 | 20.0 | 14.0 | 14.0 |

### Score distributions

- Actual Yes scores: {"count": 126, "maximum": 0.9958549, "mean": 0.17205919137777778, "median": 0.010400911, "minimum": 0.0016026678, "p90": 0.9641194900000001, "p95": 0.9885015825000001, "p99": 0.9954039450000001, "standard_deviation": 0.3298439899022727}
- Actual No scores: {"count": 2016, "maximum": 0.99564725, "mean": 0.023526889410515874, "median": 0.0041333025, "minimum": 0.0014819037, "p90": 0.010965626499999999, "p95": 0.01671475075, "p99": 0.921107444999997, "standard_deviation": 0.11821785865508698}
- Scores below 0.01 / 0.05 / 0.10: 84.78% / 95.47% / 96.03%; scores above 0.50: 2.52%.
- Mean predicted positive probability: 0.0323; actual prevalence: 0.0588.

## Human-review retrieval (exploratory)

| K | True Ambiguous Retrieved | Precision At K | Recall At K | Lift Over Positive Prevalence |
| --- | --- | --- | --- | --- |
| 10.0 | 6.0 | 0.6000 | 0.0476 | 10.200 |
| 25.0 | 11.0 | 0.4400 | 0.0873 | 7.480 |
| 50.0 | 18.0 | 0.3600 | 0.1429 | 6.120 |
| 100.0 | 32.0 | 0.3200 | 0.2540 | 5.440 |
| 200.0 | 52.0 | 0.2600 | 0.4127 | 4.420 |

## Family-level results

| Family Id | Rows | Positive Labels | Positive Prevalence | Average Precision | Roc Auc | Ap Prevalence Lift |
| --- | --- | --- | --- | --- | --- | --- |
| draft-ietf-6man-rfc2460bis | 1 | 0 | 0.000 | undefined | undefined | undefined |
| draft-ietf-httpbis-cache-header | 5 | 1 | 0.200 | 1.0000 | 1.0000 | 5.000 |
| draft-ietf-httpbis-http2bis | 239 | 13 | 0.054 | 0.3074 | 0.7859 | 5.651 |
| draft-ietf-httpbis-semantics | 385 | 32 | 0.083 | 0.2280 | 0.8011 | 2.743 |
| draft-ietf-quic-http | 314 | 10 | 0.032 | 0.2811 | 0.7007 | 8.827 |
| draft-ietf-quic-recovery | 162 | 11 | 0.068 | 0.2389 | 0.8206 | 3.519 |
| draft-ietf-quic-tls | 189 | 9 | 0.048 | 0.4206 | 0.7383 | 8.832 |
| draft-ietf-quic-transport | 793 | 46 | 0.058 | 0.2054 | 0.6927 | 3.542 |
| draft-ietf-tls-tls13 | 54 | 4 | 0.074 | 0.6412 | 0.8750 | 8.656 |

## Error examples and interpretation

Complete example rows, target clauses, labels, probabilities, folds, and predictions at illustrative thresholds are in `error_analysis_examples.csv`. These are held-out examples only; no annotations were changed. Semantic categories (vagueness, subjective quantity, unspecified implementation choice, conditional complexity, semantic ambiguity, missing protocol context) require human interpretation and are not inferred from predicted scores.

## Descriptive comparison with Experiment 1

Descriptive only: labels and prevalence differ between annotation schemes, and this pilot uses one seed versus three in Experiment 1.

- Experiment 1 A2 all-seed means: mean-fold AP 0.1322; mean pooled AP 0.0603; mean pooled ROC-AUC 0.6737.
- Experiment 1 A2 predicted-Yes counts at 0.5 by seed: [0, 0, 0]. Matched Experiment 1 A2 seed-2718 pooled and fold metrics are in `experiment_1_reference.json`; this comparison remains confounded by changed labels and prevalence.

## Interpretation and limitations

1. Positive predictions at threshold 0.5: 54.
2. Predictions first emerge at the lowest listed active threshold: 0.001.
3. Nonzero true positives occur at: 0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5.
4. Precision generally trades off against recall as the fixed threshold is lowered; use the complete table rather than the post-hoc maximum as a decision rule.
5. AP/prevalence lift is 3.964; any apparent ranking signal remains a one-seed exploratory finding.
6. Family AP ranges from 0.2054 to 1.0000; the family-wise sample sizes and positive counts vary sharply.
7. The revised annotations contain more resolved Yes labels than the original Dataset A effective cohort, but the changed target definition and single-seed pilot prevent a definitive claim that this annotation scheme is more learnable.
8. One appropriate next controlled experiment is a pre-registered nested family-aware calibration study: choose operating thresholds using fitting-side inner OOF predictions only, then evaluate once on untouched outer families; do not reuse this pilot's exploratory outer-fold optimum.

## Plots

- `plots/precision_recall_curve.png`
- `plots/roc_curve.png`
- `plots/probability_histogram.png`
- `plots/predicted_yes_vs_threshold.png`
- `plots/precision_recall_vs_threshold.png`
- `plots/f1_f2_vs_threshold.png`
- `plots/per_fold_average_precision.png`

## Provenance

- Frozen source snapshot SHA-256: `18b6f4130329fac7c9f4e23f01731cbdc7832026ef8aa617d6bfad60126151d2`.
- Protocol hash: `242f98c56d88a11dea3b355c4c3b14c693029e3d8a607fdd7a067f29af630b36`.
- Inventory: `D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\final_annotation_ready_pilot_v1\audit_logs\run_management\completion_inventory.json`.
- Per-run manifests, checkpoint hashes, source hashes, model/tokenizer hashes, and dependency/CUDA metadata are retained alongside their run outputs.

## Analysis-only recovery

The validated held-out prediction artifacts intentionally omit clause text. For error-example reporting only, `target_clause` was joined by `record_id` from the protocol-pinned effective dataset; no training, predictions, metrics, or run artifacts were changed. Recovery adapter SHA-256: `c26387b45a79e2b4326f63f0f8b11d0451e5c23db885974df99d911cb7c2e03b`.
