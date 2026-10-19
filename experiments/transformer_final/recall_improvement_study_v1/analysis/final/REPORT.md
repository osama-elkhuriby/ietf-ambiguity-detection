# Recall Improvement Study — Final Report

## Scope and status

All methods use the frozen, family-aware Dataset A cohort. The reported fold-specific operating point is chosen only from baseline inner OOF predictions; other ablations are compared at fixed threshold 0.5. Outer labels are reserved for evaluation. All results are exploratory and based on one seed; they do not establish generalization beyond the frozen families.

## Method comparison

| Model | Operating point | AP | ROC-AUC | Precision | Recall | F2 | Brier |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| weighted_ce | threshold_0.5 | 0.2332 | 0.7286 | 0.3704 | 0.1587 | 0.1792 | 0.0603 |

## Stage execution status

| Stage | Status |
| --- | --- |
| A_family_recall_diagnostics | COMPLETE |
| B_nested_threshold_calibration | SKIPPED_TIME_BUDGET |
| C_local_context | SKIPPED_TIME_BUDGET |
| D_focal_loss | SKIPPED_TIME_BUDGET |
| E_error_analysis | COMPLETE |
| G2_partial_2 | SKIPPED_TIME_BUDGET |
| G_partial_4 | SKIPPED_TIME_BUDGET |


## Stage A diagnostic

Experiment 2 baseline Fold 2 has AP 0.2280 and ROC-AUC 0.8011, but all positive scores are below 0.5 (median 0.0128, maximum 0.0205), so threshold 0.5 yields zero recall. This supports testing fitting-side thresholds; it does not prove probability calibration.

## Retrieval at fixed review budgets

See `top_k_comparison.csv` for model-specific top-K precision, recall, and lift at the predeclared review budgets.

## Family-stratified metrics

See `family_metrics.csv`. Family-level AP/AUC are undefined when a family has only one label class and are recorded as null.

## Error analysis

See `../error_analysis/selected_error_examples.csv` for threshold-near false positives and false negatives. Lexical cues are deterministic review aids, not expert-assigned ambiguity categories.

## Study limitations

- One random seed and five family folds limit precision of method comparisons.
- Grouped family cross-validation is not an external deployment evaluation.
- Inner thresholds optimize a fixed finite candidate grid and a declared precision constraint; fallback thresholds maximize F2 when infeasible.
- Partial freezing preserves the model's standard training-mode dropout.
- CPU synthetic tests verify study helpers only; they do not verify GPU optimizer or full-model behavior.

## Artifacts and reproducibility

The frozen protocol, split manifests, per-cell run manifests, complete artifact hashes, treatment audits, logs, and execution inventory are preserved under the study namespace. No historical run artifacts are modified.
