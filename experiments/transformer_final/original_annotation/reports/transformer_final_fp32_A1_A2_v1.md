# Dataset A Transformer Experiment 1 — Frozen FP32 Results

## Execution status

- Required cells: 30 (15 A1 and 15 A2).
- Verified completed cells: 30.
- Failed cells: 0; remaining cells: 0.
- A1 and A2 each contain five folds by three seeds.

## Frozen protocol and environment

- Protocol: `original_annotation_experiment_1_fp32_v1`; SHA-256 `01bb12c5cca274daaaa78b38353b4457a72a7b9c42130f5f1b2d691f84a5eba5`.
- Protocol file: `experiments\transformer_final\original_annotation\configs\fp32_protocol_v1.json`.
- Dataset: Dataset A, source hash `a34df11a87d36dfd14c7d251d96a52a0cac8bb2016db6e4d0475a858f8b382cb`.
- Effective Dataset A development rows: 2,142; positive labels: 45.
- Model: microsoft/deberta-v3-base, freshly initialized from the same local base checkpoint for every cell.
- Parameter storage and forward/loss precision: FP32; autocast disabled.
- Optimizer: AdamW; learning rate 1e-5; weight decay 0.01.
- Scheduler: linear decay with 10% warmup; 5 fixed epochs; batch 8; accumulation 1; gradient clipping norm 1.0.
- Seeds: 1729, 2718, 31415; folds: the existing five family-based splits.
- GPU and software versions are recorded in each future run manifest and the frozen protocol.

## Reuse of the completed diagnostic cell

A1 fold 1 / seed 1729 was reused from the immutable `seed_1729_diagnostic_numerics_v2` directory. Its condition, standard cross-entropy, seed/fold, FP32 checkpoint tensors, data/fold/config hashes, local base-model hashes, predictions, history, and checkpoint hashes were verified. Its original configuration uses the experiment-v1 hash and a CLI FP32 override; the explicit mapping marks this exact legacy cell as compatible with this frozen FP32 protocol. No historical source-code hash is asserted: its executed code provenance remains incomplete because relevant source files were untracked.

## Results

Primary metric is average precision (PR-AUC). Pooled metrics are calculated from each seed's complete out-of-fold predictions; means and sample standard deviations below summarize the three seeds.

| Metric | A1 standard CE | A2 weighted CE |
|---|---:|---:|
| Mean-fold AP | 0.0777 (SD 0.0196) | 0.1322 (SD 0.0471) |
| Pooled OOF AP | 0.0392 (SD 0.0089) | 0.0603 (SD 0.0310) |
| Pooled OOF ROC-AUC | 0.6056 (SD 0.0596) | 0.6737 (SD 0.0268) |
| Pooled precision @ 0.5 | 0.0000 (SD 0.0000) | 0.0000 (SD 0.0000) |
| Pooled recall @ 0.5 | 0.0000 (SD 0.0000) | 0.0000 (SD 0.0000) |
| Pooled F1 @ 0.5 | 0.0000 (SD 0.0000) | 0.0000 (SD 0.0000) |
| Pooled F2 @ 0.5 | 0.0000 (SD 0.0000) | 0.0000 (SD 0.0000) |
| Pooled MCC @ 0.5 | 0.0000 (SD 0.0000) | 0.0000 (SD 0.0000) |
| Pooled AP / prevalence | 1.8637 (SD 0.4226) | 2.8714 (SD 1.4738) |
| Pooled accuracy @ 0.5 | 0.9790 (SD 0.0000) | 0.9790 (SD 0.0000) |
| Pooled false positives | 0.0000 (SD 0.0000) | 0.0000 (SD 0.0000) |
| Pooled false negatives | 45.0000 (SD 0.0000) | 45.0000 (SD 0.0000) |

### Paired A2 minus A1 differences

Intervals are descriptive normal 95% intervals over the 15 paired fold/seed cells or the three paired seed-level pooled metrics; they are not significance tests.

| Metric | Mean difference | Sample SD | Descriptive 95% interval |
|---|---:|---:|---:|
| accuracy | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| ap_prevalence_lift | 4.7743 | 11.2450 | [-0.9164, 10.4651] |
| average_precision | 0.0546 | 0.1317 | [-0.0121, 0.1212] |
| f1 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| f2 | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| false_negatives | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| false_positives | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| mcc | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| precision | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| recall | 0.0000 | 0.0000 | [0.0000, 0.0000] |
| roc_auc | 0.0332 | 0.0624 | [0.0017, 0.0648] |

### Fold positive counts and A2 fitting-only weights

| Seed | Fold | Held-out positives | A1 AP | A2 AP | A2 class weights (No, Yes) |
|---:|---:|---:|---:|---:|---|
| 1729 | 1 | 18 | 0.0464 | 0.0304 | `{"0": 1.0, "1": 48.925925925925924}` |
| 1729 | 2 | 9 | 0.0532 | 0.0587 | `{"0": 1.0, "1": 47.80555555555556}` |
| 1729 | 3 | 3 | 0.0363 | 0.3436 | `{"0": 1.0, "1": 42.45238095238095}` |
| 1729 | 4 | 8 | 0.0733 | 0.3444 | `{"0": 1.0, "1": 48.729729729729726}` |
| 1729 | 5 | 7 | 0.1571 | 0.1167 | `{"0": 1.0, "1": 46.10526315789474}` |
| 2718 | 1 | 18 | 0.0246 | 0.0262 | `{"0": 1.0, "1": 48.925925925925924}` |
| 2718 | 2 | 9 | 0.0489 | 0.0437 | `{"0": 1.0, "1": 47.80555555555556}` |
| 2718 | 3 | 3 | 0.0183 | 0.0159 | `{"0": 1.0, "1": 42.45238095238095}` |
| 2718 | 4 | 8 | 0.0742 | 0.0615 | `{"0": 1.0, "1": 48.729729729729726}` |
| 2718 | 5 | 7 | 0.1373 | 0.2756 | `{"0": 1.0, "1": 46.10526315789474}` |
| 31415 | 1 | 18 | 0.0466 | 0.0423 | `{"0": 1.0, "1": 48.925925925925924}` |
| 31415 | 2 | 9 | 0.0685 | 0.0814 | `{"0": 1.0, "1": 47.80555555555556}` |
| 31415 | 3 | 3 | 0.0548 | 0.3414 | `{"0": 1.0, "1": 42.45238095238095}` |
| 31415 | 4 | 8 | 0.2070 | 0.0812 | `{"0": 1.0, "1": 48.729729729729726}` |
| 31415 | 5 | 7 | 0.1187 | 0.1207 | `{"0": 1.0, "1": 46.10526315789474}` |

## Scientific interpretation and limitations

- Dataset A has only 45 positive examples across nine development families; effective prevalence is 45/2,142 (2.10%). Some held-out folds have very few positives, so fold-level AP can be highly variable.
- Accuracy is secondary and may appear high simply because the negative class dominates. Threshold-0.5 precision/recall/F1/F2 and confusion counts are reported alongside ranking metrics.
- AP should be read relative to the effective positive prevalence; AP/prevalence lift is included. Raw AP should not be compared directly with Dataset B, whose prevalence differs.
- Decreasing training loss is not evidence of held-out generalization. No outer-fold labels were used for tuning or threshold selection.
- Results are descriptive across five family folds and three seeds. The intervals are not inferential significance tests; family heterogeneity and small positive counts limit certainty.
- A2 class weights are verified from each fold's fitting partition only. A1 and A2 differ only in the pre-registered loss weighting under the shared FP32 protocol.
- Dataset B, TCP, and DNS were excluded. Held-out predictions were validated against the Dataset A fold scoring IDs; the reserved families are absent from the effective development dataset.
- A1 fold 1 / seed 1729 has incomplete executed-code provenance, as described above. New runs record deterministic source-file hashes, dependency versions, model/data/config/fold hashes, and CUDA environment details.

## Artifact locations

- Durable completion inventory: `experiments\transformer_final\original_annotation\run_management\fp32_protocol_v1\completion_inventory.json`.
- A1 aggregate: `D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\original_annotation\aggregate_metrics\A1`.
- A2 aggregate: `D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\original_annotation\aggregate_metrics\A2`.
- Paired comparison: `D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\original_annotation\aggregate_metrics\A1_vs_A2_fp32_v1`.
- A1 mean fold AP across seeds: 0.0777.
- A2 mean fold AP across seeds: 0.1322.
- Family AP means (A1, descriptive range): draft-ietf-quic-http 0.0365 to draft-ietf-tls-tls13 nan.
