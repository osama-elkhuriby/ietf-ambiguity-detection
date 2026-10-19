# Final annotation-ready Transformer pilot

This namespace is isolated from the original-annotation Experiment 1. It uses
`results/annotation/final_annotation_ready.csv`, a one-seed (2718), five-fold,
FP32 DeBERTa-v3-base pilot with weighted cross-entropy. Every cell initializes
freshly from the hash-pinned local base model and trains for five fixed epochs.
No Experiment 1 fine-tuned checkpoint is used.

## Dataset authority and exclusions

The source has one binary Yes/No outcome column, `final_label`; `final_source`
and `final_status` are provenance fields. The source contains 2,425 rows:
2,424 resolved labels (148 Yes, 2,276 No) and one unresolved record,
item_id `2161`. That row has a placeholder `No` label but
`final_status=unresolved_blank_manual_verdict` and `final_source=unresolved`;
it is excluded explicitly rather than treated as ground truth. The source has
no duplicate item IDs or blank target clauses.

The effective cohort is built with the existing approved metadata joins and
family assignments. It contains 2,142 records (126 Yes, 2,016 No) across the
same nine development families as Experiment 1. The 121 candidates absent
from the approved annotation cohort remain excluded. TCP and DNS records are
reserved and never enter fitting, threshold evaluation, or calibration.
Fitting-side duplicate/leakage exclusions and context sanitization are kept.

The exact source SHA-256, dataset and fold-manifest hashes, model/tokenizer
identity, source-code hashes, dependency versions, CUDA environment, training
settings, and exploratory threshold list are pinned in
`configs/fp32_protocol_v1.json`. The source audit and immutable prepared data
are stored under `audit_logs/` and `manifests/`.

## Execution

Freeze the dataset and protocol before training:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PATH='C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\Library\bin;' + $env:PATH
& 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' -m models.transformer_final.experiment2 `
  --config experiments\transformer_final\final_annotation_ready_pilot_v1\configs\experiment_v1.json `
  --protocol experiments\transformer_final\final_annotation_ready_pilot_v1\configs\fp32_protocol_v1.json `
  --model-path 'D:\RFC\models\deberta-v3-base'
```

The sequential runner validates hashes and GPU/environment identity, then runs
folds 1–5 at seed 2718 one at a time. A failed or interrupted cell is preserved
and is not automatically retried:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PATH='C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\Library\bin;' + $env:PATH
& 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' -m models.transformer_final.experiment2_runner `
  --config experiments\transformer_final\final_annotation_ready_pilot_v1\configs\experiment_v1.json `
  --protocol experiments\transformer_final\final_annotation_ready_pilot_v1\configs\fp32_protocol_v1.json `
  --model-path 'D:\RFC\models\deberta-v3-base'
```

## Interpretation

Thresholds are fixed in advance for reporting:
`0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15,
0.20, 0.30, 0.40, 0.50`. All are applied to held-out outer-fold probabilities
and are exploratory diagnostics only. No threshold is selected or calibrated
from these results, and no training setting is changed in response.

The completed analysis contains per-fold, pooled OOF, and one-seed threshold
tables; the pooled/overall tables represent the same unique cohort. It also
contains ranking and probability diagnostics, top-K retrieval, family/error
analyses, seven charts, a descriptive Experiment 1 comparison, and the final
report. The single-seed results are a pilot, not confirmatory evidence.
