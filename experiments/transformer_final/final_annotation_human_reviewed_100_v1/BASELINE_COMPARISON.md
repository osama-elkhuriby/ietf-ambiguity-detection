# Frozen baseline comparison

The scientific baseline is `final_annotation_661_v1_2`, with its validated
Focal/Partial-4 calibration and ECDF behavior corroborated by
`method_specific_calibration_v1_1`. V1.2 is the most recent valid recovery of
the complete nested protocol and pins the same current shared training core.

| Dimension | Frozen successful baseline | This experiment |
|---|---|---|
| Source | `final_annotation.csv`, SHA `8e395e...e22e1d` | `final_annotation_ready.csv`, SHA `8ffa65...65bdbf` |
| Source distribution | 2,406; 159 Yes / 2,247 No | 2,425; 160 Yes / 2,265 No |
| Effective development | 2,125; 136 Yes / 1,989 No | 2,143; 138 Yes / 2,005 No |
| `item_id=2161` | unresolved and excluded | resolved/manual conflict verdict; included as No |
| Model/tokenizer | local `microsoft/deberta-v3-base` | unchanged |
| Model SHA-256 | `206ee7...17dd7` | unchanged |
| Input | target-only, max 256, deterministic head-tail | unchanged |
| Seed | 2718 | unchanged |
| Optimizer/schedule | AdamW, 1e-5, wd 0.01, linear, warmup 0.1 | unchanged |
| Batches/epochs | train 8, eval 16, accumulation 1, 5 epochs | unchanged |
| Precision | FP32 | unchanged |
| Focal | gamma 2, fitting-side class weights, batch mean | unchanged |
| Partial-4 | weighted CE; layers 8-11 + pooler/classifier trainable | unchanged |
| CV | 5 outer family folds, 3 inner family folds | unchanged |
| Ensemble | method-specific fitting-side ECDF; equal fixed weights | unchanged |
| Thresholds | inner OOF only; precision targets .20/.30/.50 | unchanged |
| Execution | V1.2 reused validated outer artifacts during recovery | all 40 cells fresh from base |
| Inner provenance | V1.2 compatibility fix uses outer fitting IDs | preserved; each inner root has a complete manifest |
| Resume | none | explicitly false in config and every cell spec |

No hyperparameter, architecture, tokenizer, evaluation metric, threshold rule,
or ensemble rule was changed. Necessary compatibility work is limited to the new
dataset identity/counts, resolved-2161 policy, fresh output paths, and retaining
the V1.2 inner-manifest/fitting-ID correction.

