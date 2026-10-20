# Step 4 factorial report

All values are development out-of-fold agreement with the frozen 
LLM-consensus operational proxy gold labels, not human-verified accuracy.
TCP and the sealed DNS test were not used.

## Conditions

| Condition | Mean-fold PR-AUC | Pooled PR-AUC | 95% row CI | Referral at 90% recall | Runtime (min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard_target_only | 0.3022 | 0.2874 | [0.2506, 0.3321] | 0.858 | 14.9 |
| soft_target_only | 0.3014 | 0.2997 | [0.2644, 0.3434] | 0.738 | 16.2 |
| hard_local_context | 0.2652 | 0.2779 | [0.2422, 0.3194] | 0.797 | 22.4 |
| soft_local_context | 0.2735 | 0.2903 | [0.2547, 0.3321] | 0.755 | 23.7 |

## Frozen selection

Selected condition: `hard_target_only`. Top observed condition: `hard_target_only`; runner-up: `soft_target_only`. Top-minus-runner-up mean paired difference: 0.0008; paired-fold SE: 0.0063.

## Pre-registered contrasts

| Contrast | Mean fold difference | Paired-row 95% CI | Section-cluster 95% CI |
| --- | ---: | ---: | ---: |
| soft_target_effect | -0.0008 | [-0.0100, 0.0337] | [-0.0101, 0.0330] |
| context_effect_hard | -0.0370 | [-0.0421, 0.0214] | [-0.0410, 0.0219] |
| context_effect_soft | -0.0279 | [-0.0340, 0.0145] | [-0.0339, 0.0164] |
| soft_by_context_interaction | 0.0091 | [-0.0295, 0.0309] | [-0.0294, 0.0286] |

## Matched architecture contrast

The hard-target target-only DeBERTa control minus class-weighted TF-IDF trained on the exact same nested fitting IDs has mean-fold difference 0.0259. Its pooled paired-row 95% CI is [-0.0276, 0.0467], and section-cluster 95% CI is [-0.0272, 0.0447].

## Limitations

The reference labels are an LLM consensus and no multi-annotator human gold set is currently available. Folds still contain related protocol clusters, so estimates can be optimistic for a genuinely new protocol. Probabilities are uncalibrated; ECE and Brier are secondary diagnostics. No threshold, calibration, TCP verification, or DNS evaluation was performed in this step.
