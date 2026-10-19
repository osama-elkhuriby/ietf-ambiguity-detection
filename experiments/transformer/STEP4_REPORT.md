# Step 4: soft-target and local-context factorial

Status: complete. The experiment followed the design frozen in
`EXPERIMENT_PLAN.md` before Step 4 scores were produced. TCP was not used, and the
sealed DNS clause file was not opened.

## Interpretation boundary

A multi-annotator human gold set is not currently feasible. The frozen LLM-consensus
labels are therefore the project's operational proxy gold standard. Every metric below
measures out-of-fold agreement with that reference, not human-verified ambiguity
detection accuracy.

## Data governance

The rationales-free vote artifact maps all 2,253 training rows one-to-one; every row
has at least two valid votes, so soft-target training excludes zero rows. Soft targets
use positive valid votes divided by all valid votes. Invalid votes are omitted from the
denominator, and vote information never enters model text or inference features.

The fold-specific context audit removes exact or near target reproduction across
fitting, inner-selection, and outer-validation roles. Depending on the fold, it blanks
5 to 39 previous/next context sides: 5 to 32 exact matches and 0 to 7 near matches at
normalized word-trigram Jaccard similarity >= 0.85. Targets are never altered. The
pinned 256-token policy truncates zero targets and context on seven rows per fold.

## Frozen experiment

The control reuses 15 completed class-weighted target-only cells. The run adds 45
cells: three remaining conditions x seeds 1729/2718/31415 x five frozen outer folds.
All cells use DeBERTa-v3-small revision
`a36c739020e01763fe789b4b85e2df55d6180012`, LR 1e-5, fp32, full-encoder fine-tuning,
the frozen inner-family checkpoint selector, and hard inner-label PR-AUC for checkpoint
selection. Outer folds are scored once after selection.

| Condition | Seed-averaged mean-fold PR-AUC | Pooled logit-average PR-AUC | Row-bootstrap 95% CI | Referral rate at 90% recall |
| --- | ---: | ---: | ---: | ---: |
| Hard target, target only | 0.3022 | 0.2874 | [0.2506, 0.3321] | 0.858 |
| Soft target, target only | 0.3014 | 0.2997 | [0.2644, 0.3434] | 0.738 |
| Hard target, local context | 0.2652 | 0.2779 | [0.2422, 0.3194] | 0.797 |
| Soft target, local context | 0.2735 | 0.2903 | [0.2547, 0.3321] | 0.755 |

The frozen primary statistic is seed-averaged mean-fold PR-AUC. Hard target-only is
the observed leader by 0.0008 over soft target-only, with paired-fold SE 0.0063. The
two are within one SE, so the frozen simplicity rule selects **hard target-only**.
Pooled PR-AUC does not override this rule.

The pre-registered mean-fold effects are:

| Contrast | Mean difference | Paired-fold SE | Paired-row 95% CI | Section-cluster 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Soft target effect | -0.0008 | 0.0063 | [-0.0100, 0.0337] | [-0.0101, 0.0330] |
| Context effect with hard labels | -0.0370 | 0.0152 | [-0.0421, 0.0214] | [-0.0410, 0.0219] |
| Context effect with soft labels | -0.0279 | 0.0136 | [-0.0340, 0.0145] | [-0.0339, 0.0164] |
| Soft-target x context interaction | +0.0091 | 0.0078 | [-0.0295, 0.0309] | [-0.0294, 0.0286] |

The bootstrap intervals are pooled-PR-AUC contrasts and therefore need not equal the
mean-fold point estimate. They all include zero. The primary fold-balanced estimates
show no benefit from soft targets and a substantial context penalty, but the available
data do not support a precise nonzero pooled effect.

## Matched TF-IDF comparison

The new class-weighted target-only TF-IDF comparator uses the exact nested fitting IDs
used by the transformer, including exclusion of the inner family. It has mean-fold
PR-AUC 0.2763 and pooled PR-AUC 0.2794. Hard target-only DeBERTa exceeds it by 0.0259
on seed-averaged mean-fold PR-AUC and by 0.0080 on pooled PR-AUC. The paired-row 95%
interval for the pooled difference is [-0.0276, 0.0467], and the section-cluster
interval is [-0.0272, 0.0447]. Thus the matched comparison is directionally favorable
to DeBERTa but does not establish a reliable transformer advantage; H1 remains
unsupported.

## Routing and recommendation

Soft target-only is not the primary classifier winner, but it reduces the development
90%-recall referral estimate from 85.8% to 73.8%, corresponding to estimated local
handling of about 26.2% rather than 14.2% of rows. This is a useful routing sensitivity
result, not permission to replace the selected classifier after seeing the data.

Keep hard target-only, class-weighted DeBERTa as the selected Step 4 classifier. Do not
adopt local context in its current form. Preserve soft target-only as a named routing
candidate for the next pre-registered calibration comparison. The next professional
step is to fit temperature scaling and routing thresholds from pooled development OOF
logits, then use TCP once to verify the frozen >=90% recall constraint. DNS must remain
sealed until calibration, routing policy, code, and reporting are frozen.

## Runtime and artifacts

The 45 new cells used about 62.4 aggregate GPU-minutes. Target-only peaked at about
2,950 MB allocated GPU memory; context peaked at about 3,686 MB. All 60 condition cells
preserve raw logits, checkpoints, histories, run configurations, and manifests.
Machine-readable results are under
`experiments/transformer/results/factorial-v1-analysis/`.
