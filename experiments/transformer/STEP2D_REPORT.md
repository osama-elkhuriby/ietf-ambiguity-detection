# Step 2d CPU-only ceiling audit

Status: complete before Step 3 training. No GPU was used, labels were not changed,
and the sealed DNS artifact was not opened. The audit completed in 1.67 CPU wall-clock
seconds; its initial v1 manifest hashes the outputs but does not itself store runtime.

## Vote source and safeguards

The model-ready development CSVs do not contain individual votes. Rationales-free
votes are stored in `corpus/processed/llm_annotations/v1/consensus.csv` under
`votes_json`. The audit read only `record_id` and `votes_json`, retained only record IDs
from the frozen train/TCP development allowlist, and exported no rationale fields.
Votes remain forbidden from every model input and feature.

The three models are Gemma 3 27B, GPT-OSS 20B, and Qwen3 32B. Leave-one-LLM-out
evaluation uses a held-out model's binary vote as the score and includes a row only
when the other two models both have valid, agreeing votes.

## Leave-one-LLM-out OOF PR-AUC

| Tier | Held-out model | Eligible rows | Other-two positives | PR-AUC |
| --- | --- | ---: | ---: | ---: |
| ALL | Gemma | 1,741 | 18 | 0.0157 |
| ALL | GPT-OSS | 802 | 32 | 0.1047 |
| ALL | Qwen3 | 1,054 | 372 | 0.3825 |
| HIGH | Gemma | 699 | 17 | 1.0000 |
| HIGH | GPT-OSS | 699 | 17 | 1.0000 |
| HIGH | Qwen3 | 699 | 17 | 1.0000 |
| MEDIUM | Gemma | 1,042 | 1 | 0.0010 |
| MEDIUM | GPT-OSS | 103 | 15 | 0.1456 |
| MEDIUM | Qwen3 | 355 | 355 | undefined |

HIGH is perfect by construction because all three votes agree. MEDIUM is conditioned
on majority disagreement or missing votes, so leave-one-out subsets become highly
imbalanced and can be single-class. Consequently these are approximate silver-label
reproducibility bounds, not a single usable ceiling for model performance. Complete
development, train, TCP, pooled OOF, tier, and fold results are saved in
`governance/step2d_ceiling_v1/leave_one_llm_out.csv`.

## Pairwise OOF agreement

| Pair | Eligible rows | Agreement | Cohen's kappa |
| --- | ---: | ---: | ---: |
| Gemma / GPT-OSS | 2,200 | 0.4795 | 0.1130 |
| Gemma / Qwen3 | 2,216 | 0.3696 | 0.0165 |
| GPT-OSS / Qwen3 | 2,203 | 0.7921 | 0.0462 |

The high raw agreement for GPT-OSS/Qwen3 coexists with low kappa because their positive
rates are strongly imbalanced. This audit therefore shows substantial model-specific
label behavior rather than a clean shared ceiling.

## Vote-fraction rank correlation

Vote fraction is yes votes divided by the three fixed models, yielding thirds; invalid
votes contribute no yes vote. On 2,221 OOF-scored rows:

| Model | Spearman rho | p-value |
| --- | ---: | ---: |
| DeBERTa two-seed logit average | 0.2340 | 5.19e-29 |
| TF-IDF unweighted | 0.2473 | 2.70e-32 |

Both models weakly track graded LLM agreement, with TF-IDF slightly higher. Results on
the 2,199 rows with all three valid votes are nearly unchanged.

## Error-analysis export

`governance/step2d_ceiling_v1/oof_extreme_errors.csv` contains 160 rows: the 40
highest-scoring OOF negatives and 40 lowest-scoring OOF positives for each of DeBERTa
logit-average and unweighted TF-IDF. It contains record ID, family, tier, both model
scores, and clause text. It contains neither rationales nor individual votes, and it
does not modify any label.
