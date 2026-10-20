# Hybrid models

## Final rank-level ensemble

`final_rank_ensemble.py` compares the V4 Qwen/MiniLM semantic ensemble, the
DeBERTa-v3-base pilot, and the best available classical model by development AP:
class-balanced Logistic Regression with engineered features plus TF-IDF. It
evaluates fixed equal-weight averages of within-cohort percentile ranks. Rank
averaging avoids assuming that the three models' probability scales are
interchangeable. No labels are used to tune ensemble weights or a decision threshold.

The inputs are the saved grouped out-of-fold predictions from:

- `results/LLM/qwen3-32b-final-annotation-ready-v4/analysis-final/oof_predictions.csv`
- `results/transformer/final_studies/final_annotation_ready_pilot_v1/aggregate_metrics/pooled_oof_predictions.csv`
- `results/ML/v2_engineered_tfidf/oof_predictions.csv`

Run from the repository root after both input artifacts are available:

```powershell
python -m models.hybrid.final_rank_ensemble
```

The run writes `metrics.csv`, `oof_predictions.csv`, and `manifest.json` under
`results/hybrid/final_rank_ensemble_v2/`. Existing output directories are not
overwritten; pass a new `--output-dir` when rerunning. V4 and DeBERTa must match on
item IDs, family, fold, and label. The ML run used an earlier annotation artifact:
17 current development records are absent
from it and 28 shared records have conflicting labels. The experiment excludes
those 45 records and compares every model on the same 2,097 rows with matching labels
(116 positives). It also verifies that each ML validation group maps to one document
family.

| Model | Average precision | ROC-AUC |
| --- | ---: | ---: |
| V4 semantic ensemble | 0.3373 | 0.8341 |
| DeBERTa-v3-base pilot | 0.2517 | 0.7535 |
| Logistic Regression, engineered + TF-IDF | 0.4567 | 0.7980 |
| V4 + DeBERTa, equal-rank average | 0.3546 | 0.8381 |
| V4 + ML, equal-rank average | 0.4582 | **0.8451** |
| V4 + DeBERTa + ML, equal-rank average | **0.4619** | 0.8414 |

The three-model ensemble has the highest AP of these fixed comparisons on the
label-compatible subset; combining V4 with ML alone has slightly higher ROC-AUC.
This is exploratory evidence, not a fair same-label retraining comparison: the
classical model was trained using the earlier annotation artifact, whose labels
differ on the excluded records and potentially on its training families. V4 and
ensemble design were also developed using this cohort, DeBERTa is a single-seed
pilot, and labels are not independently human-verified. TCP validation and sealed
DNS test results are not included. Do not select a production threshold from these
pooled OOF scores; retrain/re-evaluate ML on the frozen final annotation artifact
and validate the frozen method on reserved TCP before making a transfer claim.

## LLM routing

The proposal's SS6.1 routing component uses a calibrated high-recall selector
(classical ML or transformer) to route likely or uncertain clauses to an LLM
verifier while resolving clear low-risk clauses without an LLM call. This is the
component relevant to H2 and RQ4. The rank ensemble above is a final offline
comparison, not a calibrated routing policy.
