# Hybrid routing versus LLM-only

The primary hybrid preserves detection effectiveness versus LLM-only at 79.0% fewer LLM calls under the default (0.4, 0.7) band.

Family-cluster bootstrap: 1000 valid resamples; 0 one-class resamples skipped.

## Results (calls_saved is the fraction of clauses not sent to the LLM)

| system                 |   calls_saved |   pr_auc |    f2 |   recall |   precision |
|:-----------------------|--------------:|---------:|------:|---------:|------------:|
| ML-only                |         1.000 |    0.326 | 0.418 |    0.716 |       0.157 |
| LLM-only               |         0.000 |    0.372 | 0.447 |    0.569 |       0.241 |
| Hybrid                 |         0.790 |    0.330 | 0.457 |    0.624 |       0.221 |
| Random-budget LLM-only |         0.790 |    0.302 | 0.423 |    0.685 |       0.167 |

The default (0.4, 0.7) band is primary; the remaining bands are sensitivity analyses.
Paired hybrid-versus-LLM-only confidence intervals:

| metric          |   difference |   ci_low |   ci_high | significant   |
|:----------------|-------------:|---------:|----------:|:--------------|
| pr_auc          |       -0.042 |   -0.305 |     0.085 | False         |
| f2              |        0.010 |   -0.059 |     0.057 | False         |
| recall_at_20pct |        0.018 |   -0.064 |     0.057 | False         |
| roc_auc         |       -0.032 |   -0.068 |     0.012 | False         |

The primary band routes 441 clauses, including 33 positives. The LLM flips 230
ML decisions: 199 flips are correct and 31 are wrong. The "preserves"
classification follows the requested CI rule; intervals containing zero do not
prove equivalence or formal non-inferiority.

## Band-sweep trade-off

|   low |   high |   calls_saved |   pr_auc |    f2 |
|------:|-------:|--------------:|---------:|------:|
| 0.450 |  0.550 |         0.927 |    0.327 | 0.436 |
| 0.400 |  0.700 |         0.790 |    0.330 | 0.457 |
| 0.300 |  0.800 |         0.630 |    0.352 | 0.472 |
| 0.200 |  0.900 |         0.443 |    0.384 | 0.478 |
| 0.000 |  1.000 |         0.000 |    0.372 | 0.447 |

`routing_tradeoff.png` plots call fraction against PR-AUC and F2, with ML-only and LLM-only references.
`per_family.csv` reports each family separately; pooled performance does not guarantee transfer.

## Caveats

- Exploratory development comparison on OOF predictions; V4 and the models were tuned during development, so this is not a sealed-test result.
- LLM answers are simulated from stored V4 probabilities; no new LLM calls were made.
- The ML OOF file comes from an earlier annotation artifact, so rows missing from it or with conflicting labels were excluded; report how many.
- Labels are an operational reference, not ground truth; scores are rankings, not calibrated probabilities (no calibration metrics were recorded), so the 0.4/0.7 band is a heuristic.
- Only report "improves" if the paired CI supports it.

Excluded from the aligned cohort: 46 rows (0 missing from ML OOF and 46 label conflicts).
