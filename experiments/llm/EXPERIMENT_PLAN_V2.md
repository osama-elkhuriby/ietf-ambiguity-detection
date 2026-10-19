# Qwen3 32B strict in-context comparison experiment

Status: prompt/model audit and six-request smoke evaluation complete. The full grouped
development run is pending. No model parameter updates are part of this experiment.
V1 remains frozen under `qwen3-32b-final-annotation-ready-v1`; V2 writes only to a new
versioned result directory.

## Motivation

V1 retained high recall but produced excessive false positives. Its best-ranking
condition was eight-shot (pooled PR-AUC 0.196), while the generated binary label had
precision 0.098, recall 0.778, and F1 0.174. V2 tests whether a stricter decision
boundary, prevalence-aware demonstrations, fold-safe hard negatives, and a separately
estimated decision threshold improve development performance.

## Frozen inputs

- Labels: `results/annotation/final_annotation_ready.csv`, SHA-256
  `c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712`
- Development rows: 2,142, including 126 positives across nine document families
- Model: local Ollama `qwen3:32b`, digest
  `030ee887880fc378860c2dd35101da424377520441ae4bfe7be6deff8ade7840`
- Hard-negative ranking source: frozen V1 eight-shot out-of-fold predictions, SHA-256
  `d371c7a1dd560d7cfdae923adf7e2019cd8314d900910a70bf053796738e036c`
- TCP remains reserved; DNS remains sealed.

## V2 conditions

The strict prompt defaults to `no` unless the wording creates a specific unresolved
implementation conflict. Explicitly permitted discretion, MAY alternatives,
implementation-defined policy, references, technical complexity, and domain knowledge
alone are not ambiguity.

Each outer fold uses examples only from its fitting rows. Demonstrations cannot share
previous, target, or following-clause text with evaluated rows. Negative examples are
the V1 false-positive candidates with the largest ambiguity scores, selected with
family diversity. Positive examples prefer manually resolved rows and family diversity.

| Condition | Positive examples | Negative examples | Positive share |
| --- | ---: | ---: | ---: |
| 8-shot | 1 | 7 | 12.5% |
| 16-shot | 1 | 15 | 6.25% |
| 24-shot | 2 | 22 | 8.33% |

The development prevalence is 5.88%. The 16-shot condition most closely matches it;
the other conditions test the tradeoff between class-prior fidelity and positive-example
coverage.

## Metrics and threshold protocol

Mean grouped-fold PR-AUC remains the primary condition-selection metric. V2 also
reports raw precision, recall, F1, F2, accuracy, balanced accuracy, confusion matrices,
label-source slices, and predicted-positive rate.

For every condition, thresholds optimizing F1, F2, and balanced accuracy are estimated
with leave-one-grouped-fold-out cross-fitting: each fold is classified with a threshold
selected using only the other four folds. These are leakage-controlled internal
threshold estimates for each fixed condition; they are not external-test estimates,
and selecting among conditions on the same folds remains adaptive. After selecting the
condition, a final F2 threshold is fitted on all development rows solely for a future,
separately approved transfer evaluation; its development resubstitution score is
explicitly not treated as performance evidence.

The final analysis compares the selected and best-PR V1 conditions with raw and
cross-fitted V2 results in CSV, JSON, confusion-matrix SVGs, and a comparison SVG.

## Commands

```powershell
.\scripts\run_llm_in_context_v2.ps1 -Stage audit
.\scripts\run_llm_in_context_v2.ps1 -Stage smoke
.\scripts\run_llm_in_context_v2.ps1 -Stage all
```

Use `-Stage run` to stop after resumable inference and `-Stage analyze` to analyze an
already completed run. The full experiment makes 6,426 local requests. Do not delete a
partial run; matching response identities are reused.

## Claim boundary

This is a second-round development experiment designed after examining V1 development
errors. V2 can be compared with V1 on development data, but the comparison is adaptive
and is not final-test evidence. The operational labels lack independent human
verification, and Qwen3 32B contributed annotation-support votes. TCP and DNS must not
be opened until the complete V2 condition and threshold are frozen.
