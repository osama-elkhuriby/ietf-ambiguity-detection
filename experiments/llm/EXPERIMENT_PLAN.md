# Qwen3 32B zero/few-shot experiment plan

Status: implementation and preflight validation. No fine-tuning is part of this plan.
TCP validation and the sealed DNS test remain unavailable until the development
condition has been selected and a separate final-evaluation protocol is approved.

## Frozen inputs

- Source: `results/annotation/final_annotation_ready.csv`
- Source SHA-256: `c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712`
- Resolved rows: 2,424; excluded unresolved placeholder: `item_id=2161`
- Development: 2,142 rows, 126 positive, nine families
- Reserved TCP: 131 rows, 5 positive
- Sealed DNS: 151 rows, 17 positive
- Model: local Ollama `qwen3:32b`, digest
  `030ee887880fc378860c2dd35101da424377520441ae4bfe7be6deff8ade7840`

## Development protocol

Evaluate zero-shot, four-shot, and eight-shot conditions across five grouped
document-family folds. Every development clause is scored once per condition. Few-shot
demonstrations come only from the corresponding fitting families, and no previous,
target, or next-clause text may overlap an evaluated prompt.

The model receives the ordinary-engineer ambiguity rubric plus previous, target, and
following clauses. It returns exactly `yes` or `no`. The continuous ambiguity score is
`P(yes)/(P(yes)+P(no))`, calculated from native Ollama top-token log-probabilities at
the generated label position. Runs fail instead of inventing a score when both label
alternatives are unavailable.

PR-AUC is primary. Also report precision, recall, F1, F2, accuracy, balanced accuracy,
confusion matrices, per-fold results, and label-source slices. Accuracy remains
secondary because the development positive prevalence is only 5.88%.

Select the shot count by mean grouped-fold PR-AUC, preferring fewer demonstrations when
a condition lies within one paired-fold standard error of the top condition. This is
development selection only; TCP and DNS are not read.

## Commands

```powershell
.\scripts\run_llm_in_context.ps1 -Stage audit
.\scripts\run_llm_in_context.ps1 -Stage smoke
.\scripts\run_llm_in_context.ps1 -Stage all
```

The full run is sequential and resumable per clause to keep GPU memory use predictable.
A completed response is reused only when
its model digest, config, evaluator, prompt messages, and generation options have the
same cache identity.

## Claim boundary

The labels were manually annotated with assistance from two AI models but were not
independently verified by a large human panel. Results measure agreement with this
operational reference annotation and must not be described as universal ambiguity
accuracy. Qwen3 32B also contributed model votes during the annotation-support
workflow, so this experiment is not independent of label construction and may
overestimate agreement relative to a separately human-annotated external test set.
