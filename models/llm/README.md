# LLM zero/few-shot evaluation

The LLM stage uses the locally installed Ollama model `qwen3:32b`. V1 evaluates
zero-shot, four-shot, and eight-shot inference. V2 is a separate strict-prompt
8/16/24-shot comparison with cross-fitted decision-threshold estimation. V3 combines
the frozen V1 zero-shot score with a nested word/character TF-IDF logistic classifier.
V4 adds frozen MiniLM target and local-context semantic features. Qwen and MiniLM are
never fine-tuned; V3/V4 fit lightweight classifiers and are therefore supervised
hybrids rather than pure prompting experiments.

The evaluation source is
`results/annotation/final_annotation_ready.csv` (SHA-256
`c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712`). The one
unresolved placeholder (`item_id=2161`) is excluded. The resulting isolated
development cohort contains 2,142 clauses and 126 positive labels. TCP is reserved and
DNS remains sealed.

## Components

- `final_annotation_dataset.py` validates the source, excludes the unresolved row,
  writes isolated partitions, and creates five family-grouped development folds.
- `prompting.py` contains the versioned ordinary-engineer ambiguity rubric and renders
  previous/target/next clause context.
- `ollama_client.py` validates exact `yes`/`no` responses and derives a continuous
  ambiguity score from Ollama's native token log-probabilities.
- `in_context_evaluate.py` selects fold-safe demonstrations, runs resumable inference,
  and produces PR-AUC, precision, recall, F1, F2, accuracy, balanced accuracy,
  confusion matrices, per-source metrics, and SVG comparisons.
- `in_context_evaluate_v2.py` preserves V1, selects fold-safe hard negatives from
  hash-pinned V1 out-of-fold scores, cross-fits F1/F2/balanced-accuracy thresholds,
  and creates the V1/V2 comparison artifacts.
- `hybrid_evaluate_v3.py` performs nested grouped tuning of the target-clause lexical
  model, zero-shot blend weight, and inner-only decision thresholds, then creates
  V1/V2/V3 comparisons and precision-recall visualizations.
- `semantic_ensemble_v4.py` fits grouped out-of-fold target and context semantic
  classifiers over a pinned local MiniLM encoder, blends them with V3, and creates
  V1-V4 comparisons and threshold sensitivities.

Few-shot examples are selected only from the fitting families of each outer fold. Four
shot uses one `Yes` and three `No` examples; eight shot uses two `Yes` and six `No`
examples. This exposes the positive class without imposing the unrealistic 50/50 prior
used by the retired baseline.

## Run

From the repository root:

```powershell
.\scripts\run_llm_in_context.ps1 -Stage audit
.\scripts\run_llm_in_context.ps1 -Stage smoke
.\scripts\run_llm_in_context.ps1 -Stage all
```

The audit checks the source/fold hashes, installed model name and digest, prompt
construction, and demonstration isolation without running inference. The smoke stage
runs six requests: two clauses under each of zero, four, and eight shots. The full run
is resumable at the individual-clause level.

The full development evaluation requires 6,426 local requests (2,142 clauses under
three conditions). Qwen3 32B can have substantial inference latency;
do not delete a partial result directory because rerunning skips every response whose
cache identity still matches.

Results are written to
`results/LLM/qwen3-32b-final-annotation-ready-v1/` and are ignored by
Git. The model digest, Ollama version, source/config/code hashes, prompt token counts,
raw response envelope, and `yes`/`no` log-probabilities are retained for audit.

Run the isolated V2 experiment with:

```powershell
.\scripts\run_llm_in_context_v2.ps1 -Stage audit
.\scripts\run_llm_in_context_v2.ps1 -Stage smoke
.\scripts\run_llm_in_context_v2.ps1 -Stage all
```

V2 writes to `results/LLM/qwen3-32b-final-annotation-ready-v2/` and never
reuses a V1 response as a V2 response. See `experiments/llm/EXPERIMENT_PLAN_V2.md` for
the class compositions, hard-negative boundary, threshold protocol, and claim limits.

Run V3 with:

```powershell
.\scripts\run_llm_in_context_v3.ps1 -Stage all
```

V3 reuses the hash-pinned V1 zero-shot scores and therefore requires no Ollama calls.
Its outputs are written to
`results/LLM/qwen3-32b-final-annotation-ready-v3/`. See
`experiments/llm/EXPERIMENT_PLAN_V3.md` for the nested protocol and limitations.

Run V4 with:

```powershell
.\scripts\run_llm_in_context_v4.ps1 -Stage all
```

V4 runs the pinned local `all-MiniLM-L6-v2` encoder offline and reuses a hash-recorded
embedding cache. Outputs are written to
`results/LLM/qwen3-32b-final-annotation-ready-v4/`. See
`experiments/llm/EXPERIMENT_PLAN_V4.md` for the adaptive-selection limitation.

The labels combine human annotation with assistance from two AI models. They are the
project's operational reference labels, not independently human-verified external
ground truth. Because Qwen3 32B also contributed votes during the annotation-support
workflow, its measured agreement is not independent of label construction and must be
reported with that limitation.
