# Recall Improvement Study v1

This isolated study starts from the frozen annotation-ready Experiment 2
baseline. It does not modify that baseline, its source data, frozen folds, or
historical run artifacts. Stage A is a completed CPU-only analysis of the
verified Experiment 2 out-of-fold predictions.

## Frozen cohort and design

- Cohort: 2,142 resolved Dataset A records; TCP and DNS remain reserved.
- Existing five family-aware outer folds and Experiment 2 protocol are pinned
  by source, effective-dataset, fold, model, and tokenizer hashes.
- Training is FP32 weighted cross-entropy, seed 2718, five epochs, and the
  existing configuration. The only planned variations are local context,
  partial fine-tuning, and (if time permits) focal loss.
- Stage B uses three deterministic, family-disjoint `GroupKFold` partitions
  inside each outer fitting set. Its threshold is selected only from inner OOF
  scores; each corresponding outer fold is scored once afterward.
- Every model cell starts from local base weights. There are no automatic
  retries, resumes, early stopping, or precision fallbacks.

## Workflow

Run from the repository root using the configured Conda interpreter:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PATH='C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\Library\bin;' + $env:PATH
python -m experiments.transformer_final.recall_improvement_study_v1.study_protocol
python -m experiments.transformer_final.recall_improvement_study_v1.study_runner
```

The protocol command verifies the frozen Experiment 2 identity, makes all 15
inner manifests, audits the local tokenizer and model architecture on CPU, and
writes immutable protocol files. The runner checks hashes, GPU capacity, and
competing compute processes before each cell. On Windows WDDM, it distinguishes
CUDA compute processes from the known desktop graphics processes; unknown or
compute processes block execution. It executes one cell at a time;
validates outputs before proceeding; and records per-cell logs, an inventory,
and actual wall-time use. It stops on any failed cell and never retries it.

Stage order is nested threshold calibration, local context, Partial-4,
CPU-only error analysis, focal loss, and Partial-2. The runner honors the
authorized deadline and retains a reporting reserve; a later optional stage
may be skipped if there is insufficient time to complete it.

## Results and limits

Generated outputs are stored only under this directory: frozen protocols and
splits, immutable run cells, audit JSON, logs, nested threshold results,
selected error examples, family metrics, plots, and a final report. Analysis
and threshold selection are CPU-only. Error-cue labels are deterministic
lexical aids for manual review, not expert ambiguity annotations.

The study is exploratory and uses one seed. CPU synthetic tests validate split,
threshold, focal-loss, and freezing helpers, but cannot certify the GPU
optimizer path or full pretrained-model behavior.
