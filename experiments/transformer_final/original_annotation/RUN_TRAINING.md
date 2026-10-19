# DeBERTa-v3-base Experiment 1: manual execution

This protocol uses only Dataset A (`final_label`), excludes unresolved item `2161`, and reserves TCP and DNS from fitting, model selection, threshold selection, and calibration. The primary input is the target clause alone. Every run initializes a fresh `microsoft/deberta-v3-base` from the same local directory, trains for exactly five epochs, saves the final epoch, and evaluates its outer fold once. There is no early stopping or outer-fold checkpoint selection.

Run all commands from `D:\RFC\ietf-ambiguity-detection` in Windows PowerShell. Stop at the first nonzero exit code. Commands never download files: tokenizer and model calls enforce `local_files_only=True` and offline environment variables.

## Step A — preflight

```powershell
Set-Location 'D:\RFC\ietf-ambiguity-detection'
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m models.transformer_final.preflight `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --model-path 'D:\RFC\models\deberta-v3-base' `
  --require-cuda `
  --output experiments/transformer_final/original_annotation/audit_logs/preflight_v1
```

CPU/GPU: CPU file/dependency checks plus a non-computational CUDA visibility query. Loads pretrained weights: no. Downloads: no. Writes: immutable `preflight_v1/preflight_report.json`. It hashes every local model/tokenizer file and fails if required files or CUDA are missing. It does not initialize a tokenizer or model.

The shipped configuration uses `revision: null` because the explicit local directory is pinned by a complete per-file hash manifest. If your local directory is an identified Hugging Face snapshot, set `model.revision` to its exact commit hash before creating dataset/run artifacts; the loader passes that revision while remaining offline.

If `preflight_v1` already exists, inspect it or choose a new versioned output name; the command will not overwrite it.

## Step B — Dataset A and fold validation

The immutable Dataset-A manifest is part of this implementation. Verify it against its recorded hashes:

```powershell
python -m models.transformer_final.dataset `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --verify-existing
```

CPU/GPU: CPU. Loads pretrained weights: no. Downloads: no. Writes: none. This validates strict joins, the one authorized unresolved exclusion, effective labels, unique scoring IDs, all five family folds, and all artifact hashes. It never opens the sealed-test artifact and does not report DNS row-level labels.

For a clean checkout where `dataset_v1` is intentionally absent, omit `--verify-existing` to create it once. Never point `--output` at an old Transformer directory:

```powershell
python -m models.transformer_final.dataset `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json
```

CPU/GPU: CPU. Loads pretrained weights: no. Downloads: no. Writes: immutable Dataset-A dataset, validation report, leakage audit, and five fold manifests under `original_annotation/manifests/dataset_v1`. It fails rather than overwriting an existing directory.

## Step C — tokenizer-length audit

```powershell
python -m models.transformer_final.tokenization `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --model-path 'D:\RFC\models\deberta-v3-base' `
  --input-mode target_only `
  --max-length 256 `
  --output experiments/transformer_final/original_annotation/audit_logs/tokenizer_target_only_256_v1
```

CPU/GPU: CPU. Loads pretrained weights: no; loads only the local tokenizer. Downloads: no. Writes: immutable `tokenizer_length_audit.json`. Review `target_truncated_rows` before training. The deterministic overflow policy keeps the head and tail of an overlength target; context is always discarded before target text is truncated.

## Step D — one-cell training smoke run

Run exactly fold 1, seed 1729, A1 first:

```powershell
python -m models.transformer_final.train `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --condition A1 `
  --fold 1 `
  --seed 1729 `
  --model-path 'D:\RFC\models\deberta-v3-base' `
  --precision auto
```

CPU/GPU: GPU training (CPU preprocessing). Loads pretrained weights: yes, only from the explicit local base directory. Downloads: no. Writes: one immutable run at `training_outputs/A1/fold_1/seed_1729`. This is a real five-epoch run, not a reduced scientific protocol. Review its manifest, history, metrics, predictions, and final checkpoint before continuing. Rerunning the same command fails rather than overwriting it.

If `--precision auto` resolves to BF16 and the run stops on a non-finite loss, retry that same cell in FP32 without changing any other setting:

```powershell
python -m models.transformer_final.train `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --condition A1 `
  --fold 1 `
  --seed 1729 `
  --model-path 'D:\RFC\models\deberta-v3-base' `
  --precision fp32
```

The failed attempt is preserved in a hidden `.seed_1729.failed-*` directory; because no completed `seed_1729` output exists, this retry can use the same immutable output location.

Training diagnostics validate binary labels, distinguish non-finite logits from non-finite cross-entropy, add precision/epoch/batch/optimizer-step context to backward and gradient-clipping failures, and check model parameters after pretrained loading and every optimizer update. Model parameters are converted to FP32 before optimizer construction; BF16 compute remains selected through autocast when requested or resolved by `auto`. Failures do not trigger an automatic retry or precision fallback, and exception messages do not include clause text or token IDs.

For a manually authorized numerical investigation, append `--diagnostic-numerics` to the one-cell training command. This opt-in mode records compact gradient statistics before and after clipping, model parameter statistics before the update, scheduler-adjusted optimizer group settings/dtypes, and non-finite optimizer-state summaries on failure. It does not change optimizer settings or recover from failures.

## Step E — remaining A1 cells

This loop intentionally skips the completed smoke cell and runs the other 14 cells sequentially:

```powershell
$seeds = @(1729, 2718, 31415)
foreach ($seed in $seeds) {
  foreach ($fold in 1..5) {
    if (($seed -eq 1729) -and ($fold -eq 1)) { continue }
    python -m models.transformer_final.train `
      --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
      --condition A1 `
      --fold $fold `
      --seed $seed `
      --model-path 'D:\RFC\models\deberta-v3-base' `
      --precision auto
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
}
```

CPU/GPU: GPU training. Loads pretrained weights: yes, freshly for every cell. Downloads: no. Writes: 14 new immutable A1 run directories. No fold resumes from another fold.

## Step F — all A2 weighted-CE cells

```powershell
$seeds = @(1729, 2718, 31415)
foreach ($seed in $seeds) {
  foreach ($fold in 1..5) {
    python -m models.transformer_final.train `
      --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
      --condition A2 `
      --fold $fold `
      --seed $seed `
      --model-path 'D:\RFC\models\deberta-v3-base' `
      --precision auto
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }
}
```

CPU/GPU: GPU training. Loads pretrained weights: yes, freshly from the same original base for every cell. Downloads: no. Writes: 15 immutable A2 run directories. Each class ratio is computed only from that cell's post-exclusion fitting records and is persisted in its run manifest.

## Step G — aggregate results

Run only after all 15 cells for each condition are complete:

```powershell
python -m models.transformer_final.aggregate `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --condition A1

python -m models.transformer_final.aggregate `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --condition A2
```

CPU/GPU: CPU. Loads pretrained weights: no. Downloads: no. Writes: immutable `aggregate_metrics/A1` and `aggregate_metrics/A2`. Aggregation requires all cells, verifies one out-of-fold prediction per effective record per seed, preserves raw logits/probabilities, and reports per-fold AP, mean-fold AP, pooled OOF metrics, per-seed results, across-seed mean/sample standard deviation, AP/prevalence lift, and family metrics.

An optional exploratory threshold table is deliberately separate from unbiased reporting:

```powershell
python -m models.transformer_final.threshold_analysis `
  --predictions experiments/transformer_final/original_annotation/aggregate_metrics/A1/oof_predictions.csv `
  --output experiments/transformer_final/original_annotation/aggregate_metrics/A1_threshold_exploratory_v1
```

CPU/GPU: CPU. Loads pretrained weights: no. Downloads: no. Writes: an explicitly marked exploratory artifact. Never present a threshold chosen here as an unbiased result on these same OOF predictions.

## Step H — compare A1 and A2

```powershell
python -m models.transformer_final.compare `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --output experiments/transformer_final/original_annotation/aggregate_metrics/A1_vs_A2_v1
```

CPU/GPU: CPU. Loads pretrained weights: no. Downloads: no. Writes: a paired fold/seed table and summary for AP, precision, recall, and F1, including sample standard deviation, standard error, a descriptive normal 95% interval, and prevalence. The report explicitly treats differences as descriptive rather than proof of superiority.

## Step I — files to share for interpretation

Send another AI these files; checkpoints are not necessary:

- `original_annotation/manifests/dataset_v1/manifest.json`
- `original_annotation/manifests/dataset_v1/validation_report.json`
- `original_annotation/manifests/dataset_v1/leakage_audit.json`
- all five `original_annotation/manifests/dataset_v1/folds/fold_*.json` files (not the private key)
- `original_annotation/aggregate_metrics/A1/summary.json`
- `original_annotation/aggregate_metrics/A1/per_fold_seed_metrics.csv`
- `original_annotation/aggregate_metrics/A1/per_seed_metrics.csv`
- the equivalent three A2 files
- `original_annotation/aggregate_metrics/A1_vs_A2_v1/comparison_summary.json`
- optionally `per_family_seed_metrics.csv` and `oof_predictions.csv` when row-level error analysis is desired
- any affected cell's `run_manifest.json` and `history/training_history.csv` when diagnosing a run

Do not share `results/annotation/key_PRIVATE.csv`. The generated manifests intentionally omit private votes and reviewer metadata.

## Protocol notes

The remaining Dataset A Experiment 1 cells use the frozen `original_annotation/configs/fp32_protocol_v1.json` protocol. It pins FP32 parameter storage, FP32 forward computation without autocast, and FP32 cross-entropy while retaining the existing folds, seeds, model source, optimizer, scheduler, and fixed five-epoch settings. The explicit `run_path_overrides_fp32_v1.json` mapping reuses the verified A1 fold 1 / seed 1729 diagnostic run without moving or rewriting it.

To run the sequential, fail-closed experiment manager after reviewing the frozen protocol:

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:PATH='C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\Library\bin;' + $env:PATH
python -m models.transformer_final.experiment_runner `
  --config experiments/transformer_final/original_annotation/configs/experiment_v1.json `
  --protocol experiments/transformer_final/original_annotation/configs/fp32_protocol_v1.json `
  --run-overrides experiments/transformer_final/original_annotation/configs/run_path_overrides_fp32_v1.json `
  --model-path 'D:\RFC\models\deberta-v3-base'
```

The runner executes one GPU cell at a time, verifies each immutable cell before recording completion, stores separate per-cell stdout/stderr logs and a durable inventory under `original_annotation/run_management/fp32_protocol_v1/`, then aggregates A1 before starting A2. A failed or interrupted cell is preserved and stops the sequence; the runner does not retry it. New run manifests hash the Transformer source files, protocol, dependencies, data/fold artifacts, model files, checkpoint, and CUDA environment. The reused diagnostic run has no retroactively assigned source hash.

Five epochs are pre-registered as a selection-free starting policy, not an optimized value. Revisiting epoch count requires a new versioned experiment with inner family-held-out development data; candidate epochs must be chosen entirely inside each outer fitting partition, followed by exactly one outer evaluation. Results from that new protocol must not be merged with this experiment.

PyTorch deterministic algorithms, fixed Python/NumPy/PyTorch seeds, fixed data-loader generators, zero workers, and deterministic preprocessing are enabled. Complete bitwise reproducibility is still not guaranteed across CUDA, cuDNN, GPU driver, PyTorch, or Transformers versions; every run records those versions and the GPU model. `warn_only=True` is used for operations without deterministic CUDA implementations so such operations warn rather than silently changing the scientific protocol.

Dataset B is not referenced by any command. A future Experiment 2 must use a new namespace, newly versioned manifests, and fresh initialization from the original pretrained base—not an Experiment 1 checkpoint.
