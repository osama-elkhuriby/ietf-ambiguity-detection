# Recall Improvement Study V3 — Manual GPU Execution Guide

This guide is maintained under `docs/gpu/` to keep operational documentation
separate from the repository root. Run its commands from the repository root as
described below.

This package runs exactly one immutable training cell per command. It has no unattended runner, no stage loop, no retry path, and no background-job support. Run commands only in the order below, and start the next command only after you have verified the previous cell completed successfully.

## Frozen identities

- Conda environment: `ietf-ambiguity-detection`
- Frozen interpreter: `C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe`
- GPU: `NVIDIA RTX 5000 Ada Generation`
- V3 protocol SHA-256: `9502e659c1f246f433ce534fa7165959a6519c6637c71b3be0289f6638e72c51`
- V3 source snapshot SHA-256: `22af9f00506241a4d78918b3b1fc65f1a787281a685c966fc881df99671ee8c4`
- Manual launcher: `D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py`

The original failed V2 Partial-4 Fold 1 specification, stderr log, and `FAILED_NO_AUTOMATIC_RETRY` status remain unchanged. Its stderr SHA-256 is `bf3dd00e4d6e39ce920862e56c04b5c0952e807cd8cc82b6b1a786865d6f8f0b`.

## Before every cell

Open PowerShell in `D:\RFC\ietf-ambiguity-detection`. Confirm the interpreter and CPU-only package validation:

```powershell
Set-Location 'D:\RFC\ietf-ambiguity-detection'
& 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' -c "import sys; print(sys.executable); print(sys.version)"
& 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --validate-only
```

The validation must report 15 cells, the protocol/source hashes above, and `"cuda_used": false`. A previously successful cell will report `COMPLETE`; cells not yet run will report `PENDING_SEPARATE_GPU_AUTHORIZATION`.

Check the GPU immediately before each cell:

```powershell
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader
```

The GPU must be exactly `NVIDIA RTX 5000 Ada Generation`, have at least 18,000 MiB free, and have no other compute process. The launcher repeats these checks and also verifies that PyTorch sees exactly one CUDA device. Do not bypass the check if it refuses to start.

## Training order and exact commands

Each command sets the required authorization variable only for that command, invokes exactly one immutable specification, checks the exit code, and removes the variable afterward. Do not paste more than one command at a time.

### 1. Partial-4 Fold 1

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\partial4_fold1_optimizer_fix_attempt.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-4 Fold 1 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

### 2. Partial-4 Fold 2

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial4-fold2-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-4 Fold 2 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

### 3. Partial-4 Fold 3

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial4-fold3-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-4 Fold 3 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

### 4. Partial-4 Fold 4

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial4-fold4-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-4 Fold 4 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

### 5. Partial-4 Fold 5

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial4-fold5-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-4 Fold 5 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

### 6–10. Focal Loss Folds 1–5

#### Focal Loss Fold 1

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-focal-loss-fold1-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Focal Loss Fold 1 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Focal Loss Fold 2

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-focal-loss-fold2-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Focal Loss Fold 2 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Focal Loss Fold 3

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-focal-loss-fold3-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Focal Loss Fold 3 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Focal Loss Fold 4

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-focal-loss-fold4-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Focal Loss Fold 4 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Focal Loss Fold 5

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-focal-loss-fold5-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Focal Loss Fold 5 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

### 11–15. Partial-2 Folds 1–5

#### Partial-2 Fold 1

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial2-fold1-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-2 Fold 1 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Partial-2 Fold 2

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial2-fold2-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-2 Fold 2 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Partial-2 Fold 3

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial2-fold3-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-2 Fold 3 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Partial-2 Fold 4

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial2-fold4-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-2 Fold 4 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

#### Partial-2 Fold 5

```powershell
$env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED='1'; try { & 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --spec 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\cell_specs\manual-v1-partial2-fold5-seed2718.json'; if ($LASTEXITCODE -ne 0) { throw "Partial-2 Fold 5 failed with exit code $LASTEXITCODE" } } finally { Remove-Item Env:RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED -ErrorAction SilentlyContinue }
```

## Successful-run evidence

A successful command exits with code 0 and prints these three lines:

- `Completed exactly one V3 cell: <immutable run path>`
- `Cell logs: <stdout path> ; <stderr path>`
- `Completion manifest: <completion manifest path>`

The run directory contains `run_manifest.json`, `run_config.json`, the final checkpoint, history, predictions, metrics, and artifact hashes. The treatment audit is under `recall_improvement_study_v3_optimizer_fix\audits\manual_execution_v1`. Logs are under `recall_improvement_study_v3_optimizer_fix\logs\manual_execution_v1`. Completion manifests are under `recall_improvement_study_v3_optimizer_fix\completion_manifests\manual_execution_v1`.

Run the CPU-only validator after each successful cell before proceeding:

```powershell
& 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\manual_cell_launcher.py' --validate-only
```

The just-completed cell must be `COMPLETE`. The validator checks its manifest, treatment audit, artifact hashes, immutable authorization/specification hashes, protocol/source hashes, dataset identity, leakage gate, and fold membership without using CUDA.

## Expected output path pattern

- Partial-4: `...\runs\G_partial_4\fold_<N>\seed_2718\attempt_<attempt-id>`
- Focal Loss: `...\runs\D_focal_loss\fold_<N>\seed_2718\attempt_manual-v1-focal-loss-fold<N>-seed2718`
- Partial-2: `...\runs\G2_partial_2\fold_<N>\seed_2718\attempt_manual-v1-partial2-fold<N>-seed2718`

The exact path for every cell is recorded in its immutable cell specification and in `EXPERIMENT_INVENTORY.json`.

## If a cell fails

Stop the study sequence. Do not rerun the same command, edit its specification, delete its marker, reuse its output directory, or start the next cell. The launcher marks it `FAILED_NO_AUTOMATIC_RETRY` and writes an immutable failure manifest under `failure_manifests\manual_execution_v1`; its logs and any `.building-*`/`.failed-*` training directory are diagnostic evidence and must remain in place.

Send Codex the failed cell ID plus these paths:

- `experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\EXPERIMENT_INVENTORY.json`
- the cell's `.stderr.log` and `.stdout.log`
- the cell's failure manifest
- any preserved `.building-*` or `.failed-*` directory beside the intended run path

No retry is automatic. A genuinely new attempt would require a new specification, attempt ID, authorization file, logs, marker, and output directory.

## How to stop safely

If you must stop an active cell, press `Ctrl+C` once and wait for Python to exit. Treat that attempt as failed and do not rerun it. Avoid terminating the process from another shell because an uncatchable termination may leave the inventory at `IN_PROGRESS`; even then, preserve all files and report the state rather than editing or deleting anything.

Between cells, simply do not run the next command. Nothing is scheduled or running in the background.

## Final CPU-only analysis

After all 15 cells validate as `COMPLETE`, first validate the completed inputs:

```powershell
& 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\analyze_completed_v1.py' --validate-only
```

Then create the immutable final analysis once:

```powershell
& 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe' 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\manual_execution_v1\analyze_completed_v1.py' --output 'D:\RFC\ietf-ambiguity-detection\experiments\transformer_final\recall_improvement_study_v3_optimizer_fix\analysis\manual_execution_v1_final' --require-all
```

The analysis is CPU-only. It produces pooled AP, mean-fold AP, ROC-AUC, precision, recall, F1, F2, TP/FP/TN/FN, Top-K retrieval, family performance, nested threshold-calibration performance, GPU runtime comparisons, and full-versus-Partial-4/Partial-2 comparisons. `--require-all` refuses to create a final report while any V3 stage is incomplete.

To return results to Codex, provide the final analysis directory and the V3 inventory path. The analysis manifest carries the original Experiment 2 baseline provenance, V1 nested-run provenance, V2 context provenance, and V3 manual-run manifest hashes without relabeling historical artifacts.
