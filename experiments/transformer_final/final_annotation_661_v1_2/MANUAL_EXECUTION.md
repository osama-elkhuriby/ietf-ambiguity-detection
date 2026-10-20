# Manual execution — final_annotation_661_v1_2

Run all commands from `D:\RFC\ietf-ambiguity-detection` in PowerShell. Preparation
is complete. None of the preparation, validation, or test commands trains a model
or loads model weights.

```powershell
$Python = "C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe"
$Module = "experiments.transformer_final.final_annotation_661_v1_2.cli"
```

## 1. V1.2 preflight

```powershell
& $Python -m $Module preflight
```

The command must report `status: pass`, `valid_reused_outer_cells: 10`,
`inner_partitions: 15`, `inner_execution_specs: 30`, and
`training_executed: false`.

## 2. Validate the ten reused outer cells

```powershell
& $Python -m $Module validate-outer --require-all
```

This hashes and validates the immutable V1.1 references. It does not load a
checkpoint or run inference.

## 3. Manually execute the 30 inner GPU cells

This is the first command that trains. Run it only after explicit user approval,
in the approved CUDA environment. It starts each cell from the approved base
model and stops immediately on failure; it never retries automatically.

For a deliberate fresh attempt of the formerly failed logical cell only:

```powershell
& $Python -m $Module train-inner --method focal --outer-fold 1 --inner-fold 1
```

After that cell validates, run the remaining cells one at a time with the same
command while varying `--method` (`focal`, `partial4`), `--outer-fold` (1–5), and
`--inner-fold` (1–3). A completed cell is validated and skipped if included in a
later all-cells command. To execute all pending cells sequentially:

```powershell
& $Python -m $Module train-inner --all
```

The exact single-cell template is:

```powershell
& $Python -m $Module train-inner --method <focal|partial4> --outer-fold <1..5> --inner-fold <1..3>
```

## 4. Validate all inner cells

```powershell
& $Python -m $Module validate-inner --require-all
```

Do not proceed to analysis unless all 30 cells validate.

## 5. Run and validate final CPU analysis

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
& $Python -m $Module analyze
& $Python -m $Module validate-analysis
```

The analysis command refuses incomplete cell sets. It reads stored predictions
only and produces the nested metrics and Top-20 error reports; it performs no
model inference.

## CPU-only regression tests

```powershell
$env:CUDA_VISIBLE_DEVICES = ""
& $Python -m pytest experiments/transformer_final/final_annotation_661_v1_2/tests -q
```

