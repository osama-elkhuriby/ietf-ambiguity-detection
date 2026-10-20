# Manual execution — final_annotation_human_reviewed_100_v1

All transformer training is manual, sequential, CUDA-only, FP32, and fail-fast.
No preparation command starts training. The frozen protocol SHA-256 is
`122057289686ab6fae0edde521e30f480020b5a76eb4dd2bf84543f9367b5f77`.

```powershell
Set-Location -LiteralPath 'D:\RFC\ietf-ambiguity-detection'
$python = 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe'

# 1–3. Read-only preflight and fold-count verification
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli preflight
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.environment_preflight
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli status
Import-Csv -LiteralPath 'experiments\transformer_final\final_annotation_human_reviewed_100_v1\FOLD_CLASS_COUNTS.csv' | Format-Table

# 4–6. Ten outer cells, sequential; no retries
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli train-outer --all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli status
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli validate-outer --require-all

# 7–9. Thirty inner cells, sequential; no retries
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli train-inner --all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli status
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli validate-inner --require-all

# 10–12. CPU-only analysis, Top-20 generation, integrity validation
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli analyze
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli validate-analysis

# 13. Inspect failures and attempt logs
Get-ChildItem -LiteralPath 'experiments\transformer_final\final_annotation_human_reviewed_100_v1\failure_manifests' -File
Get-ChildItem -LiteralPath 'experiments\transformer_final\final_annotation_human_reviewed_100_v1\logs' -File

# 14. After investigating an interruption, run only the still-pending cell
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli train-outer --method focal --fold 1
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli train-inner --method focal --outer-fold 1 --inner-fold 1

# 15. Re-verify frozen source/protocol and historical immutability
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli preflight
git status --short -- experiments/transformer_final/recall_improvement_study_v3_optimizer_fix experiments/transformer_final/method_specific_calibration_v1 experiments/transformer_final/method_specific_calibration_v1_1
```

If a cell fails, inspect its log and failure manifest. The runner does not retry,
does not overwrite completed artifacts, and stops the sequence immediately. A
completed cell is skipped only after its manifest and held-out predictions pass
validation. Do not delete or replace a completed run; create a new experiment
version if frozen code or protocol must change.

Interpretation warning: the 100 reviewed clauses were selected from historical
OOF errors. Results on these previously reviewed families are not a fully
independent estimate of generalization improvement.
