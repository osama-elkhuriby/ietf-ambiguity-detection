# Output paths

No path below exists as a completed run yet. Every cell uses an exclusive
`attempt_<cell_id>` directory and refuses overwrite/resume.

## Training artifacts

- Outer run root: `runs/outer/<method>/fold_<N>/attempt_<cell_id>/`
- Inner run root: `runs/inner/<method>/outer_<N>/inner_<M>/attempt_<cell_id>/`
- Checkpoint per cell: `<run-root>/checkpoints/final_epoch/`
- Checkpoint metadata: `<run-root>/checkpoints/checkpoint_metadata.json`
- Per-clause held-out OOF predictions: `<run-root>/predictions/held_out_predictions.csv`
- Fold metrics: `<run-root>/metrics/metrics.json` and `per_family_metrics.csv`
- Training history: `<run-root>/history/training_history.csv`
- Logs: `logs/<cell_id>.stdout.log` and `.stderr.log`
- Completion proof: `completion_manifests/<cell_id>.json`
- Method/freeze audit: `audits/<cell_id>.json`

The five outer prediction files for each method form that method's raw pooled
outer OOF export; coverage validation requires every effective record exactly
once. Ensemble ranking scores are computed separately from raw probabilities.

## Post-training evaluation

- `OUTER_FOLD_METRICS.csv`
- `INNER_FOLD_CALIBRATION.csv`
- `SELECTED_THRESHOLDS.csv`
- `FINAL_NESTED_METRICS.csv` (includes fold and pooled outer OOF results)
- `ENSEMBLE_VALIDATION.csv`
- `TOP_K_COMPARISON.csv`
- `FAMILY_ROBUSTNESS.csv`
- `UNCERTAINTY_INTERVALS.csv`
- `GPU_TRAINING_TIME.csv`
- Per-method/ensemble Top-20 clause exports and disagreement reports
- `FINAL_REPORT.md`, `SCIENTIFIC_CONCLUSIONS.md`, and `TOP20_ERROR_ANALYSIS.md`

All paths are relative to this experiment directory. The exact absolute run path
for every cell is pinned in its `cell_specs/*.json` file.

