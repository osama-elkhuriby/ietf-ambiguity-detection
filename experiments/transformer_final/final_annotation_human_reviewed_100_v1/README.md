# final_annotation_human_reviewed_100_v1

Prepared, frozen, and preflight-validated; training has not started.

Protocol SHA-256:
`122057289686ab6fae0edde521e30f480020b5a76eb4dd2bf84543f9367b5f77`.

Start with `MANUAL_EXECUTION.md`. Run the read-only preflight commands first,
then manually run all ten outer cells and thirty inner cells. Analyze only after
both validation commands pass with `--require-all`. The launcher is sequential,
CUDA-only, offline, fail-fast, and has no automatic retry or checkpoint resume.

Key evidence:

- `DATASET_AUDIT.md` / `PREPARATION_AUDIT.json`
- `EFFECTIVE_COHORT_AND_SPLITS.md` / `FOLD_CLASS_COUNTS.csv`
- `LEAKAGE_PREVENTION_CHECKLIST.md` / `LEAKAGE_AUDIT.json`
- `BASELINE_COMPARISON.md`
- `GPU_NUMERICAL_PREFLIGHT.md`
- `RUN_MATRIX.csv` / `cell_specs/`
- `OUTPUT_PATHS.md`

Do not edit frozen `.py`, `config.json`, manifests, cell specs, or protocol files.
If scientific code/configuration must change, create a new experiment version.

