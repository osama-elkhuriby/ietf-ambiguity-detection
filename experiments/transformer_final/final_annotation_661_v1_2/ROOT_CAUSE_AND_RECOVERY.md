# V1.2 inner-manifest recovery

## Exact root cause

`models/transformer_final/train.py::run_cell()` treats the fifth value returned by
`load_fold_data()` as a complete prepared dataset root. At final run-manifest
construction it hashes both `dataset_dir/manifest.json` and
`dataset_dir/folds/fold_N.json`.

The V1.1 adapter returned `manifests/inner/outer_N` as `dataset_dir` so the fold
hash would describe the custom inner split. V1.1 preparation placed `dataset.csv`,
`leakage_audit.json`, and three fold JSON files there, but never produced
`manifest.json`, `validation_report.json`, or fold assignment CSVs. It instead
passed the global manifest object and global dataset path in each inner spec.
Training and final-epoch serialization therefore completed before provenance
assembly raised `FileNotFoundError` for the missing inner `manifest.json`.

The historical nested runner in `recall_improvement_study_v1/cell_launcher.py`
returned the global dataset root. That avoided the missing file but made the
historical run manifest hash a global outer-fold file rather than the actual
custom inner split. V1.1 corrected the root selection without completing the
dataset-root contract.

The authoritative schema and producer are
`models/transformer_final/dataset.py::prepare_dataset()` (schema version 1).
V1.2 uses its exact manifest field set and records the producer source hash in
`SOURCE_PROVENANCE.json`.

## Additional fitting-partition incompatibility

V1.1 built each inner dataset from all rows whose global outer-fold number was
different from the held-out fold. That ignored the outer fold manifest's
label-blind `fitting_exclusions`. The resulting inner inputs contained eight
records not used to fit their completed outer models: one for outer 1, three for
outer 3, three for outer 4, and one for outer 5. V1.2 derives every inner dataset
from the authoritative outer `fitting_record_ids` and retains the established
inner-family assignment for every remaining record. It regenerates the inner
leakage audit and three effective fold manifests.

## Why preflight and tests missed it

V1.1 preflight validated only the global dataset manifest, inner fold hashes, and
class-count summaries. Its tests counted partitions and exercised utility/static
guards but never asserted the complete inner dataset-root interface returned to
`train.run_cell()`. Neither path required an inner `manifest.json`, so the defect
was reached only after checkpoint and prediction serialization.

## Failed-attempt and outer-artifact disposition

The failed cell has an unpublished final-epoch checkpoint (`model.safetensors`),
predictions, metrics, history, configuration, and tokenizer files inside the
V1.1 hidden `.failed-*` directory. It has no `run_manifest.json`, audit, or
completion manifest and is not reusable or complete. Hashes for all 11 preserved
files are pinned in `V1_1_IMMUTABILITY.json`.

All ten outer cells are referenced directly from V1.1. V1.2 does not copy them.
Validation covers dataset identity, effective fitting and held-out IDs, labels,
model/configuration semantics, split and artifact hashes, prediction provenance,
checkpoint files, audits, and completion manifests. The evidence is recorded in
`OUTER_REUSE_MANIFEST.json`.

