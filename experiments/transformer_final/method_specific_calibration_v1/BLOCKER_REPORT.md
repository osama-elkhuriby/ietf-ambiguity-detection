# Method-Specific Calibration V1 — Blocker Report

## Execution status

The GPU sequence stopped after the first planned cell, as required by the frozen
stop-on-first-validation-failure policy. No retry occurred and no later cell was
started.

- Cell: `focal_loss/outer_1/inner_1/seed_2718`
- Attempt: `cal-v1-focal-loss-outer1-inner1-seed2718`
- Study status: `FAILED_VALIDATION`
- Remaining cells: 29 `PLANNED`
- Training subprocess return code: 0
- Recorded GPU training time: 145.1016 seconds
- Peak allocated GPU memory: 3645.9922 MiB
- Peak reserved GPU memory: 5638.0 MiB

## Exact mismatch

The generic training implementation writes `run_manifest.json` field
`fold_manifest_sha256` from the canonical dataset file named by the numeric
training fold. For this run it wrote:

`684cac99acb8eca3d34698265f23a9f22e879de9f1ec25280e02cf61bd87643e`

The method-specific cell specification correctly identifies the injected custom
inner split with:

`df7bcbb6af0d6e781f8a55f0101c8820366f3435718e3180fdd12aa501d61b38`

The treatment audit field `custom_split_manifest_sha256` also equals
`df7bcbb6af0d6e781f8a55f0101c8820366f3435718e3180fdd12aa501d61b38`.

The V1 validator incorrectly compared the generic legacy field directly with the
custom inner-split hash. This caused a false validation failure after successful
training.

## Evidence that remains valid but is not promoted

- Fitting rows: 910/910 expected.
- Scoring rows: 439/439 expected.
- Saved prediction rows: 439.
- Focal loss: gamma 2 with fitting-side class weights.
- Seed: 2718; FP32; five epochs; fresh pretrained initialization.
- Source dataset and effective cohort hashes match.
- Custom split hash in the treatment audit matches the immutable specification.

The run is not counted as complete because the frozen validator rejected it.

## Required recovery design

Create a separately versioned V1.1 protocol and validator that:

1. Preserves this failed-validation attempt and all logs/manifests unchanged.
2. Validates the generic `fold_manifest_sha256` against its documented legacy
   meaning.
3. Validates the injected inner split through
   `custom_split_manifest_sha256`, record IDs, family isolation, and row counts.
4. Reuses this successful GPU artifact by immutable reference only if every V1.1
   compatibility check passes; it must not retrain or retry the cell.
5. Generates new unique attempt identities for the remaining 29 cells and stops
   on any subsequent failure.

Final calibrated metrics and validated ensemble results cannot be generated until
all required inner OOF prediction cells are complete and compatible.
