# Method-Specific Calibration V1.1 — Compatibility Fix

## Scope

V1.1 changes only compatibility validation and execution infrastructure. Dataset
labels, cohorts, folds, model methods, hyperparameters, thresholds, and ensemble
methodology are unchanged.

## Hash semantics

- Legacy hash `684cac99acb8eca3d34698265f23a9f22e879de9f1ec25280e02cf61bd87643e` is the SHA-256 of the canonical dataset fold
  file loaded by the generic trainer for numeric `train_fold=1`.
- Custom hash `df7bcbb6af0d6e781f8a55f0101c8820366f3435718e3180fdd12aa501d61b38` is the SHA-256 of the stable JSON representation
  of the immutable outer-1/inner-1 fitting/scoring manifest.
- These values correctly identify different artifacts. Both are independently
  validated; no assertion was merely removed.

## Frozen split compatibility

All 15 frozen inner splits are checked against their authoritative records. The
check reconstructs family assignments, fitting exclusions, lineage and duplicate
evidence, context-overlap evidence, row and label counts, and complete one-time
OOF coverage. A later infrastructure cleanup omits context-sanitization entries
for fitting candidates that are already excluded; V1 retains those documented
historical entries. They cannot affect this study because every calibration cell
uses target-only input. No assignment or scientific setting was changed.

## First-cell disposition

- Historical attempt: `cal-v1-focal-loss-outer1-inner1-seed2718`.
- Training: `TRAINING_COMPLETED`.
- Original V1 validation: `V1_VALIDATION_FAILED` (preserved).
- V1.1 compatibility: `V1_1_COMPATIBILITY_VALIDATED`.
- Retrained: no.
- New GPU execution fabricated: no.
- Fitting/scoring/prediction rows: 910/439/439.
- Runtime: 145.1016 seconds.
- Peak allocated/reserved GPU memory: 3645.99/5638.00 MiB.

TCP and DNS remain sealed.
