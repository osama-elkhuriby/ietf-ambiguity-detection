# Leakage-prevention checklist

- [x] Outer folds use the frozen RFC-family mapping; no family spans fit and score.
- [x] Inner GroupKFold uses RFC family and only the corresponding outer
  manifest's leakage-filtered `fitting_record_ids`.
- [x] Outer held-out labels and predictions are unavailable to checkpoint,
  threshold, ECDF, weight, and calibration fitting.
- [x] Fixed five-epoch training uses final epoch only; no checkpoint selection or
  early stopping uses held-out observations.
- [x] Focal and weighted-CE class weights are computed inside each cell from its
  fitting labels only.
- [x] Input mode is target-only. Labels, provenance fields, and model scores are
  not tokenizer inputs.
- [x] Label-blind exact-target, near-target (0.85), lineage, document,
  BCP14-skeleton, and target/context controls are applied within every design.
- [x] Outer audit has zero cross-fold exact duplicates, document overlaps,
  lineage conflicts, BCP14-skeleton conflicts, and multi-fold families.
- [x] Detected near-target/context relationships are handled by fitting
  exclusions/context sanitization recorded in each manifest.
- [x] TCP and DNS are absent from fitting, calibration, threshold selection,
  ensemble fitting, and analysis.
- [x] Focal and Partial-4 inner OOF predictions must align one-to-one before the
  ensemble is formed.
- [x] Each method-specific ECDF is fit on matching inner OOF scores and applied
  unchanged to untouched outer scores; weights are fixed 0.5/0.5.
- [x] Raw `probability_yes` remains separate from ensemble `yes_score`, which is
  explicitly a ranking score rather than a calibrated probability.
- [x] Resume is disabled; every cell initializes from the approved local base.
- [x] Run paths are exclusive and completion artifacts are immutable/fail-fast.

## Selection-bias limitation

The 100 reviewed clauses were selected from historical out-of-fold errors.
Therefore, even this newly trained family-safe nested evaluation on the same
families is not a fully independent estimate of improvement. Results may be
reported as performance on the revised development annotation, but not as an
unbiased generalization gain. A genuinely independent evaluation set is needed
for that claim.

