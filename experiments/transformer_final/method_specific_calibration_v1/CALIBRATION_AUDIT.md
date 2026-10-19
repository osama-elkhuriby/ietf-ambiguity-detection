# Calibration Audit

## Verdict

**Qualified.** Historical outer predictions, cohort identity, labels, fold membership, score direction, and artifact hashes validate. The existing nested thresholds are valid only for the Full Fine-Tuning model that generated the inner OOF scores. Their later application to Focal Loss, Partial-4, Partial-2, and Local Context is a cross-method threshold transfer and is not a valid method-specific calibrated operating point.

## Verified facts

- Source CSV SHA-256: `c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712`.
- Canonical effective cohort SHA-256: `a25e99563563f71570e48f63b9515fbb51ff845762ff718763d98e0b3eefae2f`.
- The prompt supplied the truncated value `a25e99563563f71570e48f63b9515fbb51ff845762ff718763d98e0b`; the canonical value above matches all frozen repository manifests.
- Development cohort: 2142 rows, 126 positive, 2016 negative, five family-aware outer folds.
- Every completed method has exactly one held-out prediction per eligible record, with matching labels, families, and folds.
- Positive class is integer `1` and score direction is probability of `Yes`/ambiguous.
- No TCP or DNS family occurs in the development predictions or nested split IDs.
- Existing Full Fine-Tuning thresholds by outer fold are: 0.010, 0.030, 0.020, 0.010, 0.010.
- The V3 aggregate analyzer maps those same fold thresholds onto every method. This is the exact source of the transferred-threshold result.
- Fifteen inner OOF cells exist for Full Fine-Tuning only. No compatible Focal Loss or Partial-4 inner OOF predictions exist.
- Thirty new cells are therefore required: 5 outer folds × 3 inner folds × 2 methods.
- Historical V3 protocol SHA-256 remains `9502e659c1f246f433ce534fa7165959a6519c6637c71b3be0289f6638e72c51`.

## Existing prediction complementarity at threshold 0.5

- Positive detections: `{"both": 8, "focal_only": 37, "missed_by_both": 79, "partial4_only": 2}`.
- False-positive overlap: `{"focal_only": 77, "partial4_only": 5, "shared": 8}`.
- Score Spearman correlation: 0.6992.

## Finding classification

- `CURRENT_MODEL_COMPARISON.csv`, fixed threshold 0.5, AP/AUC, and existing Top-K: **valid existing evaluation**.
- Score distributions, calibration bins, complementarity, and newly examined ensemble combinations: **exploratory diagnostic**.
- Focal Loss/Partial-4 method-specific operating points and the pre-registered ECDF ensemble: **require additional fitting-side training** until all 30 inner cells validate.

## Governance limitation

Approximately 88% of the operational labels originate from agreement between two AI annotators. They are not an independently human-verified gold standard. Correlated annotator errors can inflate agreement-based performance and hide systematic ambiguity categories.
