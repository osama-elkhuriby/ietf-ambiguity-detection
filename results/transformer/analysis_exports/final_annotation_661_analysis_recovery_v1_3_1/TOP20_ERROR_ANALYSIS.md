# Top-20 error analysis

Primary decisions use each outer fold's method-specific threshold selected only from inner family-OOF predictions at precision target 0.30. Ensemble scores are ranking scores, not calibrated probabilities.

## FOCAL

- Primary status: COMPLETE.
- False positives / listed: 54 / 20.
- False negatives / listed: 104 / 20.
- Feasible threshold range: 0.13639061 to 0.82267815.
- Strongest FP score: 0.9644029; strongest FN (lowest) score: 0.058541905.
- Recurring semantic patterns are not asserted automatically; review the preserved clauses with independent human adjudication.

## PARTIAL4

- Primary status: COMPLETE.
- False positives / listed: 72 / 20.
- False negatives / listed: 83 / 20.
- Feasible threshold range: 0.036686487 to 0.16249503.
- Strongest FP score: 0.9532324; strongest FN (lowest) score: 0.0022145924.
- Recurring semantic patterns are not asserted automatically; review the preserved clauses with independent human adjudication.

## ENSEMBLE

- Primary status: COMPLETE.
- False positives / listed: 62 / 20.
- False negatives / listed: 84 / 20.
- Feasible threshold range: 0.8443708609271523 to 0.9607035928143712.
- Strongest FP score: 1.0; strongest FN (lowest) score: 0.08624577226606539.
- Recurring semantic patterns are not asserted automatically; review the preserved clauses with independent human adjudication.

## Cross-model priorities

Review records flagged by two or three methods first, then large-margin threshold-dependent errors. Model disagreement is evidence of difficulty, not proof of an annotation error.
