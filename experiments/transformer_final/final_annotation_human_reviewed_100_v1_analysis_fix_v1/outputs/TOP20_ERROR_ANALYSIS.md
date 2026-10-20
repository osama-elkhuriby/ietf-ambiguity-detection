# Top-20 error analysis

Primary decisions use each outer fold's method-specific threshold selected only from inner family-OOF predictions at precision target 0.30. Ensemble scores are ranking scores, not calibrated probabilities.

## FOCAL

- Primary status: COMPLETE.
- False positives / listed: 162 / 20.
- False negatives / listed: 37 / 20.
- Feasible threshold range: 0.07992093 to 0.17875244.
- Strongest FP score: 0.96709186; strongest FN (lowest) score: 0.038082175.
- Recurring semantic patterns are not asserted automatically; review the preserved clauses with independent human adjudication.

## PARTIAL4

- Primary status: COMPLETE.
- False positives / listed: 140 / 20.
- False negatives / listed: 40 / 20.
- Feasible threshold range: 0.012029999 to 0.026225513.
- Strongest FP score: 0.99568343; strongest FN (lowest) score: 0.0007481466.
- Recurring semantic patterns are not asserted automatically; review the preserved clauses with independent human adjudication.

## ENSEMBLE

- Primary status: COMPLETE.
- False positives / listed: 160 / 20.
- False negatives / listed: 37 / 20.
- Feasible threshold range: 0.7970974808324206 to 0.8600682593856656.
- Strongest FP score: 0.9988872403560831; strongest FN (lowest) score: 0.11641541038525963.
- Recurring semantic patterns are not asserted automatically; review the preserved clauses with independent human adjudication.

## Cross-model priorities

Review records flagged by two or three methods first, then large-margin threshold-dependent errors. Model disagreement is evidence of difficulty, not proof of an annotation error.
