# Heuristic baseline

Per proposal SS6.1: normative-keyword conflicts, vague quantifiers, undefined terms,
unresolved references, complex conditionals, pronoun/anaphora patterns, exception
density, long dependency chains.

This is the detector used in the week-1 pilot alongside one LLM prompt — keep it simple
and fast to run; it exists to give the hybrid router (`models/hybrid/`) a cheap first pass.

## Annotation heuristic

Run `python models/heuristic/ambiguity_heuristic.py` to score the deduplicated
annotation set with fixed, interpretable rules over the v2 engineered features. The
rules are not fitted or tuned against the labels. A missing alternative for a
conditional, anaphora, mixed modal levels, and vague terms increase the score;
numeric values, neighboring-clause overlap, and explicit section references reduce
it. The fixed score threshold is 0.5.

Evaluation uses the same three-fold stratified group split by draft family and reports
PR-AUC, precision, recall, F1, F2, and ROC-AUC. Results are written to
`results/heuristic/final_annotation_v1/`: `metrics.json`, `predictions.csv`, and
`engineered_features.csv`. Pass `--input` or `--output-dir` to select different paths.
