# Heuristic baseline

Per proposal SS6.1: normative-keyword conflicts, vague quantifiers, undefined terms,
unresolved references, complex conditionals, pronoun/anaphora patterns, exception
density, long dependency chains.

This is the detector used in the week-1 pilot alongside one LLM prompt — keep it simple
and fast to run; it exists to give the hybrid router (`models/hybrid/`) a cheap first pass.

## Annotation heuristic

Run `python models/heuristic/ambiguity_heuristic.py` to score the deduplicated
annotation set with fixed, interpretable rules over the engineered features from
`annotation_logreg_experiment2.py`. The rules are not fitted or tuned against the
labels. A condition without an exception marker, deictic expressions, mixed modal
levels, subjective terms, and exception markers increase the score. Numeric bounds
and target references reduce it. These are the signals available in this feature
schema; it does not provide the prior heuristic's distinct hedge, pronoun-position,
or neighboring-clause-overlap features. The fixed score threshold is 0.5.

Evaluation uses the same three-fold stratified group split by draft family and reports
PR-AUC, precision, recall, F1, F2, and ROC-AUC. Results are written to
`results/heuristic/final_annotation_v1/`: `metrics.json`, `predictions.csv`, and
`engineered_features.csv`. Pass `--input` or `--output-dir` to select different paths.
