# Classical ML

Per proposal SS6.1: logistic regression, SVM, random forest, gradient-boosted trees,
using linguistic, structural, cross-reference, and revision features from `alignment/`.

Note from the risk table: ambiguous clauses are likely rare relative to total clauses.
Plan class weighting / negative sampling from the start rather than retrofitting it —
this is called out explicitly as a High/High risk (class imbalance and optimistic leakage).

## Feature preparation

Build deterministic feature artifacts from the existing LLM-consensus binary
dataset without training a model:

```powershell
python -m models.classical.feature_pipeline
```

The default input is
`corpus/processed/llm_annotations/v1/training_dataset.csv`; it is read-only.
The output is `corpus/processed/features/v1/` and contains:

- `matrices/<split>_tfidf.npz`: word unigram/bigram TF-IDF.
- `matrices/<split>_engineered.npz`: scaled numerical clause features.
- `matrices/<split>_combined.npz`: TF-IDF followed by engineered features.
- `audit/engineered_features.csv`: unscaled, interpretable numerical values.
- `targets/`, `weights/`, and `rows/`: labels, sample weights, and row metadata
  kept outside all feature matrices.
- `preprocessing/`: the train-fitted TF-IDF vectorizer and numerical scaler.
- `feature_names.json` and `manifest.json`: stable schemas and provenance.

Existing family-level split assignments are preserved. TF-IDF vocabulary,
IDF values, and numerical scaling are fitted only on `train`. Validation and
test rows are transformed with those frozen components.

The reviewed v2 model split replaces the contaminated HTTP/2bis holdout with DNS
while preserving the original LLM annotation artifact. Build matching classical
features with:

```powershell
python -m corpus.build_model_dataset
python -m models.classical.feature_pipeline `
  --input corpus/processed/model_datasets/v2/training_dataset.csv `
  --output-dir corpus/processed/features/v3 `
  --syntax disabled
```

Syntax features use the locally installed `en_core_web_sm` model when
available. Use `--syntax disabled` for a dependency-light deterministic run,
or `--syntax required` when a missing parser should be an error. No model is
downloaded by this command.

## Frozen TF-IDF comparator

Run the target-only grouped-CV comparator with:

```powershell
python -m models.classical.tfidf_comparator
```

It fits a fresh TF-IDF vocabulary and logistic regression inside each frozen
`folds_v1.json` fitting partition. The four supported fold-local weighting modes are
unweighted, class, confidence, and combined. Exact cross-boundary duplicate targets
remain available to the fitting partition but are excluded from that fold's reported
score, matching the transformer protocol. The immutable output includes raw logits,
probabilities, per-fold manifests, percentile bootstrap intervals, and pooled OOF
summaries under `experiments/classical/results/tfidf-logreg-v1/`.

## Final annotation logistic regression

Train and validate a combined numerical-feature and TF-IDF logistic regression
on `results/annotation/final_annotation.csv` with:

```powershell
python -m models.classical.train_annotation_logreg
```

Validation uses five stratified folds grouped by draft family, balanced class
weights, and fold-local TF-IDF/scaling. The out-of-fold predictions and fold
metrics are saved under `results/annotation/logistic_regression/`; the final
model refit on all annotations is saved there as `final_model.joblib`.

The v2 annotation experiments use the same 23 ambiguity-focused numerical
features, class-balanced L1 logistic regression, and three-fold validation
grouped by draft family. TF-IDF vocabulary and all scalers are fit within each
training fold. Results should be treated as exploratory because there are only
49 positive labels across 11 draft families.

Run engineered features plus TF-IDF:

```powershell
python -m models.classical.train_annotation_logreg_v2
```

Then run the three text-only alternatives:

```powershell
python -m models.classical.train_annotation_tfidf_only
python -m models.classical.train_annotation_embeddings_only
python -m models.classical.train_annotation_tfidf_embeddings
```

Outputs are saved separately under `results/annotation/v2_engineered_tfidf/`,
`results/annotation/v2_tfidf_only/`,
`results/annotation/v2_embeddings_only/`, and
`results/annotation/v2_tfidf_embeddings/`. The frozen MiniLM encoder is used
offline for embedding modes; no model download is attempted. Each run also
refits a final model on all rows for downstream use, while out-of-fold
predictions and metrics report validation performance.

For a single engineered-plus-TF-IDF train/test run with draft families held
out, use:

```powershell
python -m models.classical.train_annotation_engineered_tfidf_holdout
```

Preprocessing and the model are fit on the training families only. Metrics are
reported once on the held-out test families and saved under
`results/annotation/v2_engineered_tfidf_holdout/`.

## Exploratory experiments

The experiment notebook and standalone Python runners are collected in this
directory:

- `ambiguity_feature_cleanup.ipynb`: exploratory classical-model comparisons
  and same-revision previous-clause context.
- `run_signal_check.py`: revision-predecessor coverage and quick LOFO signal
  checks. Outputs and local caches remain under `results/revision_features/`.
- `run_embeddings_context.py`: offline MiniLM/BGE embedding comparisons.
  Outputs remain under `results/embeddings_context/`; embeddings are cached
  under its ignored `cache/` directory.
- `audit_vote_weights.py`: vote-weight distribution and stop-gate audit.
  Outputs remain under `results/vote_weights/`.

Run the standalone scripts from the repository root, for example:

```powershell
python -m models.classical.run_embeddings_context
```

The runners use the existing v3 feature artifacts and do not rewrite them.
