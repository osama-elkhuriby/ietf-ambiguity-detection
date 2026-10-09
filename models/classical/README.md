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

Syntax features use the locally installed `en_core_web_sm` model when
available. Use `--syntax disabled` for a dependency-light deterministic run,
or `--syntax required` when a missing parser should be an error. No model is
downloaded by this command.
