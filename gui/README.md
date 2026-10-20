# Local RFC Review GUI

A local browser app for reviewing rendered RFC plaintext. It first uses the
project's evidence-first clause parser to retain clauses classified as
**NORMATIVE**: clauses with explicit uppercase BCP 14 requirement keywords that
are not quoted, imported, or terminology explanations. The saved engineered
features + TF–IDF Logistic Regression model scores only those clauses. The
results table and CSV show only YES (potentially ambiguous) predictions.
Qwen3 32B through Ollama can optionally explain up to five YES predictions.

This is a research demonstration, not a validated ambiguity detector. Positive
results use the saved model's threshold; the score is not a calibrated
probability or an adjudicated ambiguity judgment. Contextual clauses, excluded
text, lowercase modals, and section-context candidates without explicit BCP 14
keywords are never treated as normative by this GUI and are not scored.

## Requirements

- The project Python environment and dependencies.
- The local model artifact and provenance files under
  `results/ML/v2_engineered_tfidf/`:
  `final_model.joblib`, `manifest.json`, and `metrics.json`. The model artifact
  is local and is not included in Git.
- Ollama and the pinned `qwen3:32b` model for optional explanations.

Check the installed model with:

```powershell
ollama list
```

The GUI checks the model digest when a Qwen explanation is requested. It does
not switch to another model if the pinned version is unavailable.

## Run

From the repository root:

```powershell
python -m gui.app
```

The local-only server listens at `http://127.0.0.1:8765/` and opens the browser.
Press `Ctrl+C` in the terminal to stop it. To use another port or leave the
browser closed:

```powershell
python -m gui.app --port 8766 --no-browser
```

Upload a rendered RFC `.txt` file (maximum 5 MB). The GUI automatically scores
only clauses the parser classifies as normative based on explicit BCP 14
keywords, then displays and exports only the positive ambiguity predictions.
Search clause text, filter by section, or change the sort order. Select up to
five displayed results for optional Qwen explanations. The upload is parsed
from a temporary file that is removed after parsing; recent positive results
are held in process memory only for explanations. No document text is written
to the repository or sent to a remote service.

This is a deliberately conservative normativity gate, not a comprehensive
detector of every possible requirement. RFCs using only lowercase modals,
implicit section context, or other wording without explicit BCP 14 keywords
may produce no scored clauses. The GUI does not ask reviewers to confirm
normativity, and it does not promote those uncertain candidates into the
ambiguity classifier.

## Model provenance and limitations

The chosen Logistic Regression model reported pooled out-of-fold average
precision (AP) 0.5225 and ROC-AUC 0.8122 on 2,406 annotated clauses from 11
draft families. These are development estimates against operational reference
labels, not independent expert-adjudicated results. The saved model was refit on
all 2,406 rows, and independent production evaluation has not been performed.
Results from different cohorts or studies are not directly comparable.

The saved artifact was produced with scikit-learn 1.9.1. If the active
environment uses another version, the GUI reports the compatibility warning
raised by scikit-learn. Match the training version before relying on results.
The positive-result list is a research result, not a claim that a clause is
definitively ambiguous.
