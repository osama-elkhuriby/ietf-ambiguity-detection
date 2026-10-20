# Local RFC Review GUI

A local browser app for ranking clauses in a rendered RFC plaintext file. The
saved engineered-feature + TF–IDF Logistic Regression model scores explicitly
normative clauses; optionally, Qwen3 32B through Ollama generates a short
human-review note for up to five selected clauses.

This is a research demonstration, not a validated ambiguity detector. The
classifier's score is used for ranking, not as a calibrated probability or a
definitive yes/no decision. Manual-review, contextual, and excluded clauses are
counted but not scored. Review all outputs against the RFC.

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

Upload a rendered RFC `.txt` file (maximum 5 MB), review the ranked clauses,
optionally select up to five for Qwen notes, and download a CSV for human
assessment. The upload is parsed from a temporary file that is removed after
parsing; recent parsed results are held in process memory only for explanations.
No document text is written to the repository or sent to a remote service.

## Model provenance and limitations

The chosen Logistic Regression model reported pooled out-of-fold average
precision (AP) 0.5225 and ROC-AUC 0.8122 on 2,406 annotated clauses from 11
draft families. These are development estimates against operational reference
labels, not independent expert-adjudicated results. The saved model was refit on
all 2,406 rows, and independent production evaluation has not been performed.
Results from different cohorts or studies are not directly comparable.

The saved artifact was produced with scikit-learn 1.9.1. If the active
environment uses another version, the GUI reports the compatibility warning
raised by scikit-learn; match the training version before relying on
predictions. The ranked list is a review aid, not a claim that a clause is
ambiguous.
