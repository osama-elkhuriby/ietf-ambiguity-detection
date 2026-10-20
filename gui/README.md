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

## GUI model selection

The currently wired single-RFC scorer is class-balanced Logistic Regression
with engineered features and TF-IDF. Among its comparable standalone
classifiers, it had the strongest pooled grouped out-of-fold average precision:

| Classifier | Pooled OOF average precision | Pooled OOF ROC-AUC |
| --- | ---: | ---: |
| Logistic Regression, engineered + TF-IDF | 0.5225 | 0.8122 |
| Linear SVM, engineered + TF-IDF | 0.5109 | 0.8111 |

Both results use 2,406 clauses from 11 draft families. At the 0.5 threshold,
the Logistic Regression model's pooled OOF recall was 0.677, versus 0.367 for
the SVM. The SVM had higher precision (0.725 versus 0.215), but the Logistic
Regression ranking is preferred for this screening interface, where surfacing
more possible positives is useful. These are development metrics against operational labels, not independent
expert validation.

The human-reviewed transformer experiment reported ensemble AP 0.6736 and
ROC-AUC 0.9118 on 2,143 clauses across nine development families. This is a
research candidate only: its evaluation cohort differs from the saved
classical model's cohort, the result does not establish a reliable advantage,
and there is no corresponding single-RFC inference artifact wired into this
GUI. The GUI shows this result separately and does not use it to score uploads.
Its ensemble score is a ranking score, not a calibrated probability.

In a separate comparison on 2,097 label-compatible rows, the V4 Qwen/MiniLM
semantic ensemble had AP 0.3373, compared with AP 0.4567 for Logistic
Regression and 0.4619 for their three-way rank ensemble with DeBERTa. These
are exploratory OOF results on yet another cohort, not a direct comparison
with the human-reviewed transformer experiment or the active model. Pure
Qwen prompting had pooled AP 0.1513–0.1960 in its separate evaluation. In
this GUI Qwen remains an optional explanation aid; its explanations do not
affect predictions.

The revision-aware Logistic Regression experiment reached pooled OOF AP
0.5172, but requires prior-draft lineage that is not available from a single
uploaded RFC. The GUI therefore continues to use the saved classical model
for live scanning while presenting the newer ensemble as a research result
awaiting same-cohort evaluation and inference integration.
