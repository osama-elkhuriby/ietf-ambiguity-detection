# Local RFC Review GUI

A local browser app for reviewing rendered RFC plaintext. It first uses the
project's evidence-first clause parser to retain clauses classified as
**NORMATIVE**: clauses with explicit uppercase BCP 14 requirement keywords that
are not quoted, imported, or terminology explanations. The five-fold DeBERTa-v3
Focal + Partial-4 ensemble scores only those clauses. A clause is flagged when
at least three of the five fold classifiers vote YES. The results table and CSV
show only positive predictions. Qwen3 32B through Ollama can optionally explain
up to five YES predictions.

This is a research demonstration, not a validated ambiguity detector. Positive
results use fold-specific ECDF thresholds selected on inner OOF data for a
0.30 precision target, followed by a 3-of-5 vote. The vote aggregation has not
been separately evaluated, so the target is not a production precision
guarantee. The displayed mean ECDF score is for ranking, not a calibrated
probability or an adjudicated ambiguity judgment. Contextual clauses, excluded
text, lowercase modals, and section-context candidates without explicit BCP 14
keywords are never treated as normative by this GUI and are not scored.

## Requirements

- The project Python environment and dependencies.
- The complete current-source experiment artifacts under
  `experiments/transformer_final/final_annotation_human_reviewed_100_v1/`,
  including all 10 outer checkpoints, all 30 inner prediction cells,
  manifests, and the frozen protocol. The GUI validates these artifacts and
  verifies the hashes of the deployed checkpoints at startup.
- The pinned local DeBERTa-v3 base model at
  `D:/RFC/models/deberta-v3-base`. The GUI runs inference offline and does not
  download or train model weights.
- PyTorch, Transformers, and the other project Python dependencies.
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

The active human-reviewed transformer experiment pins the current annotation
source SHA-256
`8ffa654c2c405dd3eef72901c600f1ef3823e6dbff663020bb824c57ff65bdbf`. Its
nested grouped OOF development cohort contains 2,143 clauses, 138 positives,
and nine draft families; reserved TCP and DNS families are excluded. The
Focal + Partial-4 ECDF ensemble reported AP 0.6736 and ROC-AUC 0.9118. These
are development estimates against operational reference labels, not
independent expert-adjudicated results.

For the experiment's individual outer-fold classifiers, thresholds were
selected from their respective inner OOF predictions to maximize recall while
meeting the 0.30 fitting-side precision objective. The pooled nested OOF
results for that per-fold policy were precision 0.5063 and recall 0.6212.
At runtime, an unseen RFC is scored by all five outer-fold classifiers and
flagged only when at least three vote YES. This majority-vote deployment rule
and the runtime mean ECDF ranking score are not the policies evaluated by
those pooled OOF metrics, which score each clause with its corresponding
held-out outer-fold model. Neither deployment aggregate has measured precision
or recall, and neither the 0.30 target nor the reported OOF metrics guarantee
performance on new RFCs.

Each fold's Focal and Partial-4 probabilities are mapped through that fold's
inner-OOF empirical CDFs and equally averaged. The resulting mean ECDF score is
a ranking score, not a probability. The GUI retains only explicitly normative
clauses identified by its parser, and a positive result is a suggestion for
human review, not a claim that a clause is definitively ambiguous.

## GUI model selection

The GUI now uses the current-source human-reviewed transformer ensemble. The
saved classical model is retained as a separate comparison; its cohort differs
from the transformer experiment, so the results below are not directly
comparable:

| Classifier | Pooled OOF average precision | Pooled OOF ROC-AUC |
| --- | ---: | ---: |
| Logistic Regression, engineered + TF-IDF | 0.3265 | 0.8082 |
| Linear SVM, engineered + TF-IDF | 0.3584 | 0.8070 |

Both classical results use 2,425 clauses from 11 draft families. At the 0.5 threshold,
the Logistic Regression model's pooled OOF recall was 0.706, versus 0.313 for
the SVM. The SVM had higher precision (0.455 versus 0.190). These are
development metrics against operational labels, not independent expert
validation.

In a separate comparison on 2,097 label-compatible rows, the V4 Qwen/MiniLM
semantic ensemble had AP 0.3373, compared with AP 0.4567 for Logistic
Regression and 0.4619 for their three-way rank ensemble with DeBERTa. These
are exploratory OOF results on yet another cohort, not a direct comparison
with the human-reviewed transformer experiment or the active GUI model. Pure
Qwen prompting had pooled AP 0.1513–0.1960 in its separate evaluation. In this
GUI Qwen remains an optional explanation aid; its explanations do not affect
predictions.

The matched revision-aware Logistic Regression experiment reached pooled OOF AP
0.3185, but requires prior-draft lineage that is not available from a single
uploaded RFC, so it is not used for live scanning.
