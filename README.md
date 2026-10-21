# Pre-Publication Ambiguity Detection in Internet-Drafts

Research software for identifying potentially ambiguous normative clauses in IETF
Internet-Drafts, with a focus on revision history, leakage control, and human-review
prioritization.

**Queen's University School of Computing**

## Project status

This is an active research project, not a validated production detector. The repository
contains corpus and alignment tooling, operational annotation workflows, classical and
transformer baselines, local-LLM experiments, and hybrid experiments.

| Area | Status |
| --- | --- |
| Corpus | Current manifest: 264 draft revisions and 13 RFCs; retained technical-errata filter: 31 records |
| Parsing and alignment | Plaintext/XML parsing, clause segmentation, and revision-lineage tooling implemented |
| Annotation | AI-assisted operational labels prepared; independent expert adjudication is not available |
| Model development | Classical, transformer, LLM, and hybrid studies completed on several distinct data artifacts |
| Hybrid rank comparison | Three-model V4/DeBERTa/classical comparison completed on a label-compatible subset; exploratory |
| Independent evaluation | No untouched project-level TCP/DNS holdout remains: historical LOFO and current all-family classical studies include both; they remain excluded from the nine-family transformer/Qwen tracks |

Scores measure agreement with operational reference labels, not adjudicated ambiguity
accuracy. Results from separate cohorts, label versions, and validation protocols must
not be treated as a single model leaderboard.

## Latest hybrid experiment

The current exploratory comparison uses out-of-fold predictions from:

- V4: Qwen/MiniLM semantic ensemble;
- DeBERTa-v3-base pilot; and
- class-balanced Logistic Regression with engineered features and TF-IDF.

Because the classical model used an earlier annotation artifact, the comparison excludes
17 records absent from that artifact and 28 shared records with conflicting labels.
Metrics below use the remaining 2,097 matching records (116 positives).

| Model | Average precision | ROC-AUC |
| --- | ---: | ---: |
| Logistic Regression, engineered + TF-IDF | 0.4567 | 0.7980 |
| V4 semantic ensemble | 0.3373 | 0.8341 |
| DeBERTa-v3-base pilot | 0.2517 | 0.7535 |
| V4 + ML, fixed equal-rank average | 0.4582 | **0.8451** |
| V4 + DeBERTa + ML, fixed equal-rank average | **0.4619** | 0.8414 |

This is exploratory development evidence, not an independent test result. The ML model
should be retrained on the frozen final annotation artifact before making a final
comparison claim. The model and data limitations are described in
[`models/hybrid/README.md`](models/hybrid/README.md).

## Repository map

| Path | Purpose |
| --- | --- |
| [`corpus/`](corpus/) | Corpus acquisition, metadata, annotation, and model-dataset builders |
| [`alignment/`](alignment/) | Parsing, clause segmentation, version alignment, and lineage |
| [`models/classical/`](models/classical/) | Classical ML feature pipelines and experiments |
| [`models/transformer/`](models/transformer/) | Transformer training, analysis, governance, and evaluation |
| [`models/llm/`](models/llm/) | Local Ollama prompting and supervised lexical/semantic hybrids |
| [`models/hybrid/`](models/hybrid/) | Cross-model ensemble experiment and paper-figure generator |
| [`experiments/`](experiments/) | Experiment plans, protocol configs, and reproducibility support |
| [`gui/`](gui/) | Local browser app for RFC clause ranking and optional Qwen explanations |
| [`scripts/`](scripts/) | Environment checks and workflow launchers; see [`scripts/README.md`](scripts/README.md) |
| [`docs/`](docs/) | Operator guides and project documentation index |
| [`results/`](results/) | Compact reports, metrics, and publication figures; local runs are ignored |
| [`tests/`](tests/) | Offline unit, integration, and governance tests |
| [`paper/`](paper/) | Manuscript source, bibliography, and figures |
| [`DECISIONS.md`](DECISIONS.md) | Dated methodological decisions |

Detailed result conventions and data-handling guidance are in
[`results/README.md`](results/README.md). The paper's scope, build instructions, and
submission checklist are in [`paper/README.md`](paper/README.md). Other operator
guides are indexed in [`docs/README.md`](docs/README.md).

## Setup

The supported environment targets Python 3.11. Create it with Conda:

```powershell
conda env create -f environment-windows.yml
conda activate ietf-ambiguity-detection
python scripts/check_environment.py
```

On Linux, use `environment-linux.yml`. The environment check is offline. Some
workflows additionally require local model weights, GPU resources, or network access;
see the corresponding model and experiment guides before running them.

## Common workflows

Run commands from the repository root unless noted.

### Review an RFC in the local GUI

The GUI classifies explicit normative clauses in rendered RFC plaintext with
the current-source human-reviewed DeBERTa-v3 Focal + Partial-4 ensemble. All
five outer-fold model pairs score each clause; a clause is flagged when at
least three folds exceed their own inner-OOF calibrated threshold. The GUI
shows positive predictions, fold-vote support, and a mean ECDF score for
ranking. The mean score is not a probability.

#### Before the first run

1. Create and activate the project environment using the [Setup](#setup)
   instructions.
2. Make the complete local experiment bundle available:
   - The approved annotation source `results/annotation/final_annotation_ready.csv`,
     private key, canonical context and metadata files, and frozen family-fold
     source specified by the experiment config. These inputs are not all
     distributed in Git.
   - The frozen experiment files, manifests, cell specifications, outer and
     inner run outputs under
     `experiments/transformer_final/final_annotation_human_reviewed_100_v1/`.
     GUI startup validates all 10 outer and 30 inner cells and verifies the
     checkpoint hashes before loading.
   - The pinned DeBERTa-v3 base model at the local path configured by
     `model.default_local_path` in the experiment's `config.json`. It must
     match the configured model-source hash. In the original Windows setup
     this was `D:\RFC\models\deberta-v3-base`. The config and model path are
     part of the frozen experiment; relocating the model requires creating a
     new experiment version rather than silently editing the frozen config.
3. For optional explanations only, install Ollama, pull the pinned
   `qwen3:32b` model, and ensure its digest matches the GUI config. Classification
   itself is local and does not require Ollama.

#### Start and use

From the repository root, run:

```powershell
python -m gui.app
```

The app opens at `http://127.0.0.1:8765/` and binds only to the local computer.
Upload a rendered RFC `.txt` file (up to 5 MB), review the positive results,
and optionally search, filter by section, sort, or download the positive-result
CSV. Select up to five results to request Qwen explanations. Press `Ctrl+C` in
the terminal to stop the server. For a different port or no browser launch:

```powershell
python -m gui.app --port 8766 --no-browser
```

The parser only scores clauses it recognizes as explicitly normative from
uppercase BCP 14 keywords; it does not promote contextual or lowercase-modal
clauses. See [`gui/README.md`](gui/README.md) for the UI behavior and
limitations.

### Reproduce the human-reviewed transformer experiment

The GUI uses already-trained checkpoints; launching it does not train models.
To recreate the experiment, use the frozen protocol and complete local inputs
described by
[`experiments/transformer_final/final_annotation_human_reviewed_100_v1/`](experiments/transformer_final/final_annotation_human_reviewed_100_v1/)
and its [`MANUAL_EXECUTION.md`](experiments/transformer_final/final_annotation_human_reviewed_100_v1/MANUAL_EXECUTION.md).
The annotation source, private key, derived manifests, and model weights are
local artifacts and must be supplied from approved project storage. Use the
project Python 3.11 environment, the pinned DeBERTa-v3-base model, and a
CUDA-capable GPU. Training is offline, FP32, sequential, fail-fast, and does
not retry or resume checkpoints.

The experiment uses the current annotation source SHA-256
`8ffa654c2c405dd3eef72901c600f1ef3823e6dbff663020bb824c57ff65bdbf`.
Its nine-family development cohort has 2,143 clauses (138 positives); the
TCP and DNS families are excluded from this experiment. The frozen design is
five outer family folds and three inner folds per outer fitting partition.
It trains two methods in every split: full fine-tuning with gamma-2 focal
loss, and Partial-4 fine-tuning of the final four DeBERTa layers plus task
head. Inputs are target-clause text only, truncated with the frozen
head-tail policy at 256 tokens.

Run the commands in `MANUAL_EXECUTION.md` in order. The essential sequence is:

```powershell
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli preflight
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.environment_preflight
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli status

# Train and validate all 10 outer cells.
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli train-outer --all
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli validate-outer --require-all

# Then train and validate all 30 inner cells.
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli train-inner --all
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli validate-inner --require-all

# Analyze only after both cell sets validate.
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli analyze
python -m experiments.transformer_final.final_annotation_human_reviewed_100_v1.cli validate-analysis
```

Run the training commands only when the corresponding immutable runs do not
already exist. If a project artifact bundle includes completed runs, start
with `status`, then validate the available outer and inner cells; do not
retrain or overwrite them. A fresh reproduction needs a clean experiment
output area and the approved source and model inputs. The commands intentionally
fail rather than resume or overwrite existing cells.

Each training run writes an immutable checkpoint, held-out predictions,
metrics, audit, and completion manifest under the experiment's `runs/`,
`audits/`, and `completion_manifests/` directories. Do not overwrite or resume
completed runs. If frozen protocol or code must change, create a new experiment
version. Analysis outputs include fold and pooled metrics, calibration
thresholds, robustness and uncertainty summaries, and error-analysis exports.

The reported Focal + Partial-4 equal-weight ECDF ensemble achieved nested OOF
AP 0.6736 and ROC-AUC 0.9118. Each outer-fold threshold is selected from
inner-OOF predictions to maximize recall subject to a fitting-side precision
target of 0.30. The pooled OOF precision/recall for this per-fold policy were
0.5063/0.6212. These metrics do **not** evaluate the GUI's runtime 3-of-5
majority-vote policy or its mean ranking score, which have not been measured.
The target is not a production precision guarantee. Labels are operational
references rather than independent expert judgments; treat the study as
development evidence, not a validated detector or a direct leaderboard
comparison with models trained on different cohorts.

#### What remains before research or submission claims

- Measure the runtime 3-of-5 majority-vote policy and mean ECDF ranking score
  on a suitable evaluation set; current nested OOF metrics do not evaluate
  either deployment aggregate.
- Independently evaluate the detector on an approved held-out family or time
  slice with expert-adjudicated labels. Current labels are operational
  references, not independent expert validation.
- Document the cohort, annotation source, calibration, vote rule, and
  limitations in any paper or presentation. The GUI is a research
  demonstration, not a validated production detector.

See [`gui/README.md`](gui/README.md) for additional inference, provenance,
limitation, and privacy details.

### Build and validate corpus metadata

```powershell
python corpus/build_from_manifest.py
python corpus/sync_metadata.py --refresh-manifest --write-audit
python corpus/validate_corpus.py
```

Corpus collection may require network access. Existing committed sources allow many
parsing and test workflows to run offline.

### Build model datasets

```powershell
python -m corpus.build_model_dataset
python -m models.transformer.dataset
python -m models.transformer.seal_dataset
```

Generated datasets are versioned and immutable. Keep private annotations, derived
labels, and local model data out of public commits.

### Run final-annotation LLM experiments

The approved `final_annotation_ready.csv` input is local-only and is not distributed
with this repository. Supply it locally, then run the staged workflow:

```powershell
.\scripts\run_llm_in_context.ps1 -Stage audit
.\scripts\run_llm_in_context.ps1 -Stage smoke
.\scripts\run_llm_in_context.ps1 -Stage all
```

V2, V3, and V4 have separate runners and protocols. See
[`models/llm/README.md`](models/llm/README.md) and
[`experiments/llm/`](experiments/llm/).

### Reproduce the final hybrid comparison

The OOF inputs are local experiment outputs and must exist before running:

```powershell
python -m models.hybrid.final_rank_ensemble
python -m models.hybrid.plot_final_rank_ensemble
```

The first command writes local metrics, predictions, and a provenance manifest under
`results/hybrid/`. The second writes PNG and PDF figures under `paper/figures/`.
Existing output directories and figure files are not overwritten. See
[`models/hybrid/README.md`](models/hybrid/README.md) for input requirements and cohort
limitations.

### Run research utilities

Focused analysis and annotation-preparation helpers are grouped under `scripts/`:

```powershell
python -m scripts.analysis.analyze_votes
python -m scripts.analysis.per_model_targets
python -m scripts.annotation.make_annotation_file --check 5
```

These utilities require the local annotation/features described in their source and
may write private exports under `results/annotation/`. Review
[`corpus/LLM_ANNOTATION.md`](corpus/LLM_ANNOTATION.md) before creating annotation
worksheets.

### Test and lint

```powershell
python -m pytest tests/ -q
python -m ruff check alignment corpus models eval tests
```

## Data and experiment boundaries

- Some annotation exports, private review material, transformer weights, and large
  generated run directories are local-only. The active classical GUI artifact and
  selected result tables are tracked; check each artifact rather than assuming.
- Aggregate metrics, experiment protocols, tests, and publication figures may be
  committed when they do not expose private labels or reserved-set results.
- TCP and DNS were the proposal's intended holdouts, but historical LOFO and the
  current all-family classical study include both. They are excluded only from
  particular nine-family transformer/Qwen studies, not reserved project-wide.
- Model selection, preprocessing, and thresholds must use fitting-side data only.
  Report out-of-fold and development results as exploratory where appropriate.
- The current annotation labels are human/AI-assisted operational references, not
  independent expert gold labels.

See [`.gitignore`](.gitignore) for local-only paths and
[`results/README.md`](results/README.md) before adding generated outputs.

## Reused resources

The project builds on [RFCScope](https://github.com/HIPREL-Group/RFCScope),
[SAGE](https://github.com/USC-NSL/sage), and
[ietfdata](https://github.com/glasgow-ipl/ietfdata). See
[`paper/references.bib`](paper/references.bib) and
[`Related Work/`](Related%20Work/) for citations and background.

## License

See [`LICENSE`](LICENSE).
