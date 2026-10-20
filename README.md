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
| Corpus | 253 draft revisions, 12 RFCs, and 31 verified technical errata collected for selected families |
| Parsing and alignment | Plaintext/XML parsing, clause segmentation, and revision-lineage tooling implemented |
| Annotation | AI-assisted operational labels prepared; independent expert adjudication is not available |
| Model development | Classical, transformer, LLM, and hybrid studies completed on several distinct data artifacts |
| Hybrid rank comparison | Three-model V4/DeBERTa/classical comparison completed on a label-compatible subset; exploratory |
| Independent evaluation | Reserved TCP validation and sealed DNS testing have not been opened |

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

The GUI runs locally and classifies clauses in rendered RFC plaintext. It
displays only positive (potentially ambiguous) predictions from the saved
engineered-feature + TF-IDF Logistic Regression model. The 0.5 threshold
produces the binary result; scores are not calibrated probabilities or
confirmed ambiguity judgments.

#### Before the first run

1. Create and activate the project environment using the [Setup](#setup)
   instructions.
2. Restore the model files `final_model.joblib`, `manifest.json`, and
   `metrics.json` into `results/ML/v2_engineered_tfidf/` from an approved local
   backup or model-generation workflow. Model artifacts are not included in
   Git. The artifact was created with scikit-learn 1.9.1; use the matching
   version to avoid the cross-version loading warning.
3. For optional explanations, install Ollama, pull the pinned `qwen3:32b`
   model, and ensure its digest matches the project configuration. The
   classifier itself does not require Ollama; explanations are optional and
   limited to five selected positive results.

#### Start and use

From the repository root, run:

```powershell
python -m gui.app
```

The app opens at `http://127.0.0.1:8765/` and binds only to the local computer.
Upload a rendered RFC `.txt` file (up to 5 MB), review the positive results,
and optionally search, filter by section, sort, or download the positive-result
CSV. Select up to five results to request Qwen explanations. Press `Ctrl+C` in
the terminal to stop the server.

#### What remains before research or submission claims

- Match the environment to the saved artifact's scikit-learn version and verify
  model provenance before relying on predictions.
- Independently evaluate the detector on an approved, held-out dataset with
  expert-adjudicated labels. Current scores are development estimates against
  operational reference labels, not independent expert validation.
- Follow the sealed evaluation protocol for reserved TCP and DNS data; do not
  inspect or use those partitions outside the approved protocol.
- Document the evaluation cohort, label source, threshold, and limitations in
  any paper or presentation. The GUI is a research demonstration, not a
  validated production detector.

See [`gui/README.md`](gui/README.md) for model provenance, limitations, and
privacy details.

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

- Annotation exports, private review material, row-level predictions, model weights,
  and generated run directories are local-only and excluded from version control.
- Aggregate metrics, experiment protocols, tests, and publication figures may be
  committed when they do not expose private labels or reserved-set results.
- TCP is reserved for transfer verification; DNS SVCB/HTTPS is a sealed final test.
  Do not inspect or evaluate either partition outside its approved protocol.
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
