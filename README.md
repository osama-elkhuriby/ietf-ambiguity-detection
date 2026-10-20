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
| [`results/`](results/) | Compact reports, metrics, and publication figures; local runs are ignored |
| [`tests/`](tests/) | Offline unit, integration, and governance tests |
| [`paper/`](paper/) | Manuscript source, bibliography, and figures |
| [`DECISIONS.md`](DECISIONS.md) | Dated methodological decisions |
| [`MANUAL_GPU_EXECUTION_GUIDE.md`](MANUAL_GPU_EXECUTION_GUIDE.md) | GPU experiment run instructions |

Detailed result conventions and data-handling guidance are in
[`results/README.md`](results/README.md). The paper's scope, build instructions, and
submission checklist are in [`paper/README.md`](paper/README.md).

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
