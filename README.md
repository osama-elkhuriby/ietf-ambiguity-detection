# Pre-Publication Ambiguity Detection in Internet-Drafts

Revision-aware, evidence-grounded ambiguity detection for IETF Internet-Drafts.

**Queen's University School of Computing**

Supervised by **Prof. Yuanzhu Chen** and co-advised by **Dr. Ismail Anwar**.

This repository implements a leakage-controlled research pipeline for identifying
potentially ambiguous normative clauses before an Internet-Draft becomes an RFC:

**Corpus → parsing and clause segmentation → revision alignment → candidate labels →
classical/transformer models → calibrated routing → evaluation**

## Current status

The earlier classical and transformer studies remain available for provenance. The
current LLM work includes a completed Qwen3 32B zero/4/8-shot baseline (V1), a completed
strict-prompt 8/16/24-shot experiment (V2), and a nested lexical/Qwen zero-shot hybrid
(V3), and a frozen-MiniLM semantic ensemble (V4). V1 and V2 do not fit parameters. V3
and V4 do not fine-tune Qwen; they fit lightweight classifiers and must be described as
supervised hybrid modeling. All four use `final_annotation_ready.csv` as their label
artifact.

| Component | Current state |
| --- | --- |
| Corpus | Complete for 12 selected RFC families: 253 draft revisions, 12 RFCs, and 31 verified technical errata |
| Parsing and clause segmentation | Implemented for plaintext and XML2RFC sources |
| Revision alignment | Implemented with immutable lineage artifacts, lexical/semantic matching, and review queues |
| Candidate review sets | Generated for 11 draft families with revision histories |
| LLM annotation | Complete for three local models with resumable, auditable responses |
| Model-ready labels | `final_annotation_ready.csv`: 2,424 resolved binary labels plus one explicitly excluded unresolved placeholder |
| Classical baseline | Leakage-controlled TF-IDF logistic regression complete |
| Transformer | Earlier nested-CV experiments complete; results belong to their pinned label versions |
| Qwen3 LLM V1 | Full grouped development evaluation complete; retained as immutable comparison baseline |
| Qwen3 LLM V2 | Full strict-prompt 8/16/24-shot grouped development comparison complete |
| Hybrid V3 | Full nested grouped development experiment complete; Qwen remains frozen |
| Semantic V4 | Full grouped development experiment complete; Qwen and MiniLM remain frozen |
| Final evaluation | Not started; TCP validation and the sealed DNS test remain untouched by final evaluation |
| Independent human gold evaluation | Not available |

The current labels combine human annotation with assistance from two AI models. They are
the project's operational reference labels, but they are not an independently
human-verified external gold standard. Reported results therefore measure agreement
with this annotation process and must not be presented as universal ambiguity-detection
accuracy.

## Dataset snapshot

The new LLM experiment derives the following immutable partitions from the local
`results/annotation/final_annotation_ready.csv` artifact:

| Partition | Families | Rows | Positive (`yes`) | Purpose |
| --- | ---: | ---: | ---: | --- |
| Development | 9 | 2,142 | 126 | Grouped zero/few-shot evaluation and shot-count selection |
| Reserved validation | TCP | 131 | 5 | Later independent transfer verification only |
| Sealed test | DNS SVCB/HTTPS | 151 | 17 | One final evaluation after all choices are frozen |

Annotation exports and row-level labels are intentionally excluded from GitHub; this
path is a local prerequisite and is not present in a fresh clone. The source contains
2,425 rows. `item_id=2161` has an unresolved status and a placeholder
`No` value, so it is explicitly excluded instead of being treated as ground truth. The
remaining 2,424 labels contain 148 positives. Document-family isolation leaves 126
positives in development. Dataset and configuration manifests pin the source,
partition, fold, prompt, and local-model hashes.

The reviewed v2 split replaced HTTP/2bis as the test family after a leakage audit found
cross-family overlap. The current split has zero reported exact cross-split duplicates,
near duplicates, or target-as-context overlaps. The development and sealed-test
artifacts are structurally separated.

## Latest modeling results

The primary metric is PR-AUC because positive ambiguity labels are uncommon.

### Final-annotation LLM development experiments

V3 materially improves the grouped-development ranking and classification metrics. Its
F1-specific threshold is learned only from each outer fold's inner predictions.

| Experiment | PR-AUC | Precision | Recall | F1 | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| V1 best raw (8-shot) | 0.1960 | 0.0981 | 0.7778 | 0.1742 | 0.6654 |
| V2 best raw (24-shot) | 0.1035 | 0.1111 | 0.0635 | 0.0808 | 0.5159 |
| V3 nested hybrid, raw threshold | **0.2554** | 0.1737 | 0.5556 | 0.2647 | 0.6952 |
| V3 nested hybrid, inner-selected F1 threshold | **0.2554** | **0.2366** | 0.4206 | **0.3029** | 0.6679 |
| V4 semantic ensemble, raw threshold | **0.3252** | 0.2366 | 0.5238 | 0.3259 | 0.7091 |
| V4 semantic ensemble, cross-fitted F1 threshold | **0.3252** | **0.3952** | 0.3889 | **0.3920** | 0.6758 |

These are adaptive development results, not sealed-test evidence. V3 and V4 were
designed after earlier development results were inspected, the labels are not an
independent human gold standard, TCP has not been opened for these experiments, and DNS
remains sealed.

The values below are from the earlier label artifact and are retained for provenance;
they are not results for the new final-annotation LLM dataset and must not be compared
as though they used the same labels and split protocol.

### Transformer Step 3

The frozen experiment uses target-only `microsoft/deberta-v3-small`, learning rate
`1e-5`, fp32, five grouped outer folds, nested family-level checkpoint selection, and
three seeds.

| Weighting | Mean-fold PR-AUC | Pooled three-seed PR-AUC |
| --- | ---: | ---: |
| Unweighted | 0.2955 | 0.2663 |
| Class | **0.3022** | 0.2874 |
| Confidence | 0.2907 | 0.2708 |
| Combined | 0.2949 | 0.3016 |

The frozen selection rule chose **class weighting**. However, its section-cluster
confidence interval against the matching TF-IDF comparator includes zero, so the
hypothesis that the transformer outperforms the classical surface-text baseline remains
unsupported.

At 90% development recall, the selected DeBERTa configuration refers approximately
85.8% of clauses to an LLM, while the selected confidence-weighted TF-IDF configuration
refers approximately 73.3%. These are uncalibrated development estimates, not deployment
guarantees.

See:

- `results/transformer/reports/STEP3_REPORT.md`
- `experiments/transformer/EXPERIMENT_PLAN.md`
- `models/transformer/README.md`

## Repository structure

```text
corpus/                  Corpus acquisition, metadata, label construction, and datasets
alignment/
  segmentation/         Format-independent document parsing
  clause_segmentation/  Clause refinement and normative classification
  version_alignment/    Consecutive-revision matching and lineage construction
models/
  classical/            Engineered features and TF-IDF logistic-regression comparator
  transformer/          DeBERTa training, audits, analyses, sealing, and final evaluator
  llm/                  Ollama Qwen3 zero/few-shot evaluation
  heuristic/            Rule-based detector documentation
  hybrid/               Planned calibrated routing layer
eval/                    Alignment and later end-to-end evaluation protocols
experiments/
  classical/            Frozen classical-model configuration
  llm/                  LLM experiment plans and configurations
  transformer/          Experiment plans and configurations
  transformer_final/    Reproduction code, protocols, and run specifications
results/
  ML/                   Classical-model metrics, plots, and study outputs
  LLM/                  LLM experiment outputs (local-only)
  transformer/          Transformer results, reports, and run outputs
  heuristic/, ...       Other model and project result summaries
tests/                   Offline unit, integration, leakage, and governance tests
Related Work/            Project literature
Proposal/                Project proposal and literature review
DECISIONS.md             Dated methodological decision log
```

Generated outputs are organized under `results/`, not mixed into experiment source
directories. Large checkpoints, row-level predictions, private labels, and derived
corpus artifacts remain local and are ignored by Git; see [results/README.md](results/README.md).

## Environment

The source-of-truth environments target Python 3.11:

- `environment-linux.yml`
- `environment-windows.yml`

Create the appropriate Conda environment:

```powershell
conda env create -f environment-windows.yml
conda activate ietf-ambiguity-detection
python scripts/check_environment.py
```

For Linux, use `environment-linux.yml` in the same commands. The environment checker is
offline; it verifies imports, the spaCy model, command-line tools, and the available
PyTorch device.

## Main workflows

Run commands from the repository root.

### Build and validate the corpus

```powershell
python corpus/build_from_manifest.py
python corpus/sync_metadata.py --refresh-manifest --write-audit
python corpus/validate_corpus.py
```

Network access is required for collection from the IETF Datatracker and RFC Editor.
Existing committed sources allow parsing, alignment, and tests to run offline.

### Generate LLM-consensus labels

The local Ollama annotation pipeline is resumable and records each model response:

```powershell
python corpus/annotate_with_ollama.py
```

The configured models are Gemma 3 27B, GPT-OSS 20B, and Qwen3 32B. Invalid or missing
responses do not vote. Unanimous valid votes receive `HIGH` confidence, majority votes
receive `MEDIUM`, and outcomes without a valid majority remain audit-only `UNCERTAIN`.

### Build model artifacts

```powershell
python -m corpus.build_model_dataset
python -m models.classical.feature_pipeline `
  --input corpus/processed/model_datasets/v2/training_dataset.csv `
  --output-dir corpus/processed/features/v3 `
  --syntax disabled
python -m models.transformer.dataset
python -m models.transformer.seal_dataset
```

Generated directories are immutable. Use a new versioned output path instead of
overwriting an existing artifact.

### Prepare and evaluate the final-annotation LLM

The prepared dataset is hash-pinned to the local-only
`results/annotation/final_annotation_ready.csv` artifact. Supply this approved dataset
locally before running the workflow; annotation data is not distributed in this
repository. The LLM workflow contains no
fine-tuning. First audit the data, grouped folds, prompts, and locally installed model;
then run the six-request smoke test before starting the resumable full evaluation:

```powershell
.\scripts\run_llm_in_context.ps1 -Stage audit
.\scripts\run_llm_in_context.ps1 -Stage smoke
.\scripts\run_llm_in_context.ps1 -Stage all
```

On Windows, the runner automatically selects a runnable project Conda interpreter if
Application Control blocks `.venv\Scripts\python.exe`. Use `-PythonPath` only when an
explicit interpreter override is needed.

It reports grouped-fold PR-AUC, precision, recall, F1, F2, accuracy, balanced accuracy,
per-row scores, label-source slices, precision-recall tables, SVG metric comparisons,
and confusion matrices. Demonstrations come only from each fold's fitting rows.
Continuous scores are calculated from Ollama's native `yes`/`no` token
log-probabilities; no self-reported confidence or fitted threshold is used. Full results
are saved under
`results/LLM/qwen3-32b-final-annotation-ready-v1/analysis-final/`.

Qwen3 32B also contributed votes during the annotation-support workflow. Its metrics
therefore measure non-independent agreement with the operational reference labels and
must not be presented as performance on an external human gold standard.

The full run requires 6,426 local Qwen3 32B requests. It writes each valid response
immediately and resumes from matching cache identities after interruption. See
`experiments/llm/EXPERIMENT_PLAN.md` for the complete protocol and claim boundary.

The separate V2 comparison uses a stricter prompt, prevalence-aware examples,
fold-safe V1 hard negatives, and cross-fitted threshold estimates:

```powershell
.\scripts\run_llm_in_context_v2.ps1 -Stage audit
.\scripts\run_llm_in_context_v2.ps1 -Stage smoke
.\scripts\run_llm_in_context_v2.ps1 -Stage all
```

V2 results are written under
`results/LLM/qwen3-32b-final-annotation-ready-v2/`. Its final analysis
creates `v1_v2_comparison.csv`, threshold metrics, comparison visualizations, and raw
and thresholded confusion matrices. See `experiments/llm/EXPERIMENT_PLAN_V2.md`.

V3 changes the method instead of modifying the prompt again. It combines nested,
target-only word/character TF-IDF logistic regression with the frozen, label-independent
Qwen zero-shot score. It does not call Ollama or fine-tune Qwen:

```powershell
.\scripts\run_llm_in_context_v3.ps1 -Stage all
```

Results and V1/V2/V3 visual comparisons are written under
`results/LLM/qwen3-32b-final-annotation-ready-v3/analysis-final/`. See
`experiments/llm/EXPERIMENT_PLAN_V3.md` for the nested protocol and claim boundary.

V4 adds frozen MiniLM target and local-context representations to V3. MiniLM and Qwen
remain frozen; only class-balanced logistic classifiers are fitted:

```powershell
.\scripts\run_llm_in_context_v4.ps1 -Stage all
```

Results and V1-V4 comparisons are written under
`results/LLM/qwen3-32b-final-annotation-ready-v4/analysis-final/`. See
`experiments/llm/EXPERIMENT_PLAN_V4.md`. The V4 configuration was selected adaptively
on development data, so TCP transfer verification is required before claiming that the
improvement generalizes.

### Run tests and lint

```powershell
python -m pytest tests/ -q
python -m ruff check alignment corpus models eval tests
```

On a Windows machine whose global temporary directory is inaccessible, use a new
workspace-local pytest directory:

```powershell
python -m pytest tests/ -q -p no:cacheprovider --basetemp .test-tmp/local-run
```

## Experiment governance

The repository deliberately separates model development from final evaluation:

- Training and cross-validation use nine families only.
- TCP cannot select the architecture, learning rate, weighting, checkpoint, temperature,
  or routing thresholds; it is reserved for independent transfer verification.
- DNS SVCB/HTTPS is stored as a sealed final-test artifact.
- `models/transformer/final_evaluate.py` requires an explicit `--final` flag and claims
  an exclusive lock before loading the sealed rows.
- A second final-test run is refused.
- Model inputs exclude labels, tiers, weights, subtypes, family IDs, split names, source
  hashes, LLM votes, and rationales.
- All run manifests record input/configuration hashes and raw logits.

Do not run the final DNS evaluator until the selected input variant, global epoch policy,
ensemble strategy, calibration temperature, routing thresholds, code hashes, and final
command have all been frozen.

## Next steps

The remaining planned stages are:

1. Review V4's fold-level instability and error slices before treating it as the
   development winner.
2. Freeze a single full-development V4 fit and a separate TCP transfer-evaluation
   protocol before opening TCP.
3. Use TCP once for transfer verification without changing the method from its TCP
   results.
4. Freeze the complete final configuration and evaluate DNS exactly once.
5. If resources become available, build an independently annotated human evaluation
   set and report agreement before adjudication.
6. Run protocol-cluster and temporal generalization sensitivity analyses.

## Reused resources

The project distinguishes reused resources, adapted methods, and novel contributions.
Important public foundations include:

- [RFCScope](https://github.com/HIPREL-Group/RFCScope) for errata-derived ambiguity work
  and the seven-subtype vocabulary.
- [SAGE](https://github.com/USC-NSL/sage) for prior specification-analysis work.
- [ietfdata](https://github.com/glasgow-ipl/ietfdata) for Datatracker and RFC-index access.

Major methodological decisions and their rationale are recorded in `DECISIONS.md`.

## Reproducibility principles

- Keep raw sources, derived artifacts, model outputs, and human adjudication separate.
- Preserve uncertain and invalid annotation outcomes for audit, but never treat them as
  binary training labels.
- Fit preprocessing and loss weights only on the relevant training partition.
- Split by document family, not by randomly sampled clauses.
- Record hashes, model digests, seeds, token log-probabilities, exclusions, and runtime
  metadata.
- Prefer immutable versioned outputs over in-place modification.
- Report negative results and protocol deviations explicitly.

## License

See `LICENSE`.
