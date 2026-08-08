# Pre-Publication Ambiguity Detection in Internet-Drafts

Revision-aware, evidence-grounded ambiguity detection for IETF Internet-Drafts.

**Queen's University School of Computing**
Supervised by **Prof. Yuanzhu Chen**
Co-advised by **Dr. Ismail Anwar**

This repository is the research codebase for the five-stage pipeline described in the project proposal:

**Corpus → Alignment → Candidate Models → LLM Verification → Evaluation**

The project is currently in the **pilot phase**. The initial goal is to validate the data collection, revision alignment, annotation, and ambiguity-detection assumptions on a small DNS-family dataset before scaling to the full corpus.

---

## Structure

```text
corpus/                 Stage 1 — IETF Internet-Draft, RFC, and errata
                         data collection and Datatracker access

alignment/              Stage 2 — clause segmentation, version matching,
                         and revision-aware features

models/
  heuristic/            Rule-based ambiguity detectors:
                         normative-keyword conflicts, vague quantifiers,
                         undefined terms, and related patterns

  classical/            Classical ML models:
                         logistic regression, SVM, random forest,
                         gradient-boosted trees

  transformer/          Fine-tuned transformer classifiers:
                         DeBERTa, RoBERTa, and related models

  llm/                  Zero/few-shot LLM detection with evidence spans
                         and structured rationales

  hybrid/               Calibrated routing that sends uncertain clauses
                         to the LLM verifier

eval/                   Stage 5 — evaluation protocols, metrics,
                         and leakage-resistant dataset splits

docs/                   Project-wide documentation and methodology

experiments/
  pilot/                Current pilot experiment
    docs/               Pilot-specific documentation
    data/               Pilot raw, processed, and aligned data
    labels/             Pilot annotation files
    notebooks/          Pilot exploratory analysis
    results/             Pilot outputs and experiment results

notebooks/              General exploratory notebooks.
                         Final results should not depend on notebooks.

DECISIONS.md            Project-wide decision log
```

The `corpus/`, `alignment/`, `models/`, and `eval/` directories contain the core project infrastructure. The `experiments/` directory contains experiment-specific data, configurations, annotations, notebooks, and results.

---

## Reproducible Environment

The project uses **Conda** to provide a reproducible development environment across Linux and Windows.

Two OS-specific environment files are maintained:

```text
environment-linux.yml
environment-windows.yml
```

These files are the source of truth for the project's dependencies.

### Linux

```bash
conda env create -f environment-linux.yml
conda activate ietf-ambiguity-detection
```

### Windows

```powershell
conda env create -f environment-windows.yml
conda activate ietf-ambiguity-detection
```

After creating the environment, verify the installation:

```bash
python --version
python -c "import ietfdata; print('ietfdata OK')"
```

> **Note:** The Linux and Windows environment files may contain platform-specific dependencies where required. Do not manually install or change project dependencies without updating the corresponding environment file.

If a dependency is changed, update the environment using the appropriate file:

```bash
conda env update -f environment-linux.yml --prune
```

or on Windows:

```powershell
conda env update -f environment-windows.yml --prune
```

---

## Quickstart

### 1. Create the environment

Use the environment file corresponding to your operating system.

**Linux:**

```bash
conda env create -f environment-linux.yml
conda activate ietf-ambiguity-detection
```

**Windows:**

```powershell
conda env create -f environment-windows.yml
conda activate ietf-ambiguity-detection
```

### 2. Fetch a draft's revision history

The pilot currently uses a small DNS-family example.

For example:

```bash
python corpus/fetch_datatracker.py draft-ietf-dnsop-svcb-https
```

The script requires normal Internet access to the IETF Datatracker and will not work from a sandboxed or network-restricted environment.

Pilot data should be stored under:

```text
corpus/raw/<draft-name>/
```

Optionally mirror fetches under `experiments/pilot/data/raw/` using `--output-dir` (see `experiments/pilot/README.md`).

The fetched data includes the submitted revisions and a `versions.json` manifest containing revision metadata such as submission dates, revision numbers, and word counts.

Replace the draft name with the DNS-family draft selected by the team for the pilot.

---

## Where to Start

The immediate objective is **not** to build the complete five-stage pipeline. The first phase is a small end-to-end pilot designed to test the project's highest-risk assumptions.

1. Read `DECISIONS.md` to understand the current pilot scope, tooling decisions, and reuse strategy.
2. Create the Conda environment using the appropriate OS-specific environment file.
3. Run the Datatracker fetcher against one or two small DNS-family drafts.
4. Inspect the downloaded revision history and raw text before implementing alignment logic.
5. Hand-align a small set of clauses across revisions.
6. Hand-label approximately 15–20 clauses using:
   `experiments/pilot/docs/annotation_guide_v0.md`
7. Store the pilot annotations in:
   `experiments/pilot/labels/`
8. Run the initial heuristic detector and a single LLM-based detector on the pilot data.
9. Record annotation difficulties, data-quality issues, and initial findings in:
   `experiments/pilot/docs/pilot_memo_template.md`
10. Review the pilot findings with the advisors before committing to the full Phase 2 corpus build.

For the detailed pilot workflow, see:

```text
experiments/pilot/README.md
```

---

## Current Status

The repository is currently focused on **Phase 1: pilot validation**.

The pilot is intended to answer several practical questions before scaling:

* Can the Datatracker revision history be collected reliably?
* Can clauses be aligned meaningfully across revisions?
* Does the annotation scheme produce consistent labels?
* How much verified ambiguity data can realistically be obtained?
* Can simple heuristic and LLM detectors identify useful candidate signals?
* What changes between revisions are unrelated to ambiguity?

The answers will be used to refine the methodology before the full corpus and modeling stages are implemented.

---

## Reuse of Existing Work

This project builds on public prior art rather than re-deriving existing capabilities from scratch.

* **RFCScope** (Pawagi et al., ASE 2025) — public dataset and code:
  https://github.com/HIPREL-Group/RFCScope

* **SAGE** (Yen et al., SIGCOMM 2021) — public code:
  https://github.com/USC-NSL/sage

* **ietfdata** (Glasgow IPL) — Python library for accessing the IETF Datatracker and RFC index:
  https://github.com/glasgow-ipl/ietfdata

The project will explicitly distinguish between **reused resources, adapted methods, and novel contributions**.

See `DECISIONS.md` for the current reuse plan and the rationale behind methodological and tooling decisions.

---

## Research Reproducibility

To keep the research traceable throughout development:

* Environment specifications are version-controlled.
* Major methodological decisions are recorded in `DECISIONS.md`.
* Pilot data, annotations, and results are kept separate from the core pipeline.
* Annotation guidelines are maintained as living documents.
* Raw and processed data are kept separate.
* Experiments should record their configuration and results.
* Final results should be reproducible from committed code, environment specifications, and documented data-processing steps.

The goal is to make it possible to determine not only **what result was obtained**, but also **which data, code, environment, and methodological decision produced it**.
