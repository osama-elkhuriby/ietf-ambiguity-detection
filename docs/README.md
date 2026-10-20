# Project documentation

This folder contains operator and workflow guides that do not belong in the
project-wide README or beside a model implementation.

## Guides

| Path | Purpose |
| --- | --- |
| [`gpu/MANUAL_GPU_EXECUTION_GUIDE.md`](gpu/MANUAL_GPU_EXECUTION_GUIDE.md) | Sequential, manually authorized GPU execution for the Recall Improvement Study V3. |

## Where other documentation lives

- [`../README.md`](../README.md) — project overview, setup, primary workflows, and
  repository map.
- [`../DECISIONS.md`](../DECISIONS.md) — dated methodology and protocol decisions.
- [`../corpus/README.md`](../corpus/README.md) and
  [`../corpus/LLM_ANNOTATION.md`](../corpus/LLM_ANNOTATION.md) — corpus and
  annotation workflows.
- [`../models/`](../models/) — model implementation notes and evaluation protocols.
- [`../experiments/`](../experiments/) — experiment plans and frozen configurations.
- [`../results/README.md`](../results/README.md) — result organization, provenance,
  and sharing rules.
- [`../paper/README.md`](../paper/README.md) — manuscript build and submission
  checklist.

Keep focused implementation instructions beside their owning package. Keep broad
project guidance here or in the root README, and do not duplicate experiment results
across documentation pages.
