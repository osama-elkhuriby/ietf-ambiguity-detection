# Annotation pipeline v2

This package runs a separately versioned ambiguity-annotation experiment. It is
deny-by-default: only families frozen in the transformer `folds_v1.json` development
mapping may be opened. TCP, sealed DNS, and every non-allowlisted lineage are outside
scope. V1 artifacts are read only by exact allowed record ID and are never modified.

The semantic labels are `AMBIGUOUS`, `CLEAR`, and `INSUFFICIENT_CONTEXT`.
Operational failures remain null semantic labels. The conservative three-model panel
policy is defined in `aggregation.py`; a separately named v1-compatible view is for
comparison only.

Run in order:

```powershell
python -m models.annotation_v2.cli freeze
python -m models.annotation_v2.cli dry-run
python -m models.annotation_v2.cli review-packet
python -m models.annotation_v2.cli preflight
python -m models.annotation_v2.cli audit-preflight
python -m models.annotation_v2.cli annotate --scope pilot
python -m models.annotation_v2.cli report-pilot
```

The full governed development run is resumable but is never launched automatically:

```powershell
python -m models.annotation_v2.cli annotate --scope full --confirm-full
```

Each attempt is a separate append-only JSON file. A valid cell result references its
accepted attempt. Cache identity covers model digest, rendered prompts, schema,
retrieval manifest, template, Ollama version, adapter/runner versions, and all decoding
and reasoning settings.

For the manually operated standalone Ollama 0.40.4 workflow on port 11435, use
`experiments/annotation_v2/manual_execution_v1/README.md`. That package reuses the
saved Qwen GPU verification, requires a separately saved one-request Gemma check,
keeps GPT-OSS-120B blocked, and permits only frozen pilot inputs. Its live commands
must be run by the operator; preparation does not execute them.
