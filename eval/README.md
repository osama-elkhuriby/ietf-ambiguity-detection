# Stage 5 — Evaluation

Per proposal SS7: **not** random clause splitting as the primary result (adjacent versions
of the same draft are too similar). Also avoid splitting **adjacent revisions of the same draft**
across train and test — that leaks near-duplicate clauses and inflates metrics.

The current TLS-only milestone adds `alignment_metrics.py` for provisional alignment
and workload reporting. Accuracy remains unmeasured without complete reviewed gold
scopes. See [`alignment/version_alignment/README.md`](../alignment/version_alignment/README.md).
This milestone does not implement or claim the generalization protocols below.

Three protocols, all required for the later full benchmark:

1. **document-family split** — all versions of a draft family in one partition
2. **temporal split** — train on earlier drafts, test on later publication periods
3. **protocol-family holdout** — test transfer to an unseen protocol area

Report results for the full test set *and* separately by ambiguity subtype, revision stage,
and protocol family. Metrics: precision/recall/macro-F1/F2/PR-AUC/false-negative rate/per-subtype
F1 (not accuracy — ambiguous clauses are rare), plus calibration, abstention risk-coverage,
latency, cost, and explanation-faithfulness metrics per SS7.

This folder is genuinely shared across the team, not one person's stage — whoever built a
detector needs to agree on what "leakage-resistant" means for their own inputs.
