# Conclusions

- Best comparable LOFO result is the combined TF-IDF + engineered baseline: AP 0.291 on the 9-family cohort (0.178 prevalence), about 1.63× prevalence. That is weak-to-moderate ranking performance in practice, not strong detection.
- Across revision, frozen-embedding, TF-IDF, and engineered-feature experiments, AP is roughly 0.17–0.29; more precisely the revision-only minimum is 0.1649 and the all-family combined baseline is 0.293. The results do not show a large or robust signal beyond the combined text baseline.
- Frozen embeddings did not improve on TF-IDF in this small-data setting: MiniLM/BGE target-plus-engineered AP is about 0.245–0.246, and adding previous/next clause embeddings yields about 0.204–0.212. This does not establish that context is useless.
- Same-revision previous-clause text changed LOFO AP only from 0.291 to 0.292, while F1@0.5 decreased from 0.365 to 0.325 (notebook-reported).
- Basic revision predecessor signals are at/below prevalence: R1/H1 AP 0.168 and R2/H0 AP 0.165 versus prevalence 0.178.
- Positive labels have low annotator agreement: 435/455 (95.6%) have weight 0.5, corresponding to 2–1 votes; only 20 positives are 3–0. This may plausibly bound attainable performance against the consensus labels, but it is not a measured ceiling.
- Per-family AP varies widely, approximately 0.11 to 0.40; this and the small number of families argue against a strong generalization claim.
- Next steps: recover raw per-LLM votes and calculate kappa; annotate a stratified human gold subset of 50–100 rows while blind to LLM labels; then evaluate an LLM detector or hybrid against that gold subset.

## Other findings and limitations

**Exploratory fixed-split results (not comparable to LOFO; single 154-row test family, 34 positives):** engineered AP 0.290, TF-IDF AP 0.325, combined AP 0.315. An earlier revision-history heuristic reported validation AP 0.332, test AP 0.306, and test F1 0.230 at a validation-selected threshold of 0.53. These values are notebook-reported and are not directly comparable to LOFO.

The basic predecessor matcher does not handle split/merge relationships and could not be validated against row-level TLS alignment links because those records were not retained. Coverage-gate differences above 10 percentage points between labels occur in `draft-ietf-quic-http` and `draft-ietf-tcpm-rfc793bis`. Candidate source was isolated to N1, a selection-artifact diagnostic, not a candidate model.

Vote-weighted training was **not run** because the stop rule triggered. Per-LLM vote columns were not found in processed outputs; unanimous/2–1 subset metrics, kappa, and per-LLM dissent analysis are unavailable. The full-dataset weighted-positive fraction was 14.2557% (about 14.3%) versus 17.9% unweighted; the 9-family weighted fraction was 14.2298%.

Dataset discrepancy to fix: the current dataset has 2,546 rows and only 5 errata rows (all label 0 and all TLS), while the progress report stated 121 errata positives and 242 TLS records.

## Assumptions and provenance

- E0's headline is the 9-family LOFO run (453/2,540 positives, prevalence 0.178), matching the later embedding and revision evaluations. The original notebook's 11-family LOFO scores use prevalence 0.179 (455/2,546); these cohorts are not directly comparable.
- Bootstrap and paired-bootstrap intervals are reproduced only where already present in the result CSVs; no new intervals were calculated.
- Lift values in `final_table.csv` and `final_table.md` are descriptive AP/prevalence ratios, not separately estimated metrics.
- Notebook-only results are identified as notebook-reported in the final table. Fixed-split metrics and the previous-clause LOFO comparison are explicitly treated as exploratory.
- Source details: 9-family LOFO metrics and intervals are in `results/embeddings_context/results.csv`; revision APs, CIs, and top-20% recall are in `results/revision_features/signal_check.csv`; coverage and matcher caveats are in `results/revision_features/summary.md`; weight distribution and the stop decision are in `results/vote_weights/summary.md`; notebook-only figures cite notebook cell IDs in `final_table.md`.
