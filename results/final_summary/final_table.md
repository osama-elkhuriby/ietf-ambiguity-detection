# Final results table

## Main comparison

The table is restricted to family-held-out LOFO results. The original notebook reports all-11-family LOFO results; later contextual-embedding and revision-signal files follow the rule that families with fewer than 30 rows stay in training and are not held out, giving 9 evaluated families. Their evaluation cohorts differ, so compare rows within a cohort rather than treating small differences across 9- and 11-family rows as controlled model effects. E0 is listed for both cohorts to make the protocol distinction explicit.

Lift is pooled OOF AP divided by the corresponding prevalence: 0.178 for the 9-family cohort (453/2,540) and 0.179 for all 11 families (455/2,546). Lifts are ratios calculated from the cited AP and prevalence; no new confidence intervals were computed. `n/a` means the requested statistic was not present in the saved result files or notebook report. Notebook-derived figures are marked as notebook-reported in the source column.

| Experiment | Features | Families evaluated | Pooled OOF AP | 95% bootstrap CI | Lift vs prevalence | Paired AP diff vs E0 with 95% CI | F1 at 0.5 | Recall at top-20% reviewed | Takeaway | Source (file and row/cell) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| E0 combined (headline; small families train-only) | Target TF-IDF + engineered | 9 | 0.291 | 0.260–0.335 | 1.63 | 0.000 [0.000, 0.000] | 0.365 | 0.351 | Best matched 9-family reference; about 1.6× prevalence | `results/embeddings_context/results.csv`, `E0_combined_reference`; notebook cell `319865a7` |
| E0 combined (original all-family LOFO) | Target TF-IDF + engineered | 11 | 0.293 | n/a | 1.64 | n/a | 0.367 | n/a | Historical all-11-fold baseline; differs from the 9-family held-out protocol | Notebook-reported, cell `d0f0d147` |
| Engineered only | Engineered numerical | 11 | 0.275 | n/a | 1.54 | n/a | 0.362 | n/a | Below combined baseline | Notebook-reported, cell `d0f0d147` |
| TF-IDF only | Target TF-IDF | 11 | 0.273 | n/a | 1.53 | n/a | 0.325 | n/a | Below combined baseline | Notebook-reported, cell `d0f0d147` |
| LinearSVC + TF-IDF | Target TF-IDF | 11 | 0.254 | n/a | 1.42 | n/a | n/a (not F1@0.5) | n/a | Notebook F1 used SVC margin threshold 0; no F1@0.5 reported | Notebook-reported, cell `4468cd5c` |
| RandomForest + engineered | Engineered numerical | 11 | 0.240 | n/a | 1.34 | n/a | 0.151 | n/a | Low recall at the default threshold | Notebook-reported, cell `4468cd5c` |
| MiniLM E1 | Target embedding | 9 | 0.219 | 0.195–0.250 | 1.23 | −0.072 [−0.108, −0.038] | 0.304 | 0.258 | Below E0 | `results/embeddings_context/results.csv`, `MiniLM_E1_target_only` |
| MiniLM E2 | Target embedding + engineered | 9 | 0.246 | 0.219–0.280 | 1.38 | −0.045 [−0.074, −0.016] | 0.324 | 0.300 | Best fixed embedding configuration; below E0 | `results/embeddings_context/results.csv`, `MiniLM_E2_target_plus_engineered` |
| MiniLM E3 | Target + previous + next embeddings | 9 | 0.204 | 0.183–0.230 | 1.14 | −0.087 [−0.123, −0.055] | 0.263 | 0.241 | Context embeddings did not help in this setup | `results/embeddings_context/results.csv`, `MiniLM_E3_target_prev_next` |
| MiniLM E4 | Target + previous + next embeddings + engineered | 9 | 0.212 | 0.190–0.240 | 1.19 | −0.079 [−0.111, −0.048] | 0.278 | 0.252 | Below E0 | `results/embeddings_context/results.csv`, `MiniLM_E4_context_plus_engineered` |
| MiniLM E5 | Target + context embeddings + engineered + target TF-IDF | 9 | 0.212 | 0.190–0.240 | 1.19 | −0.079 [−0.113, −0.050] | 0.276 | 0.258 | Full stack remained below E0 | `results/embeddings_context/results.csv`, `MiniLM_E5_full_stack` |
| BGE-small E1 | Target embedding | 9 | 0.224 | 0.199–0.256 | 1.26 | −0.067 [−0.103, −0.033] | 0.287 | 0.269 | Below E0 | `results/embeddings_context/results.csv`, `BGE-small_E1_target_only` |
| BGE-small E2 | Target embedding + engineered | 9 | 0.245 | 0.218–0.281 | 1.38 | −0.046 [−0.082, −0.015] | 0.321 | 0.296 | Best BGE fixed configuration; below E0 | `results/embeddings_context/results.csv`, `BGE-small_E2_target_plus_engineered` |
| BGE-small E3 | Target + previous + next embeddings | 9 | 0.204 | 0.183–0.229 | 1.14 | −0.087 [−0.123, −0.056] | 0.255 | 0.267 | Context embeddings did not help in this setup | `results/embeddings_context/results.csv`, `BGE-small_E3_target_prev_next` |
| BGE-small E4 | Target + previous + next embeddings + engineered | 9 | 0.210 | 0.186–0.238 | 1.18 | −0.081 [−0.115, −0.049] | 0.257 | 0.258 | Below E0 | `results/embeddings_context/results.csv`, `BGE-small_E4_context_plus_engineered` |
| BGE-small E5 | Target + context embeddings + engineered + target TF-IDF | 9 | 0.211 | 0.189–0.239 | 1.18 | −0.080 [−0.114, −0.051] | 0.255 | 0.258 | Full stack remained below E0 | `results/embeddings_context/results.csv`, `BGE-small_E5_full_stack` |
| Nested inner-CV selected embedding | Per-family selected embedding configuration + C | 9 | 0.232 | 0.207–0.265 | 1.30 | −0.058 [−0.089, −0.030] | 0.313 | 0.298 | Nested selection was leakage-safe but remained below E0 | `results/embeddings_context/results.csv`, `Nested_inner_CV_selected_embedding` |
| Same-revision previous-clause context | Previous clause text + target TF-IDF + engineered | 9 | 0.292 | n/a | 1.64 | approximately +0.001; paired CI n/a | 0.325 | n/a | AP rose slightly but F1 at 0.5 fell from 0.365 to 0.325 | Notebook-reported, cell `319865a7` |
| R1 predecessor (H1) | `has_predecessor` + relation + similarity | 9 | 0.168 | 0.152–0.190 | 0.94 | n/a | n/a | 0.203 | At prevalence; basic predecessor/similarity features only | `results/revision_features/signal_check.csv`, `R1_H1_predecessor_only` |
| R2 predecessor (H0) | `has_predecessor` + relation + similarity | 9 | 0.165 | 0.149–0.184 | 0.92 | n/a | n/a | 0.190 | At/below prevalence; basic predecessor/similarity features only | `results/revision_features/signal_check.csv`, `R2_H0_predecessor_only` |
| N1 selection-artifact diagnostic (not a candidate model) | Candidate source only; isolated diagnostic | 9 | 0.179 | 0.160–0.199 | 1.00 | n/a | n/a | 0.185 | Diagnostic only; not a detector and not combined with R1/R2 | `results/revision_features/signal_check.csv`, `N1_selection_artifact_diagnostic` |

The embedding CIs and paired AP-difference CIs are the existing row-bootstrap intervals in `results.csv`; the revision CIs are the existing pooled AP intervals in `signal_check.csv`. No confidence intervals were available for the older notebook-only model comparisons or the previous-clause-context row.

**Mandatory interpretation:** Labels are LLM-consensus labels on a candidate cohort selected using future information; AP and prevalence do not transfer to the full clause population.

## Exploratory fixed-split results (not comparable to LOFO; single 154-row test family, 34 positives)

These are the existing fixed test-split results, not LOFO results. Thresholds shown for test F1 were selected on validation and are included only as exploratory notebook results.

| Representation | Test AP | Test F1 (not necessarily at 0.5) | Provenance |
|---|---:|---:|---|
| Engineered | 0.290 | 0.347 | Notebook-reported, cell `ce50dcd5`; test family/count in same cell output |
| TF-IDF | 0.325 | 0.412 | Notebook-reported, cell `ce50dcd5` |
| Combined | 0.315 | 0.403 | Notebook-reported, cell `ce50dcd5` |
| Earlier revision-history heuristic | Validation AP 0.332; test AP 0.306 | Test F1 0.230 at validation-selected threshold 0.53 | Notebook-reported, cell `revision-result` |

## Interpretation and caveats

- The previous-clause context result is specifically AP 0.291 → 0.292 and LOFO F1@0.5 0.365 → 0.325 (notebook cell `319865a7`).
- The predecessor experiment is a quick/basic matcher with no split/merge handling. TLS matcher validation against row-level alignment records was unavailable, because the retained artifacts did not include row-level alignment links (`results/revision_features/summary.md`, “TLS matcher validation”).
- Coverage differs by more than 10 percentage points between labels in `draft-ietf-quic-http` and `draft-ietf-tcpm-rfc793bis` under both H1 and H0; no candidate-source-level disparities exceeded the gate (`results/revision_features/summary.md`, “Coverage gate”).
- The vote-weight experiment was **not run** because the stop rule triggered. The source audit reports 435/455 positive rows at weight 0.5 (95.6%; 2–1 majority), only 20 positive rows at weight 1.0 (3–0), and 1,325/2,091 negative rows at weight 0.5 (63.4%). The weighted positive fraction is 14.2557% versus 17.9% unweighted (`results/vote_weights/summary.md`, “Summary” and “Distribution by label”).
- Per-LLM vote columns were not found in processed outputs; unanimous/2–1 subgroup metrics, kappa, and dissenting-LLM analysis are unavailable (`results/embeddings_context/summary.md`, “Protocol and assumptions”).
- Dataset discrepancy to reconcile: the current dataset has 5 errata rows, all label 0 and all TLS, in 2,546 total rows; the progress report stated 121 errata positives and 242 TLS records (`results/revision_features/summary.md`, discrepancy paragraph).

## Assumptions and provenance conventions

1. The primary E0 is the 9-family result because it matches the small-families-train-only protocol used by the embedding and revision result files; the original 11-family notebook result is retained as a separate historical row.
2. Prevalence is 453/2,540 = 0.178 for 9 evaluated families and 455/2,546 = 0.179 for the all-11-family cohort. The older model rows use the 11-family baseline; later embedding, context, and revision rows use the 9-family prevalence.
3. Lift is a descriptive ratio AP/prevalence, rounded to two decimals; no confidence intervals were computed for lift.
4. `n/a` means no value was available in the listed artifacts. In particular, the SVC notebook F1 uses margin threshold 0 rather than probability threshold 0.5.
5. The context AP/F1, fixed-split AP/F1, earlier revision-history values, and original notebook LOFO figures are notebook-reported, not recomputed from the saved metric CSVs.
6. The LOFO 9-family and 11-family runs have different held-out families; their results are not treated as a paired comparison.
7. Revision features shown here are only the existing quick-check features (predecessor availability, relation, similarity); no later/full revision feature experiment is represented.
