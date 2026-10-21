# Proposal completion matrix

Status vocabulary: **CONFIRMED** means a completed artifact supports the stated limited claim; **NOT EVALUATED** means the proposed outcome was not measured; **UNRESOLVED** means repository evidence cannot determine completion.

| Proposal element | Status | Delivered evidence | Defensible conclusion / gap |
|---|---|---|---|
| Gold annotation | NOT EVALUATED | AI-assisted consensus labels, 270 `manual_conflict_verdict` rows, and 100 `human_review_user_confirmed` rows; `results/annotation/final_annotation_ready.csv` | The operational reference is usable for development, but there is no documented blinded, independent double annotation, adjudicator identity, or inter-human agreement. Do not call it expert gold. |
| Draft-stage / temporal evaluation | PARTLY CONFIRMED | Past-only feature guard in `models/classical/revision_features.py`; family-grouped OOF studies | Feature availability is chronologically guarded. There is no true chronological train-on-past/test-on-future experiment, so temporal transfer is not measured. |
| Family evaluation | CONFIRMED | Five outer draft-lineage folds in `experiments/transformer_final/final_annotation_human_reviewed_100_v1/`; three grouped classical folds | These test transfer across held-out draft lineages represented in the study. They do not test an unseen protocol suite, and two outer folds pool several small lineages. |
| Independent TCP/DNS evaluation | NOT EVALUATED AS PROPOSED | Historical LOFO and current all-family classical outputs contain both lineages; transformer/Qwen tracks exclude them | Reservation was experiment-specific, not project-wide. No untouched TCP/DNS test remains. A new family or later time slice is needed. |
| Revision-feature improvement | CONFIRMED NEGATIVE/MIXED | Matched 2,425-row classical ablation in `results/ML/figures/experiment_metrics.csv` | AP decreases for logistic/SVM/RF by 0.0081/0.0012/0.0015; F2 changes are 0/+0.0110/+0.0227. The general improvement hypothesis is unsupported. |
| Cross-reference improvement | NOT EVALUATED | Cross-reference counts are part of engineered features | There is no isolated cross-reference ablation. |
| Detector comparison | PARTLY CONFIRMED | Heuristic/classical matched on 2,425 rows; transformer components matched on 2,143 rows; Qwen/hybrids matched on 2,142 rows | Within-study comparisons are valid. The project-wide table is a study map, not a common leaderboard. |
| Fine-tuned transformer comparison | CONFIRMED AS DEVELOPMENT EVIDENCE | Reviewed-label focal/partial/ECDF ensemble: AP 0.5823/0.5761/0.6736 | Same rows/folds and complete OOF coverage support a within-cohort descriptive comparison. One seed and selective review prevent confirmatory superiority claims. |
| Routed verifier vs score-fusion hybrid | NOT EVALUATED | V3/V4 and final ensembles fuse scores already computed for all rows | These are score-fusion models, not conditional LLM routers. |
| LLM-call reduction | NOT EVALUATED | Referral estimates exist, but Qwen scores were already generated for the full cohort | No avoided-call count or causal effectiveness/cost comparison exists. |
| Latency, energy, token, and monetary cost | NOT EVALUATED | Some run-time telemetry exists for individual pipelines | No standardized same-hardware, same-cohort end-to-end benchmark was completed. |
| Calibration | NOT EVALUATED | Inner-fold threshold selection and ECDF ranking are implemented | Threshold selection is not probability calibration. No Brier score, ECE, reliability diagram, or calibrated probabilities were reported. |
| Explanation faithfulness/usefulness | NOT EVALUATED | Optional Qwen explanation path in `gui/app.py` | No blinded human ratings, source-faithfulness check, hallucinated-reference audit, or usefulness/clarity measure exists. |
| Alignment coverage | CONFIRMED | TLS pilot: 30 documents, 29 pairs, 52,953 relationships; deterministic validation | This establishes artifact coverage and operational acceptance only. |
| Alignment accuracy | NOT EVALUATED | 150-record alignment sample is unadjudicated; accuracy fields are null | Precision, recall, relation accuracy, candidate recall, and split/merge accuracy remain unknown. |
| Prototype implementation | CONFIRMED | `gui/app.py`, `gui/rfc_scanner.py`, tracked classical artifact, and `tests/test_gui_rfc_review.py` | Local `.txt` upload, normative gating, classical score, CSV display/export, and optional local explanation are implemented. Transformer ensemble integration is absent. |
| Prospective deployment accuracy | NOT EVALUATED | Development OOF studies only | No independent expert-gold or prospective evaluation supports production claims. |

## Change in primary outcome

The proposal foregrounded thresholded $F_2$ on a temporal gold test. The completed evidence does not contain that test or expert gold. The report therefore uses pooled outer-OOF average precision as the primary development ranking metric because prevalence is low and review is budgeted. This change is methodologically reasonable for prioritization but is adaptive and must not be described as preregistered. Thresholded precision, recall, $F_1$, and $F_2$ remain secondary operating-point diagnostics selected on inner OOF data.

## Primary evidence selection

The reviewed-label DeBERTa follow-up is the primary report study because it has a hash-pinned 2,143-row effective cohort, identical rows and folds across the compared models, complete outer and inner OOF artifacts, fitting-side ECDF/threshold construction, and a validated analysis-only repair. It was not selected because it has the largest score. Its limitations—one seed, nine lineages, AI-supported labels, and 100 error-selected reviews—remain explicit.
