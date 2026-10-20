# Speaker guide — 25-minute presentation

**Talk:** *Finding ambiguity before publication: leakage-controlled, model-assisted prioritization of IETF Internet-Draft clauses*

The deck presents the current report, including the latest 2,425-row classical rerun and revision-feature ablation. Timings total **25 minutes** before Q&A.

## 1. Finding ambiguity before publication — 0:30

Open with the practical goal: help standards authors and reviewers find clauses that deserve attention before publication. Set the claim boundary immediately: this is a research-stage prioritization aid, not an autonomous standards judge.

## 2. One sentence can lead to different implementations — 1:20

Use the illustrative retry sentence to show how precise BCP 14 keywords can sit inside underspecified prose. The system does not determine which behavior is correct; it ranks clauses where implementations may diverge.

## 3. The research question is about prioritization—not verdicts — 1:20

State the question: can draft clauses be ranked using only information available at that draft stage? The output is a review queue, not a certificate of correctness. Success means useful attention allocation under a defensible evaluation.

## 4. Three traps can make an ambiguity detector look better than it is — 1:30

Explain future-information leakage, near-duplicate language across revisions, and rare/noisy positive labels. These concerns drive the temporal guards, family grouping, and emphasis on average precision and review-budget retrieval.

## 5. The project connects a corpus to several distinct studies — 1:30

The corpus contains 253 draft revisions, 12 RFCs, and 31 verified technical errata. Keep the study tracks separate: the latest classical cohort has 2,425 rows and 160 positives across 11 families; the Qwen/hybrid cohort has 2,142 rows and 126 positives across nine families; the reviewed-label DeBERTa cohort has 2,143 rows and 138 positives across nine families. These are not one shared leaderboard.

## 6. The pipeline makes the information boundary explicit — 1:30

Walk from public sources through parsing, alignment, annotation, modeling, grouped evaluation, and ranked review. Later drafts, RFC text, errata, labels, votes, rationales, and split metadata are excluded from prediction features. Correct the earlier project plan: TCP and DNS entered the latest all-family classical rerun, so they are no longer untouched project-level holdouts.

## 7. The label is an operational reference—not a definition of truth — 1:40

Emphasize the disagreement audit. In the earlier 2,546-row snapshot, 435 of 455 positive labels were two-to-one votes and only 20 were unanimous. Pairwise Cohen's kappa ranged from 0.017 to 0.113. This does not tell us which model was correct; it shows that the operational target is noisy and model-dependent. No independently double-annotated, adjudicated expert gold set exists yet.

## 8. The project tests multiple model families—not one magic model — 1:05

Orient the audience to the fixed heuristic, classical ML, DeBERTa, and Qwen/hybrid tracks. Frozen embeddings and revision-derived signals were also studied. The breadth tests modeling ideas; it does not make scores from different cohorts directly comparable.

## 9. Evaluation is grouped to test transfer across protocol families — 1:25

Related clauses stay within family groups, and preprocessing is fit inside training folds. The transformer uses five family-isolated folds; the latest classical rerun uses three stratified group folds. Average precision is primary because positives are rare. These remain development evaluations, and the newest transformer has only one seed.

## 10. Classical text features give the strongest result in their own cohort — 1:45

On 2,425 rows with 160 positives, engineered features plus TF–IDF achieve AP 0.3584 with Linear SVM, 0.3363 with random forest, and 0.3265 with logistic regression. The fixed heuristic reaches 0.0873 against 0.0660 prevalence. Threshold behavior differs: logistic regression has recall 0.706 at precision 0.190, while SVM has recall 0.313 at precision 0.455. The cohort includes TCP and DNS and is not an untouched holdout.

## 11. Latest DeBERTa comparison: promising ranking, wide uncertainty — 1:45

On the shared 2,143-row reviewed-label cohort, focal loss reaches AP 0.5823, partial fine-tuning 0.5761, and the equal-weight fitting-side ECDF rank ensemble 0.6736. The ensemble retrieves 19 of the top 20 and 46 of the top 50 operational positives. Its family-bootstrap AP interval is 0.5920–0.7641, but intervals overlap, only one seed was run, and 100 human-confirmed records were selected from historical OOF errors. The ECDF value is a rank, not a calibrated probability.

## 12. Revision history did not improve ranking AP in the matched rerun — 1:45

This is the controlled RQ2 ablation. Adding twelve past-only revision features changes AP by −0.0081 for logistic regression, −0.0012 for SVM, and −0.0015 for random forest. F2 is unchanged for logistic regression and increases slightly for SVM and forest. The result does not support a general ranking benefit from revision features. Cross-reference features were not isolated and calibration metrics were not recorded, so RQ2 is only partly answered.

## 13. The report answers some questions and leaves two open — 1:40

Summarize the proposal questions. RQ1 supports development-stage feasibility with qualifications. RQ2 is partly answered and its AP hypothesis is unsupported. RQ3 compares effectiveness and workload only under separate protocols. Qwen eight-shot prompting reaches AP 0.1960 and the fitted Qwen/TF–IDF/MiniLM V4 reaches 0.3252 on their own cohort. RQ4 remains open because no conditional router reduced LLM calls; RQ5 remains open because explanation faithfulness was not evaluated by experts.

## 14. The main limitations are scientific—not just engineering — 1:45

Cover construct validity, enriched sampling, nine-family/single-seed generalization, adaptive model selection, and the loss of the planned holdouts. Also mention pretraining contamination and the error-selected reviewed subset if time permits. These limitations prevent a claim of validated ambiguity accuracy or deployment readiness.

## 15. GUI example: RFC 9000 review results — 1:20

The local GUI found 112 suggestions among 516 explicitly normative clauses across 69 sections. The shown score of 0.5433 is a ranking signal, not a probability or verdict. The GUI uses a saved logistic-regression artifact, not the research DeBERTa ensemble; the capture also contains a scikit-learn artifact/runtime version warning.

## 16. What evidence would earn a stronger claim? — 1:30

First build an expert-adjudicated reference. Repeat selected transformers across seeds and align comparisons on common labels and folds. Implement the intended conditional LLM router and measure calls, latency, cost, calibration, and explanation quality. Finally, preregister a genuinely untouched evaluation based on unseen families or later time-sliced revisions.

## 17. The contribution is a careful path from draft text to review — 0:55

The strongest contribution is methodological: a revision-aware corpus, explicit temporal boundaries, family-aware evaluation, and an honest claim boundary. The tested revision features did not improve AP, and current labels and evaluations do not support autonomous use. The next milestone is independent evidence.

## 18. Discussion — 0:20

Invite questions with the boundary question: where should automated triage end and human judgment begin?

## Timing summary

| Slides | Minutes |
| --- | ---: |
| 1–4 | 4:40 |
| 5–8 | 5:45 |
| 9–12 | 6:40 |
| 13–16 | 6:15 |
| 17–18 | 1:15 |
| **Total** | **24:35** |

The remaining 25 seconds is intentional transition buffer.
