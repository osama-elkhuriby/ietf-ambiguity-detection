# Speaker guide — 25-minute presentation

**Talk:** *Finding ambiguity before publication: leakage-controlled, model-assisted
prioritization of IETF Internet-Draft clauses*

The slide deck is deliberately concise. Use this guide as the talk track; timings
total **25 minutes** and exclude the audience Q&A after the final slide. Edit the
slides and notes freely for the venue and audience.

## 1. Finding ambiguity before publication — 0:30

Open with the practical motivation: protocol standards are implemented by different
people, often in different codebases, and a small gap in prose can become divergent
behavior. Introduce the work as a research project on finding clauses that deserve
closer review while a specification is still a draft. Establish the boundary early:
this is a human-review prioritization aid, not an automated standards judge. The
presentation will focus on the evaluation story and on what the evidence can—and
cannot—support.

## 2. One sentence can lead to different implementations — 1:30

Use the example to make the problem concrete. Words such as MUST, SHOULD, and MAY
have defined normative meaning, but they do not automatically specify every actor,
condition, exception, or recovery path. A sentence that says to retry “when
appropriate” might lead one implementer to retry immediately, another to use
backoff, and a third to retry only for selected errors. Those behaviors can affect
interoperability. Emphasize that this is an illustration rather than a quotation
from the corpus. The system's job is not to decide which implementation is correct;
it is to help a human find clauses where interpretations may diverge.

## 3. The research question is about prioritization—not verdicts — 1:30

State the question plainly: can we rank clauses likely to be operationally ambiguous
using only information available at that draft stage? The output is a ranked queue,
not a binary certification. That framing matters because “ambiguity” is a
human-interpreted property and the reference labels are imperfect. Also explain
what the question excludes: the model should not use the future RFC or later
corrections to decide which earlier clauses look risky. The intended contribution
is a defensible workflow and evidence base, not a claim of full semantic
understanding.

## 4. Three traps can make an ambiguity detector look better than it is — 1:30

Walk through the three threats. First, future-information leakage: later RFC text
and errata may contain the resolution, but that information did not exist when the
draft was reviewed. Second, repeated language: revisions and related documents can
reuse nearly identical clauses, so random row-level splitting can leak near-duplicates
across train and test. Third, the positive class is rare and labels are uncertain.
The latest DeBERTa cohort is only 6.4% positive, and the labels are operational
AI-assisted references rather than expert adjudication. These are not details to
hide in an appendix; they determine what the reported numbers mean.

## 5. The project connects a corpus to several distinct studies — 1:30

The corpus work covers 253 Internet-Draft revisions, 12 RFCs, and 31 verified
technical errata. Then distinguish the three highlighted model cohorts: classical
ML uses 2,406 clauses from 11 families; the Qwen/hybrid experiment uses 2,142
clauses from nine families; the latest DeBERTa study uses 2,125 clauses from nine
families. These are not progressively larger or smaller versions of one identical
test set. Annotation snapshots and eligibility rules differ. This distinction
prevents an attractive but invalid cross-model leaderboard.

## 6. The pipeline makes the information boundary explicit — 1:30

Narrate the pipeline from public sources through parsing, revision alignment,
annotation, modeling, family-grouped evaluation, and ranked review. Parsing and
alignment supply structure and lineage; the model uses only allowed contemporaneous
features. The red information boundary is central: future drafts, final RFCs,
errata, labels, votes, rationales, and split metadata must not become model input.
The final purpose is prioritization by a reviewer. Close by noting that TCP transfer
validation and sealed DNS testing are reserved and remain unopened. They are not
results that can be implied from development cross-validation.

## 7. The label is an operational reference—not a definition of truth — 1:15

Explain that the project has a repeatable annotation workflow, recorded provenance,
and consensus rules, which are useful engineering assets. But a consensus from
AI-assisted annotators is not equivalent to independent experts applying a settled
rubric. The project does not yet have a blinded, independently double-annotated and
adjudicated gold set. So when a model gets a positive “right,” the exact statement
is that it agrees with the operational reference. It is not yet proof that a clause
is objectively ambiguous or that the system measures real-world defect risk.

## 8. The project tests multiple model families—not one magic model — 1:15

Briefly orient the audience to the model families. The heuristic uses transparent
rules. Classical models combine sparse word-level TF–IDF with engineered clause
features. The transformer experiments fine-tune DeBERTa. Local-LLM experiments
include Qwen prompting and supervised lexical/semantic hybrids. Earlier studies
also examined embeddings and revision-derived signals. The purpose of this breadth
is not to imply every model was evaluated on a common split. Instead, it tests
different modeling ideas on their documented artifacts and records the limitations
of each comparison.

## 9. Evaluation is grouped to test transfer across protocol families — 1:30

Describe family-grouped cross-validation: related clauses from a protocol family
stay together, which is more demanding than a random clause split and better
approximates transfer to unseen families. Preprocessing and feature selection are
fit within training folds. Average precision is emphasized because positives are
rare; fixed-budget retrieval is also meaningful for a reviewer with limited time.
Be careful: five-fold OOF is still development evaluation. In the latest DeBERTa
study, one seed means the interval over families does not tell us how much training
randomness changes the result.

## 10. Classical text features give the strongest result in their own cohort — 1:30

Present the 2,406-clause classical cohort and its 6.57% prevalence. Logistic
regression with engineered features plus TF–IDF reaches AP 0.5225; the fixed
heuristic reaches 0.1049. This suggests lexical patterns and structured clause
features contain ranking signal beyond a simple hand-set baseline in this study.
Do not compare the 0.5225 directly with the DeBERTa or Qwen numbers: cohort,
annotation source, and protocol differ. A strong within-cohort AP does not alone
establish transfer, calibration, or independent accuracy.

## 11. Latest DeBERTa study: encouraging ranking, wide uncertainty — 1:45

This study contains 2,125 clauses, 136 positives, and nine families. The equal-weight
empirical-CDF rank ensemble reaches AP 0.3711, compared with a 0.064 prevalence
baseline, and retrieves 26 operational positives among the top 50 ranked clauses.
Translate that into workflow terms: if a reviewer can inspect only 50 clauses from
this study cohort, the OOF ranking places 26 reference positives in that shortlist.
Do not call those 26 “confirmed ambiguities.” The family-cluster interval is wide
(about 0.315 to 0.508), covers only nine families, and does not include training-seed
variation. It is promising development evidence, not a test-set guarantee.

## 12. The three-model hybrid adds only a small AP increment — 1:45

This is the project's exploratory comparison combining the V4 Qwen/MiniLM semantic
ensemble, DeBERTa, and classical logistic regression. The models' within-cohort
percentile ranks are averaged equally; the result is a ranking score, not a calibrated
probability. On 2,097 label-compatible rows with 116 positives, the three-model
combination has AP 0.4619; ML alone is 0.4567. That is a 0.0052 AP difference,
which is modest and should not be described as a reliable gain. The ML source used
an earlier annotation artifact: 17 current records were absent and 28 shared records
had conflicting labels, so 45 were excluded. In addition, V4 and the hybrid were
developed after seeing development results. This is an exploratory comparison, not
independent confirmation.

## 13. What can we responsibly conclude from the model studies? — 1:30

Synthesize without turning the results into a winner chart. First, there is ranking
signal: text and engineered features can prioritize some clauses that match the
operational labels. Second, complexity is not automatically better: the tested
embeddings and revision-only signals were not consistently stronger, and the
three-model hybrid's incremental AP over ML alone is small. Third, study cohorts
matter. A score's interpretation depends on label source, candidate construction,
and evaluation protocol. The most defensible result is a disciplined shortlist and
a transparent account of uncertainty—not “ambiguity solved.”

## 14. The main limitations are scientific—not just engineering — 2:00

Pause on the five limitations. Construct validity: AI-assisted consensus is not
independent expert gold. Sampling: the candidate cohort is enriched and does not
represent every clause in all Internet-Drafts. Generalization: the latest
transformer study contains nine families and one seed. Model selection: several
comparisons are adaptive development analyses. External testing: reserved TCP and
sealed DNS have not been evaluated. Together, these limitations mean the project
cannot claim validated ambiguity accuracy or production readiness. This is not a
failure of the project; identifying the claim boundary is part of responsible ML
research.

## 15. The right role is an explainable reviewer aid — 1:30

Describe a careful human-in-the-loop deployment concept, not a deployment claim.
The system ranks clauses; a reviewer returns to the exact contemporaneous draft
text and context; the author or reviewer makes the interpretive decision. A false
positive may waste review time; a false negative may create false reassurance.
Therefore the system should expose uncertainty and provenance, should never
automatically reject a draft, and should not treat generated explanations as proof.
The user remains responsible for the specification decision.

## 16. What evidence would earn a stronger claim? — 1:30

The next steps are sequential. First, create a stratified sample for independent
double annotation and adjudication. Second, retrain the classical model against the
same frozen label source as the other candidates, and repeat finalist transformer
configurations across multiple seeds. Third, freeze the model, threshold or routing
policy, code snapshot, and analysis plan. Only then evaluate the reserved TCP set.
DNS should be opened once the decision rule is fixed. The sealed set is valuable
because it can test transfer without becoming another tuning loop.

## 17. The contribution is a careful path from draft text to review — 1:00

End with the project-level contribution: a revision-aware corpus and workflow,
explicit temporal boundaries, family-aware evaluation across model families, and
honest reporting of label and generalization limits. The model results are promising
enough to motivate the next study but not enough to claim reliable detection. The
next milestone is independent evidence, not simply a more complex architecture.
Repeat the core phrase: a research-stage prioritization aid, not an autonomous
ambiguity judge.

## 18. Discussion — 0:30

Invite questions. If discussion starts immediately, use the closing question on the
slide: where should the boundary sit between automated triage and human judgment?
The listed sources are starting points; the full project bibliography is in the
manuscript source.

## Timing summary

| Slides | Minutes |
| --- | ---: |
| 1–4 | 5:00 |
| 5–8 | 5:30 |
| 9–12 | 6:30 |
| 13–16 | 6:30 |
| 17–18 | 1:30 |
| **Total** | **25:00** |
