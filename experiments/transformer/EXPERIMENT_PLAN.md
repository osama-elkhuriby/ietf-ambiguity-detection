# Transformer experiment plan

Status: governance frozen before the learning-rate screen. A multi-annotator human
gold set is not currently feasible, so the frozen LLM-consensus labels are the
operational reference labels for this project. They are a proxy gold standard, not
human-verified ground truth; every result and limitation statement must preserve that
distinction.

## Selection rule and deviations

The original test family was HTTP/2bis. A pre-training leakage audit found 13 exact
and three near-identical cross-split clauses involving HTTP/2bis and QUIC families, so
HTTP/2bis was moved into training and DNS SVCB/HTTPS became the sealed test family.
This replacement was made before any model was evaluated on DNS.

DNS contributes exactly 34 positive silver labels because every eligible row from the
family is included; no rows were sampled to manufacture that count. The family was
selected from the available corpus after its family size and label prevalence were
known, partly because it supplied a distinct protocol-family holdout with a positive
count comparable to the contaminated test. This is a researcher degree of freedom.
The clean audit does not remove that selection discretion, so it is disclosed rather
than presenting DNS as a randomly selected holdout.

The DNS lineage audit flags eight exact BCP 14 boilerplate pairs, one against each
training RFC that contains the standard RFC 2119/8174 key-word paragraph. They are
reported separately from substantive copied normative text and are not treated as
protocol lineage or silently deleted. This handling is consistent with the clause
audit: if any such paragraph is itself a candidate clause duplicated across a fitting
and scoring boundary, the ordinary exact-duplicate rule still excludes it from the
score. Boilerplate status therefore explains the lineage finding; it does not create
an exemption from model-evaluation leakage controls.

The learning-rate rule is frozen before viewing the remaining screen results. Select
the rate with the highest arithmetic mean PR-AUC across the five held-out folds. For
the top two rates, define the paired difference in each fold and its standard error as
`sample_SD(differences) / sqrt(5)`. If their mean difference is less than one standard
error, select the numerically lower learning rate and rerun only those two rates with
seed 2718 as a stability check. The second seed does not silently change the fixed
choice; any material contradiction is reported and work stops for a governance
decision. Paired fold differences and fixed-seed bootstrap intervals are always saved.

The completed fold-1 run at 1e-5 (PR-AUC 0.3778) predates this governance freeze. It
is retained as a provisional result and comparison reference, not silently rewritten.
The official learning-rate screen will either rerun it under the frozen artifacts or
record a justified equivalence decision before aggregation.

## Scientific claim

The model predicts agreement with LLM-consensus silver labels. It does not establish
human ambiguity-detection accuracy until it is evaluated on a blinded human gold set.
The primary comparison is against a TF-IDF logistic-regression baseline on exactly the
same family split.

## Frozen data roles

| Partition | Family role | Rows | Positive labels | Permitted use |
| --- | --- | ---: | ---: | --- |
| Train | Nine families | 2,253 | 395 | Fitting and grouped CV only |
| Validation | TCP | 139 | 26 | Independent calibration/routing verification only |
| Test | DNS SVCB/HTTPS | 154 | 34 | One final evaluation after all choices are frozen |

The split artifact is versioned separately from the original annotations. The
partitioned v1 artifact stores train and TCP in a development directory with no test
path in its manifest. DNS is stored under a separate sealed directory known only to
the final-evaluation module. That module requires `--final`, atomically claims a lock
before opening DNS, and refuses a second claim. The cross-split audit must report zero
unreviewed exact duplicates before training starts.

## Development protocol

Use the hashed `folds_v1.json` five-fold family assignment over the nine training
families. Grouped five-fold is preferred to leave-one-family-out because the 6man and
Cache Header families contain only one and five rows. A target
clause that exactly matches text in that fold's fitting partition is retained for
training but excluded from that fold's validation score and reported by record ID.
All class ratios and loss weights are computed from the fitting portion of each fold.
TCP is outside CV and cannot influence architecture, learning rate, weights, checkpoint
selection, temperature, or routing thresholds.

### Nested checkpoint-selection correction

The original `lr-screen-v1` implementation evaluated every epoch on the held-out outer
fold, used that outer-fold PR-AUC for early stopping and best-checkpoint selection, and
then reported the selected checkpoint on the same rows. This creates checkpoint-selection
optimism inside every CV cell. The v1 screen remains immutable but is superseded for all
model-selection and comparison claims.

This was corrected after the pre-registered v1 learning-rate decision, but before any
weighting experiment and before any DNS evaluation. `lr-screen-v2` uses a hash-pinned
inner-family validation partition inside each outer training fold. Only the inner
partition controls early stopping and checkpoint selection. The selected checkpoint is
then evaluated exactly once on the untouched outer fold. The outer fold is not tokenized
into an evaluation loader until checkpoint selection is complete.

The inner assignment is frozen in `inner_selection_v1.json`. Families require at least
30 positives before and 25 after duplicate exclusion and no more than 500 rows. Outer
folds are processed in numerical order; eligible families with the fewest previous
assignments are preferred, with SHA-256 ordering as the deterministic tie-break. The
selected inner families for outer folds 1 through 5 are QUIC HTTP, QUIC Recovery, HTTP
Semantics, QUIC Recovery, and HTTP/2bis. The assignment was frozen before any v2 model
was trained.

| Fold | Held-out families | Rows before exclusion | Positive labels |
| ---: | --- | ---: | ---: |
| 1 | QUIC Transport | 823 | 170 |
| 2 | HTTP Semantics | 407 | 80 |
| 3 | QUIC HTTP | 339 | 49 |
| 4 | 6man, Cache Header, HTTP/2bis, TLS 1.3 | 322 | 43 |
| 5 | QUIC Recovery, QUIC TLS | 362 | 53 |

Checkpoint selection uses PR-AUC, with early stopping patience 2. F2 is reported but
is not used to select checkpoints. Token length is fixed at 256 because every target
fits without truncation; only seven training rows require context truncation. A real
development cell produced non-finite logits under bf16, so primary runs are frozen to
fp32. Do not use fp16; the upstream checkpoint's stored fp16 tensors are explicitly
loaded into fp32 parameters before training.

## Bounded experiment sequence

Run the stages in order and reuse completed control cells. Do not evaluate a full
Cartesian product.

1. Establish TF-IDF logistic-regression and simple heuristic baselines.
2. Screen learning rate on target-only, HIGH+MEDIUM, unweighted data: three rates ×
   five folds = 15 runs, one seed.
3. Freeze the rate. Compare unweighted, class-weighted, confidence-weighted, and
   combined weighting under the three-seed design specified below.
4. Freeze weighting. Compare hard HIGH+MEDIUM targets with soft vote-fraction targets
   only after a versioned vote-fraction field exists. HIGH-only remains a tier-level
   evaluation slice and is never used as a training regime because training contains
   only 17 HIGH positives.
5. Freeze the label regime. Compare target-only with local context: five new runs.
6. Rerun only the selected configuration with seeds 1729, 2718, and 31415. Report
   every seed plus mean and standard deviation; use the three-seed ensemble for routing.
7. After freezing a global epoch policy, regenerate honest out-of-fold predictions
   without selecting each checkpoint on its own held-out labels. Fit temperature
   scaling on unweighted OOF NLL and derive routing thresholds from pooled OOF
   predictions. Use TCP only to verify transfer to an unseen family.
8. Freeze code, hashes, thresholds, and the test command. Evaluate DNS once.

When configurations are statistically indistinguishable, prefer the simpler choice in
this order: target-only, unweighted, HIGH+MEDIUM. This prevents extra complexity from
being selected on noise.

### Frozen Step 3 weighting design

Use learning rate 1e-5, the existing five outer folds, the same inner-family
assignment, and seeds 1729, 2718, and 31415. Seed 31415 is fixed before any weighting
ablation result is observed. Reuse the completed unweighted 1e-5 cells for seeds 1729
and 2718. Run only missing cells: five unweighted seed-31415 cells and all three seeds
for class, confidence, and combined weighting, for 50 new cells total.

Select weighting by the highest seed-averaged mean-fold PR-AUC. Average each fold's
PR-AUC over the three seeds first. For the best two strategies, compute paired
per-fold differences between these seed-averaged scores and their standard error as
`sample_SD(differences) / sqrt(5)`. If the mean difference is less than one standard
error, choose the simpler weighting according to: unweighted, class, confidence,
combined. Report, regardless of the winner, these clean contrasts: unweighted versus
confidence, class versus combined (the confidence effect at more comparable positive
mass), and class versus unweighted.

Apply the same four weighting strategies and paired-fold one-standard-error comparison
to TF-IDF. TF-IDF is deterministic, so it has one score per fold rather than artificial
seed replicates; its per-fold values are the inputs to the otherwise identical rule.
Keep transformer and TF-IDF choices separate because their objectives and fitted data
are not identical.

## Routing objective

Select the two-threshold abstention band from honest pooled OOF predictions to minimize
LLM referral rate subject to ambiguous-clause recall of at least 90%. Verify the frozen
temperature and thresholds on TCP without refitting them. Also report precision above
the positive threshold, risk-coverage, the no-abstention operating point, and the
LLM-only reference. If TCP does not preserve the constraint, report that result instead
of weakening the target after looking at outcomes.

## Final reporting

Always save record IDs, both raw logits, uncalibrated and calibrated probabilities,
the selected threshold, abstention status, labels, tiers, and inclusion/exclusion flags.
Report precision, recall, F1, F2, PR-AUC, MCC, false-negative rate, Brier score, ECE,
confusion matrix, and bootstrap confidence intervals. Report HIGH-only and all-row
results separately.

The silver test contains only 34 positives, so its recall interval will be wide. Build
a blinded human set with at least 100 positive examples when feasible. Annotators must
not see model outputs, LLM votes, rationales, or tiers; report agreement before
adjudication and keep uncertainty-enriched cases separate from the random sample.

Two design limitations must remain visible beside aggregate results. First, the five
CV folds separate named families but still contain related protocol clusters, notably
QUIC and HTTP. Pooled out-of-fold performance can therefore be optimistic relative to
a genuinely new protocol; run a later cluster-level holdout sensitivity analysis and
report it separately. Second, the sealed test is a single document, RFC 9460, with
154 rows and 34 positives. Its confidence intervals describe that RFC and do not by
themselves establish broad cross-protocol generalization.

## Current completion boundary

Step 2b is complete. The original `lr-screen-v1` is retained but superseded because
outer-fold PR-AUC selected each checkpoint. Across its 15 cells, selected-epoch
outer-fold PR-AUC exceeded final-epoch PR-AUC by 0.0314 on average, exceeded epoch-3
PR-AUC by 0.0385, and exceeded epoch-5 PR-AUC by 0.0195 where those fixed epochs were
available. These are diagnostic differences, not a formal bias estimate.

The corrected target-only, unweighted, fp32 `lr-screen-v2` at seed 1729 achieved mean
outer-fold PR-AUC 0.2817 at 1e-5, 0.2683 at 2e-5, and 0.2600 at 3e-5. The top-two
paired mean difference was 0.0134 with standard error 0.0251. The frozen rule therefore
selected the lower rate, 1e-5, and triggered only the required seed-2718 stability
reruns for 1e-5 and 2e-5. Their mean-fold PR-AUC values were 0.3153 and 0.2865,
respectively; 1e-5 remained higher, so the stability run did not contradict the frozen
choice. The second seed confirms stability but does not redefine the selection rule.

The nested protocol creates an important comparator limitation. TF-IDF has no learned
checkpoint-selection stage and is fitted on every family in each outer training fold.
The transformer reserves one additional outer-training family for inner early stopping
and checkpoint selection, so it is fitted on fewer families and rows. Report this
training-data asymmetry beside every direct TF-IDF-versus-transformer comparison; it
prevents attributing the observed difference to architecture alone.

At seed 1729, selected 1e-5 had pooled OOF PR-AUC 0.2563 versus 0.2712 for unweighted
TF-IDF, a difference of -0.0149. Its 95% paired-row bootstrap interval was
[-0.0461, 0.0165], and its section-cluster interval was [-0.0448, 0.0168]. At seed
2718 the corresponding transformer pooled PR-AUC was 0.2774 and the difference from
TF-IDF was 0.0063, with section-cluster interval [-0.0235, 0.0362]. Neither supports a
transformer advantage at this stage. The v1-to-v2 correction reduced selected-rate
mean-fold PR-AUC by 0.0762 and pooled PR-AUC by 0.0499.

Accordingly, H1 (contextual transformer semantics outperform surface-feature
classical models) is unsupported at this stage. This is a valid interim negative
result, not a reason to change the frozen comparison protocol.

Confidence-only weighting is a documented class confound: on the full training set it
reduces positive effective-weight share from 0.1753 unweighted to 0.1389 because most
positive examples are MEDIUM tier. Class weighting gives share 0.5000 and combined
weighting gives 0.4314. The same quantities are saved fold-locally before any weighting
experiment is run.

Step 2c is also complete. Under a single-threshold policy that refers every row at or
above the boundary score, pooled 90%-recall referral rates are 0.7546 for unweighted
TF-IDF, 0.7510 for class-weighted TF-IDF, and 0.8186 for the two-seed DeBERTa logit
average. Thus a 90% recall target leaves measurable but modest call savings: about 25%
for TF-IDF and 18% for the current transformer ensemble. At 95% recall those savings
fall to about 15%-17% and 11%, respectively. These are development OOF estimates, not
deployment guarantees; fold variability and bootstrap intervals must accompany them.

The two-seed logit-average ensemble has mean-fold PR-AUC 0.3061 and pooled PR-AUC
0.2771. Against unweighted TF-IDF its pooled difference is 0.0060, with 95% paired-row
interval [-0.0214, 0.0346] and section-cluster interval [-0.0221, 0.0360]. This does
not alter the conclusion that H1 is unsupported. The ensemble is a feasibility result,
not a replacement for the pre-registered single-seed learning-rate decision.

Step 2d found that rationales-free per-model votes exist only in the annotation source
`corpus/processed/llm_annotations/v1/consensus.csv`, not in model-ready development
inputs. They remain audit-only metadata and must never enter model text, features,
weights, calibration, or routing inputs. OOF pairwise Cohen's kappa spans only 0.0165
to 0.1130, demonstrating strong model-specific voting behavior. HIGH-tier
leave-one-LLM-out PR-AUC is 1.0 by construction, while MEDIUM-tier leave-one-out
subsets are frequently sparse or single-class because majority-label selection is
conditional on disagreement. These values are approximate silver-label reproducibility
bounds, not biological or human-annotation ceilings, and no single aggregate ceiling
should be claimed from them. Spearman correlation with the fixed-denominator yes-vote
fraction is 0.2340 for the DeBERTa two-seed ensemble and 0.2473 for unweighted TF-IDF.

Implemented: reviewed split derivation, immutable input materialization, leakage and
lineage audits, structurally isolated development/test artifacts, hashed grouped folds,
one-shot final-test lock, target-preserving tokenization, fold-local weighting,
raw-logit output, checkpoint manifests, fp32 CUDA training, an end-to-end smoke run,
and the TF-IDF logistic-regression comparator over all frozen folds and weighting
strategies, plus the superseded v1 learning-rate screen, nested-CV correction, corrected
v2 screen, required second-seed check, Step 2c feasibility/noise audit, and Step 2d
silver-label ceiling/error audit. The TF-IDF pooled OOF PR-AUC is
0.2712 unweighted, 0.2785 class-weighted, 0.2741 confidence-weighted, and 0.2852
combined; these are comparison results, not a weighting-selection decision.

Step 3 is complete under the frozen design. Seed-averaged mean-fold DeBERTa PR-AUC is
0.2955 unweighted, 0.3022 class-weighted, 0.2907 confidence-weighted, and 0.2949
combined. Class minus unweighted is 0.0067 with paired-fold standard error 0.0055, so
the frozen one-standard-error fallback does not trigger and class weighting is selected.
The pooled three-seed class-weighted ensemble PR-AUC is 0.2874. Its pooled difference
from class-weighted TF-IDF is 0.0089 with section-cluster interval [-0.0296, 0.0451],
so H1 remains unsupported. At 90% recall, selected DeBERTa refers 85.8% of rows
([81.0%, 88.3%] row-bootstrap interval), leaving only about 14% estimated call savings.
TF-IDF selects confidence weighting by the same frozen rule and refers 73.3% of rows.
All probability calibration metrics remain secondary and uncalibrated at this stage.

Pending: soft-target sensitivity after a vote-fraction artifact exists, context
ablation, temperature scaling, final routing-threshold optimization, final DNS
execution, and human gold annotation/evaluation. None begins as part of Step 3.

## Frozen Step 4 factorial: label uncertainty and local context

This section was frozen before any Step 4 scores were produced. Its purpose is to
test two plausible causes of the weak transformer advantage without expanding the
search space after looking at outcomes: hard consensus targets may discard annotator
disagreement, and target-only input may omit context needed to resolve ambiguity.

Use DeBERTa-v3-small at immutable revision
`a36c739020e01763fe789b4b85e2df55d6180012`, learning rate 1e-5, fp32, class
weighting, seeds 1729/2718/31415, the five frozen outer folds, and the frozen inner
family assignments. Fine-tune the full encoder and classification head. Checkpoint
selection remains the original hard-label PR-AUC on the inner family for every cell;
the outer fold is evaluated once after selection. Reuse the existing hard-target,
target-only class-weighted cells and train only the 45 missing cells in this matrix:

| Condition | Training target | Input |
| --- | --- | --- |
| Control | Hard consensus label | Target only |
| Soft-target | Valid-vote positive fraction | Target only |
| Context | Hard consensus label | Target + local context |
| Combined | Valid-vote positive fraction | Target + local context |

The soft target is `positive_valid_votes / valid_votes`, using only rationales-free
vote fields. Invalid or missing votes are excluded from the denominator, never treated
as negative. At least two valid votes are required. The artifact builder must verify a
one-to-one mapping to development rows and report any exclusion before training. Votes,
vote fractions, and rationales never enter tokenized text or inference features.

To isolate the target change, class weights in soft-target cells are computed from the
unchanged hard labels in each fitting partition: positive weight `N_negative/N_positive`, negative
weight 1, followed by normalization to mean 1. The weighted soft cross-entropy is the
per-row binary soft-label cross-entropy multiplied by those same row weights and
divided by their sum. Do not derive class mass from soft labels. A one-hot soft-target
test must match hard cross-entropy in both value and gradients.

Local context uses the existing target-first sentence-pair policy: target is segment
one, previous and next same-section clauses form segment two, and context truncates
before the target. Because development context can exactly or nearly reproduce a
target in another nested partition, each fold receives a versioned context policy.
For fitting, inner-selection, and outer-validation rows, blank any context side whose
normalized word-trigram Jaccard similarity to a target in either of the other roles is
at least 0.85. Exact matches are therefore always removed. Preserve target text and
record the matched target IDs, roles, similarity, source hashes, and every removal.
This unsupervised sanitation may inspect text but never labels in another role.

Primary factorial estimands are computed from seed-averaged fold PR-AUC:

- soft-target effect: soft target-only minus hard target-only;
- context effect under hard labels: hard context minus hard target-only;
- context effect under soft labels: soft context minus soft target-only;
- interaction: `(soft context - soft target-only) - (hard context - hard target-only)`.

Report each seed and fold, seed-averaged mean-fold PR-AUC, pooled PR-AUC from the
three-seed logit average, paired fold differences and their standard errors, paired-row
bootstrap intervals, section-cluster bootstrap intervals, 90%-recall single-threshold
referral rate, runtime, and peak GPU memory. ECE and Brier remain uncalibrated secondary
diagnostics. The primary selection statistic is seed-averaged mean-fold PR-AUC. If the
top two differ by less than one paired-fold standard error, choose the simpler option
in this frozen order: hard target-only, soft target-only, hard context, soft context.
Pooled PR-AUC cannot override that rule.

Run one matched class-weighted target-only TF-IDF comparator using the exact transformer
fitting IDs, which excludes the inner family, and the same outer scoring exclusions.
Keep its hyperparameters fixed. Report it separately from the original TF-IDF result;
this matched comparator isolates the architecture contrast, while the original remains
the operational classical baseline trained on all outer-training families.

Do not tune on or evaluate TCP during Step 4. Do not open the sealed DNS clause file,
invoke final evaluation, write predictions for DNS, or claim human-verified accuracy.
The frozen consensus labels are treated as operational proxy gold only because human
annotation is currently infeasible. A family-count learning curve and cluster-level
holdout remain later exploratory analyses, not part of this selection experiment.
