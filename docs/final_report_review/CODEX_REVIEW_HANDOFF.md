# RFC ambiguity-detection final-report audit handoff

Audit date: 2026-10-21  
Repository: `D:\RFC\ietf-ambiguity-detection`  
Status vocabulary: **CONFIRMED**, **CORRECTED**, **UNRESOLVED**, **NOT EVALUATED**

## Executive assessment

The repository supports a substantial and defensible **development** study, but
not a confirmatory estimate on independent expert gold. The strongest single
piece of evidence is the reviewed-label transformer study: a hash-pinned
2,143-clause, 138-positive, nine-lineage cohort; complete five-fold outer OOF
predictions; three inner folds per outer fold; a fitting-side ECDF ensemble; and
a validated analysis-only repair. Its ensemble AP is 0.6736. This study is the
report's primary evidence because its compared models share rows, labels, folds,
and nested analysis—not because it has the highest score.

The principal threats are material. The annotation reference is AI-assisted,
the 100 human-confirmed clauses were deliberately selected from historical model
errors, the exact human-review process is undocumented, only one transformer
training seed is complete, and only nine draft lineages are evaluated. Historical
LOFO work and the current all-family classical rerun used TCP and DNS, so those
lineages are **not untouched project-level holdouts**. The prototype is a local
classical prioritization tool; it does not integrate the transformer ensemble.
No chronological test, calibration study, routed-verifier call-reduction study,
alignment-accuracy study, standardized latency/cost study, or explanation-
faithfulness evaluation was completed.

The report has been edited to state these boundaries, reconcile cohorts, correct
a threshold-metric contradiction and corpus-count error, add confusion counts
and fixed-budget recall, replace invalid confidence-interval reasoning with a
paired exploratory bootstrap, and make reproducibility limitations explicit.
The tracked PDF is stale and could not be rebuilt because no TeX executable is
installed locally.

## Authority and inventory

No `AGENTS.md` exists. The working tree was clean at audit start. Authority was
established from manifests and completed artifacts, not modification time.

| Area | Selected authority | Basis |
|---|---|---|
| Current report | `paper/main.tex`, `paper/references.bib`, `paper/figures/` | `paper/README.md:3` names `main.tex`; lines 30–31 state tracked `main.pdf` is stale |
| Existing compiled report | `paper/main.pdf` | Present, but **not current authority** and not distributable without rebuild (`paper/README.md:30-31`) |
| Proposal | `Proposal/Group 14_Initial_Project_Proposal.pdf`; `Proposal/Literature_Review.pdf` | Original proposal/literature artifacts; local PDF text extraction was unavailable |
| Phase-1/alignment report | `alignment/version_alignment/reports/TLS_Revision_Alignment_Stage_Report.md` and `.docx` | Contains the executed TLS alignment coverage inventory and explicit unadjudicated sample |
| Final presentation | `paper/presentation/ietf_ambiguity_detection_presentation.pptx`, source generator, speaker notes, and 18 rendered slide PNGs | Complete editable/rendered local presentation bundle; separate from proposal and alignment stage report |
| Annotation definition | `corpus/annotate_with_ollama.py:40-75`; `models/llm/prompting.py:10-45` | Executable frozen prompt wording and later prompt variants |
| Current annotation source | `results/annotation/final_annotation_ready.csv` | SHA-256 `8ffa654c2c405dd3eef72901c600f1ef3823e6dbff663020bb824c57ff65bdbf`; pinned in primary config field `data.source_sha256` |
| Current primary cohort | `experiments/transformer_final/final_annotation_human_reviewed_100_v1/manifests/dataset_v1/dataset.csv` | Effective SHA-256 `4e20a2571bfe71f38d04e5677e067ce4d2eda2008f00c297349b929bb662b4af`; manifest fields `rows=2143`, labels `138/2005`, nine families |
| Primary executed predictions | Parent study `.../final_annotation_human_reviewed_100_v1/runs/` | Ten outer and 30 inner completion artifacts, all validated by the repair |
| Primary analysis authority | `experiments/transformer_final/final_annotation_human_reviewed_100_v1_analysis_fix_v1/` | `ANALYSIS_REPAIR_MANIFEST.json` limits repair to metadata identity merge and reuses parent predictions |
| Current classical study | `results/ML/figures/experiment_metrics.csv`, per-run `results/ML/*/metrics.json` and `manifest.json` | Completed 2,425-row/160-positive all-family outputs |
| Current heuristic | `results/heuristic/final_annotation_v1/` | Completed deterministic score output on current all-family source |
| Corpus count | `corpus/metadata/final_rfc_manifest.csv` | Current 13-row lineage manifest; draft counts sum to 264 |
| Historical corpus audit | `corpus/metadata/corpus_audit.json` | Preserved but stale at 12 RFCs/253 drafts; predates Cache-Status |
| Prototype | `gui/app.py`, `gui/rfc_scanner.py`, `gui/README.md`, `tests/test_gui_rfc_review.py` | Code, tracked artifact path, documented tests/limitations; the full GUI suite is not audit-safe because one test reads reserved RFC 9460 text |

No separate report supplied only through the ChatGPT conversation was available;
the closest and documented current source is `paper/main.tex`.

## A. Dataset lineage and cohort reconciliation

The complete machine-readable registry is `docs/final_report_review/DATASET_STUDY_REGISTRY.csv`.
The operations below explain the requested counts.

| Cohort | Status and operation |
|---|---|
| 2,546 / 455 historical consensus | **CONFIRMED.** Eleven labeled families; strict-majority local-model votes. Candidate-enriched and not expert gold. `results/vote_weights/summary.md` reports 435/455 positives as 2–1 votes. |
| 2,540 / 453 LOFO scored | **CONFIRMED.** Starts from 2,546. IPv6 (one row) and Cache-Status (five rows) are below the 30-row holdout threshold and remain fitting-only, leaving 2,540 scored rows over nine held-out families. `results/revision_features/summary.md` and its per-family CSVs are authority. |
| 2,406 / 158 classical/heuristic | **CONFIRMED, superseded.** Older annotation export, eleven families, three `StratifiedGroupKFold` folds. Preserve as history; do not mix its displayed metrics with the 2,425-row rerun. |
| 2,424 source → 2,142 / 126 development | **CONFIRMED.** Source has 148 positives plus unresolved item 2161. Remove TCP 131/5, DNS 151/17, and unresolved 2161: 2,424−282=2,142; 148−22=126. Protocol authority: `experiments/transformer_final/recall_improvement_study_v3_optimizer_fix/protocols/recall_improvement_protocol_v3_optimizer_fix.json`. |
| Initial transformer 2,142 / 45 | **CONFIRMED.** Same row count, different earlier labels. It must not be conflated with the later 2,142/126 cohort; `paper/main.tex` study map identifies the separate target. |
| Earlier target-only 2,125 / 136 | **CONFIRMED.** Source 2,406/159, then reservation, unresolved-row removal, and label-blind fitting-side exclusions. Authority: `experiments/transformer_final/final_annotation_661_v1_2/DATASET_MANIFEST.json` (`original` hash prefix `8e395...`, effective `19364...`). |
| Cross-model 2,097 / 116 | **CONFIRMED.** From the 2,142-row development frame, removes 17 rows absent from an older classical OOF artifact and 28 label disagreements. The intersection therefore changes the evaluated population. Authority: `results/hybrid/final_rank_ensemble_v2/manifest.json`. |
| Updated 2,425 / 160 source | **CONFIRMED.** Contains 2,055 `model_agreement`, 270 `manual_conflict_verdict`, and 100 `human_review_user_confirmed` records. Item 2161 is resolved as No. |
| Updated 2,143 / 138 primary development | **CONFIRMED.** Excludes TCP 131/5 and DNS 151/17 from the updated source: 2,425−282=2,143 and 160−22=138. No unresolved row remains. |

The current corpus has **13 RFC lineages**, not 12: 11 have complete draft
histories and labeled candidate rows, while OSPFv2 and BGP-4 are RFC-only. The
11 full histories contain 264 drafts. The primary transformer evaluates nine of
the 11 labeled lineages because TCP and DNS are excluded in that track. The
family map is:

- Development folds: IPv6/rfc2460bis, Cache-Status, HTTP/2, HTTP semantics,
  QUIC HTTP, QUIC recovery, QUIC TLS, QUIC transport, TLS 1.3.
- Labeled but excluded from transformer/Qwen development: TCP/rfc793bis and DNS
  SVCB/HTTPS.
- Corpus-only RFC families: OSPFv2 and BGP-4.

In these folds, “family” means an Internet-Draft lineage—not a whole RFC series
or broad QUIC/HTTP/transport protocol suite (`paper/main.tex:242`). Row identity
is `record_id` in model-ready/fold artifacts and `item_id` at annotation-source
joins; both are retained where cross-version joins occur.

## B. Human review and annotation validity

**CONFIRMED:** `results/analysis/old_ensemble_top50_error_review_v1/TOP100_ENSEMBLE_ERRORS_COMBINED.csv`
contains exactly 50 false positives and 50 false negatives. Selection used the
earlier DeBERTa-v3-base focal and Partial-4 outer-OOF predictions, an equal ECDF
ensemble, and each fold's inner-selected minimum-precision-0.30 threshold. The
selection table includes item/document identity, target and adjacent context,
old reference label, model decision, ensemble and component scores, and decision
threshold. The records already existed; this was selective relabeling, not new-
case collection.

**CONFIRMED transitions:** 29 No→Yes, 17 Yes→No, 21 No→No, and 33 Yes→Yes;
therefore 46 changed and 54 did not. The reviewed subset is now 62 Yes/38 No.
Fold counts are 44, 13, 12, 15, and 16 for outer folds 1–5. All 100 current rows
have `final_source=human_review_user_confirmed`.

**UNRESOLVED:** no repository artifact records the reviewer identity, exact UI or
subset of selection fields actually shown, expertise, whether judgments were
independent, whether there were multiple reviewers, or an adjudication process.
The available selection CSV contains model outputs, so blindness cannot be
inferred. `human_review_user_confirmed` therefore means only that a user-confirmed
label entered the source; it does not establish blinded expert gold.

`manual_conflict_verdict` is **CONFIRMED only as a provenance tier** accepted by
dataset construction. No operational protocol for the 270 rows was found, so it
cannot be expanded into “independent adjudication.” Item 2161 was previously an
unresolved No placeholder and excluded; it is now resolved
`manual_conflict_verdict`, label No, and included in fold 1. The fresh reviewed-
label transformer parent and its analysis repair use the revised labels. The
current classical/heuristic outputs use the updated 2,425/160 cohort counts;
the active GUI artifact's differing source hash prevents a stronger exact-byte
claim. Earlier 2,406/2,424/2,125/2,142 artifacts retain historical snapshots.

The frozen definition (`corpus/annotate_with_ollama.py:40-75`) calls a clause
ambiguous only if its language allows at least two reasonable, materially
different implementation behaviours, or a concrete missing constraint prevents
a unique operational interpretation. Difficult/awkward wording, style, ordinary
domain knowledge, subjective words without operational divergence, and intended
MAY/SHOULD discretion do not suffice. Missing external context is Uncertain;
supplied context may resolve the wording.

The dependence is explicit: a historical model enriched the review sample with
its most severe errors, and the revised labels were then used to evaluate a
closely related model family on the same lineages. This is model-guided selective
relabeling and target repair, not an unbiased estimate of prevalence or future
error. The 100 cases were not error-enriched **new** cases.

## C. Holdout integrity and information boundaries

**CORRECTED:** TCP and DNS are not untouched project-wide holdouts.

- Historical LOFO outputs in `results/revision_features/` hold each sufficiently
  large family out in turn and include both TCP and DNS in reported evaluation.
- The current 2,425-row classical/heuristic study includes all 11 labeled families.
  In grouped CV, TCP/DNS also enter fitting when another family is held out.
- The Qwen/hybrid 2,142-row and transformer 2,125/2,143-row tracks exclude TCP
  and DNS from fitting, tuning, calibration, and evaluation. Primary config
  fields `partitions.train_families`, `tcp_family`, and `dns_family` document that
  experiment-scoped boundary (`.../config.json:29-41`).

Accurate report claim: “TCP and DNS remain excluded from the nine-family
transformer and Qwen development studies, but historical LOFO and current
classical work consumed them, so no untouched project-level TCP/DNS test remains.”
Whether anyone accessed additional previously unopened protected row-level files
is **UNRESOLVED by design**; this audit inspected only permitted code, manifests,
documentation, and already-open development evidence.

Past-only guards in `models/classical/revision_features.py` reject later revisions,
RFC endpoints, later errata, labels, votes, candidate source, and rationales from
features. That is a future-information guard, not chronological evaluation. The
actual outer folds hold out draft lineages observed in one common historical
corpus; they do not train on an earlier time window and test on a later one.

## D. Metric contradictions and source authority

**CORRECTED:** prose values precision 0.456, recall 0.382, F1 0.416 are valid for
the **earlier 2,125-row/136-positive label snapshot**, not the updated cohort.
Their source is `experiments/transformer_final/final_annotation_661_v1_2/FINAL_NESTED_METRICS.csv`,
row fields `method=ensemble`, `fold=pooled`, `minimum_fitting_precision=0.3`:
TP=52, FP=62, TN=1,927, FN=84, precision 0.456140, recall 0.382353, and F1
0.416000. The updated table values derive from the repaired primary output
`.../outputs/FINAL_NESTED_METRICS.csv`, row fields
`method=ensemble`, `scope=pooled_outer_oof`, `minimum_fitting_precision=0.3`:

| Method | TP | FP | TN | FN | Precision | Recall | F1 | F2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Focal | 101 | 162 | 1,843 | 37 | 0.3840 | 0.7319 | 0.5037 | 0.6196 |
| Partial-4 | 98 | 140 | 1,865 | 40 | 0.4118 | 0.7101 | 0.5213 | 0.6203 |
| Ensemble | 101 | 160 | 1,845 | 37 | 0.3870 | 0.7319 | 0.5063 | 0.6212 |

Verified threshold-free pooled OOF results:

| Method | AP | trapezoidal PR-AUC | ROC-AUC |
|---|---:|---:|---:|
| Focal | 0.582310 | 0.580229 | 0.881481 |
| Partial-4 | 0.576129 | 0.574198 | 0.892385 |
| Ensemble | 0.673573 | 0.672897 | 0.911842 |

The report now uses “AP” for sklearn stepwise average precision and does not call
it trapezoidal PR-AUC. `docs/final_report_review/REVIEWED_TRANSFORMER_METRIC_AUDIT.csv`
confirms 2,143 unique records, no missing/non-finite scores, higher score=Yes,
pooled one-score-per-row evaluation, and inverted-score AP near 0.035. Score ties
occur on 0 focal, 1 Partial-4, and 424 ensemble rows; Top-K ties use score
descending then `record_id` ascending. Ensemble ties are expected from empirical
CDF ranks.

Fixed budgets: ensemble Top-20 has 19 positives, precision 0.95 and recall
19/138=0.1377; Top-50 has 46 positives, precision 0.92 and recall 46/138=0.3333.

Ensemble family results from `.../outputs/FAMILY_ROBUSTNESS.csv`:

| Family | Rows / Yes | AP |
|---|---:|---:|
| IPv6/rfc2460bis | 1 / 0 | undefined |
| Cache-Status | 5 / 2 | 1.0000 |
| HTTP/2 | 239 / 11 | 0.5771 |
| HTTP semantics | 385 / 34 | 0.7961 |
| QUIC HTTP | 314 / 12 | 0.5791 |
| QUIC recovery | 162 / 11 | 0.7310 |
| QUIC TLS | 189 / 8 | 0.5706 |
| QUIC transport | 794 / 57 | 0.6456 |
| TLS 1.3 | 54 / 3 | 0.9167 |

## E. OOF and nested-protocol correctness

**CONFIRMED by code and completed artifacts:** preprocessing/vocabulary/scaling
and L1 selection in classical pipelines are fit inside training folds. For the
primary transformer, outer test rows are isolated; raw component outer scores
come from held-out prediction tables; inner OOF scores are created only within
each outer fitting design. `analysis.py:455-505` constructs component inner OOF,
fits each empirical CDF on those fitting-side scores (`:481-490`), applies it to
outer scores, and selects thresholds on inner data (`:502-510`). Ensemble weights
are fixed 0.5/0.5, never fit.

Leakage guards are label-blind (`models/transformer_final/leakage.py:71-72,
196-204`). Exact targets, near duplicates (word-trigram Jaccard ≥0.85), shared
lineages, BCP14 skeletons, and target-as-context occurrences produce fitting-side
exclusions/sanitizations while held-out evaluation rows are retained
(`leakage.py:229-327`).

The repair validates exactly ten outer and 30 inner cells
(`...analysis_fix_v1/analysis.py:179-185`) and checks identical, unique outer OOF
coverage (`:199-232`). Recalculation confirms exactly one finite score per method
per one of 2,143 rows.

Runtime/report consistency: seed 2718, FP32, batch 8/eval 16, AdamW, learning rate
1e-5, five epochs, no early stopping, focal gamma 2, and Partial-4 upper-layer
training match `config.json:70-88` and completed `run_config.json` artifacts.

Adaptive history is now disclosed. The V1.1 inner fitting-ID defect was corrected
in V1.2; an earlier outer-score checkpoint screen was retired; later V3/V4 hybrid
choices and the 100-error review were development adaptations. No repository
evidence supports calling these preregistered.

## F. Statistical interpretation

The original uncertainty output uses 2,000 family-cluster bootstrap resamples,
seed 2718. Each replicate samples nine family IDs with replacement and concatenates
all rows for every selected family (`analysis.py:405-431`). AP is undefined only
if no positive is present; ROC-AUC additionally requires both classes. The saved
output has 2,000/2,000 defined replicates for every method/metric. The IPv6 family
has no positives, but its presence does not invalidate a replicate unless all
sampled families jointly lack positives/classes.

Marginal 95% percentile intervals are ensemble AP [0.5920, 0.7641], focal
[0.5355, 0.6900], and Partial-4 [0.5210, 0.6982]. **CORRECTED:** overlap/non-overlap
of separate intervals is not a formal paired model comparison.

A new paired analysis resamples identical family draws for ensemble and each
component (`scripts/analysis/audit_reviewed_transformer_metrics.py`). Results:

- Ensemble−focal AP = 0.0913, exploratory 95% percentile interval [0.0192, 0.1332].
- Ensemble−Partial-4 AP = 0.0974, interval [0.0047, 0.1297].

All 2,000 paired replicates were defined. These intervals describe fixed completed
OOF predictions only. They do not include model-training variation, annotation
uncertainty, selective-review uncertainty, or a superpopulation of protocol
families. Nine clusters, shared CV training structure, and one seed make them
exploratory rather than confirmatory.

## G. Classical and cross-model comparisons

Historical LOFO uses nine held-out families, seed 42, and lexical/engineered/
embedding/context/revision feature sets; current classical uses three
`StratifiedGroupKFold` folds over 2,425 rows. Fixed model settings are in
`models/classical/annotation_classifier_experiments.py:60-83` and the fold-local
fit at `:244`; there is no nested inner hyperparameter search. Choosing the best
displayed feature row after OOF evaluation is adaptive development selection.

SVM “probability” is `sigmoid(LinearSVC.decision_function)` and threshold 0.5 is
margin zero (`annotation_classifier_experiments.py:52,84-90`); it is not calibrated.

MiniLM-only and TF–IDF+MiniLM logistic results do not have exactly identical AP:
0.1438528 versus 0.1438207, with only a tiny ROC-AUC difference. Their displayed
0.5-threshold metrics and fold selected-feature counts (213/213/214) match.
**UNRESOLVED:** saved aggregates do not retain selected feature identities, so
“the sparse block was eliminated or immaterial” is a plausible inference, not a
verified cause.

The 2,097-row cross-model ensemble aligns a current-label Qwen/hybrid component,
a DeBERTa component, and an older-label classical component. It removes 17
missing rows and 28 label disagreements. This makes scores joinable but changes
the population and does not make original training labels/folds identical.
Genuinely matched comparisons are: methods within the current 2,425-row classical
study; focal/Partial-4/ensemble within the primary 2,143-row transformer study;
and Qwen/hybrid rows within their common 2,142-row track. Cross-study score tables
are study maps, not superiority tests.

## H. Revision and alignment evidence

The apparent 242-versus-five TLS discrepancy is a staged filtering result:

1. `tls13_silver_label_report.json` creates 242 automatic **Uncertain silver
   candidates**: 121 errata-supported candidates and 121 stable controls. These
   are not confirmed ambiguity positives.
2. Normativity filtering, lineage deduplication, and latest-representative
   selection produce the 55-row `review/all_candidates_v1` packet: five
   errata-derived plus 50 controls (`manifest.json` fields and README).
3. Reviewer A/B/adjudication CSVs have all 55 rows, but the human judgment fields
   are blank. No completed human ambiguity review was found.
4. An older LOFO artifact happens to contain five errata-source rows, all negative,
   but no identity manifest proves that those are the same five packet rows.

The full TLS alignment artifact covers 30 documents, 29 consecutive pairs, and
52,953 relationships over 52,137 clause-version occurrences. Its 150-link sample
is unadjudicated; accuracy fields are absent/null. Coverage and operational
acceptance therefore do not equal measured link accuracy.

The current classical revision detector uses 12 H0/H1 features from
`models/classical/revision_features.py`: lexical/semantic similarity, change,
length/token deltas, normativity/keyword changes, and predecessor availability
for t−1→t and t−2→t−1. Its one-to-one same-section/heading lexical+MiniLM matcher
does not support the full validated alignment pipeline's split/merge relation
graph. Thus the revision-feature result evaluates a separate simplified matcher.

## I. Prototype and software evidence

**CONFIRMED by safe code inspection:** the server accepts one UTF-8 rendered RFC
`.txt` upload up to 5,000,000 bytes; parses locally; admits only explicit uppercase
BCP14 normative candidates; caps the review set at 5,000 clauses; scores with the
tracked engineered+TF–IDF logistic artifact at threshold 0.5; displays and exports
only predicted-Yes rows as CSV. It binds `127.0.0.1:8765` (`gui/app.py:364-370`).
Document text stays in process memory except OS-managed upload temporaries; no
remote service is required for classification.

Optional explanation generation calls a pinned local Ollama `qwen3:32b` model
for at most five displayed Yes clauses (`gui/app.py:100-218`). It is not part of
the prediction and has no faithfulness evidence. The transformer ensemble is not
integrated.

The active tracked artifact is `results/ML/v2_engineered_tfidf/final_model.joblib`
(SHA-256 `bf30fd62659f003d7e00244d21c7bdeb6374730a6c27caabf941240d93096733`,
size 450,581 bytes); its current aggregate metrics are AP
0.3265, ROC-AUC 0.8082, precision 0.1902, recall 0.7063 at 0.5. **UNRESOLVED:** its
manifest source hash begins `66d901...`, whereas the current annotation source is
`8ffa654...`, despite matching 2,425/160 counts. The artifact is runnable evidence,
but exact source-byte reconstruction is not established.

Safe checks include the synthetic revision-feature, engineered-feature extraction,
and result-plot tests; schema/hash validators; report figure generation; and
static imports that do not load the model. Do **not** run the full GUI suite in
this audit context: `test_rfc_scan_ranks_only_normative_clauses` reads RFC 9460
text. `test_annotation_classifier_experiments.py` also fits synthetic SVM/RF
models and is outside the no-training validation set. This audit did not start
the GUI or load the classifier.

## J. Proposal completion and reproducibility

`docs/final_report_review/PROPOSAL_COMPLETION_MATRIX.md` maps every proposed item.
In summary:

- Gold annotation: **NOT EVALUATED**; operational reference only.
- Family transfer: **CONFIRMED as development evidence**; draft-lineage transfer,
  not broad-suite or prospective transfer.
- Chronological transfer: **NOT EVALUATED**; only feature-time guards exist.
- Revision improvement: **CONFIRMED negative/mixed** on the current classical
  cohort; AP decreases for logistic/SVM/RF, while some F2 changes are small.
- Detector comparison: **PARTLY CONFIRMED** within matched studies only.
- Routed verifier versus score fusion and LLM-call reduction: **NOT EVALUATED**;
  completed hybrids compute component scores for all rows.
- Standardized latency/cost, calibration, explanation faithfulness, and alignment
  accuracy: **NOT EVALUATED**.
- Prototype flow and alignment coverage: **CONFIRMED**.

The proposal foregrounded F2 on a temporal expert-gold test. That test was not
completed. The report's primary outcome changed adaptively to pooled outer-OOF AP
on development rankings, appropriate to rare positives and review prioritization
but not preregistered. Inner-selected precision/recall/F1/F2 remain secondary.

A fresh clone can reproduce static code checks, regenerate evidence-derived
figures when tracked CSVs are present, and validate aggregate tables. It cannot
fully reproduce all experiments without private annotation inputs, the external
offline DeBERTa snapshot at `D:/RFC/models/deberta-v3-base`, large run/checkpoint
artifacts, local Qwen/Ollama assets, and environment-specific historical inputs.
Examiner access to this repository gives stronger internal traceability than
public reproducibility. Exact aggregate claims remain linked to manifests and
hashes, but hashes do not replace absent distributable inputs.

## Confirmed errors and corrections

1. **CORRECTED corpus count:** 253 drafts/12 RFCs was stale; current authority is
   264 drafts/13 RFCs. Updated report, root README, and cohort figure.
2. **CORRECTED threshold prose:** replaced .456/.382/.416 with authoritative
   ensemble .387/.732/.506 and F2 .621; added TP/FP/TN/FN.
3. **CORRECTED fixed-budget reporting:** added recall to Top-20/Top-50 precision.
4. **CORRECTED metric name:** distinguished AP from trapezoidal PR-AUC.
5. **CORRECTED CI interpretation:** removed overlap-as-test claim; added paired
   family bootstrap with explicit exploratory scope.
6. **CORRECTED holdout claim:** TCP/DNS exclusion is experiment-scoped, not
   project-wide.
7. **CORRECTED annotation claim:** human-confirmed/manual-conflict provenance is
   not described as blinded expert adjudication.
8. **CORRECTED TLS silver wording:** 242 candidates, not positives; reconciled
   121 errata-supported candidates to the five-row post-filter packet.
9. **CORRECTED prototype documentation:** current metrics, threshold, local flow,
   and source-hash limitation now match evidence.

## Bibliography check

All 17 citation keys used by `paper/main.tex` have one matching entry in
`paper/references.bib`; no citation key is missing. Primary DOI/proceedings records
were checked for BCP 14 (RFC 2119/8174), SAGE (`10.1145/3452296.3472910`),
NRFCKG (`10.1016/j.isatra.2023.04.025`), PROSPER
(`10.1145/3626111.3628205`), RFCScope (`10.1109/ASE63991.2025.00106`),
TAPHSIR, DeBERTaV3, focal loss, Sentence-BERT, Qwen3, and the AP/ROC references.
No verified bibliographic defect required an edit. The repository artifact entry
is a project self-citation rather than independent peer-reviewed evidence.

## Files changed

- `paper/main.tex`
- `paper/generate_report_figures.py`
- `paper/figures/project_cohorts.png`
- `paper/README.md` (already identified the stale PDF; retained as build authority)
- `README.md`
- `gui/README.md`
- `models/classical/annotation_classifier_experiments.py`
- `models/heuristic/ambiguity_heuristic.py`
- `scripts/analysis/audit_reviewed_transformer_metrics.py`
- `scripts/analysis/prepare_primary_development_baselines.py`
- `scripts/analysis/prepare_blinded_development_review_packet.py`
- `scripts/analysis/prepare_transformer_seed_repetitions.py`
- `docs/final_report_review/REVIEWED_TRANSFORMER_METRIC_AUDIT.csv`
- `docs/final_report_review/PAIRED_FAMILY_BOOTSTRAP.csv`
- `docs/final_report_review/DATASET_STUDY_REGISTRY.csv`
- `docs/final_report_review/REPORT_CLAIM_EVIDENCE.csv`
- `docs/final_report_review/PROPOSAL_COMPLETION_MATRIX.md`
- `docs/final_report_review/REMAINING_EXPERIMENTS.md`
- this handoff

## Validation performed

- Revalidated all ten outer and 30 inner primary prediction artifacts through the
  existing repair code; reconstructed pooled scores without inference.
- Recomputed AP, trapezoidal PR-AUC, ROC-AUC, score direction, missingness, ties,
  and paired family-bootstrap differences from existing permitted development
  OOF predictions.
- Regenerated and visually inspected `project_cohorts.png`,
  `classical_ap_summary.png`, and `revision_feature_ablation.png`; labels, values,
  legends, and clipping were readable.
- Ran AST syntax checks, command-line help smoke checks, and the explicitly
  non-training synthetic tests listed in the report appendix: six passed. The
  first attempt had one pass and five setup errors because pytest could not use
  the sandboxed default temp/cache directory; rerunning with an explicit empty
  workspace `--basetemp` and cache disabled passed all six, after which the audit
  temp directory was removed. No classifier fit, model loading/inference, or
  protected fixture was used.
- Rendered both planned transformer seed packages (1729 and 31415) in memory and
  syntax-checked every generated Python module; no package file, model, or run
  artifact was created.
- TeX/PDF rebuild: **NOT EVALUATED** because `pdflatex`, `latexmk`, `xelatex`, and
  `pdftotext` are absent. The tracked PDF remains stale; no post-edit PDF was
  visually inspected.

## Prioritized next actions

1. Rebuild the report in a TeX environment, review warnings/overflow/citations,
   and visually inspect every page before submission.
2. Have the student/supervisor verify the contribution/acknowledgement wording
   and supply the missing human-review protocol facts, if records exist.
3. Run the prepared matched heuristic/SVM/RF commands on the primary cohort and
   frozen five folds; report them only after artifact validation.
4. If compute permits, run both preselected transformer seed repetitions and
   report all seeds rather than selecting the best.
5. Conduct the prepared two-reviewer development annotation study; reserve a new
   prospective lineage/time block for any confirmatory claim.
6. Do not use TCP/DNS as “untouched” without a formal new reservation decision;
   preferably select genuinely unexamined data.

Exact Windows commands, safeguards, and resource estimates are in
`docs/final_report_review/REMAINING_EXPERIMENTS.md`. No command in that document
was executed during this audit.

## Questions for the student/supervisor

1. Who reviewed the 100 error-selected clauses, what exact fields/UI did each
   reviewer see, were judgments independent, and was there adjudication?
2. What operational process produced the 270 `manual_conflict_verdict` rows, and
   is there a dated protocol or locked reviewer output that can be archived?
3. Is there a verified source snapshot corresponding to the active GUI model's
   `66d901...` manifest hash, and may it be included for examiner reproducibility?
4. Which individual made each documented contribution? Add a contribution
   statement only from verifiable records; do not infer attribution from commits.
5. Does the university require a declaration on AI assistance, data/software
   availability, ethics, or individual contributions beyond the current template?

## Boundary confirmation

This audit performed **no model training or fine-tuning, no model inference, no
GPU job, no LLM/paid API call, no annotation job, no model download, and no
reserved-set evaluation**. It did not open or evaluate reserved TCP/DNS clause
text, row-level labels, predictions, or previously unopened result artifacts.
Historical exposure findings use permitted code, manifests, documentation, and
already-existing aggregate/development evidence only. Nothing was committed,
pushed, published, or uploaded.
