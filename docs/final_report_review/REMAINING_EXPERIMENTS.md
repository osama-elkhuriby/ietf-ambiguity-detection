# Remaining experiments — prepared, not executed

Status: **PREPARED / NOT EVALUATED**. None of the commands labelled TRAINING or
INFERENCE below was run during this audit. Run from
`D:\RFC\ietf-ambiguity-detection` in PowerShell. The verified project interpreter
is `C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe` (Python
3.11.16). All paths are offline repository/local-model paths.

## 1. Matched heuristic and classical baselines

Purpose: evaluate deterministic heuristic, linear SVM, and random forest on the
same frozen 2,143-row/138-positive development cohort and the same five outer
family folds as the primary transformer study. `--fold-column validation_fold`
is essential: omitting it silently returns to the historical three-fold design.

The conversion is schema-only and has hard guards for the source SHA-256,
cohort counts, family membership, folds, and TCP/DNS exclusion.

```powershell
$python = 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe'
$input = 'results\planned\primary_development_baselines\primary_development_input.csv'

# SAFE PREPARATION — no fitting or inference
& $python scripts\analysis\prepare_primary_development_baselines.py

# EVALUATION ONLY — deterministic rules; no model fitting
& $python -m models.heuristic.ambiguity_heuristic --input $input --fold-column validation_fold --output-dir results\planned\primary_development_baselines\heuristic

# TRAINING — CPU classical OOF evaluation plus final all-row fitted artifact
& $python -m models.classical.annotation_classifier_experiments --input $input --fold-column validation_fold --skip-final-fit --classifier linear_svm --feature-mode engineered_tfidf --output-dir results\planned\primary_development_baselines\classical
& $python -m models.classical.annotation_classifier_experiments --input $input --fold-column validation_fold --skip-final-fit --classifier random_forest --feature-mode engineered_tfidf --output-dir results\planned\primary_development_baselines\classical
```

Before reporting, verify that all three prediction files contain exactly one row
per item, validation folds match the input, and compare pooled AP on identical
item IDs. Threshold metrics must state that the SVM score is a sigmoid transform
of an uncalibrated margin. Expected resource envelope: heuristic under one CPU
minute; each classical run roughly 5–30 CPU minutes and 4–12 GB RAM depending on
TF–IDF sparsity. These are planning estimates, not observed measurements.

## 2. Selected transformer seed repetitions

Rationale: the primary transformer result has one training seed. Repeat the
complete focal and Partial-4 nested protocol at seeds 1729 and 31415; do not
change labels, folds, hyperparameters, threshold policy, or ensemble rule. Each
new seed gets a separately versioned package so existing frozen evidence is not
modified.

```powershell
$python = 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe'

# SAFE VALIDATION — renders/syntax-checks both packages in memory only
& $python scripts\analysis\prepare_transformer_seed_repetitions.py --seed 1729 --seed 31415 --validate-only

# SAFE PREPARATION — copies/version-controls source and config only
& $python scripts\analysis\prepare_transformer_seed_repetitions.py --seed 1729 --seed 31415

# SAFE PREFLIGHT/PREPARATION — hashes local inputs and creates cell specifications; no training
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed1729_v1.cli preflight
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed1729_v1.cli prepare
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed31415_v1.cli preflight
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed31415_v1.cli prepare

# TRAINING — GPU, 40 explicit cells per seed (10 outer + 30 inner)
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed1729_v1.cli train-outer --all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed1729_v1.cli train-inner --all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed31415_v1.cli train-outer --all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed31415_v1.cli train-inner --all

# CPU ANALYSIS — consumes only completed new-seed artifacts
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed1729_v1.cli validate-outer --require-all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed1729_v1.cli validate-inner --require-all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed1729_v1.cli analyze --bootstrap-replicates 2000
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed31415_v1.cli validate-outer --require-all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed31415_v1.cli validate-inner --require-all
& $python -m experiments.transformer_final.final_annotation_human_reviewed_100_seed31415_v1.cli analyze --bootstrap-replicates 2000
```

The completed seed-2718 run used 8,737.2 aggregate GPU-seconds (2.43 hours) over
40 sequential cells, with maximum recorded allocation/reservation of 3,976/5,844
MB on an RTX 5000 Ada; its run tree is 27.8 GiB. Two repetitions therefore plan
for about 4.9 GPU-hours and 56 GiB if runtime/storage scale similarly. Those are
extrapolations, not measurements of the new seeds. Preserve per-seed results and
report dispersion; do not choose the best seed.

## 3. Blinded development annotation packet

Purpose: measure agreement with the frozen ambiguity definition, not detector
performance or corpus prevalence. The script samples up to ten Yes and ten No
reference cases per development family, randomizes reviewer order separately,
and places identities/reference labels only in `key_PRIVATE.csv`. It cannot read
TCP/DNS because their families are hard-rejected.

```powershell
$python = 'C:\Users\mosta\.conda\envs\ietf-ambiguity-detection\python.exe'

# SAFE PACKET PREPARATION — no model fitting/inference; do not give the key to reviewers
& $python scripts\analysis\prepare_blinded_development_review_packet.py --per-stratum 10 --seed 20261021
```

Use two independent reviewers, lock both original CSVs before comparison, and
adjudicate disagreements in a third file without editing either review. Report
raw agreement and a chance-corrected coefficient with uncertainty, plus the
Yes/No/Uncertain confusion table. Because sampling is balanced by the existing
reference label and uses development data, it is not an independent generalization
test and cannot estimate prevalence.

### Future independent evaluation protocol (do not instantiate now)

1. Freeze the definition, sampling frame, identity/deduplication rules, reviewer
   instructions, adjudication rule, primary metric, threshold, and analysis code
   before any protected labels or predictions are opened.
2. Have a data custodian—not a model developer—sample one untouched future
   lineage/time block. TCP/DNS may be candidates only after the supervisor
   explicitly reclassifies their reservation and records who accessed what and
   when. This audit did not do so.
3. Reviewers see only clause/context and an anonymous case ID. They must not see
   model outputs, old labels, candidate source, or each other's decision.
4. Lock independent judgments; adjudicate separately; retain Uncertain rather
   than coercing unresolved cases into No.
5. Execute exactly once. Report AP and predeclared fixed-budget/operating-point
   metrics with confusion counts, and disclose every exclusion.

## 4. Optional efficiency and explanation-quality study

The GUI supports local Qwen explanations for at most five displayed positive
predictions, so a limited study is meaningful only after a human packet and test
protocol are frozen. Record end-to-end parser/classifier latency separately from
optional explanation latency; warm-up, hardware, repetitions, and peak memory
must be logged. For explanation quality, reviewers should score factual grounding,
whether both materially different behaviours are stated, and unsupported claims.

This is **NOT YET EXECUTION-READY**: the repository has no batch explanation
evaluation harness or frozen rubric, and the current route would invoke local
`qwen3:32b` inference. Do not use GUI requests as evidence. First implement a
fixture-driven harness that stores prompts, model digest, raw/validated outputs,
timings, failures, and reviewer-blinded IDs. Then obtain explicit authorization
before running local inference. No paid API is needed or permitted.

## Stop conditions

- Never point any command at reserved TCP/DNS data or old protected predictions.
- Never overwrite a completed experiment directory; version a new package.
- Stop on source, model, fold, or protocol hash mismatch.
- Do not regenerate the report as a side effect of training/inference.
- Keep preparation, GPU training, analysis, and report compilation as separate
  explicit commands.
