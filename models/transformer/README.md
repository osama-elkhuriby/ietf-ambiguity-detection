# Fine-tuned transformer

The transformer stage is a leakage-controlled binary ambiguity classifier. A
multi-annotator human set is not currently feasible, so frozen LLM-consensus labels
are the project's operational proxy gold standard. Results measure agreement with
that reference source; they are not human-verified ambiguity-detection accuracy.

## Current implementation status

First derive the reviewed family split without changing the original LLM annotation
dataset, then materialize immutable transformer inputs:

```powershell
python -m corpus.build_model_dataset
python -m models.transformer.dataset
python -m models.transformer.seal_dataset
```

`corpus/processed/transformer_inputs/v3/` is the audited staging artifact. The
training-ready artifact is
`corpus/processed/transformer_partitioned/v1/development/` and contains only:

- `train.csv`: the nine-family fitting/CV partition.
- `validation.csv`: TCP, reserved for independent calibration/routing verification.
- `manifest.json`: hashes and an allowlist containing only those two roles.

DNS is under the separate `sealed_test/` sibling. Only `final_evaluate.py` contains
that path. It requires `--final`, claims an exclusive lock before reading DNS, and
refuses every second run.

An unreviewed exact target-text duplicate across splits marks the artifact
`training_blocked=true`. Review legitimate duplicates and pass a JSON allowlist with
`--duplicate-allowlist`; do not silently delete or ignore them. Output directories are
immutable, so rebuilding requires a new versioned path.

The v2 split policy uses nine training families, TCP only to verify frozen calibration
and routing, and DNS SVCB/HTTPS as the sealed test family. Its full audit reports no
exact or near cross-split duplicate and no target-as-cross-split-context overlap. The
lineage audit also reports no shared draft/RFC chain or copied non-boilerplate normative
text. The superseded v1 audit found HTTP/2bis overlap with QUIC HTTP/Transport and
remains a local audit record, not a trainable dataset.

Audit the pinned tokenizer without modifying the immutable dataset directory:

```powershell
python -m models.transformer.tokenization `
  --output corpus/processed/transformer_audits/v4/token_length_report_256.json
```

For the pinned DeBERTa tokenizer, the 2,253 training targets have a maximum encoded
length of 131 tokens. Target plus local context has p99 185.48 and maximum 495; at a
256-token cap no target is truncated and context is truncated for seven rows.

The primary encoder is `microsoft/deberta-v3-small` at the immutable revision pinned
in each YAML config. The trainer reads only the development `train.csv`, verifies its
hash and the committed family-to-fold assignment, computes loss weights from each
fitting fold, and excludes exact train/validation text matches from fold scoring. It
saves both raw logits for every held-out prediction plus a hashed model/tokenizer
checkpoint and run manifest.

Run a cheap end-to-end architecture check with a randomly initialized tiny DeBERTa:

```powershell
python -m models.transformer.train `
  --output results/transformer/studies/smoke-clause-only-v2 `
  --learning-rate 1e-5 --fold 1 --smoke
```

Run one pinned pretrained development cell with:

```powershell
python -m models.transformer.train `
  --output results/transformer/studies/clause-lr1e-5-fold1-seed1729 `
  --learning-rate 1e-5 --fold 1 --weight-strategy unweighted `
  --precision fp32 --allow-download
```

Run directories are immutable. `--allow-download` is required only to populate a
missing pinned Hugging Face snapshot; normal reproducible runs should use the local
snapshot. The full bounded experiment sequence and decision rules are in
`experiments/transformer/EXPERIMENT_PLAN.md`.

## Corrected frozen learning-rate screen

The first screen (`lr-screen-v1`) is superseded: it used the held-out outer fold for
early stopping and checkpoint selection. The corrected trainer uses the hash-pinned
`inner_selection_v1.json` family assignment for selection, then evaluates the outer
fold once after loading the selected checkpoint. The manifest records zero outer-fold
evaluations during training, and structural tests enforce that ordering.

The corrected target-only, unweighted fp32 screen covers learning rates 1e-5, 2e-5,
and 3e-5 over all five folds at seed 1729. Analyze completed immutable run cells with:

```powershell
python -m models.transformer.lr_screen_analysis `
  --run-root results/transformer/studies/lr-screen-v2 `
  --output results/transformer/studies/lr-screen-v2/analysis-primary-final
```

The frozen one-standard-error rule selected 1e-5 and required seed 2718 reruns for
only the top two rates. The second seed again ranked 1e-5 above 2e-5. Analysis artifacts
contain pooled raw-logit OOF predictions, confidence-tier results, paired row
bootstraps, section-cluster bootstraps, protocol-version comparisons, and per-cell
runtime/GPU memory. Run `models.transformer.protocol_comparison` only into a new
immutable output directory when regenerating those comparisons.

## Step 2c feasibility audit

Routing feasibility, two-seed variance, and inner-validation diagnostics can be
regenerated from saved development OOF predictions without loading a model:

```powershell
python -m models.transformer.feasibility_audit `
  --output results/transformer/governance/step2c_feasibility_v2
```

The v1 audit uses 2,000 within-fold bootstrap replicates and saves the complete
risk-coverage curves, pooled and per-fold referral operating points, seed variance,
ensemble comparisons, and selected epochs. At 90% ambiguous recall, unweighted TF-IDF
avoids about 25% of LLM calls and the current two-seed DeBERTa logit average avoids
about 18%. These are uncalibrated development estimates; TCP verification remains a
later frozen-threshold step.

## Step 2d ceiling and error audit

The CPU-only audit reads rationales-free vote fields from the annotation-source
artifact, filters them to the frozen development allowlist, and never exposes votes to
training or inference code:

```powershell
python -m models.transformer.ceiling_audit `
  --output results/transformer/governance/step2d_ceiling_v1
```

It writes leave-one-LLM-out agreement metrics, pairwise agreement and Cohen's kappa,
score/vote-fraction rank correlations, and a rationale-free CSV of extreme OOF errors
for manual review. These are approximate silver-label reproducibility bounds, not a
human-performance ceiling.

## Step 3 weighting analysis

After all frozen LR-1e-5 cells for seeds 1729, 2718, and 31415 are complete, analyze
the immutable predictions with:

```powershell
python -m models.transformer.weighting_analysis `
  --output-dir results/transformer/studies/weighting-ablation-v1/analysis-final
```

The analysis validates every run against the frozen protocol, averages raw log-odds
across seeds, applies the pre-registered one-standard-error selection rule, reports
the three required weighting contrasts and direct TF-IDF comparisons with paired-row
and section-cluster intervals, and records 90%-recall referral rates. ECE and Brier
scores are explicitly secondary because these probabilities have not been calibrated.

## Step 4 soft-target/context factorial

Build the development-only soft-target and fold-specific context-sanitation artifacts
before training:

```powershell
python -m models.transformer.factorial_artifacts
```

Soft targets use the positive fraction among valid rationales-free votes, with at
least two valid votes required. Context sanitation blanks a previous/next clause when
it exactly or nearly reproduces a target in another nested-CV role. Neither votes nor
vote fractions become input features. The trainer keeps checkpoint selection on hard
inner-family labels so all four conditions use the same selection target.

Run the three missing conditions (the Step 3 class-weighted target-only condition is
reused as the control), then create the matched-data TF-IDF comparator and analysis:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts/run_transformer_factorial.ps1
python -m models.classical.matched_tfidf
python -m models.transformer.factorial_analysis
```

The factorial compares hard/soft targets crossed with target-only/local-context input
at LR 1e-5, fp32, three seeds, and five frozen folds. DNS and TCP are not read during
this step. Selection uses seed-averaged mean-fold PR-AUC and the pre-registered
one-standard-error simplicity rule; pooled performance is supporting evidence only.
