# V4 frozen-semantic ensemble

## Purpose

V4 tests whether dense semantic representations add information that V3's lexical and
Qwen zero-shot signals miss. It preserves V1-V3 and creates a new result namespace.

The method combines:

1. V3's grouped out-of-fold probability (weight 0.80);
2. a target-clause MiniLM semantic classifier (weight 0.10); and
3. a separate target/previous/next MiniLM semantic classifier (weight 0.10).

Scores are combined in log-odds space. `all-MiniLM-L6-v2` is loaded from the pinned
local revision, kept frozen, and used only as a feature encoder. Two class-balanced
logistic regressions are fitted with `C=0.03`. Qwen and MiniLM receive no parameter
updates.

## Validation design

- Semantic classifiers use the same five frozen document-family folds.
- Each scored row is encoded without labels and classified by a model fitted without
  its held-out family.
- Empty previous/next context maps to a zero vector.
- F1, F2, and balanced-accuracy thresholds are reported as leave-one-fold-out
  sensitivity analyses.
- TCP is not opened and DNS remains sealed.

## Important claim boundary

V4 is adaptive exploratory development work. The semantic `C` and ensemble weights
were frozen after inspecting development sweeps, and the V3 component itself was
designed on this cohort. Consequently, V4's development improvement is not an unbiased
estimate of transfer performance. The untouched TCP partition is required to test
whether the improvement generalizes. DNS must remain sealed until the method and TCP
decision rule are frozen.

MiniLM-only and naive MiniLM/Qwen models were also checked during development and did
not clearly beat V3. V4 retains MiniLM only because its errors are complementary when
combined with V3; it does not claim that the semantic model is stronger by itself.

## Run

```powershell
.\scripts\run_llm_in_context_v4.ps1 -Stage all
```

The first run generates a hash-recorded embedding cache from the local model snapshot.
Subsequent matching runs reuse it. The output includes out-of-fold predictions,
per-fold and pooled metrics, cross-fitted threshold sensitivities, confusion matrices,
an adaptive weight-sensitivity table, and V1-V4 comparison artifacts.
