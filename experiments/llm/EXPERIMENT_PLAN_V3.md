# V3 nested lexical/Qwen hybrid experiment

## Motivation

V2 overcorrected V1's false-positive problem. Its best raw condition reached only
PR-AUC 0.103, while V1's eight-shot condition reached PR-AUC 0.196 but had precision
0.098. V3 therefore changes the method rather than making another small prompt edit.

V3 combines two complementary signals:

1. the immutable, label-independent Qwen3 32B V1 zero-shot probability; and
2. a supervised word-and-character TF-IDF logistic classifier fitted only to the
   target clause.

The two probabilities are combined in log-odds space. Qwen is not fine-tuned. The
lightweight lexical classifier is trained, so V3 must be described as a hybrid
supervised experiment rather than a pure zero/few-shot result.

## Leakage controls

- The existing five document-family folds remain frozen.
- Every outer-fold lexical model is fitted without the held-out families.
- Logistic-regression `C` and the lexical/Qwen blend weight are selected using only
  inner out-of-fold predictions from the remaining four folds.
- The Qwen input is the frozen zero-shot artifact, which used no labeled examples and
  is therefore independent of all annotation labels.
- F1, F2, and balanced-accuracy thresholds are learned from the selected inner
  predictions and then applied to the untouched outer fold.
- TCP is not opened. DNS remains sealed.

The V1 eight-shot score is intentionally not used as the stacking input: although it
is out-of-fold for each scored row, demonstrations for one training fold can come from
another outer fold. Reusing it in nested stacking would therefore weaken the strict
outer-fold isolation.

## Fixed search space

- target-only word TF-IDF: 1-2 grams, `min_df=2`, at most 30,000 features;
- target-only character TF-IDF: 3-5 `char_wb` grams, `min_df=3`, at most 30,000
  features;
- class-balanced logistic regression;
- `C`: 0.03, 0.1, 0.3, 1.0, or 3.0;
- lexical blend weight: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, or 1.0;
- primary selection metric: inner grouped PR-AUC.

## Reporting

The run saves pooled and per-fold precision, recall, F1, F2, accuracy, balanced
accuracy, PR-AUC, confusion matrices, precision-recall curves, selected
hyperparameters, and direct V1/V2/V3 comparison artifacts. Because V3 was designed
after examining V1 and V2 development results, its development metrics are adaptive
exploratory evidence, not final test evidence.

Run from PowerShell:

```powershell
.\scripts\run_llm_in_context_v3.ps1 -Stage all
```

This experiment reuses the frozen Qwen zero-shot scores, so it does not call Ollama
and should finish in minutes rather than hours.
