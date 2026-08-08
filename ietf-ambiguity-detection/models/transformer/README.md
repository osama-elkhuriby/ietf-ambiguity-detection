# Fine-tuned transformer

Per proposal SS6.1: a compact encoder (DeBERTa or RoBERTa) trained for
ambiguous/unambiguous classification, and — where data permits — ambiguity-subtype
prediction. Uses PyTorch + Hugging Face Transformers (SS6.4).

Don't start fine-tuning until the pilot (see root README) has produced enough labeled
examples to know whether the label scheme itself is stable — a model trained on a
label scheme that changes in week 3 is wasted compute.
