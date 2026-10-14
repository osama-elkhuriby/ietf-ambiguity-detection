"""Analyze the 3-LLM votes. Read-only. Run:  python analyze_votes.py"""
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ANN = Path("/media/yahya/data_stuff/DATA/Final Project/ietf-ambiguity-detection/corpus/processed/llm_annotations/v1")
cons = pd.read_csv(ANN / "consensus.csv", dtype=str)
train = pd.read_csv(ANN / "training_dataset.csv", dtype=str)

# 1) how many rows were dropped as uncertain
print("consensus rows:", len(cons), "| training rows:", len(train), "| dropped:", len(cons) - len(train))
print("\nconsensus label x tier (all rows):")
print(pd.crosstab(cons.llm_is_ambiguous, cons.llm_label_tier))
print("\nconsensus_reason:")
print(cons.consensus_reason.value_counts().to_string())

# 2) parse votes: votes_json = {model: {"status":..., "vote":...}}
models = sorted(json.loads(cons.votes_json.iloc[0]).keys())
print("\nmodels:", models)


def votes_of(row):
    v = json.loads(row)
    return [v[m]["vote"] if v[m]["status"] == "valid" else "invalid" for m in models]


cons["votes"] = cons.votes_json.apply(votes_of)

# 3) vote pattern for the training positives (label yes) and negatives
def pattern(votes):
    c = Counter(votes)
    return " + ".join(f"{c[k]} {k}" for k in ("yes", "no", "uncertain", "invalid") if c[k])


in_train = cons[cons.record_id.isin(train.record_id)]
for lab in ("yes", "no"):
    sub = in_train[in_train.llm_is_ambiguous == lab]
    print(f"\nVote patterns for training rows labeled '{lab}' (n={len(sub)}):")
    print(sub.votes.apply(pattern).value_counts().to_string())

# 4) which model is the dissenter in non-unanimous training rows
dissent = Counter()
for _, r in in_train.iterrows():
    maj = r.llm_is_ambiguous
    for m, v in zip(models, r.votes):
        if v != maj:
            dissent[m] += 1
print("\nDissent count per model (vote != consensus, training rows):", dict(dissent))

# 5) agreement among rows where all 3 models gave a valid vote
valid = cons[cons.votes.apply(lambda v: "invalid" not in v)]
cats = ["yes", "no", "uncertain"]
print(f"\nrows with 3 valid votes: {len(valid)} of {len(cons)}")


def fleiss(votes_list):
    table = np.array([[v.count(c) for c in cats] for v in votes_list], dtype=float)
    n = table.shape[1] and table.sum(axis=1)[0]
    p_j = table.sum(axis=0) / table.sum()
    p_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    pe, po = (p_j ** 2).sum(), p_i.mean()
    return (po - pe) / (1 - pe)


def cohen(a, b):
    a, b = np.array(a), np.array(b)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe)


vl = list(valid.votes)
print(f"Fleiss kappa (yes/no/uncertain): {fleiss(vl):.3f}")
for (i, mi), (j, mj) in combinations(enumerate(models), 2):
    print(f"Cohen kappa {mi} vs {mj}: {cohen([v[i] for v in vl], [v[j] for v in vl]):.3f}")
print("yes-rate per model:", {m: round(np.mean([v[i] == 'yes' for v in vl]), 3) for i, m in enumerate(models)})