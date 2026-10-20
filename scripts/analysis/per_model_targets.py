"""Train/evaluate one classifier per LLM's own decisions and ordinal vote count.

Votes are TARGETS only. Features = target-clause TF-IDF + engineered (E0 feature set),
all vectorizers/scalers fit on the training families of each fold only.
Run: python -m scripts.analysis.per_model_targets.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "corpus" / "processed"
ANN, FEAT = BASE / "llm_annotations" / "v1", BASE / "features" / "v3"
OUT = ROOT / "results" / "per_model_targets"
OUT.mkdir(parents=True, exist_ok=True)
MIN_FAMILY, N_BOOT, SEED = 30, 1000, 42
MODELS = {"gemma": "gemma3:27b", "gptoss": "gpt-oss:20b", "qwen": "qwen3:32b"}

ENG = ["basic_word_count", "basic_character_count", "basic_sentence_count", "vague_term_count", "anaphora_count",
       "cross_reference_count", "conditional_count", "exception_count", "temporal_count", "ordering_count",
       "normative_bcp14_count", "normative_bcp14_unique_count", "normative_obligation_count",
       "normative_recommendation_count", "normative_permission_count", "normative_not_recommended_count",
       "normative_should_not_count", "normative_shall_not_count", "normative_must_not_count",
       "normative_recommended_count", "normative_required_count", "normative_optional_count",
       "normative_should_count", "normative_shall_count", "normative_must_count", "normative_may_count",
       "syntax_mean_dependency_distance", "syntax_max_dependency_depth", "syntax_subordinate_clause_count",
       "syntax_coordinate_conjunction_count", "context_previous_word_count", "context_previous_character_count",
       "context_previous_sentence_count", "context_next_word_count", "context_next_character_count",
       "context_next_sentence_count", "context_section_depth"]

# ---------------- data ----------------
df = pd.read_csv(ANN / "training_dataset.csv", dtype=str)
cons = pd.read_csv(ANN / "consensus.csv", dtype=str)[["record_id", "votes_json"]]
eng = pd.read_csv(FEAT / "audit" / "engineered_features.csv", dtype={"record_id": str})
df = df.merge(cons, on="record_id", validate="one_to_one").merge(eng[["record_id"] + ENG], on="record_id", validate="one_to_one")
df["label"] = df["label"].astype(int)
votes = df.votes_json.apply(json.loads)
for key, name in MODELS.items():
    df[f"valid_{key}"] = votes.apply(lambda v: v[name]["status"] == "valid")
    df[f"yes_{key}"] = votes.apply(lambda v: int(v[name]["status"] == "valid" and v[name]["vote"] == "yes"))
df["all3_valid"] = df[[f"valid_{k}" for k in MODELS]].all(axis=1)
df["yes_count"] = df[[f"yes_{k}" for k in MODELS]].sum(axis=1)
df["pattern"] = votes.apply(lambda v: " + ".join(f"{n} {k}" for k in ("yes", "no", "uncertain", "invalid")
                            for n in [sum((x["vote"] if x["status"] == "valid" else "invalid") == k for x in v.values())] if n))

sizes = df.family_id.value_counts()
EVAL = sorted(f for f in sizes.index if sizes[f] >= MIN_FAMILY)
print("rows:", len(df), "| evaluated families:", len(EVAL), "| rows evaluated:", int(df.family_id.isin(EVAL).sum()))

# ---------------- per-fold features (fit on training families only) ----------------
folds = {}
for fam in EVAL:
    te = (df.family_id == fam).to_numpy()
    tr = ~te
    tfidf = TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=2,
                            max_features=20000, sublinear_tf=True, token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9_-]*\b")
    imp, sc = SimpleImputer(strategy="median"), StandardScaler()
    Xe_tr = sc.fit_transform(imp.fit_transform(df.loc[tr, ENG]))
    Xe_te = sc.transform(imp.transform(df.loc[te, ENG]))
    X_tr = sparse.hstack([tfidf.fit_transform(df.loc[tr, "clause_text"]), sparse.csr_matrix(Xe_tr)], format="csr")
    X_te = sparse.hstack([tfidf.transform(df.loc[te, "clause_text"]), sparse.csr_matrix(Xe_te)], format="csr")
    folds[fam] = (tr, te, X_tr, X_te)


def oof_scores(y, train_mask, kind):
    """Out-of-fold scores for every row of every evaluated family. Trains only on rows in train_mask."""
    scores = np.full(len(df), np.nan)
    for fam, (tr, te, X_tr, X_te) in folds.items():
        fit_rows = train_mask[tr]
        if kind == "clf":
            m = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr[fit_rows], y[tr][fit_rows])
            scores[te] = m.predict_proba(X_te)[:, 1]
        else:
            m = Ridge(alpha=1.0).fit(X_tr[fit_rows], y[tr][fit_rows])
            scores[te] = m.predict(X_te)
    return scores


def boot_ci(y, s, fn, n=N_BOOT):
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() > 0:
            vals.append(fn(y[i], s[i]))
    return np.percentile(vals, [2.5, 97.5])


def evaluate(name, y, s, valid, binary=True):
    ev = df.family_id.isin(EVAL).to_numpy() & valid
    yy, ss, fam = y[ev], s[ev], df.family_id.to_numpy()[ev]
    prev = yy.mean()
    ap = average_precision_score(yy, ss)
    lo, hi = boot_ci(yy, ss, average_precision_score)
    fam_ap = {f: average_precision_score(yy[fam == f], ss[fam == f]) for f in EVAL if yy[fam == f].sum() > 0}
    top = np.argsort(-ss)[: int(0.2 * len(ss))]
    p, r, f1, _ = precision_recall_fscore_support(yy, ss >= 0.5, average="binary", zero_division=0)
    row = dict(target=name, n=len(yy), positives=int(yy.sum()), prevalence=round(prev, 3), AP=round(ap, 3),
               AP_CI=f"{lo:.3f}-{hi:.3f}", lift=round(ap / prev, 2), mean_family_AP=round(np.mean(list(fam_ap.values())), 3),
               sd_family_AP=round(np.std(list(fam_ap.values())), 3), precision_at_0_5=round(p, 3),
               recall_at_0_5=round(r, 3), F1_at_0_5=round(f1, 3), recall_top20=round(yy[top].sum() / yy.sum(), 3),
               CM=confusion_matrix(yy, ss >= 0.5, labels=[0, 1]).ravel().tolist())
    return row, fam_ap


# ---------------- binary targets ----------------
targets = {"cons": (df.label.to_numpy(), np.ones(len(df), bool))}
for key in MODELS:
    targets[key] = (df[f"yes_{key}"].to_numpy(), df[f"valid_{key}"].to_numpy())

rows, fam_rows, scores = [], [], {}
for name, (y, valid) in targets.items():
    scores[name] = oof_scores(y, valid, "clf")
    row, fam_ap = evaluate(name, y, scores[name], valid)
    rows.append(row)
    fam_rows += [dict(target=name, family=f, AP=round(v, 3)) for f, v in fam_ap.items()]
    print(row)
res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
pd.DataFrame(fam_rows).to_csv(OUT / "per_family_ap.csv", index=False)

# ---------------- ordinal: number of yes votes (rows with 3 valid votes) ----------------
y_cnt = df.yes_count.to_numpy().astype(float)
valid3 = df.all3_valid.to_numpy()
s_cnt = oof_scores(y_cnt, valid3, "reg")
ev = df.family_id.isin(EVAL).to_numpy() & valid3
rho = spearmanr(y_cnt[ev], s_cnt[ev])[0]
rng = np.random.default_rng(SEED)
rhos = [spearmanr(y_cnt[ev][i], s_cnt[ev][i])[0] for i in (rng.integers(0, ev.sum(), ev.sum()) for _ in range(N_BOOT))]
ord_rows = [dict(metric="Spearman", value=round(rho, 3), CI="%.3f-%.3f" % tuple(np.percentile(rhos, [2.5, 97.5])))]
for thr in (1, 2):
    yb = (y_cnt >= thr).astype(int)
    ap = average_precision_score(yb[ev], s_cnt[ev])
    ord_rows.append(dict(metric=f"AP count>={thr}", value=round(ap, 3), CI=f"prev={yb[ev].mean():.3f}, lift={ap / yb[ev].mean():.2f}"))
pd.DataFrame(ord_rows).to_csv(OUT / "ordinal.csv", index=False)
print(pd.DataFrame(ord_rows).to_string(index=False))

# ---------------- cross-target: score of model A against target B ----------------
cross = []
for a in targets:
    for b, (yb, vb) in targets.items():
        e = df.family_id.isin(EVAL).to_numpy() & vb
        ap = average_precision_score(yb[e], scores[a][e])
        cross.append(dict(trained_on=a, evaluated_on=b, AP=round(ap, 3), lift=round(ap / yb[e].mean(), 2)))
cross = pd.DataFrame(cross)
cross.to_csv(OUT / "cross_target.csv", index=False)
print(cross.pivot(index="trained_on", columns="evaluated_on", values="lift").to_string())

# ---------------- consensus-score by vote pattern ----------------
e = df.family_id.isin(EVAL).to_numpy()
pat = pd.DataFrame({"pattern": df.pattern[e], "score": scores["cons"][e]}).groupby("pattern").score.agg(["count", "mean"]).round(3)
pat.to_csv(OUT / "pattern_breakdown.csv")
print(pat.to_string())
print("\nSanity: 'cons' AP should be close to the earlier E0 (0.291); small differences come from TF-IDF settings.")