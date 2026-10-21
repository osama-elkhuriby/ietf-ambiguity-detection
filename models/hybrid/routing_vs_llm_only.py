"""Evaluate fixed ML-to-LLM band routing using existing OOF predictions only."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, fbeta_score, precision_score, recall_score, roc_auc_score

from models.hybrid.final_rank_ensemble import DEFAULT_ML_PREDICTIONS, DEFAULT_TRANSFORMER_PREDICTIONS, DEFAULT_V4_PREDICTIONS, _read_predictions

OUT = Path(__file__).resolve().parents[2] / "results/hybrid/routing_vs_llm_only"
LLM_THRESHOLD, ML_THRESHOLD = 0.5, 0.5
PRIMARY_BAND = (0.4, 0.7)
BANDS = ((0.45, 0.55), PRIMARY_BAND, (0.3, 0.8), (0.2, 0.9), (0.0, 1.0))
BOOTSTRAPS, BOOTSTRAP_SEED = 1000, 42
RANDOM_SEEDS = range(200)
TOP_K = (5, 10, 20)
PAIRED_METRICS = ("pr_auc", "f2", "recall_at_20pct", "roc_auc")
CAVEATS = [
    "Exploratory development comparison on OOF predictions; V4 and the models were tuned during development, so this is not a sealed-test result.",
    "LLM answers are simulated from stored V4 probabilities; no new LLM calls were made.",
    "The ML OOF file comes from an earlier annotation artifact, so rows missing from it or with conflicting labels were excluded; report how many.",
    "Labels are an operational reference, not ground truth; scores are rankings, not calibrated probabilities (no calibration metrics were recorded), so the 0.4/0.7 band is a heuristic.",
    'Only report "improves" if the paired CI supports it.',
]

def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()

def _band(p: np.ndarray, low: float, high: float) -> np.ndarray:
    # The full-range sweep sentinel includes p=1 so it exactly represents LLM-only.
    upper = p <= high if (low, high) == (0.0, 1.0) else p < high
    return (p >= low) & upper

def _hybrid(frame: pd.DataFrame, low: float, high: float):
    ml, llm = frame.ml_probability.to_numpy(float), frame.v4_probability.to_numpy(float)
    inside = _band(ml, low, high)
    decisions = np.where(inside, llm >= LLM_THRESHOLD, ml >= high).astype(int)
    positive, negative = ~inside & (ml >= high), ~inside & (ml < low)
    scores = np.zeros(len(frame))
    # Offsets enforce positive > routed > negative; percentile ranks break ties
    # within each tier using the score from the model that decided that clause.
    scores[negative], scores[inside], scores[positive] = (
        _rank(ml[negative]), 1 + _rank(llm[inside]), 2 + _rank(ml[positive])
    )
    return decisions, scores, inside

def _metrics(y, decision, score, calls, inside, ml_decision):
    ap = average_precision_score(y, score)
    result = {
        "precision": precision_score(y, decision, zero_division=0),
        "recall": recall_score(y, decision, zero_division=0),
        "f1": f1_score(y, decision, zero_division=0),
        "f2": fbeta_score(y, decision, beta=2, zero_division=0),
        "pr_auc": ap,
        "roc_auc": roc_auc_score(y, score),
        "ap_lift_over_prevalence": ap / y.mean(),
        "llm_calls": int(calls.sum()),
        "llm_call_fraction": calls.mean(),
        "calls_saved": 1 - calls.mean(),
        "band_clauses": int(inside.sum()),
        "band_positives": int(y[inside].sum()),
    }
    for percent in TOP_K:
        top = np.argsort(-score, kind="stable")[: max(1, int(np.ceil(len(y) * percent / 100)))]
        result[f"recall_at_{percent}pct"] = y[top].sum() / y.sum()
    flipped = decision != ml_decision
    result.update(
        llm_flipped_decisions=int(flipped.sum()),
        llm_flipped_correct=int((flipped & (decision == y)).sum()),
        llm_flipped_wrong=int((flipped & (decision != y)).sum()),
    )
    return {key: float(value) for key, value in result.items()}

def _system(frame: pd.DataFrame, name: str, band=PRIMARY_BAND):
    ml, llm = frame.ml_probability.to_numpy(float), frame.v4_probability.to_numpy(float)
    ml_decision = (ml >= ML_THRESHOLD).astype(int)
    if name == "ML-only":
        decision, score, calls, inside = ml_decision, ml, np.zeros(len(frame), bool), np.zeros(len(frame), bool)
    elif name == "LLM-only":
        decision, score, calls, inside = (llm >= LLM_THRESHOLD).astype(int), llm, np.ones(len(frame), bool), np.zeros(len(frame), bool)
    else:
        decision, score, inside = _hybrid(frame, *band)
        calls = inside
    return dict(decision=decision, score=score, calls=calls, inside=inside, ml_decision=ml_decision)

def _score_metrics(frame, data, indices):
    y = frame.label.to_numpy(int)[indices]
    return _metrics(
        y, data["decision"][indices], data["score"][indices],
        data["calls"][indices], data["inside"][indices], data["ml_decision"][indices],
    )

def _bootstrap_samples(frame):
    rng, families = np.random.default_rng(BOOTSTRAP_SEED), frame.family_id.unique()
    family_rows = {f: np.flatnonzero(frame.family_id.to_numpy() == f) for f in families}
    labels, samples, skipped, attempts = frame.label.to_numpy(int), [], 0, 0
    while len(samples) < BOOTSTRAPS:
        attempts += 1
        if attempts > BOOTSTRAPS * 100:
            raise RuntimeError("Unable to draw enough two-class family bootstrap samples")
        sample = np.concatenate([family_rows[f] for f in rng.choice(families, len(families), replace=True)])
        if np.unique(labels[sample]).size < 2:
            skipped += 1
        else:
            samples.append(sample)
    return samples, skipped

def _bootstrap_results(frame, systems, samples):
    estimates, boot = {}, {}
    full = np.arange(len(frame))
    for name, data in systems.items():
        estimates[name] = _score_metrics(frame, data, full)
        boot[name] = [_score_metrics(frame, data, indices) for indices in samples]
    rows = []
    for name, point in estimates.items():
        for metric, value in point.items():
            vals = [item[metric] for item in boot[name]]
            low, high = np.quantile(vals, [0.025, 0.975])
            rows.append(dict(system=name, metric=metric, estimate=value, ci_low=low, ci_high=high,
                             ci_method="95% family-cluster bootstrap"))
    paired = []
    for baseline in ("LLM-only", "ML-only"):
        for metric in PAIRED_METRICS:
            diffs = [a[metric] - b[metric] for a, b in zip(boot["Hybrid"], boot[baseline], strict=True)]
            low, high = np.quantile(diffs, [0.025, 0.975])
            paired.append(dict(comparison=f"Hybrid - {baseline}", metric=metric,
                               difference=estimates["Hybrid"][metric] - estimates[baseline][metric],
                               ci_low=low, ci_high=high, significant=bool(high < 0 or low > 0),
                               ci_method="paired 95% family-cluster bootstrap"))
    return pd.DataFrame(rows), pd.DataFrame(paired), estimates

def _random_budget(frame, calls):
    y, ml, llm = frame.label.to_numpy(int), frame.ml_probability.to_numpy(float), frame.v4_probability.to_numpy(float)
    ml_decision, outcomes = (ml >= ML_THRESHOLD).astype(int), []
    for seed in RANDOM_SEEDS:
        rng, chosen = np.random.default_rng(seed), np.zeros(len(y), bool)
        chosen[rng.choice(len(y), calls, replace=False)] = True
        decision = np.where(chosen, llm >= LLM_THRESHOLD, ml_decision).astype(int)
        source = np.where(chosen, llm, ml)
        score = np.zeros(len(y))
        for positive in (False, True):
            group = decision == positive
            score[group] = 2 * positive + _rank(source[group])
        outcomes.append(_metrics(y, decision, score, chosen, np.zeros(len(y), bool), ml_decision))
    mean, interval = {}, {}
    for metric in outcomes[0]:
        vals = [row[metric] for row in outcomes]
        mean[metric] = float(np.mean(vals))
        interval[metric] = np.quantile(vals, [0.025, 0.975])
    return mean, interval

def _sweep(frame):
    rows = []
    for low, high in BANDS:
        data = _system(frame, "Hybrid", (low, high))
        result = _score_metrics(frame, data, np.arange(len(frame)))
        rows.append(dict(low=low, high=high, **{key: result[key] for key in
                     ("llm_calls", "llm_call_fraction", "calls_saved", "pr_auc", "f2")}))
    return pd.DataFrame(rows)

def _per_family(frame):
    rows = []
    for family, group in frame.groupby("family_id", sort=True):
        y = group.label.to_numpy(int)
        for name in ("ML-only", "LLM-only", "Hybrid"):
            data = _system(group, name)
            rows.append(dict(family_id=family, system=name, rows=len(group), positives=int(y.sum()),
                             average_precision=average_precision_score(y, data["score"]) if np.unique(y).size == 2 else np.nan,
                             f2=fbeta_score(y, data["decision"], beta=2, zero_division=0)))
    return pd.DataFrame(rows)

def _plot(sweep, metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, metric, title in zip(axes, ("pr_auc", "f2"), ("PR-AUC", "F2"), strict=True):
        ax.plot(sweep.llm_call_fraction, sweep[metric], marker="o", label="Band sweep")
        for name, marker, fraction in (("LLM-only", "s", 1), ("ML-only", "X", 0)):
            value = metrics.loc[(metrics.system == name) & (metrics.metric == metric), "estimate"].iloc[0]
            ax.scatter(fraction, value, marker=marker, s=70, label=name)
        ax.set(xlabel="LLM call fraction", ylabel=title, title=title)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def _verdict(paired):
    rows = paired[paired.comparison == "Hybrid - LLM-only"].set_index("metric")
    ap, f2 = rows.loc["pr_auc"], rows.loc["f2"]
    if ap.ci_low > 0 and f2.ci_low > 0:
        return "improves"
    if ap.ci_high < 0 and f2.ci_high < 0:
        return "degrades"
    if ap.ci_low <= 0 <= ap.ci_high and f2.ci_low <= 0 <= f2.ci_high:
        return "preserves"
    return "has mixed evidence; no categorical verdict is supported"

def _self_check():
    tiny = pd.DataFrame(dict(
        family_id=["a", "a", "b", "b"], label=[0, 1, 1, 0],
        ml_probability=[0.1, 0.4, 0.55, 0.9], v4_probability=[0.2, 0.8, 0.3, 0.7]))
    y = tiny.label.to_numpy(int)
    d, s, inside = _hybrid(tiny, *PRIMARY_BAND)
    assert int(inside.sum()) == 2 and int(y[inside].sum()) == 2
    ml, llm = _system(tiny, "ML-only"), _system(tiny, "LLM-only")
    full, empty = _system(tiny, "Hybrid", (0, 1)), _system(tiny, "Hybrid", (0.5, 0.5))
    assert np.array_equal(full["decision"], llm["decision"])
    full_metrics = _metrics(y, full["decision"], full["score"], full["calls"], full["inside"], full["ml_decision"])
    llm_metrics = _metrics(y, llm["decision"], llm["score"], llm["calls"], llm["inside"], llm["ml_decision"])
    keys = ("precision", "recall", "f1", "f2", "pr_auc", "roc_auc",
            "ap_lift_over_prevalence", "recall_at_5pct", "recall_at_10pct", "recall_at_20pct")
    assert all(full_metrics[key] == llm_metrics[key] for key in keys)
    assert full_metrics["llm_calls"] == len(tiny)
    assert np.array_equal(empty["decision"], ml["decision"])
    assert _metrics(y, empty["decision"], empty["score"], empty["calls"], empty["inside"], empty["ml_decision"]) == \
           _metrics(y, ml["decision"], ml["score"], ml["calls"], ml["inside"], ml["ml_decision"])
    assert d.shape == s.shape == inside.shape

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def run():
    _self_check()
    frame, audit = _read_predictions(DEFAULT_V4_PREDICTIONS, DEFAULT_TRANSFORMER_PREDICTIONS, DEFAULT_ML_PREDICTIONS)
    y, dropped = frame.label.to_numpy(int), audit["v4_rows_missing_from_ml"] + audit["label_mismatch_rows_excluded"]
    print(f"Cohort audit: rows={len(frame)}, positives={y.sum()}, prevalence={y.mean():.4f}, rows dropped={dropped} "
          f"(missing={audit['v4_rows_missing_from_ml']}, label conflicts={audit['label_mismatch_rows_excluded']})")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {OUT}")
    OUT.mkdir(parents=True)
    systems = {name: _system(frame, name) for name in ("ML-only", "LLM-only", "Hybrid")}
    samples, skipped = _bootstrap_samples(frame)
    metrics, paired, estimates = _bootstrap_results(frame, systems, samples)
    random_mean, random_ci = _random_budget(frame, int(systems["Hybrid"]["calls"].sum()))
    extra = [dict(system="Random-budget LLM-only", metric=m, estimate=v,
                  ci_low=random_ci[m][0], ci_high=random_ci[m][1],
                  ci_method="95% interval across 200 random seeds") for m, v in random_mean.items()]
    metrics = pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True)
    sweep, family = _sweep(frame), _per_family(frame)
    hybrid = systems["Hybrid"]
    flipped = hybrid["decision"] != hybrid["ml_decision"]
    pred = frame[["source_item_id", "family_id", "fold", "label", "ml_probability", "v4_probability"]].copy()
    pred["in_band"], pred["hybrid_decision"], pred["hybrid_score"] = hybrid["inside"], hybrid["decision"], hybrid["score"]
    pred["ml_decision"] = hybrid["ml_decision"]
    pred["decision_flipped"] = flipped
    pred["llm_flipped_correct"] = flipped & (hybrid["decision"] == y)
    pred["llm_flipped_wrong"] = flipped & (hybrid["decision"] != y)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    paired.to_csv(OUT / "paired_differences.csv", index=False)
    sweep.to_csv(OUT / "band_sweep.csv", index=False)
    family.to_csv(OUT / "per_family.csv", index=False)
    pred.to_csv(OUT / "predictions.csv", index=False)
    _plot(sweep, metrics, OUT / "routing_tradeoff.png")
    table = metrics[metrics.system.isin(["ML-only", "LLM-only", "Hybrid"])].pivot(
        index="system", columns="metric", values="estimate").loc[["ML-only", "LLM-only", "Hybrid"]]
    table.loc["Random-budget LLM-only"] = metrics[metrics.system == "Random-budget LLM-only"].set_index("metric").estimate.reindex(table.columns)
    columns = ["calls_saved", "pr_auc", "f2", "recall", "precision"]
    verdict = _verdict(paired)
    saved = estimates["Hybrid"]["calls_saved"]
    sentence = f"The primary hybrid {verdict} detection effectiveness versus LLM-only at {saved:.1%} fewer LLM calls under the default (0.4, 0.7) band."
    paired_llm = paired[paired.comparison == "Hybrid - LLM-only"][["metric", "difference", "ci_low", "ci_high", "significant"]]
    summary = f"""# Hybrid routing versus LLM-only
{sentence}
Family-cluster bootstrap: {BOOTSTRAPS} valid resamples; {skipped} one-class resamples skipped.
## Results (calls_saved is the fraction of clauses not sent to the LLM)
{table[columns].to_markdown(floatfmt=".3f")}
The default (0.4, 0.7) band is primary; the remaining bands are sensitivity analyses.
Paired hybrid-versus-LLM-only confidence intervals:
{paired_llm.to_markdown(index=False, floatfmt=".3f")}
The primary band routes {int(hybrid["inside"].sum())} clauses, including {int(y[hybrid["inside"]].sum())} positives. The LLM flips {int(flipped.sum())} ML decisions: {int((flipped & (hybrid["decision"] == y)).sum())} are correct and {int((flipped & (hybrid["decision"] != y)).sum())} are wrong. The "preserves" classification follows the requested CI rule; intervals containing zero do not prove equivalence or formal non-inferiority.
## Band-sweep trade-off
{sweep[["low", "high", "calls_saved", "pr_auc", "f2"]].to_markdown(index=False, floatfmt=".3f")}
`routing_tradeoff.png` plots call fraction against PR-AUC and F2, with ML-only and LLM-only references.
`per_family.csv` reports each family separately; pooled performance does not guarantee transfer.
## Caveats
""" + "\n".join(f"- {item}" for item in CAVEATS) + f"""
Excluded from the aligned cohort: {dropped} rows ({audit["v4_rows_missing_from_ml"]} missing from ML OOF and {audit["label_mismatch_rows_excluded"]} label conflicts).
"""
    (OUT / "summary.md").write_text(summary, encoding="utf-8")
    manifest = dict(
        experiment="routing_vs_llm_only", created_at_utc=datetime.now(UTC).isoformat(),
        input_sha256={"v4": _sha256(DEFAULT_V4_PREDICTIONS),
                      "transformer": _sha256(DEFAULT_TRANSFORMER_PREDICTIONS),
                      "ml": _sha256(DEFAULT_ML_PREDICTIONS)},
        primary_band=PRIMARY_BAND, band_sweep=BANDS,
        constants={"llm_threshold": LLM_THRESHOLD, "ml_threshold": ML_THRESHOLD},
        seeds={"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_resamples": BOOTSTRAPS,
               "random_seeds": list(RANDOM_SEEDS), "one_class_samples_skipped": skipped},
        cohort_audit={**audit, "rows": len(frame), "positives": int(y.sum()),
                      "prevalence": float(y.mean()), "rows_dropped": dropped},
        band_effect={"clauses": int(hybrid["inside"].sum()), "positives": int(y[hybrid["inside"]].sum()),
                     "flipped": int(flipped.sum()), "flipped_correct": int((flipped & (hybrid["decision"] == y)).sum()),
                     "flipped_wrong": int((flipped & (hybrid["decision"] != y)).sum())},
        verdict=verdict, caveats=CAVEATS)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("\nSummary table:\n", table[columns].to_string(float_format="%.3f"))
    print("\n" + sentence)

if __name__ == "__main__":
    run()
