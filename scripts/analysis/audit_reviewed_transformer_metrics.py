"""Audit completed reviewed-cohort transformer predictions without inference.

This utility reads only the nine-family development experiment's frozen dataset
and already-written OOF predictions.  It writes aggregate checks; it deliberately
does not export row-level labels, text, or scores.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.transformer_final.final_annotation_human_reviewed_100_v1 import analysis as parent
from experiments.transformer_final.final_annotation_human_reviewed_100_v1 import core
from experiments.transformer_final.final_annotation_human_reviewed_100_v1_analysis_fix_v1 import analysis as repair


DEFAULT_OUTPUT = ROOT / "docs" / "final_report_review"
METHODS = (*core.METHODS, "ensemble")


def reconstruct_pooled_scores() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Recreate parent analysis score transformations from frozen OOF artifacts."""
    repair.verify_parent_artifacts()
    dataset = parent._load_dataset()
    outer_raw = {method: parent._load_predictions("outer", method) for method in core.METHODS}
    inner_raw = {method: parent._load_predictions("inner", method) for method in core.METHODS}
    outer_scores: dict[str, dict[int, pd.DataFrame]] = {method: {} for method in METHODS}

    for outer_fold in range(1, 6):
        inner_scores: dict[str, pd.DataFrame] = {}
        for method in core.METHODS:
            inner = inner_raw[method][outer_fold].copy()
            inner["yes_score"] = inner["probability_yes"].astype(float)
            inner_scores[method] = inner

            outer = repair.merge_metadata(outer_raw[method][outer_fold], dataset)
            outer["yes_score"] = outer["probability_yes"].astype(float)
            outer_scores[method][outer_fold] = outer

        focal_inner = inner_scores["focal"][["record_id", "label", "yes_score"]].rename(
            columns={"yes_score": "focal_score"}
        )
        partial_inner = inner_scores["partial4"][["record_id", "label", "yes_score"]].rename(
            columns={"yes_score": "partial4_score"}
        )
        paired_inner = focal_inner.merge(
            partial_inner, on=["record_id", "label"], validate="one_to_one"
        )

        focal_outer = outer_scores["focal"][outer_fold].rename(
            columns={"yes_score": "focal_score"}
        )
        partial_outer = outer_scores["partial4"][outer_fold][["record_id", "yes_score"]].rename(
            columns={"yes_score": "partial4_score"}
        )
        paired_outer = focal_outer.merge(partial_outer, on="record_id", validate="one_to_one")
        paired_outer["yes_score"] = 0.5 * parent.empirical_cdf(
            paired_inner["focal_score"], paired_outer["focal_score"]
        ) + 0.5 * parent.empirical_cdf(
            paired_inner["partial4_score"], paired_outer["partial4_score"]
        )
        outer_scores["ensemble"][outer_fold] = paired_outer

    pooled = {
        method: pd.concat(list(outer_scores[method].values()), ignore_index=True)
        for method in METHODS
    }
    expected_ids = set(dataset["record_id"].astype(str))
    for method, frame in pooled.items():
        if len(frame) != len(dataset) or frame["record_id"].duplicated().any():
            raise RuntimeError(f"OOF coverage is not exactly once per row for {method}")
        if set(frame["record_id"].astype(str)) != expected_ids:
            raise RuntimeError(f"OOF record set differs from frozen cohort for {method}")
        if not np.isfinite(frame["yes_score"].to_numpy(float)).all():
            raise RuntimeError(f"Non-finite OOF scores found for {method}")
    return dataset, pooled


def metric_audit(dataset: pd.DataFrame, pooled: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, frame in pooled.items():
        labels = frame["label"].to_numpy(int)
        scores = frame["yes_score"].to_numpy(float)
        precision, recall, _ = precision_recall_curve(labels, scores)
        unique_scores = np.unique(scores)
        rows.append(
            {
                "method": method,
                "rows": len(frame),
                "positives": int(labels.sum()),
                "negatives": int((labels == 0).sum()),
                "families": int(frame["family_id"].nunique()),
                "unique_record_ids": int(frame["record_id"].nunique()),
                "missing_scores": int(pd.isna(frame["yes_score"]).sum()),
                "nonfinite_scores": int((~np.isfinite(scores)).sum()),
                "unique_scores": int(len(unique_scores)),
                "rows_in_score_ties": int(len(scores) - len(unique_scores)),
                "average_precision": float(average_precision_score(labels, scores)),
                "trapezoidal_pr_auc": float(auc(recall[::-1], precision[::-1])),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "inverted_average_precision": float(average_precision_score(labels, -scores)),
                "higher_score_class": "Yes/1",
                "pooling": "one outer OOF score per row; pooled across folds",
                "top_k_tie_break": "score descending, then record_id ascending",
                "cohort_rows": len(dataset),
            }
        )
    return pd.DataFrame(rows)


def paired_family_bootstrap(
    pooled: dict[str, pd.DataFrame], *, replicates: int = 2000, seed: int = 2718
) -> pd.DataFrame:
    """Paired percentile intervals for ensemble AP minus each component AP."""
    base = pooled["ensemble"][["record_id", "family_id", "label", "yes_score"]].rename(
        columns={"yes_score": "ensemble_score"}
    )
    for component in core.METHODS:
        scores = pooled[component][["record_id", "label", "yes_score"]].rename(
            columns={"label": f"{component}_label", "yes_score": f"{component}_score"}
        )
        base = base.merge(scores, on="record_id", validate="one_to_one")
        if not base["label"].eq(base[f"{component}_label"]).all():
            raise RuntimeError(f"Label mismatch between ensemble and {component}")
        base.drop(columns=f"{component}_label", inplace=True)

    families = sorted(base["family_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    samples = {component: [] for component in core.METHODS}
    invalid = 0
    for _ in range(replicates):
        selected = rng.choice(families, size=len(families), replace=True)
        replicate = pd.concat(
            [base.loc[base["family_id"] == family] for family in selected],
            ignore_index=True,
        )
        labels = replicate["label"].to_numpy(int)
        if labels.sum() == 0:
            invalid += 1
            continue
        ensemble_ap = float(average_precision_score(labels, replicate["ensemble_score"]))
        for component in core.METHODS:
            component_ap = float(average_precision_score(labels, replicate[f"{component}_score"]))
            samples[component].append(ensemble_ap - component_ap)

    rows: list[dict[str, object]] = []
    labels = base["label"].to_numpy(int)
    ensemble_ap = float(average_precision_score(labels, base["ensemble_score"]))
    for component in core.METHODS:
        values = np.asarray(samples[component], dtype=float)
        component_ap = float(average_precision_score(labels, base[f"{component}_score"]))
        rows.append(
            {
                "comparison": f"ensemble_minus_{component}",
                "estimate_ap_difference": ensemble_ap - component_ap,
                "lower_95_percentile": float(np.quantile(values, 0.025)) if len(values) else math.nan,
                "upper_95_percentile": float(np.quantile(values, 0.975)) if len(values) else math.nan,
                "cluster_unit": "family_id",
                "independent_families": len(families),
                "requested_replicates": replicates,
                "defined_replicates": len(values),
                "invalid_no_positive_replicates": invalid,
                "seed": seed,
                "paired_resamples": True,
                "scope": "exploratory; fixed completed OOF predictions; no training variability",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2718)
    args = parser.parse_args()
    if args.replicates <= 0:
        raise ValueError("--replicates must be positive")

    dataset, pooled = reconstruct_pooled_scores()
    metrics = metric_audit(dataset, pooled)
    paired = paired_family_bootstrap(pooled, replicates=args.replicates, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "REVIEWED_TRANSFORMER_METRIC_AUDIT.csv", index=False, lineterminator="\n")
    paired.to_csv(args.output_dir / "PAIRED_FAMILY_BOOTSTRAP.csv", index=False, lineterminator="\n")
    print(metrics.to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
