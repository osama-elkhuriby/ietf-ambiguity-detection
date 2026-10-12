"""Audit routing feasibility, seed variance, and inner validation without training."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.experiment import load_development_split
from models.transformer.lr_screen_analysis import (
    CLASSICAL_ROOT,
    DEVELOPMENT_DIR,
    _paired_pr_auc_bootstrap,
    _read_transformer_run,
)

V2_ROOT = ROOT / "experiments/transformer/results/lr-screen-v2"
INNER_ASSIGNMENT = ROOT / "experiments/transformer/configs/inner_selection_v1.json"
TARGET_RECALLS = (0.80, 0.90, 0.95)
PRIMARY_SEED = 1729
SECOND_SEED = 2718
BootstrapStrategy = Literal["model", "no_skill", "oracle"]


def logit_average_probability(first: pd.DataFrame, second: pd.DataFrame) -> pd.DataFrame:
    """Average binary log-odds from two aligned prediction frames."""

    left = first[
        ["record_id", "fold", "label", "logit_0", "logit_1"]
    ].rename(columns={"logit_0": "logit_0_first", "logit_1": "logit_1_first"})
    right = second[["record_id", "logit_0", "logit_1"]].rename(
        columns={"logit_0": "logit_0_second", "logit_1": "logit_1_second"}
    )
    merged = left.merge(right, on="record_id", how="inner", validate="one_to_one")
    if len(merged) != len(first) or len(merged) != len(second):
        raise RuntimeError("Seed prediction rows do not align")
    log_odds = 0.5 * (
        merged["logit_1_first"]
        - merged["logit_0_first"]
        + merged["logit_1_second"]
        - merged["logit_0_second"]
    )
    merged["ambiguity_probability"] = 1.0 / (1.0 + np.exp(-log_odds))
    return merged[["record_id", "fold", "label", "ambiguity_probability"]]


def routing_curve(labels: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    """Return threshold routing risk versus local coverage, preserving score ties."""

    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) == 0 or labels.sum() == 0:
        raise ValueError("Routing analysis requires at least one positive row")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    total_rows = len(labels)
    total_positives = int(labels.sum())
    records: list[dict[str, float | int]] = [
        {
            "threshold": math.inf,
            "referred_rows": 0,
            "referral_rate": 0.0,
            "ambiguous_recall": 0.0,
            "local_coverage": 1.0,
            "local_selective_risk": total_positives / total_rows,
        }
    ]
    boundaries = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    cumulative_positives = np.cumsum(sorted_labels)
    for boundary in boundaries:
        referred = int(boundary + 1)
        referred_positives = int(cumulative_positives[boundary])
        local_rows = total_rows - referred
        missed_positives = total_positives - referred_positives
        records.append(
            {
                "threshold": float(sorted_scores[boundary]),
                "referred_rows": referred,
                "referral_rate": referred / total_rows,
                "ambiguous_recall": referred_positives / total_positives,
                "local_coverage": local_rows / total_rows,
                "local_selective_risk": (
                    missed_positives / local_rows if local_rows else math.nan
                ),
            }
        )
    return pd.DataFrame(records)


def routing_point(
    labels: np.ndarray, scores: np.ndarray, target_recall: float
) -> dict[str, float | int]:
    """Find the minimum single score threshold meeting a recall target."""

    curve = routing_curve(labels, scores)
    eligible = curve[curve["ambiguous_recall"] >= target_recall]
    if eligible.empty:
        raise RuntimeError("No routing threshold reaches the requested recall")
    return eligible.iloc[0].to_dict()


def _oracle_scores(labels: np.ndarray) -> np.ndarray:
    """Return a unique perfect ranking for a conceptual oracle."""

    count = len(labels)
    tie_break = np.arange(count, 0, -1, dtype=np.float64) / (count + 1)
    return labels.astype(np.float64) * 2.0 + tie_break


def _bootstrap_operating_points(
    frame: pd.DataFrame,
    *,
    strategy: BootstrapStrategy,
    repetitions: int,
    seed: int,
) -> dict[float, dict[str, dict[str, float | int]]]:
    rng = np.random.default_rng(seed)
    fold_indices = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in frame.groupby("fold", sort=True)
    ]
    samples: dict[float, dict[str, list[float]]] = {
        target: {
            "referral_rate": [],
            "ambiguous_recall": [],
            "local_coverage": [],
            "local_selective_risk": [],
        }
        for target in TARGET_RECALLS
    }
    labels_all = frame["label"].to_numpy(dtype=np.int64)
    scores_all = frame["score"].to_numpy(dtype=np.float64)
    for _ in range(repetitions):
        selected = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in fold_indices]
        )
        labels = labels_all[selected]
        if labels.sum() == 0:
            continue
        if strategy == "oracle":
            scores = _oracle_scores(labels)
        elif strategy == "no_skill":
            scores = np.full(len(labels), labels.mean(), dtype=np.float64)
        else:
            scores = scores_all[selected]
        curve = routing_curve(labels, scores)
        for target in TARGET_RECALLS:
            point = curve[curve["ambiguous_recall"] >= target].iloc[0]
            for metric in samples[target]:
                value = float(point[metric])
                if np.isfinite(value):
                    samples[target][metric].append(value)

    intervals: dict[float, dict[str, dict[str, float | int]]] = {}
    for target, metrics in samples.items():
        intervals[target] = {}
        for metric, values in metrics.items():
            if not values:
                intervals[target][metric] = {
                    "lower": math.nan,
                    "upper": math.nan,
                    "valid_bootstraps": 0,
                }
                continue
            lower, upper = np.percentile(values, [2.5, 97.5])
            intervals[target][metric] = {
                "lower": float(lower),
                "upper": float(upper),
                "valid_bootstraps": len(values),
            }
    return intervals


def _load_tfidf(strategy: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(1, 6):
        path = (
            CLASSICAL_ROOT
            / strategy
            / f"fold-{fold}"
            / "validation_predictions.csv"
        )
        frame = pd.read_csv(path)
        frame = frame[frame["included_in_scoring"]].copy()
        parts.append(
            frame[["record_id", "label", "ambiguity_probability"]].assign(fold=fold)
        )
    return pd.concat(parts, ignore_index=True)


def _load_transformer(seed: int) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in range(1, 6):
        frame, _ = _read_transformer_run(V2_ROOT, 1e-5, fold, seed)
        parts.append(frame.assign(fold=fold))
    return pd.concat(parts, ignore_index=True)


def _routing_models() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    development = load_development_split(DEVELOPMENT_DIR, "train")
    metadata = development[["record_id", "document_id", "section_ids"]].copy()
    metadata["section_cluster"] = (
        metadata["document_id"].astype(str)
        + "::"
        + metadata["section_ids"].astype(str)
    )
    first = _load_transformer(PRIMARY_SEED)
    second = _load_transformer(SECOND_SEED)
    ensemble = logit_average_probability(first, second)
    models = {
        "tfidf_unweighted": _load_tfidf("unweighted"),
        "tfidf_class": _load_tfidf("class"),
        "deberta_1e-5_seed_1729": first,
        "deberta_1e-5_seed_2718": second,
        "deberta_1e-5_logit_average": ensemble,
    }
    expected_ids = set(models["tfidf_unweighted"]["record_id"])
    for name, frame in models.items():
        if set(frame["record_id"]) != expected_ids:
            raise RuntimeError(f"Routing rows do not align for {name}")
    return models, metadata


def _risk_and_referral_outputs(
    models: dict[str, pd.DataFrame],
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_records: list[pd.DataFrame] = []
    operating_records: list[dict[str, Any]] = []
    model_items: list[tuple[str, pd.DataFrame, BootstrapStrategy]] = [
        (name, frame, "model") for name, frame in models.items()
    ]
    reference = models["tfidf_unweighted"]
    no_skill = reference.copy()
    no_skill["ambiguity_probability"] = float(no_skill["label"].mean())
    oracle = reference.copy()
    oracle["ambiguity_probability"] = _oracle_scores(
        oracle["label"].to_numpy(dtype=np.int64)
    )
    model_items.extend(
        [
            ("prevalence_matched_no_skill", no_skill, "no_skill"),
            ("oracle", oracle, "oracle"),
        ]
    )
    for model_index, (name, source, bootstrap_strategy) in enumerate(model_items):
        for scope_index, fold in enumerate((None, 1, 2, 3, 4, 5)):
            subset = source if fold is None else source[source["fold"] == fold]
            subset = subset.reset_index(drop=True).rename(
                columns={"ambiguity_probability": "score"}
            )
            if bootstrap_strategy == "no_skill":
                subset["score"] = float(subset["label"].mean())
            elif bootstrap_strategy == "oracle":
                subset["score"] = _oracle_scores(
                    subset["label"].to_numpy(dtype=np.int64)
                )
            curve = routing_curve(
                subset["label"].to_numpy(dtype=np.int64),
                subset["score"].to_numpy(dtype=np.float64),
            )
            curve.insert(0, "fold", "pooled" if fold is None else fold)
            curve.insert(0, "model", name)
            curve_records.append(curve)
            intervals = _bootstrap_operating_points(
                subset,
                strategy=bootstrap_strategy,
                repetitions=repetitions,
                seed=7100 + model_index * 100 + scope_index,
            )
            for target in TARGET_RECALLS:
                point = routing_point(
                    subset["label"].to_numpy(dtype=np.int64),
                    subset["score"].to_numpy(dtype=np.float64),
                    target,
                )
                record: dict[str, Any] = {
                    "model": name,
                    "fold": "pooled" if fold is None else fold,
                    "rows": len(subset),
                    "positives": int(subset["label"].sum()),
                    "positive_prevalence": float(subset["label"].mean()),
                    "target_ambiguous_recall": target,
                    **point,
                }
                for metric, interval in intervals[target].items():
                    record[f"{metric}_ci_lower"] = interval["lower"]
                    record[f"{metric}_ci_upper"] = interval["upper"]
                    record[f"{metric}_valid_bootstraps"] = interval[
                        "valid_bootstraps"
                    ]
                operating_records.append(record)
    return pd.concat(curve_records, ignore_index=True), pd.DataFrame(operating_records)


def _seed_variance(
    models: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    *,
    repetitions: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    first = models["deberta_1e-5_seed_1729"]
    second = models["deberta_1e-5_seed_2718"]
    ensemble = models["deberta_1e-5_logit_average"]
    tfidf_unweighted = models["tfidf_unweighted"]
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        one = first[first["fold"] == fold]
        two = second[second["fold"] == fold]
        ens = ensemble[ensemble["fold"] == fold]
        tfidf = tfidf_unweighted[tfidf_unweighted["fold"] == fold]
        labels = one["label"].to_numpy(dtype=np.int64)
        seed_scores = [
            float(average_precision_score(labels, one["ambiguity_probability"])),
            float(average_precision_score(labels, two["ambiguity_probability"])),
        ]
        rows.append(
            {
                "fold": fold,
                "rows": len(one),
                "positives": int(labels.sum()),
                "seed_1729_pr_auc": seed_scores[0],
                "seed_2718_pr_auc": seed_scores[1],
                "seed_sample_standard_deviation": float(
                    np.std(seed_scores, ddof=1)
                ),
                "logit_average_ensemble_pr_auc": float(
                    average_precision_score(labels, ens["ambiguity_probability"])
                ),
                "tfidf_unweighted_pr_auc": float(
                    average_precision_score(labels, tfidf["ambiguity_probability"])
                ),
            }
        )
    frame = pd.DataFrame(rows)

    pooled = ensemble.merge(
        metadata, on="record_id", how="left", validate="one_to_one"
    ).reset_index(drop=True)
    comparisons: dict[str, Any] = {}
    for index, comparator_name in enumerate(("tfidf_unweighted", "tfidf_class")):
        comparator = models[comparator_name][
            ["record_id", "ambiguity_probability"]
        ].rename(columns={"ambiguity_probability": "comparator_probability"})
        paired = pooled.merge(
            comparator, on="record_id", how="inner", validate="one_to_one"
        ).reset_index(drop=True)
        comparisons[comparator_name] = {
            "mean_fold_ensemble_pr_auc": float(
                frame["logit_average_ensemble_pr_auc"].mean()
            ),
            "mean_fold_comparator_pr_auc": float(
                np.mean(
                    [
                        average_precision_score(
                            paired.loc[paired["fold"] == fold, "label"],
                            paired.loc[
                                paired["fold"] == fold, "comparator_probability"
                            ],
                        )
                        for fold in range(1, 6)
                    ]
                )
            ),
            "pooled_ensemble_pr_auc": float(
                average_precision_score(
                    paired["label"], paired["ambiguity_probability"]
                )
            ),
            "pooled_comparator_pr_auc": float(
                average_precision_score(
                    paired["label"], paired["comparator_probability"]
                )
            ),
            "paired_row_bootstrap": _paired_pr_auc_bootstrap(
                paired,
                "ambiguity_probability",
                "comparator_probability",
                repetitions=repetitions,
                seed=8100 + index * 10,
            ),
            "section_cluster_bootstrap": _paired_pr_auc_bootstrap(
                paired,
                "ambiguity_probability",
                "comparator_probability",
                repetitions=repetitions,
                seed=8101 + index * 10,
                cluster_column="section_cluster",
            ),
        }
    summary = {
        "seed_sample_standard_deviation_mean_across_folds": float(
            frame["seed_sample_standard_deviation"].mean()
        ),
        "seed_sample_standard_deviation_maximum": float(
            frame["seed_sample_standard_deviation"].max()
        ),
        "ensemble_comparisons": comparisons,
    }
    return frame, summary


def _inner_validation_audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment = json.loads(INNER_ASSIGNMENT.read_text(encoding="utf-8"))
    family_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        cells: list[tuple[str, int]] = [
            ("1e-5", PRIMARY_SEED),
            ("2e-5", PRIMARY_SEED),
            ("3e-5", PRIMARY_SEED),
            ("1e-5", SECOND_SEED),
            ("2e-5", SECOND_SEED),
        ]
        manifests: list[dict[str, Any]] = []
        for rate, seed in cells:
            path = V2_ROOT / f"lr-{rate}" / f"fold-{fold}-seed-{seed}"
            manifest = json.loads(
                (path / "run_manifest.json").read_text(encoding="utf-8")
            )
            manifests.append(manifest)
            cell_rows.append(
                {
                    "outer_fold": fold,
                    "inner_family": manifest["inner_selection_family"],
                    "learning_rate": rate,
                    "seed": seed,
                    "selected_epoch": int(manifest["best_epoch"]),
                    "inner_rows_before_duplicate_exclusion": int(
                        manifest["inner_selection_rows"]
                    ),
                    "inner_scored_rows": int(manifest["inner_selection_scored_rows"]),
                    "inner_positive_labels": int(
                        manifest["inner_selection_positive_labels"]
                    ),
                    "fewer_than_20_positives": int(
                        manifest["inner_selection_positive_labels"]
                    )
                    < 20,
                }
            )
        expected_family = assignment["outer_fold_to_inner_family"][str(fold)]
        values = {
            (
                manifest["inner_selection_family"],
                int(manifest["inner_selection_rows"]),
                int(manifest["inner_selection_scored_rows"]),
                int(manifest["inner_selection_positive_labels"]),
            )
            for manifest in manifests
        }
        if len(values) != 1 or next(iter(values))[0] != expected_family:
            raise RuntimeError(f"Inconsistent inner validation metadata for fold {fold}")
        family, rows, scored_rows, positives = next(iter(values))
        family_rows.append(
            {
                "outer_fold": fold,
                "inner_family": family,
                "rows_before_duplicate_exclusion": rows,
                "scored_rows": scored_rows,
                "positive_labels": positives,
                "fewer_than_20_positives": positives < 20,
            }
        )
    return pd.DataFrame(family_rows), pd.DataFrame(cell_rows)


def run(output_dir: Path, *, bootstrap_repetitions: int = 2000) -> Path:
    """Create an immutable Step 2c audit from existing OOF artifacts."""

    if output_dir.exists():
        raise FileExistsError(f"Feasibility audit output is immutable: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}-{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        models, metadata = _routing_models()
        curves, operating = _risk_and_referral_outputs(
            models, repetitions=bootstrap_repetitions
        )
        seed_frame, seed_summary = _seed_variance(
            models, metadata, repetitions=bootstrap_repetitions
        )
        inner_families, inner_cells = _inner_validation_audit()
        curves.to_csv(
            temporary / "risk_coverage_curves.csv", index=False, lineterminator="\n"
        )
        operating.to_csv(
            temporary / "referral_operating_points.csv",
            index=False,
            lineterminator="\n",
        )
        seed_frame.to_csv(
            temporary / "seed_variance.csv", index=False, lineterminator="\n"
        )
        inner_families.to_csv(
            temporary / "inner_validation_families.csv",
            index=False,
            lineterminator="\n",
        )
        inner_cells.to_csv(
            temporary / "inner_validation_cells.csv",
            index=False,
            lineterminator="\n",
        )
        summary = {
            "schema_version": 1,
            "bootstrap_repetitions": bootstrap_repetitions,
            "routing_definition": {
                "policy": "refer rows with ambiguity score at or above one threshold",
                "local_coverage": "fraction below threshold and handled locally as negative",
                "local_selective_risk": "ambiguous fraction among locally handled rows",
                "tie_policy": "all rows tied at the boundary are referred",
                "no_skill": "constant observed-prevalence score within each scope",
                "oracle": "conceptual perfect label ranking with deterministic unique order",
            },
            "seed_variance": seed_summary,
            "inner_families_below_20_positives": int(
                inner_families["fewer_than_20_positives"].sum()
            ),
            "input_hashes": {
                "inner_assignment": sha256_file(INNER_ASSIGNMENT),
                "development_train": sha256_file(DEVELOPMENT_DIR / "train.csv"),
            },
        }
        (temporary / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(
                {"schema_version": 1, "artifacts": artifacts},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/transformer/governance/step2c_feasibility_v1",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    print(
        run(
            args.output.resolve(),
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
