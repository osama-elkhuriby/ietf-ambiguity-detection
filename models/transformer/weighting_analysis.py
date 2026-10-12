"""Analyze the frozen three-seed weighting ablation without test-set access."""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.experiment import load_development_split
from models.transformer.feasibility_audit import (
    _bootstrap_operating_points,
    routing_point,
)
from models.transformer.lr_screen_analysis import (
    CLASSICAL_ROOT,
    DEVELOPMENT_DIR,
    _paired_pr_auc_bootstrap,
)

STRATEGIES = ("unweighted", "class", "confidence", "combined")
SIMPLICITY_ORDER = {name: index for index, name in enumerate(STRATEGIES)}
SEEDS = (1729, 2718, 31415)
FOLDS = (1, 2, 3, 4, 5)
CONTRASTS = (
    ("unweighted", "confidence"),
    ("class", "combined"),
    ("class", "unweighted"),
)
LR_SCREEN_ROOT = ROOT / "experiments/transformer/results/lr-screen-v2/lr-1e-5"
WEIGHTING_ROOT = ROOT / "experiments/transformer/results/weighting-ablation-v1"


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, *, bins: int = 10
) -> float:
    """Return equal-width expected calibration error."""

    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(labels) == 0 or len(labels) != len(probabilities):
        raise ValueError("Labels and probabilities must be aligned and non-empty")
    if bins < 1 or np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Probabilities and bin count are invalid")
    assignments = np.minimum((probabilities * bins).astype(int), bins - 1)
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        if selected.any():
            ece += float(selected.mean()) * abs(
                float(labels[selected].mean()) - float(probabilities[selected].mean())
            )
    return ece


def select_weighting(fold_scores: dict[str, list[float]]) -> dict[str, Any]:
    """Apply the frozen seed-averaged mean-fold one-standard-error rule."""

    if set(fold_scores) != set(STRATEGIES):
        raise ValueError("Selection requires all four frozen weighting strategies")
    if {len(values) for values in fold_scores.values()} != {len(FOLDS)}:
        raise ValueError("Selection requires five seed-averaged fold scores")
    means = {name: float(np.mean(values)) for name, values in fold_scores.items()}
    ranked = sorted(STRATEGIES, key=lambda name: (-means[name], SIMPLICITY_ORDER[name]))
    top, runner_up = ranked[:2]
    differences = np.asarray(fold_scores[top]) - np.asarray(fold_scores[runner_up])
    standard_error = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
    mean_difference = float(np.mean(differences))
    within_one_se = mean_difference < standard_error
    selected = (
        min((top, runner_up), key=SIMPLICITY_ORDER.get) if within_one_se else top
    )
    return {
        "seed_averaged_mean_fold_pr_auc": means,
        "ranking": ranked,
        "top_strategy": top,
        "runner_up_strategy": runner_up,
        "paired_fold_differences_top_minus_runner_up": differences.tolist(),
        "mean_paired_difference": mean_difference,
        "standard_error_paired_differences": standard_error,
        "difference_is_less_than_one_standard_error": within_one_se,
        "selected_strategy": selected,
        "simplicity_order": list(STRATEGIES),
    }


def _run_directory(strategy: str, seed: int, fold: int) -> tuple[Path, bool]:
    if strategy == "unweighted" and seed in (1729, 2718):
        return LR_SCREEN_ROOT / f"fold-{fold}-seed-{seed}", True
    return WEIGHTING_ROOT / strategy / f"fold-{fold}-seed-{seed}", False


def _load_transformer_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    cells: list[dict[str, Any]] = []
    expected_ids: dict[int, set[str]] = {}
    for strategy in STRATEGIES:
        for seed in SEEDS:
            for fold in FOLDS:
                directory, reused = _run_directory(strategy, seed, fold)
                prediction_path = directory / "validation_predictions.csv"
                manifest_path = directory / "run_manifest.json"
                config_path = directory / "run_config.json"
                if not all(path.is_file() for path in (prediction_path, manifest_path, config_path)):
                    raise FileNotFoundError(f"Missing completed weighting cell: {directory}")
                frame = pd.read_csv(prediction_path)
                required = {
                    "record_id", "label", "logit_0", "logit_1",
                    "ambiguity_probability", "included_in_scoring",
                }
                if not required.issubset(frame.columns):
                    raise RuntimeError(f"Raw-logit output is incomplete: {directory}")
                frame = frame[frame["included_in_scoring"]].copy()
                ids = set(frame["record_id"].astype(str))
                if fold in expected_ids and ids != expected_ids[fold]:
                    raise RuntimeError(f"Scored rows differ for fold {fold}: {directory}")
                expected_ids.setdefault(fold, ids)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                config = json.loads(config_path.read_text(encoding="utf-8"))
                resolved = config["resolved"]
                if (
                    manifest["precision"] != "fp32"
                    or manifest["seed"] != seed
                    or manifest["fold"] != fold
                    or resolved["weight_strategy"] != strategy
                    or float(resolved["learning_rate"]) != 1e-5
                    or manifest["outer_fold_evaluations_during_training"] != 0
                    or manifest["outer_fold_evaluations_after_selection"] != 1
                ):
                    raise RuntimeError(f"Cell violates frozen protocol: {directory}")
                labels = frame["label"].to_numpy(dtype=np.int64)
                probabilities = frame["ambiguity_probability"].to_numpy(dtype=float)
                predictions.append(frame.assign(strategy=strategy, seed=seed, fold=fold))
                cells.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "fold": fold,
                        "reused_from_lr_screen_v2": reused,
                        "rows": len(frame),
                        "positives": int(labels.sum()),
                        "pr_auc": float(average_precision_score(labels, probabilities)),
                        "selected_epoch": int(manifest["best_epoch"]),
                        "runtime_seconds": float(manifest["total_runtime_seconds"]),
                        "peak_gpu_memory_allocated_mb": float(
                            manifest["peak_gpu_memory_allocated_mb"]
                        ),
                        "run_directory": str(directory.relative_to(ROOT)),
                    }
                )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(cells)


def _ensemble_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    source = predictions.copy()
    source["raw_log_odds"] = source["logit_1"] - source["logit_0"]
    identity = ["strategy", "fold", "record_id", "label"]
    grouped = source.groupby(identity, sort=True, as_index=False).agg(
        ensemble_raw_log_odds=("raw_log_odds", "mean"),
        seeds=("seed", "nunique"),
    )
    if not (grouped["seeds"] == len(SEEDS)).all():
        raise RuntimeError("Every ensemble row must contain all three frozen seeds")
    grouped["ambiguity_probability"] = 1.0 / (
        1.0 + np.exp(-grouped["ensemble_raw_log_odds"])
    )
    return grouped.drop(columns="seeds")


def _metadata() -> pd.DataFrame:
    development = load_development_split(DEVELOPMENT_DIR, "train")
    frame = development[["record_id", "document_id", "section_ids"]].copy()
    frame["section_cluster"] = (
        frame["document_id"].astype(str) + "::" + frame["section_ids"].astype(str)
    )
    return frame


def _strategy_summary(
    cells: pd.DataFrame, ensembles: pd.DataFrame, *, repetitions: int
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    rows: list[dict[str, Any]] = []
    fold_scores: dict[str, list[float]] = {}
    for index, strategy in enumerate(STRATEGIES):
        cell_subset = cells[cells["strategy"] == strategy]
        seed_fold = cell_subset.pivot(index="fold", columns="seed", values="pr_auc")
        averaged = seed_fold.mean(axis=1).reindex(FOLDS)
        fold_scores[strategy] = averaged.tolist()
        pooled = ensembles[ensembles["strategy"] == strategy].copy().reset_index(
            drop=True
        )
        labels = pooled["label"].to_numpy(dtype=np.int64)
        scores = pooled["ambiguity_probability"].to_numpy(dtype=float)
        routing_frame = pooled[["fold", "label"]].copy()
        routing_frame["score"] = scores
        point = routing_point(labels, scores, 0.90)
        intervals = _bootstrap_operating_points(
            routing_frame, strategy="model", repetitions=repetitions, seed=9200 + index
        )[0.90]["referral_rate"]
        rows.append(
            {
                "strategy": strategy,
                "seed_averaged_mean_fold_pr_auc": float(averaged.mean()),
                "pooled_three_seed_ensemble_pr_auc": float(
                    average_precision_score(labels, scores)
                ),
                "pooled_brier_secondary_uncalibrated": float(
                    brier_score_loss(labels, scores)
                ),
                "pooled_ece_10_bin_secondary_uncalibrated": expected_calibration_error(
                    labels, scores
                ),
                "recall_target": 0.90,
                "referral_rate": float(point["referral_rate"]),
                "achieved_recall": float(point["ambiguous_recall"]),
                "referral_rate_ci_lower": intervals["lower"],
                "referral_rate_ci_upper": intervals["upper"],
                "total_runtime_seconds_all_15_cells": float(
                    cell_subset["runtime_seconds"].sum()
                ),
                "incremental_runtime_seconds_this_step": float(
                    cell_subset.loc[
                        ~cell_subset["reused_from_lr_screen_v2"], "runtime_seconds"
                    ].sum()
                ),
                "reused_cells": int(cell_subset["reused_from_lr_screen_v2"].sum()),
                "new_cells": int((~cell_subset["reused_from_lr_screen_v2"]).sum()),
                "maximum_peak_gpu_memory_allocated_mb": float(
                    cell_subset["peak_gpu_memory_allocated_mb"].max()
                ),
            }
        )
    return pd.DataFrame(rows), fold_scores


def _contrast_records(
    predictions: pd.DataFrame,
    metadata: pd.DataFrame,
    fold_scores: dict[str, list[float]],
    *,
    repetitions: int,
    prefix: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    wide = predictions.pivot(
        index=["record_id", "fold", "label"],
        columns="strategy",
        values="ambiguity_probability",
    ).reset_index()
    wide = wide.merge(metadata, on="record_id", how="left", validate="one_to_one")
    for index, (first, second) in enumerate(CONTRASTS):
        differences = np.asarray(fold_scores[first]) - np.asarray(fold_scores[second])
        records.append(
            {
                "model": prefix,
                "contrast": f"{first}_minus_{second}",
                "mean_seed_averaged_fold_difference": float(differences.mean()),
                "standard_error_paired_fold_differences": float(
                    np.std(differences, ddof=1) / math.sqrt(len(differences))
                ),
                "per_fold_differences": differences.tolist(),
                "paired_row_bootstrap": _paired_pr_auc_bootstrap(
                    wide.reset_index(drop=True), first, second,
                    repetitions=repetitions, seed=9300 + index,
                ),
                "section_cluster_bootstrap": _paired_pr_auc_bootstrap(
                    wide.reset_index(drop=True), first, second,
                    repetitions=repetitions, seed=9400 + index,
                    cluster_column="section_cluster",
                ),
            }
        )
    return records


def _load_tfidf() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        for fold in FOLDS:
            path = CLASSICAL_ROOT / strategy / f"fold-{fold}" / "validation_predictions.csv"
            frame = pd.read_csv(path)
            frame = frame[frame["included_in_scoring"]].copy()
            parts.append(frame.assign(strategy=strategy, fold=fold))
    return pd.concat(parts, ignore_index=True)


def _tfidf_analysis(
    predictions: pd.DataFrame, metadata: pd.DataFrame, *, repetitions: int
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    fold_scores: dict[str, list[float]] = {}
    for index, strategy in enumerate(STRATEGIES):
        subset = predictions[predictions["strategy"] == strategy].copy().reset_index(
            drop=True
        )
        fold_values = [
            float(
                average_precision_score(
                    subset.loc[subset["fold"] == fold, "label"],
                    subset.loc[subset["fold"] == fold, "ambiguity_probability"],
                )
            )
            for fold in FOLDS
        ]
        fold_scores[strategy] = fold_values
        labels = subset["label"].to_numpy(dtype=np.int64)
        scores = subset["ambiguity_probability"].to_numpy(dtype=float)
        route = subset[["fold", "label"]].copy()
        route["score"] = scores
        point = routing_point(labels, scores, 0.90)
        interval = _bootstrap_operating_points(
            route, strategy="model", repetitions=repetitions, seed=9500 + index
        )[0.90]["referral_rate"]
        rows.append(
            {
                "strategy": strategy,
                "mean_fold_pr_auc": float(np.mean(fold_values)),
                "pooled_pr_auc": float(average_precision_score(labels, scores)),
                "pooled_brier_secondary_uncalibrated": float(
                    brier_score_loss(labels, scores)
                ),
                "pooled_ece_10_bin_secondary_uncalibrated": expected_calibration_error(
                    labels, scores
                ),
                "recall_target": 0.90,
                "referral_rate": float(point["referral_rate"]),
                "achieved_recall": float(point["ambiguous_recall"]),
                "referral_rate_ci_lower": interval["lower"],
                "referral_rate_ci_upper": interval["upper"],
                "runtime": "not recorded by the completed Step 1 comparator",
            }
        )
    return (
        pd.DataFrame(rows),
        select_weighting(fold_scores),
        _contrast_records(
            predictions, metadata, fold_scores,
            repetitions=repetitions, prefix="tfidf_logistic_regression",
        ),
    )


def _transformer_tfidf_comparisons(
    transformer: pd.DataFrame,
    tfidf: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    repetitions: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, strategy in enumerate(STRATEGIES):
        first = transformer[transformer["strategy"] == strategy][
            ["record_id", "fold", "label", "ambiguity_probability"]
        ].rename(columns={"ambiguity_probability": "transformer_probability"})
        second = tfidf[tfidf["strategy"] == strategy][
            ["record_id", "ambiguity_probability"]
        ].rename(columns={"ambiguity_probability": "tfidf_probability"})
        paired = first.merge(second, on="record_id", validate="one_to_one")
        paired = paired.merge(
            metadata, on="record_id", how="left", validate="one_to_one"
        ).reset_index(drop=True)
        fold_differences = []
        for fold in FOLDS:
            subset = paired[paired["fold"] == fold]
            fold_differences.append(
                float(
                    average_precision_score(
                        subset["label"], subset["transformer_probability"]
                    )
                    - average_precision_score(
                        subset["label"], subset["tfidf_probability"]
                    )
                )
            )
        records.append(
            {
                "strategy": strategy,
                "contrast": "deberta_three_seed_logit_average_minus_tfidf",
                "mean_fold_difference": float(np.mean(fold_differences)),
                "standard_error_paired_fold_differences": float(
                    np.std(fold_differences, ddof=1) / math.sqrt(len(fold_differences))
                ),
                "per_fold_differences": fold_differences,
                "paired_row_bootstrap": _paired_pr_auc_bootstrap(
                    paired,
                    "transformer_probability",
                    "tfidf_probability",
                    repetitions=repetitions,
                    seed=9600 + index,
                ),
                "section_cluster_bootstrap": _paired_pr_auc_bootstrap(
                    paired,
                    "transformer_probability",
                    "tfidf_probability",
                    repetitions=repetitions,
                    seed=9700 + index,
                    cluster_column="section_cluster",
                ),
            }
        )
    return records


def analyze(output_dir: Path, *, repetitions: int) -> None:
    """Validate all cells and write immutable Step 3 analysis artifacts."""

    started = time.perf_counter()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    transformer_predictions, cells = _load_transformer_cells()
    ensembles = _ensemble_predictions(transformer_predictions)
    metadata = _metadata()
    summary, fold_scores = _strategy_summary(
        cells, ensembles, repetitions=repetitions
    )
    transformer_selection = select_weighting(fold_scores)
    transformer_contrasts = _contrast_records(
        ensembles, metadata, fold_scores,
        repetitions=repetitions, prefix="deberta_three_seed_logit_average",
    )
    tfidf_predictions = _load_tfidf()
    tfidf_summary, tfidf_selection, tfidf_contrasts = _tfidf_analysis(
        tfidf_predictions, metadata, repetitions=repetitions
    )
    architecture_comparisons = _transformer_tfidf_comparisons(
        ensembles, tfidf_predictions, metadata, repetitions=repetitions
    )

    cells.to_csv(output_dir / "transformer_cells.csv", index=False)
    ensembles.to_csv(output_dir / "transformer_ensemble_oof_predictions.csv", index=False)
    summary.to_csv(output_dir / "transformer_strategy_summary.csv", index=False)
    tfidf_summary.to_csv(output_dir / "tfidf_strategy_summary.csv", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(
            {"transformer": transformer_selection, "tfidf": tfidf_selection},
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    (output_dir / "contrasts.json").write_text(
        json.dumps(
            {
                "transformer": transformer_contrasts,
                "tfidf": tfidf_contrasts,
                "transformer_vs_tfidf": architecture_comparisons,
            },
            indent=2, sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    artifacts = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "experiment": "weighting-ablation-v1-analysis",
        "learning_rate": 1e-5,
        "seeds": list(SEEDS),
        "folds": list(FOLDS),
        "strategies": list(STRATEGIES),
        "bootstrap_repetitions": repetitions,
        "analysis_runtime_seconds": time.perf_counter() - started,
        "dns_opened": False,
        "artifacts": {path.name: sha256_file(path) for path in artifacts},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    analyze(args.output_dir, repetitions=args.bootstrap_repetitions)


if __name__ == "__main__":
    main()
