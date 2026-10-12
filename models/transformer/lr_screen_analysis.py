"""Analyze the frozen DeBERTa learning-rate screen without test-set access."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from models.classical.feature_pipeline import sha256_file
from models.classical.tfidf_comparator import binary_metrics, bootstrap_intervals
from models.transformer.dataset import ROOT
from models.transformer.experiment import load_development_split, load_frozen_folds

LEARNING_RATES = (1e-5, 2e-5, 3e-5)
PRIMARY_SEED = 1729
SECOND_SEED = 2718
FOLD_ASSIGNMENT = ROOT / "experiments/transformer/configs/folds_v1.json"
DEVELOPMENT_DIR = ROOT / "corpus/processed/transformer_partitioned/v1/development"
CLASSICAL_ROOT = ROOT / "experiments/classical/results/tfidf-logreg-v1"


def lr_slug(learning_rate: float) -> str:
    """Return the stable directory label used by screen runs."""

    return f"{learning_rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def select_learning_rate(fold_scores: dict[float, list[float]]) -> dict[str, Any]:
    """Apply the frozen mean/one-standard-error rule exactly."""

    if set(fold_scores) != set(LEARNING_RATES):
        raise ValueError("Selection requires exactly the three frozen learning rates")
    if {len(values) for values in fold_scores.values()} != {5}:
        raise ValueError("Selection requires exactly five scores per learning rate")
    means = {rate: float(np.mean(values)) for rate, values in fold_scores.items()}
    ranked = sorted(LEARNING_RATES, key=lambda rate: (-means[rate], rate))
    top, runner_up = ranked[:2]
    differences = np.asarray(fold_scores[top]) - np.asarray(fold_scores[runner_up])
    standard_error = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
    mean_difference = float(np.mean(differences))
    within_one_se = mean_difference < standard_error
    selected = min(top, runner_up) if within_one_se else top
    return {
        "mean_pr_auc": {lr_slug(rate): means[rate] for rate in LEARNING_RATES},
        "ranked_learning_rates": [lr_slug(rate) for rate in ranked],
        "top_learning_rate": lr_slug(top),
        "runner_up_learning_rate": lr_slug(runner_up),
        "paired_fold_differences_top_minus_runner_up": differences.tolist(),
        "mean_paired_difference": mean_difference,
        "standard_error_paired_differences": standard_error,
        "difference_is_less_than_one_standard_error": within_one_se,
        "selected_learning_rate": lr_slug(selected),
        "second_seed_required": within_one_se,
        "second_seed": SECOND_SEED if within_one_se else None,
    }


def _paired_pr_auc_bootstrap(
    frame: pd.DataFrame,
    probability_a: str,
    probability_b: str,
    *,
    repetitions: int,
    seed: int,
    cluster_column: str | None = None,
) -> dict[str, Any]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    first = frame[probability_a].to_numpy(dtype=np.float64)
    second = frame[probability_b].to_numpy(dtype=np.float64)
    point = float(
        average_precision_score(labels, first)
        - average_precision_score(labels, second)
    )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    fold_values = sorted(frame["fold"].unique())
    cluster_indices: dict[int, list[np.ndarray]] = {}
    if cluster_column is not None:
        for fold in fold_values:
            subset = frame[frame["fold"] == fold]
            cluster_indices[int(fold)] = [
                group.index.to_numpy(dtype=np.int64)
                for _, group in subset.groupby(cluster_column, sort=True)
            ]
            if len(cluster_indices[int(fold)]) < 2:
                raise ValueError(f"Fold {fold} has fewer than two bootstrap clusters")
    fold_indices = {
        int(fold): frame.index[frame["fold"] == fold].to_numpy(dtype=np.int64)
        for fold in fold_values
    }
    for _ in range(repetitions):
        selected: list[np.ndarray] = []
        for fold in fold_values:
            if cluster_column is None:
                candidates = fold_indices[int(fold)]
                selected.append(rng.choice(candidates, size=len(candidates), replace=True))
            else:
                clusters = cluster_indices[int(fold)]
                choices = rng.integers(0, len(clusters), size=len(clusters))
                selected.extend(clusters[index] for index in choices)
        indices = np.concatenate(selected)
        sampled_labels = labels[indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        samples.append(
            float(
                average_precision_score(sampled_labels, first[indices])
                - average_precision_score(sampled_labels, second[indices])
            )
        )
    if not samples:
        raise RuntimeError("No valid paired bootstrap samples")
    lower, upper = np.percentile(samples, [2.5, 97.5])
    return {
        "point_difference": point,
        "lower": float(lower),
        "upper": float(upper),
        "valid_bootstraps": len(samples),
        "method": (
            "within-fold section-cluster paired bootstrap"
            if cluster_column
            else "within-fold paired row bootstrap"
        ),
    }


def _read_transformer_run(
    run_root: Path, learning_rate: float, fold: int, seed: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = run_root / f"lr-{lr_slug(learning_rate)}" / f"fold-{fold}-seed-{seed}"
    predictions_path = directory / "validation_predictions.csv"
    manifest_path = directory / "run_manifest.json"
    if not predictions_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing completed screen run: {directory}")
    predictions = pd.read_csv(predictions_path)
    predictions = predictions[predictions["included_in_scoring"]].copy()
    predictions["fold"] = fold
    predictions["learning_rate"] = learning_rate
    predictions["seed"] = seed
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["precision"] != "fp32" or manifest["seed"] != seed:
        raise RuntimeError(f"Run does not satisfy frozen screen protocol: {directory}")
    return predictions, manifest


def _classical_predictions(fold: int) -> pd.DataFrame:
    path = CLASSICAL_ROOT / "unweighted" / f"fold-{fold}" / "validation_predictions.csv"
    frame = pd.read_csv(path)
    frame = frame[frame["included_in_scoring"]].copy()
    return frame[["record_id", "ambiguity_probability"]].rename(
        columns={"ambiguity_probability": "tfidf_probability"}
    )


def _metric_record(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    metrics = binary_metrics(labels, probabilities)
    metrics["bootstrap_95_percentile"] = bootstrap_intervals(
        labels, probabilities, repetitions=repetitions, seed=seed
    )
    return metrics


def _safe_pr_auc(labels: np.ndarray, probabilities: pd.Series) -> float | None:
    """Return PR-AUC only when both classes are represented."""

    if len(np.unique(labels)) < 2:
        return None
    return float(average_precision_score(labels, probabilities))


def analyze(
    run_root: Path,
    output_dir: Path,
    *,
    seed: int = PRIMARY_SEED,
    bootstrap_repetitions: int = 2000,
) -> Path:
    """Build immutable screen summaries from completed fold artifacts."""

    if output_dir.exists():
        raise FileExistsError(f"Analysis output is immutable: {output_dir}")
    train = load_development_split(DEVELOPMENT_DIR, "train")
    train_manifest = json.loads(
        (DEVELOPMENT_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    folds = load_frozen_folds(
        train,
        FOLD_ASSIGNMENT,
        expected_sha256="c04dc8424ed96486f8521ef57d17e4b503d2c3f41bb116edeeb9535073dd7884",
        expected_train_sha256=train_manifest["split_files"]["train"]["sha256"],
    )
    metadata = train[
        ["record_id", "document_id", "section_ids", "label_tier"]
    ].copy()
    metadata["section_cluster"] = (
        metadata["document_id"].astype(str) + "::" + metadata["section_ids"].astype(str)
    )
    temporary = output_dir.with_name(f".{output_dir.name}-{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        fold_records: list[dict[str, Any]] = []
        tier_records: list[dict[str, Any]] = []
        runtime_records: list[dict[str, Any]] = []
        pooled_by_lr: dict[float, pd.DataFrame] = {}
        scores: dict[float, list[float]] = {rate: [] for rate in LEARNING_RATES}
        for rate_index, rate in enumerate(LEARNING_RATES):
            pooled_parts: list[pd.DataFrame] = []
            for fold in folds:
                predictions, manifest = _read_transformer_run(
                    run_root, rate, fold.number, seed
                )
                predictions = predictions.merge(
                    metadata, on=["record_id", "label_tier"], how="left", validate="one_to_one"
                )
                predictions = predictions.merge(
                    _classical_predictions(fold.number),
                    on="record_id",
                    how="left",
                    validate="one_to_one",
                )
                if predictions["tfidf_probability"].isna().any():
                    raise RuntimeError("Transformer and TF-IDF scoring rows do not align")
                labels = predictions["label"].to_numpy(dtype=np.int64)
                probabilities = predictions["ambiguity_probability"].to_numpy(
                    dtype=np.float64
                )
                metrics = _metric_record(
                    labels,
                    probabilities,
                    repetitions=bootstrap_repetitions,
                    seed=314159 + rate_index * 100 + fold.number,
                )
                tfidf_pr_auc = float(
                    average_precision_score(labels, predictions["tfidf_probability"])
                )
                scores[rate].append(float(metrics["pr_auc"]))
                fold_records.append(
                    {
                        "learning_rate": rate,
                        "fold": fold.number,
                        "held_out_families": ";".join(fold.held_out_families),
                        "tfidf_unweighted_pr_auc": tfidf_pr_auc,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key != "bootstrap_95_percentile"
                        },
                        **{
                            f"{name}_ci_lower": interval["lower"]
                            for name, interval in metrics[
                                "bootstrap_95_percentile"
                            ].items()
                        },
                        **{
                            f"{name}_ci_upper": interval["upper"]
                            for name, interval in metrics[
                                "bootstrap_95_percentile"
                            ].items()
                        },
                    }
                )
                for tier in ("HIGH", "MEDIUM"):
                    subset = predictions[predictions["label_tier"] == tier]
                    tier_labels = subset["label"].to_numpy(dtype=np.int64)
                    tier_records.append(
                        {
                            "model": "transformer",
                            "learning_rate": rate,
                            "fold": fold.number,
                            "label_tier": tier,
                            "rows": len(subset),
                            "positives": int(tier_labels.sum()),
                            "no_skill_pr_auc": float(tier_labels.mean()),
                            "pr_auc": _safe_pr_auc(
                                tier_labels, subset["ambiguity_probability"]
                            ),
                        }
                    )
                    if rate_index == 0:
                        tier_records.append(
                            {
                                "model": "tfidf_unweighted",
                                "learning_rate": None,
                                "fold": fold.number,
                                "label_tier": tier,
                                "rows": len(subset),
                                "positives": int(tier_labels.sum()),
                                "no_skill_pr_auc": float(tier_labels.mean()),
                                "pr_auc": _safe_pr_auc(
                                    tier_labels, subset["tfidf_probability"]
                                ),
                            }
                        )
                runtime_records.append(
                    {
                        "learning_rate": rate,
                        "fold": fold.number,
                        "best_epoch": manifest["best_epoch"],
                        "training_runtime_seconds": manifest[
                            "training_runtime_seconds"
                        ],
                        "total_runtime_seconds": manifest["total_runtime_seconds"],
                        "peak_gpu_memory_allocated_mb": manifest[
                            "peak_gpu_memory_allocated_mb"
                        ],
                        "peak_gpu_memory_reserved_mb": manifest[
                            "peak_gpu_memory_reserved_mb"
                        ],
                    }
                )
                pooled_parts.append(predictions)
            pooled = pd.concat(pooled_parts, ignore_index=True)
            pooled_by_lr[rate] = pooled
            pooled.to_csv(
                temporary / f"oof_predictions_lr_{lr_slug(rate)}_seed_{seed}.csv",
                index=False,
                lineterminator="\n",
            )

        fold_frame = pd.DataFrame(fold_records)
        fold_frame.to_csv(temporary / "fold_metrics.csv", index=False, lineterminator="\n")
        pd.DataFrame(tier_records).to_csv(
            temporary / "tier_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(runtime_records).to_csv(
            temporary / "runtime_gpu.csv", index=False, lineterminator="\n"
        )

        selection = select_learning_rate(scores)
        pooled_metrics: dict[str, Any] = {}
        for rate_index, rate in enumerate(LEARNING_RATES):
            pooled = pooled_by_lr[rate]
            pooled_metrics[lr_slug(rate)] = _metric_record(
                pooled["label"].to_numpy(dtype=np.int64),
                pooled["ambiguity_probability"].to_numpy(dtype=np.float64),
                repetitions=bootstrap_repetitions,
                seed=271828 + rate_index,
            )
            pooled_metrics[lr_slug(rate)]["tfidf_unweighted_pr_auc"] = float(
                average_precision_score(
                    pooled["label"], pooled["tfidf_probability"]
                )
            )

        paired_differences: dict[str, Any] = {}
        rates = list(LEARNING_RATES)
        for first_index, first_rate in enumerate(rates):
            for second_rate in rates[first_index + 1 :]:
                first = pooled_by_lr[first_rate]
                second = pooled_by_lr[second_rate][
                    ["record_id", "ambiguity_probability"]
                ].rename(columns={"ambiguity_probability": "probability_b"})
                paired = first.merge(
                    second, on="record_id", how="inner", validate="one_to_one"
                ).reset_index(drop=True)
                fold_differences = [
                    scores[first_rate][fold] - scores[second_rate][fold]
                    for fold in range(5)
                ]
                key = f"{lr_slug(first_rate)}_minus_{lr_slug(second_rate)}"
                paired_differences[key] = {
                    "per_fold_differences": fold_differences,
                    "mean_difference": float(np.mean(fold_differences)),
                    "standard_error": float(
                        np.std(fold_differences, ddof=1) / math.sqrt(5)
                    ),
                    "paired_row_bootstrap": _paired_pr_auc_bootstrap(
                        paired,
                        "ambiguity_probability",
                        "probability_b",
                        repetitions=bootstrap_repetitions,
                        seed=161803 + first_index,
                    ),
                }

        cluster_comparisons: dict[str, Any] = {}
        for rate_index, rate in enumerate(LEARNING_RATES):
            pooled = pooled_by_lr[rate].reset_index(drop=True)
            cluster_comparisons[lr_slug(rate)] = _paired_pr_auc_bootstrap(
                pooled,
                "ambiguity_probability",
                "tfidf_probability",
                repetitions=bootstrap_repetitions,
                seed=141421 + rate_index,
                cluster_column="section_cluster",
            )

        analysis = {
            "schema_version": 1,
            "seed": seed,
            "selection_rule": selection,
            "pooled_metrics": pooled_metrics,
            "paired_learning_rate_differences": paired_differences,
            "transformer_minus_tfidf_cluster_bootstrap": cluster_comparisons,
            "cluster_definition": "document_id + section_ids, resampled within fold",
            "bootstrap_repetitions": bootstrap_repetitions,
            "fold_assignment_sha256": sha256_file(FOLD_ASSIGNMENT),
            "development_train_sha256": train_manifest["split_files"]["train"][
                "sha256"
            ],
        }
        (temporary / "analysis.json").write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(temporary.iterdir())
                if path.is_file()
            },
        }
        (temporary / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
        "--run-root",
        type=Path,
        default=ROOT / "experiments/transformer/results/lr-screen-v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=PRIMARY_SEED)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    print(
        analyze(
            args.run_root.resolve(),
            args.output.resolve(),
            seed=args.seed,
            bootstrap_repetitions=args.bootstrap_repetitions,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
