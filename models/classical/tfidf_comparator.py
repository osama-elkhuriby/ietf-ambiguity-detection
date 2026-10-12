"""Leakage-controlled TF-IDF logistic-regression grouped-CV comparator."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.experiment import (
    DevelopmentFold,
    WeightStrategy,
    compute_training_weights,
    load_development_split,
    load_frozen_folds,
)

DEFAULT_CONFIG = ROOT / "experiments/classical/configs/tfidf_logreg_v1.json"
METRIC_NAMES = ("pr_auc", "precision", "recall", "f1", "f2", "mcc")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _vectorizer(config: dict[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=int(config["min_df"]),
        max_features=int(config["max_features"]),
        sublinear_tf=True,
        dtype=np.float64,
        token_pattern=r"(?u)\b[A-Za-z0-9][A-Za-z0-9_-]*\b",
    )


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Compute the frozen threshold-0.5 binary metrics."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if len(labels) != len(probabilities) or not len(labels):
        raise ValueError("Labels and probabilities must be non-empty and aligned")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("Labels must be binary")
    predicted = (probabilities >= 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    mcc = (
        float(matthews_corrcoef(labels, predicted))
        if len(np.unique(np.concatenate((labels, predicted)))) == 2
        else 0.0
    )
    return {
        "rows": len(labels),
        "positives": int(labels.sum()),
        "positive_prevalence": float(labels.mean()),
        "no_skill_pr_auc": float(labels.mean()),
        "threshold": 0.5,
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "mcc": mcc,
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def bootstrap_intervals(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, dict[str, float | int]]:
    """Ordinary paired row bootstrap percentile intervals for reported metrics."""

    point = binary_metrics(labels, probabilities)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        name: [] for name in (*METRIC_NAMES, "no_skill_pr_auc")
    }
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=len(labels))
        current = binary_metrics(labels[indices], probabilities[indices])
        for name, values in samples.items():
            value = current[name]
            if value is not None and np.isfinite(value):
                values.append(float(value))
    intervals: dict[str, dict[str, float | int]] = {}
    for name, values in samples.items():
        if not values:
            continue
        lower, upper = np.percentile(values, [2.5, 97.5])
        intervals[name] = {
            "point": float(point[name]),
            "lower": float(lower),
            "upper": float(upper),
            "valid_bootstraps": len(values),
        }
    return intervals


def fold_prevalence_tables(
    frame: pd.DataFrame, folds: list[DevelopmentFold]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build fold and held-out-family counts after duplicate-safe scoring."""

    fold_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for fold in folds:
        validation = frame.iloc[fold.validation_indices]
        scored = frame.iloc[fold.scoring_indices]
        positives = int(scored["label"].sum())
        prevalence = positives / len(scored)
        fold_rows.append(
            {
                "fold": fold.number,
                "held_out_families": ";".join(fold.held_out_families),
                "rows_before_exclusion": len(validation),
                "rows_scored": len(scored),
                "excluded_exact_duplicates": len(validation) - len(scored),
                "positives": positives,
                "positive_prevalence": prevalence,
                "no_skill_pr_auc": prevalence,
            }
        )
        scoring_ids = set(scored["record_id"].astype(str))
        for family in fold.held_out_families:
            before = validation[validation["family_id"].astype(str) == family]
            included = before[before["record_id"].astype(str).isin(scoring_ids)]
            family_positives = int(included["label"].sum())
            family_prevalence = family_positives / len(included)
            family_rows.append(
                {
                    "fold": fold.number,
                    "held_out_family": family,
                    "rows_before_exclusion": len(before),
                    "rows_scored": len(included),
                    "excluded_exact_duplicates": len(before) - len(included),
                    "positives": family_positives,
                    "positive_prevalence": family_prevalence,
                    "no_skill_pr_auc": family_prevalence,
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(family_rows)


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _flatten_result(
    strategy: str, fold: DevelopmentFold, metrics: dict[str, Any]
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "weight_strategy": strategy,
        "fold": fold.number,
        "held_out_families": ";".join(fold.held_out_families),
        "rows": metrics["rows"],
        "positives": metrics["positives"],
        "positive_prevalence": metrics["positive_prevalence"],
        "no_skill_pr_auc": metrics["no_skill_pr_auc"],
    }
    for name, interval in metrics["bootstrap_95_percentile"].items():
        row[name] = interval["point"]
        row[f"{name}_ci_lower"] = interval["lower"]
        row[f"{name}_ci_upper"] = interval["upper"]
    return row


def _fit_fold(
    frame: pd.DataFrame,
    fold: DevelopmentFold,
    strategy: WeightStrategy,
    config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = frame.iloc[fold.train_indices].reset_index(drop=True)
    validation = frame.iloc[fold.validation_indices].reset_index(drop=True)
    scoring_ids = set(frame.iloc[fold.scoring_indices]["record_id"].astype(str))
    vectorizer = _vectorizer(config["tfidf"])
    train_matrix = vectorizer.fit_transform(train["target_clause"].astype(str))
    validation_matrix = vectorizer.transform(validation["target_clause"].astype(str))
    weights = compute_training_weights(train, strategy)
    model = LogisticRegression(
        C=float(config["logistic_regression"]["c"]),
        solver=str(config["logistic_regression"]["solver"]),
        max_iter=int(config["logistic_regression"]["max_iter"]),
        random_state=int(config["seed"]),
    )
    model.fit(train_matrix, train["label"].to_numpy(dtype=np.int64), sample_weight=weights)
    logits = model.decision_function(validation_matrix)
    probabilities = model.predict_proba(validation_matrix)[:, 1]
    predictions = validation[
        ["record_id", "family_id", "label", "label_tier"]
    ].copy()
    predictions["raw_logit"] = logits
    predictions["ambiguity_probability"] = probabilities
    predictions["predicted_label_0_5"] = (probabilities >= 0.5).astype(np.int64)
    predictions["included_in_scoring"] = predictions["record_id"].astype(str).isin(
        scoring_ids
    )
    scored = predictions[predictions["included_in_scoring"]]
    labels = scored["label"].to_numpy(dtype=np.int64)
    scored_probabilities = scored["ambiguity_probability"].to_numpy(dtype=np.float64)
    metrics = binary_metrics(labels, scored_probabilities)
    metrics["bootstrap_95_percentile"] = bootstrap_intervals(
        labels,
        scored_probabilities,
        repetitions=int(config["bootstrap_repetitions"]),
        seed=int(config["bootstrap_seed"]) + fold.number,
    )
    metrics["excluded_exact_duplicate_record_ids"] = list(
        fold.excluded_exact_duplicate_record_ids
    )

    output_dir.mkdir(parents=True)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    _write_json(output_dir / "metrics.json", metrics)
    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(model, output_dir / "logistic_regression.joblib")
    run_config = {
        "variant": "target_clause_only",
        "weight_strategy": strategy,
        "fold": fold.number,
        "seed": config["seed"],
        "tfidf": config["tfidf"],
        "logistic_regression": config["logistic_regression"],
        "threshold": 0.5,
    }
    _write_json(output_dir / "run_config.json", run_config)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "training_rows": len(train),
        "training_positives": int(train["label"].sum()),
        "training_weight_mean": float(weights.mean()),
        "validation_rows": len(validation),
        "scored_rows": len(scored),
        "held_out_families": list(fold.held_out_families),
        "fold_assignment_sha256": config["fold_assignment_sha256"],
        "development_train_sha256": config["development_train_sha256"],
        "artifact_hashes": artifact_hashes,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return predictions, {**metrics, "manifest": manifest}


def run(config_path: Path, output_dir: Path) -> Path:
    """Run all frozen folds and weighting strategies into a new immutable directory."""

    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dataset_dir = (ROOT / config["development_dataset_dir"]).resolve()
    frame = load_development_split(dataset_dir, "train")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    train_hash = manifest["split_files"]["train"]["sha256"]
    if train_hash != config["development_train_sha256"]:
        raise RuntimeError("Development train hash differs from comparator config")
    assignment = (ROOT / config["fold_assignment"]).resolve()
    folds = load_frozen_folds(
        frame,
        assignment,
        expected_sha256=config["fold_assignment_sha256"],
        expected_train_sha256=train_hash,
    )
    if output_dir.exists():
        raise FileExistsError(f"Immutable run directory already exists: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}-{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        fold_table, family_table = fold_prevalence_tables(frame, folds)
        fold_table.to_csv(temporary / "fold_prevalence.csv", index=False)
        family_table.to_csv(temporary / "family_prevalence.csv", index=False)
        rows: list[dict[str, Any]] = []
        pooled: dict[str, Any] = {}
        for strategy in config["weight_strategies"]:
            strategy_predictions: list[pd.DataFrame] = []
            for fold in folds:
                predictions, metrics = _fit_fold(
                    frame,
                    fold,
                    strategy,
                    config,
                    temporary / strategy / f"fold-{fold.number}",
                )
                strategy_predictions.append(
                    predictions[predictions["included_in_scoring"]].copy()
                )
                rows.append(_flatten_result(strategy, fold, metrics))
            pooled_predictions = pd.concat(strategy_predictions, ignore_index=True)
            pooled_predictions.insert(0, "weight_strategy", strategy)
            pooled_predictions.to_csv(
                temporary / f"pooled_predictions_{strategy}.csv", index=False
            )
            labels = pooled_predictions["label"].to_numpy(dtype=np.int64)
            probabilities = pooled_predictions["ambiguity_probability"].to_numpy(
                dtype=np.float64
            )
            pooled_metrics = binary_metrics(labels, probabilities)
            pooled_metrics["bootstrap_95_percentile"] = bootstrap_intervals(
                labels,
                probabilities,
                repetitions=int(config["bootstrap_repetitions"]),
                seed=int(config["bootstrap_seed"]),
            )
            pooled[strategy] = pooled_metrics
        pd.DataFrame(rows).to_csv(temporary / "fold_results.csv", index=False)
        _write_json(temporary / "pooled_metrics.json", pooled)
        root_manifest = {
            "schema_version": 1,
            "experiment": config["experiment_name"],
            "config_sha256": sha256_file(config_path),
            "fold_assignment_sha256": sha256_file(assignment),
            "development_train_sha256": train_hash,
            "git_commit": _git_commit(),
            "weight_strategies": config["weight_strategies"],
            "bootstrap": {
                "method": "ordinary paired row resampling, percentile interval",
                "confidence_level": 0.95,
                "repetitions": config["bootstrap_repetitions"],
                "seed": config["bootstrap_seed"],
            },
            "artifacts": {
                path.name: sha256_file(path)
                for path in sorted(temporary.iterdir())
                if path.is_file()
            },
        }
        _write_json(temporary / "run_manifest.json", root_manifest)
        temporary.rename(output_dir)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/classical/results/tfidf-logreg-v1",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    output = run(args.config, args.output_dir.resolve())
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
