"""Analyze the frozen Step 4 target/context factorial without test access."""

from __future__ import annotations

import argparse
import json
import math
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from models.classical.feature_pipeline import sha256_file
from models.classical.tfidf_comparator import bootstrap_intervals
from models.transformer.dataset import ROOT
from models.transformer.experiment import load_development_split
from models.transformer.factorial_artifacts import (
    DEFAULT_OUTPUT as FACTORIAL_ARTIFACTS,
)
from models.transformer.factorial_artifacts import (
    apply_context_policy,
    load_factorial_artifacts,
)
from models.transformer.feasibility_audit import (
    _bootstrap_operating_points,
    routing_point,
)
from models.transformer.lr_screen_analysis import (
    DEVELOPMENT_DIR,
    _paired_pr_auc_bootstrap,
)
from models.transformer.tokenization import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    audit_token_lengths,
)
from models.transformer.weighting_analysis import expected_calibration_error

CONDITIONS = (
    "hard_target_only",
    "soft_target_only",
    "hard_local_context",
    "soft_local_context",
)
SIMPLICITY_ORDER = {name: index for index, name in enumerate(CONDITIONS)}
SEEDS = (1729, 2718, 31415)
FOLDS = (1, 2, 3, 4, 5)
BASELINE_ROOT = ROOT / "experiments/transformer/results/weighting-ablation-v1/class"
FACTORIAL_ROOT = ROOT / "experiments/transformer/results/factorial-v1"
MATCHED_TFIDF_ROOT = ROOT / "experiments/classical/results/tfidf-matched-nested-v1"
DEFAULT_OUTPUT = ROOT / "experiments/transformer/results/factorial-v1-analysis"
CONTRASTS = (
    ("soft_target_only", "hard_target_only", "soft_target_effect"),
    ("hard_local_context", "hard_target_only", "context_effect_hard"),
    ("soft_local_context", "soft_target_only", "context_effect_soft"),
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def select_condition(fold_scores: dict[str, list[float]]) -> dict[str, Any]:
    """Apply the pre-registered one-standard-error simplicity rule."""

    if set(fold_scores) != set(CONDITIONS):
        raise ValueError("Selection requires all four factorial conditions")
    if {len(values) for values in fold_scores.values()} != {len(FOLDS)}:
        raise ValueError("Selection requires five fold scores per condition")
    means = {name: float(np.mean(values)) for name, values in fold_scores.items()}
    ranked = sorted(CONDITIONS, key=lambda name: (-means[name], SIMPLICITY_ORDER[name]))
    top, runner_up = ranked[:2]
    differences = np.asarray(fold_scores[top]) - np.asarray(fold_scores[runner_up])
    standard_error = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
    mean_difference = float(differences.mean())
    within_one_se = mean_difference < standard_error
    selected = (
        min((top, runner_up), key=SIMPLICITY_ORDER.get) if within_one_se else top
    )
    return {
        "seed_averaged_mean_fold_pr_auc": means,
        "ranking": ranked,
        "top_condition": top,
        "runner_up_condition": runner_up,
        "paired_fold_differences_top_minus_runner_up": differences.tolist(),
        "mean_paired_difference": mean_difference,
        "standard_error_paired_differences": standard_error,
        "difference_is_less_than_one_standard_error": within_one_se,
        "selected_condition": selected,
        "simplicity_order": list(CONDITIONS),
    }


def _run_directory(condition: str, fold: int, seed: int) -> tuple[Path, bool]:
    if condition == "hard_target_only":
        return BASELINE_ROOT / f"fold-{fold}-seed-{seed}", True
    slug = {
        "soft_target_only": "soft-target-only",
        "hard_local_context": "hard-local-context",
        "soft_local_context": "soft-local-context",
    }[condition]
    return FACTORIAL_ROOT / slug / f"fold-{fold}-seed-{seed}", False


def _load_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    cells: list[dict[str, Any]] = []
    expected_ids: dict[int, set[str]] = {}
    for condition in CONDITIONS:
        expected_target = "soft" if condition.startswith("soft") else "hard"
        expected_variant = "local_context" if "local_context" in condition else "target_only"
        for seed in SEEDS:
            for fold in FOLDS:
                directory, reused = _run_directory(condition, fold, seed)
                predictions_path = directory / "validation_predictions.csv"
                manifest_path = directory / "run_manifest.json"
                config_path = directory / "run_config.json"
                if not all(
                    path.is_file()
                    for path in (predictions_path, manifest_path, config_path)
                ):
                    raise FileNotFoundError(f"Missing completed factorial cell: {directory}")
                frame = pd.read_csv(predictions_path)
                required = {
                    "record_id",
                    "label",
                    "logit_0",
                    "logit_1",
                    "ambiguity_probability",
                    "included_in_scoring",
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
                    or resolved["weight_strategy"] != "class"
                    or float(resolved["learning_rate"]) != 1e-5
                    or manifest["outer_fold_evaluations_during_training"] != 0
                    or manifest["outer_fold_evaluations_after_selection"] != 1
                ):
                    raise RuntimeError(f"Cell violates the frozen protocol: {directory}")
                if not reused and (
                    resolved["target_regime"] != expected_target
                    or resolved["variant"] != expected_variant
                    or manifest["checkpoint_selection_target"]
                    != "hard_consensus_label_pr_auc"
                ):
                    raise RuntimeError(f"Factorial cell has wrong target/input: {directory}")
                labels = frame["label"].to_numpy(dtype=np.int64)
                scores = frame["ambiguity_probability"].to_numpy(dtype=np.float64)
                prediction_parts.append(
                    frame.assign(condition=condition, seed=seed, fold=fold)
                )
                cells.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "fold": fold,
                        "rows": len(frame),
                        "positives": int(labels.sum()),
                        "pr_auc": float(average_precision_score(labels, scores)),
                        "selected_epoch": int(manifest["best_epoch"]),
                        "runtime_seconds": float(manifest["total_runtime_seconds"]),
                        "peak_gpu_memory_allocated_mb": float(
                            manifest["peak_gpu_memory_allocated_mb"]
                        ),
                        "reused_step_3_control": reused,
                        "run_directory": str(directory.relative_to(ROOT)),
                    }
                )
    return pd.concat(prediction_parts, ignore_index=True), pd.DataFrame(cells)


def _ensemble(predictions: pd.DataFrame) -> pd.DataFrame:
    source = predictions.copy()
    source["raw_log_odds"] = source["logit_1"] - source["logit_0"]
    identity = ["condition", "fold", "record_id", "label"]
    result = source.groupby(identity, as_index=False, sort=True).agg(
        raw_log_odds=("raw_log_odds", "mean"), seeds=("seed", "nunique")
    )
    if not (result["seeds"] == len(SEEDS)).all():
        raise RuntimeError("Every ensemble row must contain all three seeds")
    result["ambiguity_probability"] = 1.0 / (1.0 + np.exp(-result["raw_log_odds"]))
    return result.drop(columns="seeds")


def _metadata() -> pd.DataFrame:
    train = load_development_split(DEVELOPMENT_DIR, "train")
    metadata = train[["record_id", "document_id", "section_ids", "label_tier"]].copy()
    metadata["section_cluster"] = (
        metadata["document_id"].astype(str)
        + "::"
        + metadata["section_ids"].astype(str)
    )
    return metadata


def _tokenization_audit() -> dict[str, Any]:
    """Confirm target/context truncation after each fold's sanitation policy."""

    from transformers import AutoTokenizer

    train = load_development_split(DEVELOPMENT_DIR, "train")
    _, policy, artifact_manifest = load_factorial_artifacts(FACTORIAL_ARTIFACTS)
    tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_MODEL,
        revision=DEFAULT_REVISION,
        local_files_only=True,
        use_fast=True,
    )
    folds: dict[str, Any] = {}
    for fold in FOLDS:
        sanitized, removals = apply_context_policy(train, policy, fold=fold)
        folds[f"fold_{fold}"] = {
            **audit_token_lengths(tokenizer, sanitized, max_length=256),
            "context_removals": removals,
        }
    return {
        "model": DEFAULT_MODEL,
        "revision": DEFAULT_REVISION,
        "factorial_artifact_hashes": artifact_manifest["artifact_hashes"],
        "folds": folds,
    }


def _summaries(
    cells: pd.DataFrame, ensembles: pd.DataFrame, *, repetitions: int
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    records: list[dict[str, Any]] = []
    fold_scores: dict[str, list[float]] = {}
    for index, condition in enumerate(CONDITIONS):
        condition_cells = cells[cells["condition"] == condition]
        seed_fold = condition_cells.pivot(index="fold", columns="seed", values="pr_auc")
        averaged = seed_fold.mean(axis=1).reindex(FOLDS)
        fold_scores[condition] = averaged.tolist()
        pooled = ensembles[ensembles["condition"] == condition].reset_index(drop=True)
        labels = pooled["label"].to_numpy(dtype=np.int64)
        scores = pooled["ambiguity_probability"].to_numpy(dtype=np.float64)
        route_frame = pooled[["fold", "label"]].copy()
        route_frame["score"] = scores
        route = routing_point(labels, scores, 0.90)
        route_ci = _bootstrap_operating_points(
            route_frame,
            strategy="model",
            repetitions=repetitions,
            seed=44100 + index,
        )[0.90]["referral_rate"]
        pr_ci = bootstrap_intervals(
            labels, scores, repetitions=repetitions, seed=44200 + index
        )["pr_auc"]
        records.append(
            {
                "condition": condition,
                "seed_averaged_mean_fold_pr_auc": float(averaged.mean()),
                "pooled_three_seed_logit_average_pr_auc": float(
                    average_precision_score(labels, scores)
                ),
                "pooled_pr_auc_ci_lower": pr_ci["lower"],
                "pooled_pr_auc_ci_upper": pr_ci["upper"],
                "referral_rate_at_90_recall": float(route["referral_rate"]),
                "achieved_recall": float(route["ambiguous_recall"]),
                "referral_rate_ci_lower": route_ci["lower"],
                "referral_rate_ci_upper": route_ci["upper"],
                "pooled_brier_secondary_uncalibrated": float(
                    brier_score_loss(labels, scores)
                ),
                "pooled_ece_10_bin_secondary_uncalibrated": expected_calibration_error(
                    labels, scores
                ),
                "runtime_seconds_all_cells": float(condition_cells["runtime_seconds"].sum()),
                "incremental_runtime_seconds": float(
                    condition_cells.loc[
                        ~condition_cells["reused_step_3_control"], "runtime_seconds"
                    ].sum()
                ),
                "maximum_peak_gpu_memory_allocated_mb": float(
                    condition_cells["peak_gpu_memory_allocated_mb"].max()
                ),
            }
        )
    return pd.DataFrame(records), fold_scores


def _bootstrap_contrast(
    wide: pd.DataFrame,
    first: str,
    second: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "paired_row": _paired_pr_auc_bootstrap(
            wide, first, second, repetitions=repetitions, seed=seed
        ),
        "section_cluster": _paired_pr_auc_bootstrap(
            wide,
            first,
            second,
            repetitions=repetitions,
            seed=seed + 100,
            cluster_column="section_cluster",
        ),
    }


def _contrasts(
    ensembles: pd.DataFrame,
    metadata: pd.DataFrame,
    fold_scores: dict[str, list[float]],
    *,
    repetitions: int,
) -> list[dict[str, Any]]:
    wide = ensembles.pivot(
        index=["record_id", "fold", "label"],
        columns="condition",
        values="ambiguity_probability",
    ).reset_index()
    wide = wide.merge(metadata, on="record_id", how="left", validate="one_to_one")
    records: list[dict[str, Any]] = []
    for index, (first, second, name) in enumerate(CONTRASTS):
        differences = np.asarray(fold_scores[first]) - np.asarray(fold_scores[second])
        records.append(
            {
                "contrast": name,
                "first": first,
                "second": second,
                "mean_seed_averaged_fold_difference": float(differences.mean()),
                "standard_error_paired_fold_differences": float(
                    np.std(differences, ddof=1) / math.sqrt(len(differences))
                ),
                "per_fold_differences": differences.tolist(),
                **_bootstrap_contrast(
                    wide,
                    first,
                    second,
                    repetitions=repetitions,
                    seed=44300 + index,
                ),
            }
        )
    hard_context = np.asarray(fold_scores["hard_local_context"]) - np.asarray(
        fold_scores["hard_target_only"]
    )
    soft_context = np.asarray(fold_scores["soft_local_context"]) - np.asarray(
        fold_scores["soft_target_only"]
    )
    interaction = soft_context - hard_context
    # A PR-AUC interaction has no single paired-row equivalent, so bootstrap the
    # four score columns directly within each resample.
    labels = wide["label"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(44400)
    row_samples: list[float] = []
    cluster_samples: list[float] = []
    fold_indices = [group.index.to_numpy() for _, group in wide.groupby("fold")]
    cluster_indices = {
        int(fold): [
            group.index.to_numpy()
            for _, group in subset.groupby("section_cluster", sort=True)
        ]
        for fold, subset in wide.groupby("fold", sort=True)
    }
    for _ in range(repetitions):
        row_selected = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in fold_indices]
        )
        selected_clusters: list[np.ndarray] = []
        for fold in FOLDS:
            clusters = cluster_indices[fold]
            choices = rng.integers(0, len(clusters), len(clusters))
            selected_clusters.extend(clusters[index] for index in choices)
        cluster_selected = np.concatenate(selected_clusters)
        for selected, destination in (
            (row_selected, row_samples),
            (cluster_selected, cluster_samples),
        ):
            if len(np.unique(labels[selected])) < 2:
                continue
            pr = {
                condition: average_precision_score(
                    labels[selected], wide[condition].to_numpy()[selected]
                )
                for condition in CONDITIONS
            }
            destination.append(
                (pr["soft_local_context"] - pr["soft_target_only"])
                - (pr["hard_local_context"] - pr["hard_target_only"])
            )
    point_pr = {
        condition: average_precision_score(labels, wide[condition])
        for condition in CONDITIONS
    }
    pooled_point = (
        point_pr["soft_local_context"]
        - point_pr["soft_target_only"]
        - point_pr["hard_local_context"]
        + point_pr["hard_target_only"]
    )
    records.append(
        {
            "contrast": "soft_by_context_interaction",
            "mean_seed_averaged_fold_difference": float(interaction.mean()),
            "standard_error_paired_fold_differences": float(
                np.std(interaction, ddof=1) / math.sqrt(len(interaction))
            ),
            "per_fold_differences": interaction.tolist(),
            "pooled_point_difference": float(pooled_point),
            "paired_row": {
                "lower": float(np.percentile(row_samples, 2.5)),
                "upper": float(np.percentile(row_samples, 97.5)),
                "valid_bootstraps": len(row_samples),
            },
            "section_cluster": {
                "lower": float(np.percentile(cluster_samples, 2.5)),
                "upper": float(np.percentile(cluster_samples, 97.5)),
                "valid_bootstraps": len(cluster_samples),
            },
        }
    )
    return records


def _matched_tfidf_comparison(
    ensembles: pd.DataFrame,
    metadata: pd.DataFrame,
    transformer_fold_scores: list[float],
    *,
    repetitions: int,
) -> dict[str, Any]:
    transformer = ensembles[ensembles["condition"] == "hard_target_only"][
        ["record_id", "fold", "label", "ambiguity_probability"]
    ].rename(columns={"ambiguity_probability": "deberta"})
    parts: list[pd.DataFrame] = []
    for fold in FOLDS:
        frame = pd.read_csv(
            MATCHED_TFIDF_ROOT / f"fold-{fold}" / "validation_predictions.csv"
        )
        frame = frame[frame["included_in_scoring"]].copy()
        parts.append(
            frame[["record_id", "label", "ambiguity_probability"]]
            .rename(columns={"ambiguity_probability": "tfidf_matched"})
            .assign(fold=fold)
        )
    tfidf = pd.concat(parts, ignore_index=True)
    wide = transformer.merge(
        tfidf, on=["record_id", "fold", "label"], validate="one_to_one"
    ).merge(metadata, on="record_id", validate="one_to_one")
    if len(wide) != len(transformer) or len(wide) != len(tfidf):
        raise RuntimeError("Matched TF-IDF and transformer scoring rows do not align")
    ensemble_fold_differences: list[float] = []
    tfidf_fold_scores: list[float] = []
    for fold in FOLDS:
        current = wide[wide["fold"] == fold]
        tfidf_score = float(
            average_precision_score(current["label"], current["tfidf_matched"])
        )
        tfidf_fold_scores.append(tfidf_score)
        ensemble_fold_differences.append(
            float(average_precision_score(current["label"], current["deberta"]))
            - tfidf_score
        )
    primary_differences = np.asarray(transformer_fold_scores) - np.asarray(
        tfidf_fold_scores
    )
    return {
        "contrast": "hard_target_deberta_minus_matched_class_tfidf",
        "mean_seed_averaged_fold_difference": float(primary_differences.mean()),
        "standard_error_paired_fold_differences": float(
            np.std(primary_differences, ddof=1)
            / math.sqrt(len(primary_differences))
        ),
        "per_fold_seed_averaged_differences": primary_differences.tolist(),
        "mean_ensemble_fold_difference": float(np.mean(ensemble_fold_differences)),
        "per_fold_ensemble_differences": ensemble_fold_differences,
        **_bootstrap_contrast(
            wide,
            "deberta",
            "tfidf_matched",
            repetitions=repetitions,
            seed=44500,
        ),
    }


def _render_report(
    summaries: pd.DataFrame,
    selection: dict[str, Any],
    contrasts: list[dict[str, Any]],
    matched: dict[str, Any],
) -> str:
    lines = [
        "# Step 4 factorial report",
        "",
        "All values are development out-of-fold agreement with the frozen ",
        "LLM-consensus operational proxy gold labels, not human-verified accuracy.",
        "TCP and the sealed DNS test were not used.",
        "",
        "## Conditions",
        "",
        "| Condition | Mean-fold PR-AUC | Pooled PR-AUC | 95% row CI | Referral at 90% recall | Runtime (min) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.seed_averaged_mean_fold_pr_auc:.4f} | "
            f"{row.pooled_three_seed_logit_average_pr_auc:.4f} | "
            f"[{row.pooled_pr_auc_ci_lower:.4f}, {row.pooled_pr_auc_ci_upper:.4f}] | "
            f"{row.referral_rate_at_90_recall:.3f} | "
            f"{row.runtime_seconds_all_cells / 60:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen selection",
            "",
            (
                f"Selected condition: `{selection['selected_condition']}`. Top observed "
                f"condition: `{selection['top_condition']}`; runner-up: "
                f"`{selection['runner_up_condition']}`. Top-minus-runner-up mean paired "
                f"difference: {selection['mean_paired_difference']:.4f}; "
                "paired-fold SE: "
                f"{selection['standard_error_paired_differences']:.4f}."
            ),
            "",
            "## Pre-registered contrasts",
            "",
            "| Contrast | Mean fold difference | Paired-row 95% CI | Section-cluster 95% CI |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for contrast in contrasts:
        lines.append(
            f"| {contrast['contrast']} | "
            f"{contrast['mean_seed_averaged_fold_difference']:.4f} | "
            f"[{contrast['paired_row']['lower']:.4f}, "
            f"{contrast['paired_row']['upper']:.4f}] | "
            f"[{contrast['section_cluster']['lower']:.4f}, "
            f"{contrast['section_cluster']['upper']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Matched architecture contrast",
            "",
            (
                "The hard-target target-only DeBERTa control minus class-weighted "
                "TF-IDF trained on the exact same nested fitting IDs has mean-fold "
                f"difference {matched['mean_seed_averaged_fold_difference']:.4f}. "
                "Its pooled paired-row 95% CI is "
                f"[{matched['paired_row']['lower']:.4f}, "
                f"{matched['paired_row']['upper']:.4f}], and section-cluster 95% CI "
                f"is [{matched['section_cluster']['lower']:.4f}, "
                f"{matched['section_cluster']['upper']:.4f}]."
            ),
            "",
            "## Limitations",
            "",
            (
                "The reference labels are an LLM consensus and no multi-annotator "
                "human gold set is currently available. Folds still contain related "
                "protocol clusters, so estimates can be optimistic for a genuinely "
                "new protocol. Probabilities are uncalibrated; ECE and Brier are "
                "secondary diagnostics. No threshold, calibration, TCP verification, "
                "or DNS evaluation was performed in this step."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def analyze(output: Path, *, repetitions: int) -> Path:
    """Create immutable factorial tables and a concise machine-readable report."""

    if output.exists():
        raise FileExistsError(f"Factorial analysis output is immutable: {output}")
    predictions, cells = _load_cells()
    ensembles = _ensemble(predictions)
    metadata = _metadata()
    summaries, fold_scores = _summaries(cells, ensembles, repetitions=repetitions)
    selection = select_condition(fold_scores)
    contrasts = _contrasts(
        ensembles, metadata, fold_scores, repetitions=repetitions
    )
    matched = _matched_tfidf_comparison(
        ensembles,
        metadata,
        fold_scores["hard_target_only"],
        repetitions=repetitions,
    )
    tokenization_audit = _tokenization_audit()
    temporary = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    try:
        cells.to_csv(temporary / "cell_metrics.csv", index=False, lineterminator="\n")
        summaries.to_csv(
            temporary / "condition_summary.csv", index=False, lineterminator="\n"
        )
        ensembles.to_csv(
            temporary / "oof_ensemble_predictions.csv", index=False, lineterminator="\n"
        )
        _write_json(temporary / "selection.json", selection)
        _write_json(temporary / "factorial_contrasts.json", contrasts)
        _write_json(temporary / "matched_tfidf_comparison.json", matched)
        _write_json(temporary / "tokenization_audit.json", tokenization_audit)
        (temporary / "report.md").write_text(
            _render_report(summaries, selection, contrasts, matched), encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "scope": "development_train_only",
            "bootstrap_repetitions": repetitions,
            "conditions": list(CONDITIONS),
            "seeds": list(SEEDS),
            "folds": list(FOLDS),
            "selected_condition": selection["selected_condition"],
            "dns_opened": False,
            "tcp_used": False,
            "input_hashes": {
                "experiment_plan.md": sha256_file(
                    ROOT / "experiments/transformer/EXPERIMENT_PLAN.md"
                ),
                "train.py": sha256_file(ROOT / "models/transformer/train.py"),
                "modeling.py": sha256_file(ROOT / "models/transformer/modeling.py"),
                "factorial_artifacts.py": sha256_file(
                    ROOT / "models/transformer/factorial_artifacts.py"
                ),
                "factorial_artifact_manifest.json": sha256_file(
                    FACTORIAL_ARTIFACTS / "manifest.json"
                ),
                "matched_tfidf_manifest.json": sha256_file(
                    MATCHED_TFIDF_ROOT / "manifest.json"
                ),
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        import shutil

        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = analyze(args.output.resolve(), repetitions=args.bootstrap_repetitions)
    print(f"Factorial analysis written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
