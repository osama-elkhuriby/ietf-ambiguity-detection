"""Nested lexical/Qwen zero-shot hybrid for the third LLM experiment."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import FeatureUnion

import models.llm.in_context_evaluate as v1
import models.llm.in_context_evaluate_v2 as v2
from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT

DEFAULT_CONFIG = (
    ROOT / "experiments/llm/configs/qwen3_32b_final_annotation_ready_hybrid_v3.yaml"
)
FOLDS = (1, 2, 3, 4, 5)


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_type") != "nested_target_lexical_qwen_zero_shot_hybrid":
        raise ValueError("V3 requires the nested lexical/Qwen zero-shot hybrid")
    updates = config.get("parameter_updates", {})
    if bool(updates.get("qwen3_32b", True)):
        raise ValueError("V3 must not update Qwen parameters")
    if not bool(updates.get("lexical_classifier", False)):
        raise ValueError("V3 requires a fitted lexical classifier")
    base = config.get("base_signal", {})
    if base.get("model") != "qwen3:32b" or base.get("condition") != "zero_shot":
        raise ValueError("V3 requires the frozen Qwen3 32B zero-shot signal")
    if not bool(base.get("label_independent", False)):
        raise ValueError("Nested blending requires a label-independent base signal")
    if config["lexical_model"].get("input_field") != "target_clause":
        raise ValueError("V3 deliberately fits lexical features to target_clause only")
    selection = config["selection"]
    if selection.get("protocol") != "nested_grouped_cross_validation":
        raise ValueError("V3 requires nested grouped cross-validation")
    if int(selection.get("n_outer_folds", 0)) != len(FOLDS):
        raise ValueError("V3 requires the five frozen outer folds")
    c_grid = tuple(float(value) for value in selection["c_grid"])
    weights = tuple(float(value) for value in selection["lexical_weight_grid"])
    if not c_grid or min(c_grid) <= 0:
        raise ValueError("Every logistic-regression C must be positive")
    if not weights or min(weights) < 0 or max(weights) > 1:
        raise ValueError("Lexical blend weights must be in [0, 1]")
    if tuple(selection["threshold_objectives"]) != (
        "f1",
        "f2",
        "balanced_accuracy",
    ):
        raise ValueError("Unexpected V3 threshold objectives")


def _load_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, int]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    source = (ROOT / str(config["source_annotation"])).resolve()
    if sha256_file(source) != config["source_annotation_sha256"]:
        raise RuntimeError("Configured final_annotation_ready.csv hash mismatch")
    dataset_dir = (ROOT / str(config["dataset_dir"])).resolve()
    train_path = dataset_dir / "train.csv"
    if sha256_file(train_path) != config["development_train_sha256"]:
        raise RuntimeError("Prepared development train hash mismatch")
    frame = pd.read_csv(train_path).fillna("")
    if frame["record_id"].duplicated().any() or set(frame["split"]) != {"train"}:
        raise RuntimeError("Development data must contain unique train records")
    folds_path = (ROOT / str(config["selection"]["outer_fold_assignment"])).resolve()
    if sha256_file(folds_path) != config["selection"]["outer_fold_assignment_sha256"]:
        raise RuntimeError("Frozen fold-assignment hash mismatch")
    fold_spec = json.loads(folds_path.read_text(encoding="utf-8"))
    family_to_fold = {
        str(family): int(fold) for family, fold in fold_spec["family_to_fold"].items()
    }
    if set(frame["family_id"]) != set(family_to_fold):
        raise RuntimeError("Frozen folds do not cover exactly the development families")
    frame["fold"] = frame["family_id"].map(family_to_fold).astype(int)
    if set(frame["fold"]) != set(FOLDS):
        raise RuntimeError("Development records do not cover all five folds")
    forbidden = {"draft-ietf-tcpm-rfc793bis", "draft-ietf-dnsop-svcb-https"}
    if set(frame["family_id"]) & forbidden:
        raise RuntimeError("Reserved TCP or sealed DNS data entered V3 development")
    base_path = (ROOT / str(config["base_signal"]["predictions"])).resolve()
    if sha256_file(base_path) != config["base_signal"]["predictions_sha256"]:
        raise RuntimeError("Frozen Qwen zero-shot prediction hash mismatch")
    base = pd.read_csv(base_path)
    needed = {"record_id", "fold", "label", "ambiguity_probability"}
    if not needed.issubset(base) or base["record_id"].duplicated().any():
        raise RuntimeError("Invalid Qwen zero-shot prediction artifact")
    base = base[["record_id", "fold", "label", "ambiguity_probability"]].rename(
        columns={"ambiguity_probability": "qwen_zero_shot_probability"}
    )
    merged = frame[["record_id", "label", "fold"]].merge(
        base,
        on="record_id",
        how="outer",
        suffixes=("_data", "_base"),
        validate="one_to_one",
    )
    if len(merged) != len(frame) or merged.isna().any().any():
        raise RuntimeError(
            "Qwen zero-shot predictions do not exactly cover development"
        )
    if not np.array_equal(merged["label_data"], merged["label_base"]):
        raise RuntimeError("Qwen artifact labels differ from development labels")
    if not np.array_equal(merged["fold_data"], merged["fold_base"]):
        raise RuntimeError("Qwen artifact folds differ from frozen folds")
    if not np.isfinite(merged["qwen_zero_shot_probability"]).all():
        raise RuntimeError("Qwen artifact contains a non-finite probability")
    base = merged[["record_id", "qwen_zero_shot_probability"]]
    return config, frame, base, family_to_fold


def _vectorizer(config: dict[str, Any]) -> FeatureUnion:
    lexical = config["lexical_model"]
    word = lexical["word_tfidf"]
    char = lexical["character_tfidf"]
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    ngram_range=tuple(word["ngram_range"]),
                    min_df=int(word["min_df"]),
                    max_df=float(word["max_df"]),
                    max_features=int(word["max_features"]),
                    sublinear_tf=bool(word["sublinear_tf"]),
                    strip_accents=str(word["strip_accents"]),
                ),
            ),
            (
                "character",
                TfidfVectorizer(
                    analyzer=str(char["analyzer"]),
                    ngram_range=tuple(char["ngram_range"]),
                    min_df=int(char["min_df"]),
                    max_features=int(char["max_features"]),
                    sublinear_tf=bool(char["sublinear_tf"]),
                ),
            ),
        ]
    )


def _classifier(config: dict[str, Any], c_value: float) -> LogisticRegression:
    lexical = config["lexical_model"]
    return LogisticRegression(
        C=float(c_value),
        class_weight=str(lexical["class_weight"]),
        solver=str(lexical["solver"]),
        max_iter=int(lexical["max_iter"]),
        random_state=int(lexical["random_seed"]),
    )


def blend_probabilities(
    lexical_probability: np.ndarray,
    qwen_probability: np.ndarray,
    lexical_weight: float,
) -> np.ndarray:
    """Blend evidence in log-odds space while avoiding infinite logits."""

    epsilon = 1e-6
    lexical = np.clip(
        np.asarray(lexical_probability, dtype=float), epsilon, 1 - epsilon
    )
    qwen = np.clip(np.asarray(qwen_probability, dtype=float), epsilon, 1 - epsilon)
    weight = float(lexical_weight)
    return expit(weight * logit(lexical) + (1.0 - weight) * logit(qwen))


def select_candidate(
    labels: np.ndarray,
    qwen_probability: np.ndarray,
    lexical_predictions: dict[float, np.ndarray],
    lexical_weights: tuple[float, ...],
) -> tuple[float, float, pd.DataFrame]:
    """Choose C and blend weight by inner out-of-fold PR-AUC only."""

    rows: list[dict[str, float]] = []
    for c_value, lexical in lexical_predictions.items():
        for weight in lexical_weights:
            score = blend_probabilities(lexical, qwen_probability, weight)
            rows.append(
                {
                    "c": float(c_value),
                    "lexical_weight": float(weight),
                    "inner_pr_auc": float(average_precision_score(labels, score)),
                }
            )
    candidates = pd.DataFrame(rows).sort_values(
        ["inner_pr_auc", "c", "lexical_weight"],
        ascending=[False, True, True],
        kind="stable",
    )
    best = candidates.iloc[0]
    return float(best["c"]), float(best["lexical_weight"]), candidates


def _nested_outer_predictions(
    config: dict[str, Any], frame: pd.DataFrame, base: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = frame.merge(base, on="record_id", how="left", validate="one_to_one")
    labels = work["label"].to_numpy(dtype=np.int64)
    folds = work["fold"].to_numpy(dtype=np.int64)
    qwen = work["qwen_zero_shot_probability"].to_numpy(dtype=float)
    text = work["target_clause"].astype(str)
    c_grid = tuple(float(value) for value in config["selection"]["c_grid"])
    weights = tuple(
        float(value) for value in config["selection"]["lexical_weight_grid"]
    )
    output = work[
        ["record_id", "family_id", "label_tier", "label", "fold", "target_clause"]
    ].copy()
    output["qwen_zero_shot_probability"] = qwen
    output["lexical_probability"] = np.nan
    output["ambiguity_probability"] = np.nan
    output["selected_c"] = np.nan
    output["selected_lexical_weight"] = np.nan
    for objective in config["selection"]["threshold_objectives"]:
        output[f"threshold_{objective}"] = np.nan
        output[f"predicted_label_{objective}"] = -1
    tuning_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for outer_fold in FOLDS:
        outer_train = folds != outer_fold
        outer_test = folds == outer_fold
        inner_predictions = {
            c_value: np.full(int(outer_train.sum()), np.nan, dtype=float)
            for c_value in c_grid
        }
        outer_train_indices = np.flatnonzero(outer_train)
        local_position = {
            index: position for position, index in enumerate(outer_train_indices)
        }
        for inner_fold in sorted(set(folds[outer_train])):
            inner_train = outer_train & (folds != inner_fold)
            inner_valid = outer_train & (folds == inner_fold)
            vectorizer = _vectorizer(config)
            train_matrix = vectorizer.fit_transform(text.loc[inner_train])
            valid_matrix = vectorizer.transform(text.loc[inner_valid])
            valid_positions = [
                local_position[index] for index in np.flatnonzero(inner_valid)
            ]
            for c_value in c_grid:
                model = _classifier(config, c_value).fit(
                    train_matrix, labels[inner_train]
                )
                inner_predictions[c_value][valid_positions] = model.predict_proba(
                    valid_matrix
                )[:, 1]
        if any(np.isnan(values).any() for values in inner_predictions.values()):
            raise RuntimeError(
                f"Incomplete inner predictions for outer fold {outer_fold}"
            )
        best_c, best_weight, candidates = select_candidate(
            labels[outer_train], qwen[outer_train], inner_predictions, weights
        )
        candidates.insert(0, "outer_fold", outer_fold)
        candidates["selected"] = np.isclose(candidates["c"], best_c) & np.isclose(
            candidates["lexical_weight"], best_weight
        )
        tuning_rows.append(candidates)
        selected_inner = blend_probabilities(
            inner_predictions[best_c], qwen[outer_train], best_weight
        )
        vectorizer = _vectorizer(config)
        train_matrix = vectorizer.fit_transform(text.loc[outer_train])
        test_matrix = vectorizer.transform(text.loc[outer_test])
        model = _classifier(config, best_c).fit(train_matrix, labels[outer_train])
        lexical_probability = model.predict_proba(test_matrix)[:, 1]
        hybrid_probability = blend_probabilities(
            lexical_probability, qwen[outer_test], best_weight
        )
        output.loc[outer_test, "lexical_probability"] = lexical_probability
        output.loc[outer_test, "ambiguity_probability"] = hybrid_probability
        output.loc[outer_test, "selected_c"] = best_c
        output.loc[outer_test, "selected_lexical_weight"] = best_weight
        fold_metric = v1.classification_metrics(
            labels[outer_test],
            hybrid_probability,
            (hybrid_probability >= 0.5).astype(int),
        )
        matrix = fold_metric.pop("confusion_matrix")
        fold_rows.append(
            {
                "fold": outer_fold,
                "held_out_families": ";".join(
                    sorted(set(work.loc[outer_test, "family_id"]))
                ),
                "selected_c": best_c,
                "selected_lexical_weight": best_weight,
                **fold_metric,
                **matrix,
            }
        )
        for objective in config["selection"]["threshold_objectives"]:
            threshold = v2._best_threshold(
                labels[outer_train], selected_inner, objective
            )
            output.loc[outer_test, f"threshold_{objective}"] = threshold
            output.loc[outer_test, f"predicted_label_{objective}"] = (
                hybrid_probability >= threshold
            ).astype(int)
    if output["ambiguity_probability"].isna().any():
        raise RuntimeError("V3 outer predictions are incomplete")
    output["predicted_label_raw"] = (output["ambiguity_probability"] >= 0.5).astype(int)
    return output, pd.DataFrame(fold_rows), pd.concat(tuning_rows, ignore_index=True)


def _comparison_svg(rows: list[dict[str, Any]]) -> str:
    metrics = ("pr_auc", "precision", "recall", "f1", "balanced_accuracy")
    colors = ("#64748b", "#dc2626", "#2563eb", "#16a34a")
    width, height = 1120, 500
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="30" text-anchor="middle" font-size="22" font-family="sans-serif">V1, V2, and V3 grouped-development comparison</text>',
    ]
    for tick in range(6):
        value, y = tick / 5, 365 - tick * 58
        parts.append(
            f'<line x1="80" y1="{y}" x2="1080" y2="{y}" stroke="#e2e8f0"/><text x="68" y="{y + 4}" text-anchor="end" font-size="12">{value:.1f}</text>'
        )
    for metric_index, metric in enumerate(metrics):
        center = 100 + metric_index * 195 + 92
        parts.append(
            f'<text x="{center}" y="390" text-anchor="middle" font-size="12">{html.escape(metric.replace("_", " "))}</text>'
        )
        for row_index, row in enumerate(rows):
            value = float(row[metric])
            x, y = center - 64 + row_index * 32, 365 - value * 290
            parts.append(
                f'<rect x="{x}" y="{y:.1f}" width="25" height="{365 - y:.1f}" fill="{colors[row_index]}"/>'
            )
    for index, row in enumerate(rows):
        x = 55 + index * 265
        parts.append(
            f'<rect x="{x}" y="445" width="14" height="14" fill="{colors[index]}"/><text x="{x + 20}" y="457" font-size="12">{html.escape(str(row["series"]))}</text>'
        )
    return "".join(parts) + "</svg>\n"


def _pr_curve_svg(labels: np.ndarray, series: list[tuple[str, np.ndarray]]) -> str:
    colors = ("#64748b", "#dc2626", "#16a34a")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="520" viewBox="0 0 720 520"><rect width="100%" height="100%" fill="white"/>',
        '<text x="360" y="28" text-anchor="middle" font-size="21" font-family="sans-serif">Development precision-recall curves</text>',
    ]
    for tick in range(6):
        x, y = 75 + tick * 110, 440 - tick * 72
        parts.append(
            f'<line x1="75" y1="{y}" x2="625" y2="{y}" stroke="#e2e8f0"/><line x1="{x}" y1="80" x2="{x}" y2="440" stroke="#e2e8f0"/><text x="62" y="{y + 4}" text-anchor="end" font-size="11">{tick / 5:.1f}</text><text x="{x}" y="458" text-anchor="middle" font-size="11">{tick / 5:.1f}</text>'
        )
    for index, (name, scores) in enumerate(series):
        precision, recall, _ = precision_recall_curve(labels, scores)
        points = " ".join(
            f"{75 + 550 * r:.1f},{440 - 360 * p:.1f}"
            for p, r in zip(precision, recall, strict=True)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[index]}" stroke-width="2"/>'
        )
        y = 485 + index * 0
        x = 80 + index * 205
        parts.append(
            f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" stroke="{colors[index]}" stroke-width="3"/><text x="{x + 35}" y="{y + 4}" font-size="12">{html.escape(name)}</text>'
        )
    parts.extend(
        [
            '<text x="350" y="478" text-anchor="middle" font-size="14">Recall</text>',
            '<text x="20" y="260" text-anchor="middle" font-size="14" transform="rotate(-90 20 260)">Precision</text>',
        ]
    )
    return "".join(parts) + "</svg>\n"


def audit(config_path: Path, output: Path) -> Path:
    config, frame, base, family_to_fold = _load_protocol(config_path)
    if output.exists():
        raise FileExistsError(f"V3 audit output is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": config["experiment_name"],
        "created_at_utc": _utc_now(),
        "config_sha256": sha256_file(config_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "rows": len(frame),
        "positives": int(frame["label"].sum()),
        "families": sorted(family_to_fold),
        "fold_counts": {
            str(key): int(value)
            for key, value in frame["fold"].value_counts().sort_index().items()
        },
        "qwen_base_rows": len(base),
        "base_signal_is_label_independent": True,
        "tcp_opened": False,
        "dns_opened": False,
        "leakage_checks_passed": True,
    }
    _json(output, report)
    return output


def _series_metrics(
    name: str, labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    metrics = v1.classification_metrics(labels, scores, predictions)
    matrix = metrics.pop("confusion_matrix")
    return {"series": name, **metrics, **matrix}


def run(config_path: Path, output: Path) -> Path:
    config, frame, base, _ = _load_protocol(config_path)
    if output.exists():
        raise FileExistsError(f"V3 output is immutable: {output}")
    temporary = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        predictions, fold_metrics, tuning = _nested_outer_predictions(
            config, frame, base
        )
        labels = predictions["label"].to_numpy(dtype=np.int64)
        scores = predictions["ambiguity_probability"].to_numpy(dtype=float)
        metric_rows = [
            _series_metrics(
                "v3_nested_hybrid_raw",
                labels,
                scores,
                predictions["predicted_label_raw"].to_numpy(dtype=np.int64),
            )
        ]
        for objective in config["selection"]["threshold_objectives"]:
            metric_rows.append(
                _series_metrics(
                    f"v3_nested_hybrid_{objective}_threshold",
                    labels,
                    scores,
                    predictions[f"predicted_label_{objective}"].to_numpy(
                        dtype=np.int64
                    ),
                )
            )
        metrics = pd.DataFrame(metric_rows)
        label_rows: list[dict[str, Any]] = []
        for tier, subset in predictions.groupby("label_tier", sort=True):
            tier_labels = subset["label"].to_numpy(dtype=np.int64)
            tier_scores = subset["ambiguity_probability"].to_numpy(dtype=float)
            for series, column in [
                ("raw", "predicted_label_raw"),
                ("f1_threshold", "predicted_label_f1"),
                ("f2_threshold", "predicted_label_f2"),
            ]:
                row = _series_metrics(
                    series,
                    tier_labels,
                    tier_scores,
                    subset[column].to_numpy(dtype=np.int64),
                )
                row["label_tier"] = tier
                label_rows.append(row)
        v1_path = (ROOT / str(config["comparison"]["v1_metrics"])).resolve()
        v2_path = (ROOT / str(config["comparison"]["v2_metrics"])).resolve()
        for path, expected in [
            (v1_path, config["comparison"]["v1_metrics_sha256"]),
            (v2_path, config["comparison"]["v2_metrics_sha256"]),
        ]:
            if sha256_file(path) != expected:
                raise RuntimeError(f"Comparison artifact hash mismatch: {path}")
        old_v1 = (
            pd.read_csv(v1_path).sort_values("pooled_pr_auc", ascending=False).iloc[0]
        )
        old_v2 = pd.read_csv(v2_path).sort_values("pr_auc", ascending=False).iloc[0]
        v3_raw = metrics.iloc[0]
        v3_f1 = metrics.loc[metrics["series"] == "v3_nested_hybrid_f1_threshold"].iloc[
            0
        ]
        comparison_rows = [
            {
                "series": "V1 best raw (8-shot)",
                "pr_auc": float(old_v1["pooled_pr_auc"]),
                **{
                    key: float(old_v1[key])
                    for key in ("precision", "recall", "f1", "balanced_accuracy")
                },
            },
            {
                "series": "V2 best raw (24-shot)",
                "pr_auc": float(old_v2["pr_auc"]),
                **{
                    key: float(old_v2[key])
                    for key in ("precision", "recall", "f1", "balanced_accuracy")
                },
            },
            {
                "series": "V3 nested hybrid raw",
                **{
                    key: float(v3_raw[key])
                    for key in (
                        "pr_auc",
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                    )
                },
            },
            {
                "series": "V3 nested hybrid F1 threshold",
                **{
                    key: float(v3_f1[key])
                    for key in (
                        "pr_auc",
                        "precision",
                        "recall",
                        "f1",
                        "balanced_accuracy",
                    )
                },
            },
        ]
        predictions.to_csv(
            temporary / "oof_predictions.csv", index=False, lineterminator="\n"
        )
        fold_metrics.to_csv(
            temporary / "fold_metrics.csv", index=False, lineterminator="\n"
        )
        tuning.to_csv(temporary / "nested_tuning.csv", index=False, lineterminator="\n")
        metrics.to_csv(
            temporary / "pooled_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(label_rows).to_csv(
            temporary / "label_tier_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(comparison_rows).to_csv(
            temporary / "v1_v2_v3_comparison.csv", index=False, lineterminator="\n"
        )
        for row in metric_rows:
            matrix = {key: int(row[key]) for key in ("tn", "fp", "fn", "tp")}
            (temporary / f"confusion_matrix_{row['series']}.svg").write_text(
                v1._confusion_svg(str(row["series"]), matrix), encoding="utf-8"
            )
        (temporary / "v1_v2_v3_comparison.svg").write_text(
            _comparison_svg(comparison_rows), encoding="utf-8"
        )
        v1_score_path = (ROOT / str(config["base_signal"]["predictions"])).resolve()
        v1_predictions = pd.read_csv(v1_score_path).sort_values("record_id")
        v2_predictions = pd.read_csv(
            ROOT
            / "results/LLM/qwen3-32b-final-annotation-ready-v2/analysis-final/oof_predictions_24_shot.csv"
        ).sort_values("record_id")
        ordered = predictions.sort_values("record_id")
        if not np.array_equal(
            ordered["label"].to_numpy(), v1_predictions["label"].to_numpy()
        ):
            raise RuntimeError("V1 comparison order/labels do not match V3")
        (temporary / "precision_recall_curves.svg").write_text(
            _pr_curve_svg(
                ordered["label"].to_numpy(dtype=np.int64),
                [
                    (
                        "V1 zero-shot",
                        v1_predictions["ambiguity_probability"].to_numpy(),
                    ),
                    ("V2 24-shot", v2_predictions["ambiguity_probability"].to_numpy()),
                    ("V3 hybrid", ordered["ambiguity_probability"].to_numpy()),
                ],
            ),
            encoding="utf-8",
        )
        analysis = {
            "experiment": config["experiment_name"],
            "result_status": config["reporting"]["development_result_status"],
            "created_at_utc": _utc_now(),
            "rows": len(predictions),
            "positives": int(labels.sum()),
            "primary_metric": "pr_auc",
            "v3_raw": metric_rows[0],
            "v3_thresholded": {row["series"]: row for row in metric_rows[1:]},
            "selected_hyperparameters_by_outer_fold": fold_metrics[
                ["fold", "selected_c", "selected_lexical_weight"]
            ].to_dict(orient="records"),
            "comparison": comparison_rows,
            "qwen_parameter_updates": False,
            "lexical_classifier_fitted": True,
            "tcp_opened": False,
            "dns_opened": False,
            "interpretation": (
                "Adaptive grouped-development evidence only; reserve TCP for later validation "
                "and keep DNS sealed until the complete method is frozen."
            ),
        }
        _json(temporary / "analysis.json", analysis)
        artifact_hashes = {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file()
        }
        _json(
            temporary / "run_manifest.json",
            {
                "experiment": config["experiment_name"],
                "completed_at_utc": _utc_now(),
                "config_sha256": sha256_file(config_path),
                "evaluator_sha256": sha256_file(Path(__file__)),
                "artifacts": artifact_hashes,
                "tcp_opened": False,
                "dns_opened": False,
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    audit_parser.add_argument("--output", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.config.resolve(), args.output.resolve())
    else:
        result = run(args.config.resolve(), args.output.resolve())
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
