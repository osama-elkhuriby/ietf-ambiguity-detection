"""Adaptive V4 ensemble of V3 scores and frozen MiniLM semantic classifiers."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import models.llm.in_context_evaluate as v1
import models.llm.in_context_evaluate_v2 as v2
from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

DEFAULT_CONFIG = (
    ROOT / "experiments/llm/configs/qwen3_32b_final_annotation_ready_semantic_v4.yaml"
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
    if config.get("experiment_type") != "frozen_minilm_semantic_v3_score_ensemble":
        raise ValueError("V4 requires the frozen MiniLM/V3 ensemble")
    updates = config.get("parameter_updates", {})
    if bool(updates.get("qwen3_32b", True)):
        raise ValueError("V4 must not update Qwen")
    if bool(updates.get("sentence_encoder", True)):
        raise ValueError("V4 must keep MiniLM frozen")
    if not bool(updates.get("semantic_classifiers", False)):
        raise ValueError("V4 requires fitted semantic classifiers")
    encoder = config["sentence_encoder"]
    if encoder["repository"] != "sentence-transformers/all-MiniLM-L6-v2":
        raise ValueError("Unexpected V4 sentence encoder")
    if int(encoder["dimensions"]) != 384 or not bool(encoder["local_files_only"]):
        raise ValueError("V4 requires the pinned offline 384-dimensional encoder")
    classifier = config["semantic_classifier"]
    if float(classifier["c"]) <= 0:
        raise ValueError("Semantic classifier C must be positive")
    expected = {"target_embedding", "target_previous_next_embeddings"}
    if set(classifier["configurations"]) != expected:
        raise ValueError("V4 requires target and local-context semantic classifiers")
    weights = config["ensemble"]
    values = [
        float(weights["v3_weight"]),
        float(weights["target_semantic_weight"]),
        float(weights["context_semantic_weight"]),
    ]
    if min(values) < 0 or not np.isclose(sum(values), 1.0):
        raise ValueError("V4 ensemble weights must be non-negative and sum to one")
    selection = config["selection"]
    if int(selection["n_outer_folds"]) != 5:
        raise ValueError("V4 requires the five frozen family folds")
    if tuple(selection["threshold_objectives"]) != (
        "f1",
        "f2",
        "balanced_accuracy",
    ):
        raise ValueError("Unexpected V4 threshold objectives")


def _load_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, int]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    source = (ROOT / str(config["source_annotation"])).resolve()
    if sha256_file(source) != config["source_annotation_sha256"]:
        raise RuntimeError("Configured annotation-source hash mismatch")
    train_path = (ROOT / str(config["dataset_dir"]) / "train.csv").resolve()
    if sha256_file(train_path) != config["development_train_sha256"]:
        raise RuntimeError("Configured development-train hash mismatch")
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
        raise RuntimeError("Frozen folds do not exactly cover V4 development families")
    frame["fold"] = frame["family_id"].map(family_to_fold).astype(int)
    forbidden = {"draft-ietf-tcpm-rfc793bis", "draft-ietf-dnsop-svcb-https"}
    if set(frame["family_id"]) & forbidden:
        raise RuntimeError("Reserved TCP or sealed DNS data entered V4 development")
    base_path = (ROOT / str(config["base_signal"]["predictions"])).resolve()
    if sha256_file(base_path) != config["base_signal"]["predictions_sha256"]:
        raise RuntimeError("V3 base-prediction hash mismatch")
    base = pd.read_csv(base_path)
    required = {"record_id", "fold", "label", "ambiguity_probability"}
    if not required.issubset(base) or base["record_id"].duplicated().any():
        raise RuntimeError("Invalid V3 base-prediction artifact")
    base = base[["record_id", "fold", "label", "ambiguity_probability"]].rename(
        columns={
            "fold": "v3_fold",
            "label": "v3_label",
            "ambiguity_probability": "v3_probability",
        }
    )
    frame = frame.merge(base, on="record_id", how="outer", validate="one_to_one")
    if len(frame) != 2142 or frame.isna().any().any():
        raise RuntimeError("V3 predictions do not exactly cover V4 development")
    if not np.array_equal(frame["label"].astype(int), frame["v3_label"].astype(int)):
        raise RuntimeError("V3 labels differ from V4 development labels")
    if not np.array_equal(frame["fold"].astype(int), frame["v3_fold"].astype(int)):
        raise RuntimeError("V3 folds differ from V4 frozen folds")
    if not np.isfinite(frame["v3_probability"]).all():
        raise RuntimeError("V3 base scores contain non-finite values")
    return config, frame, family_to_fold


def blend_three_probabilities(
    v3_probability: np.ndarray,
    target_probability: np.ndarray,
    context_probability: np.ndarray,
    weights: tuple[float, float, float],
) -> np.ndarray:
    """Combine the three component scores in log-odds space."""

    epsilon = 1e-6
    arrays = [
        np.clip(np.asarray(values, dtype=float), epsilon, 1 - epsilon)
        for values in (v3_probability, target_probability, context_probability)
    ]
    return expit(sum(weight * logit(values) for weight, values in zip(weights, arrays)))


def _embedding_fingerprint(config: dict[str, Any], frame: pd.DataFrame) -> str:
    payload = {
        "record_ids": frame["record_id"].tolist(),
        "target": frame["target_clause"].astype(str).tolist(),
        "previous": frame["previous_clause"].astype(str).tolist(),
        "next": frame["next_clause"].astype(str).tolist(),
        "encoder": config["sentence_encoder"],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encode_nonempty(model: Any, texts: pd.Series, encoder: dict[str, Any]) -> np.ndarray:
    expected = (len(texts), int(encoder["dimensions"]))
    output = np.zeros(expected, dtype=np.float32)
    indices = [index for index, value in enumerate(texts.astype(str)) if value.strip()]
    if indices:
        vectors = model.encode(
            [str(texts.iloc[index]) for index in indices],
            batch_size=int(encoder["batch_size"]),
            convert_to_numpy=True,
            normalize_embeddings=bool(encoder["normalize_embeddings"]),
            show_progress_bar=True,
        ).astype(np.float32, copy=False)
        if vectors.shape != (len(indices), expected[1]):
            raise RuntimeError(f"Unexpected MiniLM embedding shape: {vectors.shape}")
        output[indices] = vectors
    return output


def load_or_create_embeddings(
    config: dict[str, Any], frame: pd.DataFrame, cache_path: Path
) -> dict[str, np.ndarray]:
    fingerprint = _embedding_fingerprint(config, frame)
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["fingerprint"].item()) != fingerprint:
            raise RuntimeError("V4 embedding cache targets different inputs")
        arrays = {name: cached[name] for name in ("target", "previous", "next")}
    else:
        # Import after pandas/sklearn on Windows; this avoids a pyarrow import-order crash.
        from sentence_transformers import SentenceTransformer

        encoder = config["sentence_encoder"]
        model = SentenceTransformer(
            str(encoder["repository"]),
            revision=str(encoder["revision"]),
            device=str(encoder["device"]),
            local_files_only=bool(encoder["local_files_only"]),
        )
        model.eval()
        arrays = {
            "target": _encode_nonempty(model, frame["target_clause"], encoder),
            "previous": _encode_nonempty(model, frame["previous_clause"], encoder),
            "next": _encode_nonempty(model, frame["next_clause"], encoder),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fingerprint=np.array(fingerprint), **arrays)
    expected = (len(frame), int(config["sentence_encoder"]["dimensions"]))
    if any(array.shape != expected or not np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Invalid V4 embedding cache")
    return arrays


def _classifier(config: dict[str, Any]) -> LogisticRegression:
    values = config["semantic_classifier"]
    return LogisticRegression(
        C=float(values["c"]),
        class_weight=str(values["class_weight"]),
        solver=str(values["solver"]),
        max_iter=int(values["max_iter"]),
        random_state=int(values["random_seed"]),
    )


def _outer_predictions(
    config: dict[str, Any], frame: pd.DataFrame, embeddings: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = frame["label"].to_numpy(dtype=np.int64)
    folds = frame["fold"].to_numpy(dtype=np.int64)
    target = embeddings["target"]
    context = np.hstack([target, embeddings["previous"], embeddings["next"]])
    target_probability = np.full(len(frame), np.nan)
    context_probability = np.full(len(frame), np.nan)
    fold_rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        fitting = folds != fold
        scoring = folds == fold
        target_scaler = StandardScaler().fit(target[fitting])
        target_model = _classifier(config).fit(
            target_scaler.transform(target[fitting]), labels[fitting]
        )
        target_probability[scoring] = target_model.predict_proba(
            target_scaler.transform(target[scoring])
        )[:, 1]
        context_scaler = StandardScaler().fit(context[fitting])
        context_model = _classifier(config).fit(
            context_scaler.transform(context[fitting]), labels[fitting]
        )
        context_probability[scoring] = context_model.predict_proba(
            context_scaler.transform(context[scoring])
        )[:, 1]
    weights = config["ensemble"]
    weight_tuple = (
        float(weights["v3_weight"]),
        float(weights["target_semantic_weight"]),
        float(weights["context_semantic_weight"]),
    )
    scores = blend_three_probabilities(
        frame["v3_probability"].to_numpy(dtype=float),
        target_probability,
        context_probability,
        weight_tuple,
    )
    output = frame[
        ["record_id", "source_item_id", "family_id", "label_tier", "label", "fold"]
    ].copy()
    output["v3_probability"] = frame["v3_probability"].to_numpy(dtype=float)
    output["target_semantic_probability"] = target_probability
    output["context_semantic_probability"] = context_probability
    output["ambiguity_probability"] = scores
    output["predicted_label_raw"] = (scores >= 0.5).astype(int)
    for fold in FOLDS:
        held_out = output["fold"] == fold
        metrics = v1.classification_metrics(
            labels[held_out], scores[held_out], output.loc[held_out, "predicted_label_raw"]
        )
        matrix = metrics.pop("confusion_matrix")
        fold_rows.append(
            {
                "fold": fold,
                "held_out_families": ";".join(
                    sorted(set(output.loc[held_out, "family_id"]))
                ),
                **metrics,
                **matrix,
            }
        )
    return output, pd.DataFrame(fold_rows)


def _metrics_row(
    series: str, labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray
) -> dict[str, Any]:
    metrics = v1.classification_metrics(labels, scores, predictions)
    matrix = metrics.pop("confusion_matrix")
    return {"series": series, **metrics, **matrix}


def _comparison_svg(rows: list[dict[str, Any]]) -> str:
    metrics = ("pr_auc", "precision", "recall", "f1", "balanced_accuracy")
    colors = ("#64748b", "#dc2626", "#2563eb", "#16a34a")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="500" viewBox="0 0 1120 500"><rect width="100%" height="100%" fill="white"/>',
        '<text x="560" y="30" text-anchor="middle" font-size="22" font-family="sans-serif">V1-V4 grouped-development comparison</text>',
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
                f'<rect x="{x}" y="{y:.1f}" width="25" height="{365-y:.1f}" fill="{colors[row_index]}"/>'
            )
    for index, row in enumerate(rows):
        x = 50 + index * 268
        parts.append(
            f'<rect x="{x}" y="445" width="14" height="14" fill="{colors[index]}"/><text x="{x+20}" y="457" font-size="12">{html.escape(str(row["series"]))}</text>'
        )
    return "".join(parts) + "</svg>\n"


def audit(config_path: Path, output: Path) -> Path:
    config, frame, families = _load_protocol(config_path)
    if output.exists():
        raise FileExistsError(f"V4 audit output is immutable: {output}")
    snapshot = (
        Path.home()
        / ".cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
        / str(config["sentence_encoder"]["revision"])
    )
    if not snapshot.is_dir():
        raise RuntimeError(f"Pinned MiniLM snapshot is unavailable locally: {snapshot}")
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(
        output,
        {
            "experiment": config["experiment_name"],
            "created_at_utc": _utc_now(),
            "config_sha256": sha256_file(config_path),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "rows": len(frame),
            "positives": int(frame["label"].sum()),
            "families": sorted(families),
            "encoder_snapshot": str(snapshot),
            "adaptive_configuration": True,
            "tcp_opened": False,
            "dns_opened": False,
        },
    )
    return output


def run(config_path: Path, output: Path, cache_path: Path) -> Path:
    config, frame, _ = _load_protocol(config_path)
    if output.exists():
        raise FileExistsError(f"V4 output is immutable: {output}")
    temporary = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        embeddings = load_or_create_embeddings(config, frame, cache_path)
        predictions, fold_metrics = _outer_predictions(config, frame, embeddings)
        labels = predictions["label"].to_numpy(dtype=np.int64)
        scores = predictions["ambiguity_probability"].to_numpy(dtype=float)
        metric_rows = [
            _metrics_row(
                "v4_semantic_ensemble_raw",
                labels,
                scores,
                predictions["predicted_label_raw"].to_numpy(dtype=np.int64),
            )
        ]
        threshold_rows: list[dict[str, Any]] = []
        for objective in config["selection"]["threshold_objectives"]:
            threshold_metrics, thresholded = v2._cross_fitted_threshold_metrics(
                predictions[
                    ["fold", "record_id", "label", "ambiguity_probability"]
                ].copy(),
                objective,
            )
            predictions[f"threshold_{objective}"] = thresholded["threshold_used"]
            predictions[f"predicted_label_{objective}"] = thresholded[
                "predicted_label_thresholded"
            ]
            matrix = threshold_metrics.pop("confusion_matrix")
            threshold_metrics.pop("fold_thresholds", None)
            threshold_metrics.pop("objective", None)
            threshold_metrics.pop("predicted_positive_rate", None)
            row = {
                "series": f"v4_semantic_ensemble_{objective}_threshold",
                **threshold_metrics,
                **matrix,
            }
            metric_rows.append(row)
            threshold_rows.append(row)
        metrics = pd.DataFrame(metric_rows)
        comparison_paths = {
            key: (ROOT / str(config["comparison"][f"{key}_metrics"])).resolve()
            for key in ("v1", "v2", "v3")
        }
        for key, path in comparison_paths.items():
            if sha256_file(path) != config["comparison"][f"{key}_metrics_sha256"]:
                raise RuntimeError(f"{key.upper()} comparison hash mismatch")
        old_v1 = pd.read_csv(comparison_paths["v1"]).sort_values(
            "pooled_pr_auc", ascending=False
        ).iloc[0]
        old_v2 = pd.read_csv(comparison_paths["v2"]).sort_values(
            "pr_auc", ascending=False
        ).iloc[0]
        old_v3 = pd.read_csv(comparison_paths["v3"]).iloc[0]
        raw = metrics.iloc[0]
        comparison = [
            {
                "series": "V1 best raw",
                "pr_auc": float(old_v1["pooled_pr_auc"]),
                **{
                    key: float(old_v1[key])
                    for key in ("precision", "recall", "f1", "balanced_accuracy")
                },
            },
            {
                "series": "V2 best raw",
                "pr_auc": float(old_v2["pr_auc"]),
                **{
                    key: float(old_v2[key])
                    for key in ("precision", "recall", "f1", "balanced_accuracy")
                },
            },
            {
                "series": "V3 nested hybrid raw",
                "pr_auc": float(old_v3["pr_auc"]),
                **{
                    key: float(old_v3[key])
                    for key in ("precision", "recall", "f1", "balanced_accuracy")
                },
            },
            {
                "series": "V4 semantic ensemble raw",
                **{
                    key: float(raw[key])
                    for key in ("pr_auc", "precision", "recall", "f1", "balanced_accuracy")
                },
            },
        ]
        sensitivity_rows: list[dict[str, float]] = []
        v3_scores = predictions["v3_probability"].to_numpy(dtype=float)
        target_scores = predictions["target_semantic_probability"].to_numpy(dtype=float)
        context_scores = predictions["context_semantic_probability"].to_numpy(dtype=float)
        for v3_weight in np.arange(0.6, 0.951, 0.05):
            remaining = 1.0 - v3_weight
            for target_weight in np.arange(0.0, remaining + 1e-9, 0.05):
                context_weight = remaining - target_weight
                candidate = blend_three_probabilities(
                    v3_scores,
                    target_scores,
                    context_scores,
                    (v3_weight, target_weight, context_weight),
                )
                sensitivity_rows.append(
                    {
                        "v3_weight": float(v3_weight),
                        "target_semantic_weight": float(target_weight),
                        "context_semantic_weight": float(context_weight),
                        "pr_auc": float(v1.average_precision_score(labels, candidate)),
                    }
                )
        sensitivity = pd.DataFrame(sensitivity_rows).sort_values(
            "pr_auc", ascending=False
        )
        predictions.to_csv(temporary / "oof_predictions.csv", index=False, lineterminator="\n")
        fold_metrics.to_csv(temporary / "fold_metrics.csv", index=False, lineterminator="\n")
        metrics.to_csv(temporary / "pooled_metrics.csv", index=False, lineterminator="\n")
        sensitivity.to_csv(
            temporary / "adaptive_weight_sensitivity.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(comparison).to_csv(
            temporary / "v1_v2_v3_v4_comparison.csv", index=False, lineterminator="\n"
        )
        for row in metric_rows:
            matrix = {key: int(row[key]) for key in ("tn", "fp", "fn", "tp")}
            (temporary / f"confusion_matrix_{row['series']}.svg").write_text(
                v1._confusion_svg(str(row["series"]), matrix), encoding="utf-8"
            )
        (temporary / "v1_v2_v3_v4_comparison.svg").write_text(
            _comparison_svg(comparison), encoding="utf-8"
        )
        analysis = {
            "experiment": config["experiment_name"],
            "created_at_utc": _utc_now(),
            "result_status": config["reporting"]["development_result_status"],
            "rows": len(predictions),
            "positives": int(labels.sum()),
            "primary_metric": "pr_auc",
            "fixed_configuration": {
                "semantic_classifier_c": float(config["semantic_classifier"]["c"]),
                "ensemble": config["ensemble"],
            },
            "raw_metrics": metric_rows[0],
            "threshold_sensitivity": threshold_rows,
            "comparison": comparison,
            "best_adaptive_weight_sensitivity": sensitivity.iloc[0].to_dict(),
            "dependency_warning": config["reporting"]["dependency_warning"],
            "qwen_parameter_updates": False,
            "sentence_encoder_updates": False,
            "semantic_classifier_fitted": True,
            "tcp_opened": False,
            "dns_opened": False,
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
                "embedding_cache_sha256": sha256_file(cache_path),
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
    run_parser.add_argument("--embedding-cache", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.config.resolve(), args.output.resolve())
    else:
        result = run(
            args.config.resolve(),
            args.output.resolve(),
            args.embedding_cache.resolve(),
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
