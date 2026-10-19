"""Versioned strict-prompt Qwen3 in-context experiment with threshold analysis."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

import models.llm.in_context_evaluate as v1
import models.llm.ollama_client as ollama_module
import models.llm.prompting as prompting_module
from models.classical.feature_pipeline import sha256_file
from models.llm.ollama_client import (
    OllamaClient,
    OllamaError,
    ScoreUnavailable,
    extract_binary_score,
)
from models.llm.prompting import render_messages, system_prompt_for_contract
from models.transformer.dataset import ROOT, normalize_for_audit
from models.transformer.experiment import load_frozen_folds

DEFAULT_CONFIG = (
    ROOT / "experiments/llm/configs/qwen3_32b_final_annotation_ready_in_context_v2.yaml"
)
FOLDS = (1, 2, 3, 4, 5)


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    _json(temporary, value)
    temporary.replace(path)


def _stable_key(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def condition_name(shots: int) -> str:
    if shots <= 0:
        raise ValueError("V2 requires a positive number of demonstrations")
    return f"{shots}_shot"


def _dependency_hashes() -> dict[str, str]:
    return {
        "evaluator_v2": sha256_file(Path(__file__)),
        "v1_shared_helpers": sha256_file(Path(v1.__file__)),
        "prompting": sha256_file(Path(prompting_module.__file__)),
        "ollama_client": sha256_file(Path(ollama_module.__file__)),
    }


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("variant") != "local_context":
        raise ValueError("The V2 evaluator requires local_context")
    if config.get("prompt_contract") != "ordinary_engineer_strict_v2":
        raise ValueError("V2 requires the strict V2 prompt contract")
    system_prompt_for_contract(str(config["prompt_contract"]))
    shots = tuple(map(int, config["in_context"]["shot_counts"]))
    if shots != (8, 16, 24):
        raise ValueError("V2 shot counts must be exactly 8, 16, and 24")
    compositions = config["in_context"]["shot_compositions"]
    for shot_count in shots:
        composition = compositions[str(shot_count)]
        if int(composition["positive"]) + int(composition["negative"]) != shot_count:
            raise ValueError(f"Shot composition does not sum to {shot_count}")
        if min(map(int, composition.values())) < 1:
            raise ValueError("Every V2 prompt must contain both labels")
    model = config.get("model", {})
    if model.get("name") != "qwen3:32b":
        raise ValueError("V2 is pinned to local qwen3:32b")
    digest = str(model.get("digest", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("The Ollama model digest must be lowercase SHA-256")
    options = config.get("ollama", {})
    if float(options.get("temperature", -1)) != 0:
        raise ValueError("Deterministic evaluation requires temperature 0")
    if bool(options.get("think", True)):
        raise ValueError("Exact binary scoring requires thinking to be disabled")
    if int(options.get("top_logprobs", 0)) != 20:
        raise ValueError("Binary scoring requires 20 top logprobs")
    if bool(config.get("parameter_updates", True)):
        raise ValueError("V2 permits no model parameter updates")
    objectives = tuple(config["selection"]["threshold_objectives"])
    if objectives != ("f1", "f2", "balanced_accuracy"):
        raise ValueError("Unexpected threshold objectives")
    if config["selection"]["threshold_objective"] not in objectives:
        raise ValueError("Primary threshold objective is not configured")


def _load_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, list[Any], dict[str, Any], pd.DataFrame]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    source = (ROOT / str(config["source_annotation"])).resolve()
    if sha256_file(source) != config["source_annotation_sha256"]:
        raise RuntimeError("Configured final_annotation_ready.csv hash mismatch")
    dataset_dir = (ROOT / str(config["dataset_dir"])).resolve()
    root_manifest = json.loads(
        (dataset_dir.parent / "manifest.json").read_text(encoding="utf-8")
    )
    if root_manifest["source_sha256"] != config["source_annotation_sha256"]:
        raise RuntimeError("Prepared data targets another annotation source")
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_role") != "development_zero_few_shot_evaluation_only":
        raise RuntimeError("Config does not target the isolated development data")
    info = manifest["split_files"]["train"]
    train_path = dataset_dir / str(info["path"])
    if sha256_file(train_path) != config["development_train_sha256"]:
        raise RuntimeError("Prepared development hash mismatch")
    frame = pd.read_csv(train_path).fillna("")
    if len(frame) != int(info["rows"]):
        raise RuntimeError("Development row count does not match its manifest")
    if frame["record_id"].duplicated().any() or set(frame["split"]) != {"train"}:
        raise RuntimeError("Development data is not an isolated unique train split")
    folds_path = (ROOT / str(config["selection"]["fold_assignment"])).resolve()
    folds = load_frozen_folds(
        frame,
        folds_path,
        expected_sha256=config["selection"]["fold_assignment_sha256"],
        expected_train_sha256=config["development_train_sha256"],
    )
    score_path = (ROOT / str(config["in_context"]["hard_negative_scores"])).resolve()
    if sha256_file(score_path) != config["in_context"]["hard_negative_scores_sha256"]:
        raise RuntimeError("V1 hard-negative score artifact hash mismatch")
    scores = pd.read_csv(score_path)
    required = {"record_id", "label", "ambiguity_probability"}
    if not required.issubset(scores.columns) or scores["record_id"].duplicated().any():
        raise RuntimeError("Invalid V1 hard-negative score artifact")
    expected_labels = frame.set_index("record_id")["label"].astype(int)
    observed_labels = scores.set_index("record_id")["label"].astype(int)
    if set(observed_labels.index) != set(expected_labels.index):
        raise RuntimeError("V1 hard-negative scores do not cover the development set")
    if not observed_labels.sort_index().equals(expected_labels.sort_index()):
        raise RuntimeError("V1 hard-negative score labels differ from prepared data")
    scores = scores[["record_id", "ambiguity_probability"]].rename(
        columns={"ambiguity_probability": "v1_ambiguity_probability"}
    )
    return config, frame, folds, root_manifest, scores


def _overlaps(row: Any, forbidden: set[str]) -> bool:
    values = {
        normalize_for_audit(str(getattr(row, column)))
        for column in ("previous_clause", "target_clause", "next_clause")
    }
    values.discard("")
    return bool(values & forbidden)


def _diverse_take(
    pool: pd.DataFrame, count: int, *, positive: bool, seed: int
) -> pd.DataFrame:
    if len(pool) < count:
        raise RuntimeError(
            f"Only {len(pool)} candidates available for {count} demonstrations"
        )
    candidates = pool.copy()
    candidates["_manual_rank"] = (
        candidates["label_tier"] != "manual_conflict_verdict"
    ).astype(int)
    candidates["_tie"] = candidates["record_id"].map(
        lambda value: _stable_key(seed, value)
    )
    if positive:
        candidates = candidates.sort_values(
            ["_manual_rank", "v1_ambiguity_probability", "_tie"],
            ascending=[True, False, True],
        )
    else:
        candidates = candidates.sort_values(
            ["v1_ambiguity_probability", "_tie"], ascending=[False, True]
        )
    selected: list[int] = []
    family_counts: Counter[str] = Counter()
    remaining = list(candidates.index)
    rank_order = {index: position for position, index in enumerate(remaining)}
    while len(selected) < count:
        best = min(
            remaining,
            key=lambda index: (
                family_counts[str(candidates.at[index, "family_id"])],
                rank_order[index],
            ),
        )
        selected.append(best)
        remaining.remove(best)
        family_counts[str(candidates.at[best, "family_id"])] += 1
    return candidates.loc[selected].drop(columns=["_manual_rank", "_tie"])


def select_demonstrations_v2(
    fitting: pd.DataFrame,
    hard_scores: pd.DataFrame,
    *,
    shots: int,
    fold: int,
    seed: int,
    composition: dict[str, int],
    forbidden_normalized_texts: set[str],
) -> pd.DataFrame:
    candidates = fitting.merge(
        hard_scores, on="record_id", how="left", validate="one_to_one"
    )
    if candidates["v1_ambiguity_probability"].isna().any():
        raise RuntimeError("Missing hard-negative score in fitting data")
    allowed = [
        not _overlaps(row, forbidden_normalized_texts)
        for row in candidates.itertuples(index=False)
    ]
    candidates = candidates.loc[allowed].copy()
    positive_count = int(composition["positive"])
    negative_count = int(composition["negative"])
    positives = _diverse_take(
        candidates.loc[candidates["label"] == 1],
        positive_count,
        positive=True,
        seed=seed + fold + shots,
    )
    negatives = _diverse_take(
        candidates.loc[candidates["label"] == 0],
        negative_count,
        positive=False,
        seed=seed + fold + shots,
    )
    output = pd.concat([positives, negatives], ignore_index=True)
    output["_order"] = output["record_id"].map(
        lambda value: _stable_key(seed, fold, shots, value)
    )
    output = output.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if len(output) != shots or int(output["label"].sum()) != positive_count:
        raise RuntimeError("V2 demonstration composition contract failed")
    if any(
        _overlaps(row, forbidden_normalized_texts)
        for row in output.itertuples(index=False)
    ):
        raise RuntimeError("A V2 demonstration overlaps evaluated prompt text")
    return output


def _fold_frames(
    frame: pd.DataFrame, fold: Any, *, smoke: bool
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    fitting = frame.iloc[fold.train_indices].reset_index(drop=True)
    outer = frame.iloc[fold.scoring_indices].reset_index(drop=True)
    if smoke:
        outer = pd.concat(
            [outer.loc[outer["label"] == label].head(1) for label in (0, 1)],
            ignore_index=True,
        )
    forbidden = v1._texts_for_rows(
        frame.iloc[fold.validation_indices]
    ) | v1._texts_for_rows(outer)
    return fitting, outer, forbidden


def _client(config: dict[str, Any]) -> OllamaClient:
    options = config["ollama"]
    return OllamaClient(str(options["base_url"]), int(options["timeout_seconds"]))


def _score_record(
    *,
    client: OllamaClient,
    config: dict[str, Any],
    config_path: Path,
    row: Any,
    messages: list[dict[str, str]],
    path: Path,
) -> dict[str, Any]:
    code_hashes = _dependency_hashes()
    cache_key = _sha256_json(
        {
            "config_sha256": sha256_file(config_path),
            "code_hashes": code_hashes,
            "model_digest": config["model"]["digest"],
            "request": v1._request_payload(config, messages),
        }
    )
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("cache_key") != cache_key:
            raise RuntimeError(f"Cached response identity mismatch: {path}")
        return existing
    attempts: list[dict[str, Any]] = []
    last_error = ""
    for attempt_number in range(1, int(config["ollama"]["max_retries"]) + 2):
        attempt_messages = list(messages)
        if last_error:
            attempt_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"The previous response was invalid: {last_error[:400]}. "
                        "Re-evaluate the original clause and return exactly one lowercase word: yes or no."
                    ),
                }
            )
        started = time.perf_counter()
        try:
            response = client.chat(v1._request_payload(config, attempt_messages))
            score = extract_binary_score(response)
            if (score.ambiguity_probability >= 0.5) != bool(score.predicted_label):
                raise ScoreUnavailable("Generated label disagrees with yes/no logprobs")
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "valid",
                    "wall_seconds": time.perf_counter() - started,
                }
            )
            result = {
                "schema_version": 2,
                "record_id": str(row.record_id),
                "cache_key": cache_key,
                "code_hashes": code_hashes,
                "model": config["model"]["name"],
                "model_digest": config["model"]["digest"],
                "predicted_label": score.predicted_label,
                "output_text": score.output_text,
                "ambiguity_probability": score.ambiguity_probability,
                "log_odds_yes_no": score.log_odds_yes_no,
                "yes_logprob": score.yes_logprob,
                "no_logprob": score.no_logprob,
                "prompt_eval_count": response.get("prompt_eval_count"),
                "prompt_eval_cached_count": response.get("prompt_eval_cached_count"),
                "eval_count": response.get("eval_count"),
                "total_duration_ns": response.get("total_duration"),
                "response": response,
                "attempts": attempts,
                "created_at": _utc_now(),
            }
            _atomic_json(path, result)
            return result
        except (OllamaError, ScoreUnavailable) as error:
            last_error = str(error)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "invalid",
                    "error": last_error,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    raise RuntimeError(f"No valid score for {row.record_id}: {last_error}")


def audit_prompts(config_path: Path, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"Audit output is immutable: {output}")
    config, frame, folds, dataset_manifest, hard_scores = _load_protocol(config_path)
    client = _client(config)
    preflight = v1._ollama_preflight(config, client)
    plan: dict[str, Any] = {}
    lengths: dict[str, list[int]] = {}
    seed = int(config["in_context"]["demonstration_seed"])
    for fold in folds:
        fitting, outer, forbidden = _fold_frames(frame, fold, smoke=False)
        for raw_shots in config["in_context"]["shot_counts"]:
            shots = int(raw_shots)
            key = f"fold_{fold.number}/{condition_name(shots)}"
            demos = select_demonstrations_v2(
                fitting,
                hard_scores,
                shots=shots,
                fold=fold.number,
                seed=seed,
                composition=config["in_context"]["shot_compositions"][str(shots)],
                forbidden_normalized_texts=forbidden,
            )
            plan[key] = demos[
                [
                    "record_id",
                    "family_id",
                    "label",
                    "label_tier",
                    "v1_ambiguity_probability",
                    "source_row_sha256",
                ]
            ].to_dict(orient="records")
            lengths[key] = [
                len(
                    _canonical_json(
                        render_messages(
                            row, demos, prompt_contract=config["prompt_contract"]
                        )
                    )
                )
                for row in outer.itertuples(index=False)
            ]
    all_lengths = [length for values in lengths.values() for length in values]
    report = {
        "schema_version": 2,
        "evaluation_mode": "qwen3_32b_strict_prompt_prevalence_aware_hard_negative_v2",
        "config_sha256": sha256_file(config_path),
        "code_hashes": _dependency_hashes(),
        "prompt_sha256": hashlib.sha256(
            system_prompt_for_contract(config["prompt_contract"]).encode("utf-8")
        ).hexdigest(),
        "source_annotation_sha256": config["source_annotation_sha256"],
        "development_train_sha256": config["development_train_sha256"],
        "hard_negative_scores_sha256": config["in_context"][
            "hard_negative_scores_sha256"
        ],
        "development_rows": len(frame),
        "expected_full_requests": sum(map(len, lengths.values())),
        "maximum_serialized_prompt_characters": max(all_lengths),
        "prompt_character_lengths": {
            key: {
                "count": len(values),
                "minimum": min(values),
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "maximum": max(values),
            }
            for key, values in sorted(lengths.items())
        },
        "demonstration_plan": plan,
        "shot_compositions": config["in_context"]["shot_compositions"],
        "ollama_preflight": preflight,
        "source_rows": dataset_manifest["source_rows"],
        "excluded_unresolved_item_ids": dataset_manifest[
            "excluded_unresolved_item_ids"
        ],
        "human_annotation_involved": True,
        "independent_human_verification": False,
        "tcp_opened": False,
        "dns_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, report)
    return output


def _run_cell(
    *,
    config: dict[str, Any],
    config_path: Path,
    client: OllamaClient,
    preflight: dict[str, Any],
    frame: pd.DataFrame,
    hard_scores: pd.DataFrame,
    fold: Any,
    shots: int,
    output: Path,
    smoke: bool,
) -> Path:
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("config_sha256") != sha256_file(config_path)
            or manifest.get("code_hashes") != _dependency_hashes()
        ):
            raise RuntimeError(f"Completed V2 cell uses another protocol: {output}")
        print(f"SKIP completed: {output}", flush=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    fitting, outer, forbidden = _fold_frames(frame, fold, smoke=smoke)
    demos = select_demonstrations_v2(
        fitting,
        hard_scores,
        shots=shots,
        fold=fold.number,
        seed=int(config["in_context"]["demonstration_seed"]),
        composition=config["in_context"]["shot_compositions"][str(shots)],
        forbidden_normalized_texts=forbidden,
    )
    _json(
        output / "demonstrations.json",
        {
            "schema_version": 2,
            "fold": fold.number,
            "shots": shots,
            "composition": config["in_context"]["shot_compositions"][str(shots)],
            "selection_method": config["in_context"]["demonstration_selection"],
            "records": demos[
                [
                    "record_id",
                    "family_id",
                    "label",
                    "label_tier",
                    "v1_ambiguity_probability",
                    "source_row_sha256",
                ]
            ].to_dict(orient="records"),
        },
    )
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(outer.itertuples(index=False), start=1):
        messages = render_messages(
            row, demos, prompt_contract=config["prompt_contract"]
        )
        scored = _score_record(
            client=client,
            config=config,
            config_path=config_path,
            row=row,
            messages=messages,
            path=output / "responses" / f"{row.record_id}.json",
        )
        rows.append(
            {
                "record_id": row.record_id,
                "family_id": row.family_id,
                "label_tier": row.label_tier,
                "label": int(row.label),
                "log_odds_yes_no": scored["log_odds_yes_no"],
                "ambiguity_probability": scored["ambiguity_probability"],
                "predicted_label": scored["predicted_label"],
                "yes_logprob": scored["yes_logprob"],
                "no_logprob": scored["no_logprob"],
                "prompt_eval_count": scored["prompt_eval_count"],
                "eval_count": scored["eval_count"],
                "total_duration_ns": scored["total_duration_ns"],
            }
        )
        if position == 1 or position % 10 == 0 or position == len(outer):
            print(
                f"V2 {condition_name(shots)} fold {fold.number}: {position}/{len(outer)}",
                flush=True,
            )
    predictions = pd.DataFrame(rows)
    metrics = v1.classification_metrics(
        predictions["label"].to_numpy(),
        predictions["ambiguity_probability"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
    )
    predictions.to_csv(
        output / "outer_predictions.csv", index=False, lineterminator="\n"
    )
    _json(output / "metrics.json", metrics)
    _json(
        manifest_path,
        {
            "schema_version": 2,
            "evaluation_mode": "qwen3_32b_strict_prompt_prevalence_aware_hard_negative_v2",
            "config_sha256": sha256_file(config_path),
            "code_hashes": _dependency_hashes(),
            "source_annotation_sha256": config["source_annotation_sha256"],
            "train_split_sha256": config["development_train_sha256"],
            "hard_negative_scores_sha256": config["in_context"][
                "hard_negative_scores_sha256"
            ],
            "model": config["model"]["name"],
            "model_digest": config["model"]["digest"],
            "ollama_version": preflight["ollama_version"],
            "fold": fold.number,
            "shots": shots,
            "condition": condition_name(shots),
            "held_out_families": list(fold.held_out_families),
            "scored_rows": len(outer),
            "smoke": smoke,
            "human_annotation_involved": True,
            "human_verified": False,
            "tcp_opened": False,
            "dns_opened": False,
            "completed_at": _utc_now(),
            "artifacts": v1._artifact_hashes(output),
        },
    )
    return output


def run_evaluation(config_path: Path, run_root: Path, *, smoke: bool) -> Path:
    config, frame, folds, _, hard_scores = _load_protocol(config_path)
    client = _client(config)
    preflight = v1._ollama_preflight(config, client)
    for raw_shots in config["in_context"]["shot_counts"]:
        shots = int(raw_shots)
        for fold in folds[:1] if smoke else folds:
            _run_cell(
                config=config,
                config_path=config_path,
                client=client,
                preflight=preflight,
                frame=frame,
                hard_scores=hard_scores,
                fold=fold,
                shots=shots,
                output=run_root
                / "cells"
                / condition_name(shots)
                / f"fold-{fold.number}",
                smoke=smoke,
            )
    return run_root


def _validate_cell(
    directory: Path, config: dict[str, Any], config_path: Path, shots: int, fold: int
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256_file(config_path),
        "code_hashes": _dependency_hashes(),
        "source_annotation_sha256": config["source_annotation_sha256"],
        "train_split_sha256": config["development_train_sha256"],
        "hard_negative_scores_sha256": config["in_context"][
            "hard_negative_scores_sha256"
        ],
        "model": config["model"]["name"],
        "model_digest": config["model"]["digest"],
        "fold": fold,
        "shots": shots,
        "condition": condition_name(shots),
        "evaluation_mode": "qwen3_32b_strict_prompt_prevalence_aware_hard_negative_v2",
        "smoke": False,
        "tcp_opened": False,
        "dns_opened": False,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"V2 cell manifest mismatch in {directory}: {mismatches}")
    for name, expected_hash in manifest["artifacts"].items():
        if sha256_file(directory / name) != expected_hash:
            raise RuntimeError(f"V2 cell artifact hash mismatch: {directory / name}")
    predictions = pd.read_csv(directory / "outer_predictions.csv")
    if predictions["record_id"].duplicated().any():
        raise RuntimeError(f"Duplicate predictions in {directory}")
    predictions.insert(0, "fold", fold)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return predictions, metrics, manifest


def _best_threshold(labels: np.ndarray, scores: np.ndarray, objective: str) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    if objective in {"f1", "f2"}:
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        beta_squared = 1.0 if objective == "f1" else 4.0
        values = (
            (1.0 + beta_squared)
            * precision[:-1]
            * recall[:-1]
            / (beta_squared * precision[:-1] + recall[:-1] + 1e-15)
        )
        best = np.flatnonzero(np.isclose(values, np.nanmax(values)))[-1]
        return float(thresholds[best])
    if objective == "balanced_accuracy":
        false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
        values = (true_positive_rate + 1.0 - false_positive_rate) / 2.0
        best_value = np.nanmax(values)
        candidates = thresholds[np.isclose(values, best_value)]
        return float(np.max(candidates))
    raise ValueError(f"Unsupported threshold objective: {objective}")


def _cross_fitted_threshold_metrics(
    predictions: pd.DataFrame, objective: str
) -> tuple[dict[str, Any], pd.DataFrame]:
    output = predictions.copy()
    output["threshold_used"] = np.nan
    output["predicted_label_thresholded"] = -1
    thresholds: dict[str, float] = {}
    for fold in FOLDS:
        fitting = output["fold"] != fold
        scoring = output["fold"] == fold
        threshold = _best_threshold(
            output.loc[fitting, "label"].to_numpy(),
            output.loc[fitting, "ambiguity_probability"].to_numpy(),
            objective,
        )
        thresholds[str(fold)] = threshold
        output.loc[scoring, "threshold_used"] = threshold
        output.loc[scoring, "predicted_label_thresholded"] = (
            output.loc[scoring, "ambiguity_probability"] >= threshold
        ).astype(int)
    metrics = v1.classification_metrics(
        output["label"].to_numpy(),
        output["ambiguity_probability"].to_numpy(),
        output["predicted_label_thresholded"].to_numpy(),
    )
    metrics["objective"] = objective
    metrics["fold_thresholds"] = thresholds
    metrics["predicted_positive_rate"] = float(
        output["predicted_label_thresholded"].mean()
    )
    return metrics, output


def _select_condition(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    means = fold_metrics.groupby("shots", sort=True)["pr_auc"].mean()
    top_shots = int(means.idxmax())
    top = (
        fold_metrics.loc[fold_metrics["shots"] == top_shots]
        .sort_values("fold")["pr_auc"]
        .to_numpy()
    )
    comparisons: list[dict[str, Any]] = []
    for shots, mean_score in means.items():
        candidate = (
            fold_metrics.loc[fold_metrics["shots"] == shots]
            .sort_values("fold")["pr_auc"]
            .to_numpy()
        )
        difference = top - candidate
        standard_error = (
            0.0
            if np.allclose(difference, 0.0)
            else float(np.std(difference, ddof=1) / math.sqrt(len(FOLDS)))
        )
        gap = float(means.loc[top_shots] - mean_score)
        comparisons.append(
            {
                "shots": int(shots),
                "condition": condition_name(int(shots)),
                "mean_fold_pr_auc": float(mean_score),
                "gap_from_top": gap,
                "paired_fold_standard_error": standard_error,
                "within_one_standard_error": gap <= standard_error + 1e-12,
            }
        )
    selected = min(
        item["shots"] for item in comparisons if item["within_one_standard_error"]
    )
    return {
        "rule": "highest mean fold PR-AUC; prefer fewer demonstrations within one paired-fold standard error",
        "top_shots": top_shots,
        "selected_shots": selected,
        "selected_condition": condition_name(selected),
        "comparisons": comparisons,
    }


def _comparison_svg(rows: list[dict[str, Any]]) -> str:
    metrics = ("pr_auc", "precision", "recall", "f1", "balanced_accuracy")
    colors = ("#64748b", "#94a3b8", "#2563eb", "#ea580c")
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="500" viewBox="0 0 1100 500"><rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="30" text-anchor="middle" font-size="22" font-family="sans-serif">Qwen3 32B V1 vs V2 development comparison</text>',
    ]
    for tick in range(6):
        value, y = tick / 5, 375 - tick * 60
        parts.append(
            f'<line x1="80" y1="{y}" x2="1060" y2="{y}" stroke="#e2e8f0"/><text x="68" y="{y + 5}" text-anchor="end" font-size="12">{value:.1f}</text>'
        )
    group_width = 190
    for metric_index, metric in enumerate(metrics):
        center = 90 + metric_index * group_width + group_width / 2
        parts.append(
            f'<text x="{center}" y="400" text-anchor="middle" font-size="12">{html.escape(metric.replace("_", " "))}</text>'
        )
        for row_index, row in enumerate(rows):
            value = float(row[metric])
            x, y = center - 60 + row_index * 30, 375 - value * 300
            parts.append(
                f'<rect x="{x}" y="{y:.1f}" width="24" height="{375 - y:.1f}" fill="{colors[row_index]}"/>'
            )
    for index, row in enumerate(rows):
        x = 80 + index * 250
        parts.append(
            f'<rect x="{x}" y="445" width="14" height="14" fill="{colors[index]}"/><text x="{x + 20}" y="457" font-size="12">{html.escape(str(row["series"]))}</text>'
        )
    return "".join(parts) + "</svg>\n"


def analyze(run_root: Path, output: Path, config_path: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"V2 analysis output is immutable: {output}")
    config, frame, _, _, _ = _load_protocol(config_path)
    temporary = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        fold_rows: list[dict[str, Any]] = []
        pooled_rows: list[dict[str, Any]] = []
        threshold_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        pooled_by_condition: dict[str, pd.DataFrame] = {}
        for raw_shots in config["in_context"]["shot_counts"]:
            shots = int(raw_shots)
            condition = condition_name(shots)
            pieces: list[pd.DataFrame] = []
            for fold in FOLDS:
                predictions, metrics, manifest = _validate_cell(
                    run_root / "cells" / condition / f"fold-{fold}",
                    config,
                    config_path,
                    shots,
                    fold,
                )
                pieces.append(predictions)
                fold_rows.append(
                    {
                        "condition": condition,
                        "shots": shots,
                        "fold": fold,
                        "held_out_families": ";".join(manifest["held_out_families"]),
                        **{
                            key: metrics[key]
                            for key in (
                                "rows",
                                "positives",
                                "prevalence",
                                "pr_auc",
                                "precision",
                                "recall",
                                "f1",
                                "f2",
                                "accuracy",
                                "balanced_accuracy",
                            )
                        },
                        **metrics["confusion_matrix"],
                    }
                )
            pooled = pd.concat(pieces, ignore_index=True)
            if pooled["record_id"].duplicated().any() or set(
                pooled["record_id"]
            ) != set(frame["record_id"]):
                raise RuntimeError(f"V2 OOF population is invalid for {condition}")
            pooled_by_condition[condition] = pooled
            raw_metrics = v1.classification_metrics(
                pooled["label"].to_numpy(),
                pooled["ambiguity_probability"].to_numpy(),
                pooled["predicted_label"].to_numpy(),
            )
            matrix = raw_metrics.pop("confusion_matrix")
            pooled_rows.append(
                {
                    "condition": condition,
                    "shots": shots,
                    "mean_fold_pr_auc": float(
                        np.mean(
                            [
                                row["pr_auc"]
                                for row in fold_rows
                                if row["condition"] == condition
                            ]
                        )
                    ),
                    **raw_metrics,
                    "predicted_positive_rate": float(pooled["predicted_label"].mean()),
                    **matrix,
                }
            )
            pooled.to_csv(
                temporary / f"oof_predictions_{condition}.csv",
                index=False,
                lineterminator="\n",
            )
            (temporary / f"confusion_matrix_{condition}_raw.svg").write_text(
                v1._confusion_svg(f"{condition} raw", matrix), encoding="utf-8"
            )
            for objective in config["selection"]["threshold_objectives"]:
                threshold_metrics, thresholded = _cross_fitted_threshold_metrics(
                    pooled, str(objective)
                )
                threshold_matrix = threshold_metrics.pop("confusion_matrix")
                threshold_rows.append(
                    {
                        "condition": condition,
                        "shots": shots,
                        **threshold_metrics,
                        **threshold_matrix,
                    }
                )
                thresholded.to_csv(
                    temporary
                    / f"oof_predictions_{condition}_{objective}_thresholded.csv",
                    index=False,
                    lineterminator="\n",
                )
                (
                    temporary
                    / f"confusion_matrix_{condition}_{objective}_thresholded.svg"
                ).write_text(
                    v1._confusion_svg(
                        f"{condition} {objective} thresholded", threshold_matrix
                    ),
                    encoding="utf-8",
                )
            for label_source, subset in pooled.groupby("label_tier"):
                source_metrics = v1.classification_metrics(
                    subset["label"].to_numpy(),
                    subset["ambiguity_probability"].to_numpy(),
                    subset["predicted_label"].to_numpy(),
                )
                source_matrix = source_metrics.pop("confusion_matrix")
                source_rows.append(
                    {
                        "condition": condition,
                        "label_source": label_source,
                        **source_metrics,
                        **source_matrix,
                    }
                )
        fold_frame = pd.DataFrame(fold_rows)
        selection = _select_condition(fold_frame)
        selected_condition = selection["selected_condition"]
        selected = pooled_by_condition[selected_condition]
        threshold_objective = str(config["selection"]["threshold_objective"])
        final_threshold = _best_threshold(
            selected["label"].to_numpy(),
            selected["ambiguity_probability"].to_numpy(),
            threshold_objective,
        )
        final_predictions = (
            selected["ambiguity_probability"] >= final_threshold
        ).astype(int)
        final_fit_metrics = v1.classification_metrics(
            selected["label"].to_numpy(),
            selected["ambiguity_probability"].to_numpy(),
            final_predictions.to_numpy(),
        )
        final_threshold_report = {
            "condition": selected_condition,
            "shots": int(selection["selected_shots"]),
            "objective": threshold_objective,
            "threshold_fitted_on_all_development": final_threshold,
            "resubstitution_metrics_not_an_unbiased_performance_estimate": final_fit_metrics,
            "use_for_future_frozen_transfer_evaluation_only": True,
        }

        baseline_analysis_path = (
            ROOT / str(config["comparison"]["baseline_analysis"])
        ).resolve()
        baseline_metrics_path = (
            ROOT / str(config["comparison"]["baseline_metrics"])
        ).resolve()
        if (
            sha256_file(baseline_analysis_path)
            != config["comparison"]["baseline_analysis_sha256"]
            or sha256_file(baseline_metrics_path)
            != config["comparison"]["baseline_metrics_sha256"]
        ):
            raise RuntimeError("Pinned V1 comparison artifacts changed")
        baseline_analysis = json.loads(
            baseline_analysis_path.read_text(encoding="utf-8")
        )
        baseline = pd.read_csv(baseline_metrics_path)
        baseline_selected_name = baseline_analysis["selection"]["selected_condition"]
        baseline_best = baseline.sort_values("pooled_pr_auc", ascending=False).iloc[0]
        baseline_selected = baseline.loc[
            baseline["condition"] == baseline_selected_name
        ].iloc[0]
        v2_raw = (
            pd.DataFrame(pooled_rows)
            .loc[lambda value: value["condition"] == selected_condition]
            .iloc[0]
        )
        v2_threshold = (
            pd.DataFrame(threshold_rows)
            .loc[
                lambda value: (
                    (value["condition"] == selected_condition)
                    & (value["objective"] == threshold_objective)
                )
            ]
            .iloc[0]
        )
        comparison_rows = [
            {
                "series": f"V1 selected {baseline_selected_name} raw",
                "pr_auc": float(baseline_selected["pooled_pr_auc"]),
                **{
                    metric: float(baseline_selected[metric])
                    for metric in (
                        "precision",
                        "recall",
                        "f1",
                        "f2",
                        "accuracy",
                        "balanced_accuracy",
                    )
                },
            },
            {
                "series": f"V1 best-PR {baseline_best['condition']} raw",
                "pr_auc": float(baseline_best["pooled_pr_auc"]),
                **{
                    metric: float(baseline_best[metric])
                    for metric in (
                        "precision",
                        "recall",
                        "f1",
                        "f2",
                        "accuracy",
                        "balanced_accuracy",
                    )
                },
            },
            {
                "series": f"V2 selected {selected_condition} raw",
                "pr_auc": float(v2_raw["pr_auc"]),
                **{
                    metric: float(v2_raw[metric])
                    for metric in (
                        "precision",
                        "recall",
                        "f1",
                        "f2",
                        "accuracy",
                        "balanced_accuracy",
                    )
                },
            },
            {
                "series": f"V2 selected {selected_condition} cross-fit {threshold_objective}",
                "pr_auc": float(v2_threshold["pr_auc"]),
                **{
                    metric: float(v2_threshold[metric])
                    for metric in (
                        "precision",
                        "recall",
                        "f1",
                        "f2",
                        "accuracy",
                        "balanced_accuracy",
                    )
                },
            },
        ]
        fold_frame.to_csv(
            temporary / "fold_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(pooled_rows).to_csv(
            temporary / "pooled_raw_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(threshold_rows).to_csv(
            temporary / "cross_fitted_threshold_metrics.csv",
            index=False,
            lineterminator="\n",
        )
        pd.DataFrame(source_rows).to_csv(
            temporary / "label_source_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(comparison_rows).to_csv(
            temporary / "v1_v2_comparison.csv", index=False, lineterminator="\n"
        )
        _json(temporary / "final_development_threshold.json", final_threshold_report)
        (temporary / "v1_v2_comparison.svg").write_text(
            _comparison_svg(comparison_rows), encoding="utf-8"
        )
        _json(
            temporary / "analysis.json",
            {
                "schema_version": 2,
                "evaluation_mode": "qwen3_32b_strict_prompt_prevalence_aware_hard_negative_v2",
                "primary_metric": "mean grouped-fold PR-AUC",
                "selection": selection,
                "pooled_raw_metrics": pooled_rows,
                "cross_fitted_threshold_metrics": threshold_rows,
                "final_development_threshold": final_threshold_report,
                "v1_v2_comparison": comparison_rows,
                "all_no_accuracy_baseline": 1.0 - float(frame["label"].mean()),
                "accuracy_warning": "Accuracy is secondary because labels are strongly imbalanced.",
                "label_warning": "Operational labels lack independent human verification and Qwen3 contributed annotation-support votes.",
                "source_annotation_sha256": config["source_annotation_sha256"],
                "development_train_sha256": config["development_train_sha256"],
                "hard_negative_scores_sha256": config["in_context"][
                    "hard_negative_scores_sha256"
                ],
                "model": config["model"],
                "code_hashes": _dependency_hashes(),
                "tcp_opened": False,
                "dns_opened": False,
            },
        )
        _json(
            temporary / "run_manifest.json",
            {"schema_version": 2, "artifacts": v1._artifact_hashes(temporary)},
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output)
    except BaseException:
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
    run_parser.add_argument("--run-root", type=Path, required=True)
    run_parser.add_argument("--smoke", action="store_true")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    analyze_parser.add_argument("--run-root", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_prompts(args.config.resolve(), args.output.resolve())
        print(f"V2 prompt/model audit written to {result}")
    elif args.command == "run":
        result = run_evaluation(
            args.config.resolve(), args.run_root.resolve(), smoke=args.smoke
        )
        print(f"V2 cells written under {result}")
    else:
        result = analyze(
            args.run_root.resolve(), args.output.resolve(), args.config.resolve()
        )
        print(f"V2 analysis and V1 comparison written to {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
