"""Leakage-controlled Qwen3 zero/few-shot evaluation through local Ollama."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import shutil
import time
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

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
    ROOT
    / "experiments/llm/configs/qwen3_32b_final_annotation_ready_in_context.yaml"
)
FOLDS = (1, 2, 3, 4, 5)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    _json(temporary, value)
    temporary.replace(path)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def condition_name(shots: int) -> str:
    if shots == 0:
        return "zero_shot"
    if shots in {4, 8}:
        return f"{shots}_shot"
    raise ValueError("Supported shot counts are 0, 4, and 8")


def _stable_key(*values: object) -> str:
    return hashlib.sha256("|".join(map(str, values)).encode("utf-8")).hexdigest()


def _texts_for_rows(frame: pd.DataFrame) -> set[str]:
    return {
        normalized
        for column in ("previous_clause", "target_clause", "next_clause")
        for value in frame[column]
        if (normalized := normalize_for_audit(str(value)))
    }


def select_demonstrations(
    fitting: pd.DataFrame,
    *,
    shots: int,
    fold: int,
    seed: int,
    forbidden_normalized_texts: set[str],
) -> pd.DataFrame:
    """Select deterministic family-diverse demos with a 1:3 Yes/No ratio."""

    condition_name(shots)
    if shots == 0:
        return fitting.iloc[0:0].copy()
    positive_count = shots // 4
    counts = {0: shots - positive_count, 1: positive_count}

    def overlaps(row: Any) -> bool:
        values = {
            normalize_for_audit(str(getattr(row, column)))
            for column in ("previous_clause", "target_clause", "next_clause")
        }
        values.discard("")
        return bool(values & forbidden_normalized_texts)

    allowed = [not overlaps(row) for row in fitting.itertuples(index=False)]
    candidates = fitting.loc[allowed].copy()
    selected: list[pd.Series] = []
    for label, count in counts.items():
        label_rows = candidates.loc[candidates["label"] == label].copy()
        if len(label_rows) < count:
            raise RuntimeError(f"Fold {fold} has too few label-{label} demonstrations")
        queues: dict[str, deque[int]] = {}
        for family, group in label_rows.groupby("family_id", sort=True):
            queues[str(family)] = deque(
                sorted(
                    group.index,
                    key=lambda index: _stable_key(
                        "demo-row-v2",
                        seed,
                        fold,
                        label,
                        family,
                        label_rows.loc[index, "record_id"],
                    ),
                )
            )
        family_order = sorted(
            queues,
            key=lambda family: _stable_key(
                "demo-family-v2", seed, fold, label, family
            ),
        )
        chosen: list[int] = []
        while len(chosen) < count:
            progressed = False
            for family in family_order:
                if queues[family]:
                    chosen.append(queues[family].popleft())
                    progressed = True
                    if len(chosen) == count:
                        break
            if not progressed:
                raise RuntimeError("Demonstration selection exhausted candidates")
        selected.extend(label_rows.loc[index] for index in chosen)
    output = pd.DataFrame(selected).reset_index(drop=True)
    output["_order"] = output["record_id"].map(
        lambda record_id: _stable_key("demo-order-v2", seed, fold, shots, record_id)
    )
    output = output.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    if len(output) != shots or int(output["label"].sum()) != positive_count:
        raise RuntimeError("Demonstration class-ratio contract failed")
    if any(overlaps(row) for row in output.itertuples(index=False)):
        raise RuntimeError("A demonstration overlaps an evaluated prompt")
    return output


def render_in_context_messages(
    row: Any, demonstrations: pd.DataFrame, *, prompt_contract: str
) -> list[dict[str, str]]:
    return render_messages(row, demonstrations, prompt_contract=prompt_contract)


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("variant") != "local_context":
        raise ValueError("The zero/few-shot evaluator requires local_context")
    system_prompt_for_contract(str(config.get("prompt_contract", "")))
    if tuple(map(int, config["in_context"]["shot_counts"])) != (0, 4, 8):
        raise ValueError("Shot counts must be exactly 0, 4, and 8")
    model = config.get("model", {})
    if model.get("name") != "qwen3:32b":
        raise ValueError("This protocol is pinned to local qwen3:32b")
    digest = str(model.get("digest", ""))
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("The Ollama model digest must be lowercase SHA-256")
    options = config.get("ollama", {})
    if float(options.get("temperature", -1)) != 0:
        raise ValueError("Deterministic evaluation requires temperature 0")
    if int(options.get("top_logprobs", 0)) != 20:
        raise ValueError("Binary scoring requires 20 top logprobs")
    if bool(config.get("parameter_updates", True)):
        raise ValueError("This protocol permits no parameter updates")


def _load_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, list[Any], dict[str, Any]]:
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
    return config, frame, folds, root_manifest


def _fold_frames(
    frame: pd.DataFrame, fold: Any, *, smoke: bool
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    fitting = frame.iloc[fold.train_indices].reset_index(drop=True)
    outer = frame.iloc[fold.scoring_indices].reset_index(drop=True)
    forbidden = _texts_for_rows(frame.iloc[fold.validation_indices])
    if smoke:
        outer = _balanced_head(outer, 1)
    return fitting, outer, forbidden


def _balanced_head(frame: pd.DataFrame, rows_per_label: int) -> pd.DataFrame:
    output = pd.concat(
        [frame.loc[frame["label"] == label].head(rows_per_label) for label in (0, 1)],
        ignore_index=True,
    )
    if len(output) != rows_per_label * 2:
        raise RuntimeError("Smoke subset requires both labels")
    return output


def _client(config: dict[str, Any]) -> OllamaClient:
    options = config["ollama"]
    return OllamaClient(str(options["base_url"]), int(options["timeout_seconds"]))


def _ollama_preflight(config: dict[str, Any], client: OllamaClient) -> dict[str, Any]:
    model_name = str(config["model"]["name"])
    tags = client.tags()
    if model_name not in tags:
        raise RuntimeError(f"Ollama model is not installed: {model_name}")
    metadata = tags[model_name]
    if metadata.get("digest") != config["model"]["digest"]:
        raise RuntimeError(f"Installed {model_name} digest differs from the config")
    shown = client.show(model_name)
    capabilities = set(shown.get("capabilities", []))
    if "completion" not in capabilities:
        raise RuntimeError("Installed model has no completion capability")
    model_info = shown.get("model_info", {})
    architecture = str(model_info.get("general.architecture", ""))
    context_length = int(model_info.get(f"{architecture}.context_length", 0))
    if context_length and int(config["ollama"]["num_ctx"]) > context_length:
        raise RuntimeError("Configured context length exceeds the installed model")
    return {
        "ollama_version": client.version(),
        "model": model_name,
        "model_digest": metadata.get("digest"),
        "model_size_bytes": metadata.get("size"),
        "model_details": metadata.get("details", {}),
        "capabilities": sorted(capabilities),
        "advertised_context_length": context_length,
    }


def audit_prompts(config_path: Path, output: Path) -> Path:
    """Validate data, folds, model identity, demos, and prompt sizes without inference."""

    if output.exists():
        raise FileExistsError(f"Audit output is immutable: {output}")
    config, frame, folds, dataset_manifest = _load_protocol(config_path)
    preflight = _ollama_preflight(config, _client(config))
    seed = int(config["in_context"]["demonstration_seed"])
    lengths: dict[str, list[int]] = {}
    demo_plan: dict[str, Any] = {}
    for fold in folds:
        fitting, outer, forbidden = _fold_frames(frame, fold, smoke=False)
        for raw_shots in config["in_context"]["shot_counts"]:
            shots = int(raw_shots)
            key = f"fold_{fold.number}/{condition_name(shots)}"
            demos = select_demonstrations(
                fitting,
                shots=shots,
                fold=fold.number,
                seed=seed,
                forbidden_normalized_texts=forbidden,
            )
            demo_plan[key] = demos[
                ["record_id", "family_id", "label", "source_row_sha256"]
            ].to_dict(orient="records")
            lengths[key] = [
                len(
                    _canonical_json(
                        render_in_context_messages(
                            row, demos, prompt_contract=config["prompt_contract"]
                        )
                    )
                )
                for row in outer.itertuples(index=False)
            ]
    all_lengths = [item for values in lengths.values() for item in values]
    report = {
        "schema_version": 1,
        "evaluation_mode": "ollama_zero_and_few_shot_no_parameter_updates",
        "config_sha256": sha256_file(config_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "prompt_sha256": hashlib.sha256(
            system_prompt_for_contract(config["prompt_contract"]).encode("utf-8")
        ).hexdigest(),
        "source_annotation_sha256": config["source_annotation_sha256"],
        "development_train_sha256": config["development_train_sha256"],
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
        "demonstration_plan": demo_plan,
        "demonstration_class_policy": "4-shot=1 Yes/3 No; 8-shot=2 Yes/6 No",
        "ollama_preflight": preflight,
        "source_rows": dataset_manifest["source_rows"],
        "excluded_unresolved_item_ids": dataset_manifest["excluded_unresolved_item_ids"],
        "human_annotation_involved": True,
        "independent_human_verification": False,
        "tcp_opened": False,
        "dns_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    _json(output, report)
    return output


def _request_payload(
    config: dict[str, Any], messages: list[dict[str, str]]
) -> dict[str, Any]:
    options = config["ollama"]
    return {
        "model": config["model"]["name"],
        "messages": messages,
        "stream": False,
        "think": options["think"],
        "logprobs": True,
        "top_logprobs": int(options["top_logprobs"]),
        "keep_alive": options["keep_alive"],
        "options": {
            "temperature": float(options["temperature"]),
            "seed": int(options["seed"]),
            "num_ctx": int(options["num_ctx"]),
            "num_predict": int(options["num_predict"]),
        },
    }


def _score_record(
    *,
    client: OllamaClient,
    config: dict[str, Any],
    config_path: Path,
    row: Any,
    messages: list[dict[str, str]],
    path: Path,
) -> dict[str, Any]:
    cache_key = _sha256_json(
        {
            "config_sha256": sha256_file(config_path),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "model_digest": config["model"]["digest"],
            "request": _request_payload(config, messages),
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
                        "Re-evaluate the original clause and return exactly one "
                        "lowercase word: yes or no."
                    ),
                }
            )
        started = time.perf_counter()
        try:
            response = client.chat(_request_payload(config, attempt_messages))
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
                "schema_version": 1,
                "record_id": str(row.record_id),
                "cache_key": cache_key,
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


def classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray, predicted_labels: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=float)
    predicted = np.asarray(predicted_labels, dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "rows": len(labels),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


def _run_cell(
    *,
    config: dict[str, Any],
    config_path: Path,
    client: OllamaClient,
    preflight: dict[str, Any],
    frame: pd.DataFrame,
    fold: Any,
    shots: int,
    output: Path,
    smoke: bool,
) -> Path:
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_sha256") != sha256_file(config_path):
            raise RuntimeError(f"Completed cell uses another config: {output}")
        print(f"SKIP completed: {output}", flush=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    fitting, outer, forbidden = _fold_frames(frame, fold, smoke=smoke)
    demos = select_demonstrations(
        fitting,
        shots=shots,
        fold=fold.number,
        seed=int(config["in_context"]["demonstration_seed"]),
        forbidden_normalized_texts=forbidden,
    )
    _json(
        output / "demonstrations.json",
        {
            "schema_version": 1,
            "fold": fold.number,
            "shots": shots,
            "selection_method": config["in_context"]["demonstration_selection"],
            "records": demos[
                ["record_id", "family_id", "label", "source_row_sha256"]
            ].to_dict(orient="records"),
        },
    )
    rows: list[dict[str, Any]] = []
    for position, row in enumerate(outer.itertuples(index=False), start=1):
        messages = render_in_context_messages(
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
                f"{condition_name(shots)} fold {fold.number}: {position}/{len(outer)}",
                flush=True,
            )
    predictions = pd.DataFrame(rows)
    metrics = classification_metrics(
        predictions["label"].to_numpy(),
        predictions["ambiguity_probability"].to_numpy(),
        predictions["predicted_label"].to_numpy(),
    )
    predictions.to_csv(output / "outer_predictions.csv", index=False, lineterminator="\n")
    _json(output / "metrics.json", metrics)
    _json(
        manifest_path,
        {
            "schema_version": 1,
            "evaluation_mode": "ollama_in_context_no_parameter_updates",
            "config_sha256": sha256_file(config_path),
            "evaluator_sha256": sha256_file(Path(__file__)),
            "source_annotation_sha256": config["source_annotation_sha256"],
            "train_split_sha256": config["development_train_sha256"],
            "model": config["model"]["name"],
            "model_digest": config["model"]["digest"],
            "ollama_version": preflight["ollama_version"],
            "fold": fold.number,
            "shots": shots,
            "condition": condition_name(shots),
            "held_out_families": list(fold.held_out_families),
            "scored_rows": len(outer),
            "excluded_exact_duplicate_record_ids": list(
                fold.excluded_exact_duplicate_record_ids
            ),
            "smoke": smoke,
            "human_annotation_involved": True,
            "human_verified": False,
            "tcp_opened": False,
            "dns_opened": False,
            "completed_at": _utc_now(),
            "artifacts": _artifact_hashes(output),
        },
    )
    return output


def run_evaluation(config_path: Path, run_root: Path, *, smoke: bool) -> Path:
    config, frame, folds, _ = _load_protocol(config_path)
    client = _client(config)
    preflight = _ollama_preflight(config, client)
    for raw_shots in config["in_context"]["shot_counts"]:
        shots = int(raw_shots)
        for fold in (folds[:1] if smoke else folds):
            _run_cell(
                config=config,
                config_path=config_path,
                client=client,
                preflight=preflight,
                frame=frame,
                fold=fold,
                shots=shots,
                output=run_root / "cells" / condition_name(shots) / f"fold-{fold.number}",
                smoke=smoke,
            )
    return run_root


def _validate_cell(
    directory: Path,
    config: dict[str, Any],
    config_path: Path,
    shots: int,
    fold: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    expected = {
        "config_sha256": sha256_file(config_path),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "source_annotation_sha256": config["source_annotation_sha256"],
        "train_split_sha256": config["development_train_sha256"],
        "model": config["model"]["name"],
        "model_digest": config["model"]["digest"],
        "fold": fold,
        "shots": shots,
        "condition": condition_name(shots),
        "evaluation_mode": "ollama_in_context_no_parameter_updates",
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
        raise RuntimeError(f"Cell manifest mismatch in {directory}: {mismatches}")
    for name, expected_hash in manifest["artifacts"].items():
        if sha256_file(directory / name) != expected_hash:
            raise RuntimeError(f"Cell artifact hash mismatch: {directory / name}")
    predictions = pd.read_csv(directory / "outer_predictions.csv")
    if predictions["record_id"].duplicated().any():
        raise RuntimeError(f"Duplicate predictions in {directory}")
    predictions.insert(0, "fold", fold)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    return predictions, metrics, manifest


def _select_shot_condition(fold_metrics: pd.DataFrame) -> dict[str, Any]:
    means = fold_metrics.groupby("shots", sort=True)["pr_auc"].mean()
    top_shots = int(means.idxmax())
    top_scores = fold_metrics.loc[fold_metrics["shots"] == top_shots].sort_values(
        "fold"
    )["pr_auc"].to_numpy()
    comparisons: list[dict[str, Any]] = []
    for shots, mean_score in means.items():
        candidate = fold_metrics.loc[fold_metrics["shots"] == shots].sort_values(
            "fold"
        )["pr_auc"].to_numpy()
        difference = top_scores - candidate
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
    selected = min(item["shots"] for item in comparisons if item["within_one_standard_error"])
    return {
        "rule": (
            "highest mean fold PR-AUC; prefer fewer demonstrations among conditions "
            "within one paired-fold standard error of the top"
        ),
        "top_shots": top_shots,
        "selected_shots": selected,
        "selected_condition": condition_name(selected),
        "comparisons": comparisons,
    }


def _confusion_svg(condition: str, matrix: dict[str, int]) -> str:
    values = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    maximum = max(max(row) for row in values) or 1
    cells: list[str] = []
    for row in range(2):
        for column in range(2):
            value = values[row][column]
            shade = 245 - round(145 * value / maximum)
            x, y = 145 + column * 150, 90 + row * 110
            cells.append(
                f'<rect x="{x}" y="{y}" width="150" height="110" '
                f'fill="rgb({shade},{shade},255)" stroke="#334155"/>'
                f'<text x="{x + 75}" y="{y + 65}" text-anchor="middle" '
                f'font-size="28" font-family="sans-serif">{value}</text>'
            )
    title = html.escape(condition.replace("_", " ").title())
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="500" height="360" '
        'viewBox="0 0 500 360"><rect width="500" height="360" fill="white"/>'
        f'<text x="250" y="35" text-anchor="middle" font-size="22" '
        f'font-family="sans-serif">{title} confusion matrix</text>'
        '<text x="295" y="335" text-anchor="middle" font-size="16" '
        'font-family="sans-serif">Predicted label</text>'
        '<text x="30" y="200" text-anchor="middle" font-size="16" '
        'font-family="sans-serif" transform="rotate(-90 30 200)">True label</text>'
        '<text x="220" y="75" text-anchor="middle">No</text>'
        '<text x="370" y="75" text-anchor="middle">Yes</text>'
        '<text x="125" y="150" text-anchor="end">No</text>'
        '<text x="125" y="260" text-anchor="end">Yes</text>'
        + "".join(cells)
        + "</svg>\n"
    )


def _metric_svg(rows: list[dict[str, Any]]) -> str:
    metrics = ("pooled_pr_auc", "precision", "recall", "f1", "balanced_accuracy")
    colors = ("#2563eb", "#16a34a", "#ea580c")
    parts = [
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="430" '
            'viewBox="0 0 900 430"><rect width="100%" height="100%" fill="white"/>'
        ),
        (
            '<text x="450" y="28" text-anchor="middle" font-size="21" '
            'font-family="sans-serif">Qwen3 32B zero/few-shot metrics</text>'
        ),
    ]
    for tick in range(6):
        value, y = tick / 5, 340 - tick * 57
        parts.append(
            f'<line x1="80" y1="{y}" x2="860" y2="{y}" stroke="#e2e8f0"/>'
            f'<text x="68" y="{y + 5}" text-anchor="end" font-size="12">{value:.1f}</text>'
        )
    for metric_index, metric in enumerate(metrics):
        center = 80 + (metric_index + 0.5) * 156
        parts.append(
            f'<text x="{center}" y="365" text-anchor="middle" font-size="12">'
            f'{html.escape(metric.replace("pooled_", "").replace("_", " "))}</text>'
        )
        for row_index, row in enumerate(rows):
            value = float(row[metric])
            x, y = center - 45 + row_index * 30, 340 - value * 285
            parts.append(
                f'<rect x="{x}" y="{y:.1f}" width="25" height="{340 - y:.1f}" '
                f'fill="{colors[row_index]}"/>'
            )
    for index, row in enumerate(rows):
        x = 210 + index * 200
        parts.append(
            f'<rect x="{x}" y="397" width="14" height="14" fill="{colors[index]}"/>'
            f'<text x="{x + 21}" y="409" font-size="13">'
            f'{html.escape(str(row["condition"]).replace("_", " "))}</text>'
        )
    return "".join(parts) + "</svg>\n"


def analyze(run_root: Path, output: Path, config_path: Path) -> Path:
    """Aggregate all 15 grouped-fold cells and render evaluation artifacts."""

    if output.exists():
        raise FileExistsError(f"Analysis output is immutable: {output}")
    config, frame, _, _ = _load_protocol(config_path)
    temporary = output.parent / f".{output.name}.building-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        fold_rows: list[dict[str, Any]] = []
        pooled_rows: list[dict[str, Any]] = []
        confusion_rows: list[dict[str, Any]] = []
        curve_rows: list[dict[str, Any]] = []
        source_rows: list[dict[str, Any]] = []
        for raw_shots in config["in_context"]["shot_counts"]:
            shots = int(raw_shots)
            condition = condition_name(shots)
            pieces: list[pd.DataFrame] = []
            for fold in FOLDS:
                directory = run_root / "cells" / condition / f"fold-{fold}"
                predictions, metrics, manifest = _validate_cell(
                    directory, config, config_path, shots, fold
                )
                pieces.append(predictions)
                fold_rows.append(
                    {
                        "condition": condition,
                        "shots": shots,
                        "fold": fold,
                        "held_out_families": ";".join(manifest["held_out_families"]),
                        **{key: metrics[key] for key in (
                            "rows", "positives", "prevalence", "pr_auc", "precision",
                            "recall", "f1", "f2", "accuracy", "balanced_accuracy"
                        )},
                        **metrics["confusion_matrix"],
                    }
                )
            pooled = pd.concat(pieces, ignore_index=True)
            if pooled["record_id"].duplicated().any():
                raise RuntimeError(f"OOF records overlap for {condition}")
            if set(pooled["record_id"]) != set(frame["record_id"]):
                raise RuntimeError(f"OOF population is incomplete for {condition}")
            labels = pooled["label"].to_numpy(dtype=np.int64)
            scores = pooled["ambiguity_probability"].to_numpy(dtype=float)
            predicted = pooled["predicted_label"].to_numpy(dtype=np.int64)
            metrics = classification_metrics(labels, scores, predicted)
            matrix = metrics.pop("confusion_matrix")
            pooled_pr_auc = metrics.pop("pr_auc")
            pooled_row = {
                "condition": condition,
                "shots": shots,
                **metrics,
                "mean_fold_pr_auc": float(np.mean([
                    row["pr_auc"] for row in fold_rows if row["condition"] == condition
                ])),
                "pooled_pr_auc": pooled_pr_auc,
                **matrix,
            }
            pooled_rows.append(pooled_row)
            confusion_rows.extend([
                {"condition": condition, "true_label": "No", "predicted_no": matrix["tn"], "predicted_yes": matrix["fp"]},
                {"condition": condition, "true_label": "Yes", "predicted_no": matrix["fn"], "predicted_yes": matrix["tp"]},
            ])
            precision, recall, thresholds = precision_recall_curve(labels, scores)
            for index, (p_value, r_value) in enumerate(zip(precision, recall, strict=True)):
                curve_rows.append({
                    "condition": condition,
                    "precision": float(p_value),
                    "recall": float(r_value),
                    "threshold": float(thresholds[index]) if index < len(thresholds) else None,
                })
            for label_source, subset in pooled.groupby("label_tier"):
                source_metrics = classification_metrics(
                    subset["label"].to_numpy(),
                    subset["ambiguity_probability"].to_numpy(),
                    subset["predicted_label"].to_numpy(),
                )
                source_matrix = source_metrics.pop("confusion_matrix")
                source_rows.append({
                    "condition": condition,
                    "label_source": label_source,
                    **source_metrics,
                    **source_matrix,
                })
            pooled.to_csv(
                temporary / f"oof_predictions_{condition}.csv",
                index=False,
                lineterminator="\n",
            )
            (temporary / f"confusion_matrix_{condition}.svg").write_text(
                _confusion_svg(condition, matrix), encoding="utf-8"
            )
        fold_frame = pd.DataFrame(fold_rows)
        selection = _select_shot_condition(fold_frame)
        fold_frame.to_csv(temporary / "fold_metrics.csv", index=False, lineterminator="\n")
        pd.DataFrame(pooled_rows).to_csv(temporary / "pooled_metrics.csv", index=False, lineterminator="\n")
        pd.DataFrame(confusion_rows).to_csv(temporary / "confusion_matrices.csv", index=False, lineterminator="\n")
        pd.DataFrame(curve_rows).to_csv(temporary / "precision_recall_curves.csv", index=False, lineterminator="\n")
        pd.DataFrame(source_rows).to_csv(temporary / "label_source_metrics.csv", index=False, lineterminator="\n")
        (temporary / "metric_comparison.svg").write_text(_metric_svg(pooled_rows), encoding="utf-8")
        _json(
            temporary / "analysis.json",
            {
                "schema_version": 1,
                "evaluation_mode": "qwen3_32b_zero_and_few_shot_no_parameter_updates",
                "primary_metric": "mean fold PR-AUC",
                "score_contract": "P(yes)/(P(yes)+P(no)) from Ollama native top-token logprobs",
                "classification_contract": "the model's generated yes/no label",
                "selection": selection,
                "pooled_metrics": pooled_rows,
                "all_no_accuracy_baseline": 1.0 - float(frame["label"].mean()),
                "accuracy_warning": "Accuracy is secondary because labels are strongly imbalanced.",
                "source_annotation_sha256": config["source_annotation_sha256"],
                "development_train_sha256": config["development_train_sha256"],
                "model": config["model"],
                "human_annotation_involved": True,
                "independent_human_verification": False,
                "tcp_opened": False,
                "dns_opened": False,
            },
        )
        _json(temporary / "run_manifest.json", {"schema_version": 1, "artifacts": _artifact_hashes(temporary)})
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
        output = audit_prompts(args.config.resolve(), args.output.resolve())
        print(f"Prompt/model audit written to {output}")
    elif args.command == "run":
        output = run_evaluation(args.config.resolve(), args.run_root.resolve(), smoke=args.smoke)
        print(f"Zero/few-shot cells written under {output}")
    else:
        output = analyze(args.run_root.resolve(), args.output.resolve(), args.config.resolve())
        print(f"Zero/few-shot analysis written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
