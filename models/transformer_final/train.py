"""Train exactly one fixed-epoch outer-fold cell from the original base checkpoint."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional

from .common import (
    git_commit,
    hash_directory,
    immutable_directory,
    read_json,
    repo_path,
    resolve_path,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
    utc_now,
    write_json,
)
from .config import DEFAULT_CONFIG, load_config, public_config
from .dataset import load_effective_dataset
from .leakage import apply_context_sanitization
from .metrics import metrics_for_predictions
from .modeling import ClauseDataset, collate_clauses, supported_model_fields
from .preflight import require_local_model_files
from .protocol import load_fp32_protocol


def compute_class_weights(labels: np.ndarray, condition: str) -> dict[int, float]:
    labels = np.asarray(labels, dtype=np.int64)
    if set(np.unique(labels)) - {0, 1}:
        raise ValueError("Class weights require binary labels")
    if condition == "A1":
        return {0: 1.0, 1: 1.0}
    if condition != "A2":
        raise ValueError(f"Unknown condition: {condition}")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Weighted CE requires both classes in the fitting partition")
    return {0: 1.0, 1: negatives / positives}


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: torch.Tensor | None,
) -> torch.Tensor:
    return functional.cross_entropy(logits.float(), labels, weight=loss_weights)


def _backward_with_diagnostics(
    backward: Callable[[], None], *, context: str
) -> None:
    try:
        backward()
    except Exception as exc:
        raise RuntimeError(f"stage=backward; {context}: {exc}") from exc


def _numerical_context(
    precision: str, epoch: int, batch_number: int, optimizer_step: int
) -> str:
    return (
        f"precision {precision}, epoch {epoch}, batch {batch_number}, "
        f"optimizer step {optimizer_step}"
    )


def _validate_training_labels(
    labels: torch.Tensor, batch_size: int, *, context: str
) -> None:
    if labels.dtype != torch.long:
        raise TypeError(f"Training labels must use torch.long ({context})")
    if labels.ndim != 1 or len(labels) != batch_size:
        raise ValueError(
            f"Training labels do not align with the input batch ({context})"
        )
    if bool(((labels < 0) | (labels > 1)).any()):
        raise ValueError(f"Training labels must be binary 0/1 ({context})")


def _nonfinite_parameter_names(model: Any) -> list[str]:
    parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.numel()
    ]
    if not parameters:
        return []
    with torch.no_grad():
        finite_checks = torch.stack(
            [torch.isfinite(parameter).all() for _, parameter in parameters]
        )
        if bool(finite_checks.all()):
            return []
        return [
            name
            for name, parameter in parameters
            if not bool(torch.isfinite(parameter).all())
        ]


def _check_forward_numerics(
    logits: torch.Tensor,
    loss: torch.Tensor,
    model: Any,
    *,
    context: str,
) -> None:
    logits_finite = torch.isfinite(logits).all()
    loss_finite = torch.isfinite(loss)
    if bool(logits_finite & loss_finite):
        return
    if not bool(logits_finite):
        parameter_names = _nonfinite_parameter_names(model)
        parameter_detail = (
            f"; non-finite model parameters: {', '.join(parameter_names)}"
            if parameter_names
            else ""
        )
        raise FloatingPointError(
            f"Non-finite model logits under {context}{parameter_detail}"
        )
    raise FloatingPointError(f"Non-finite cross-entropy loss under {context}")


def _clip_gradients(
    model: Any,
    max_grad_norm: float,
    *,
    context: str,
    diagnostic_numerics: bool = False,
) -> dict[str, Any] | None:
    before = _gradient_statistics(model) if diagnostic_numerics else None
    try:
        clipped_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
    except RuntimeError as exc:
        diagnostic_detail = f"; gradients_before_clip={before}" if before else ""
        raise FloatingPointError(
            f"Gradient validation/clipping failed under {context}"
            f"{diagnostic_detail}"
        ) from exc
    if not diagnostic_numerics:
        return None
    return {
        "before_clip": before,
        "clip_grad_norm_returned": float(clipped_norm.detach().item()),
        "after_clip": _gradient_statistics(model),
    }


def _gradient_statistics(model: Any) -> dict[str, Any]:
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None and parameter.grad.numel()
    ]
    if not gradients:
        return {
            "parameters_with_gradients": 0,
            "nonfinite_gradient_tensors": 0,
            "global_norm": 0.0,
            "max_abs_gradient": 0.0,
        }
    with torch.no_grad():
        finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
        norms = torch.stack(
            [
                torch.linalg.vector_norm(gradient.to(dtype=torch.float64))
                for gradient in gradients
            ]
        )
        maxima = torch.stack(
            [
                gradient.abs().max().to(dtype=torch.float64)
                for gradient in gradients
            ]
        )
        return {
            "parameters_with_gradients": len(gradients),
            "nonfinite_gradient_tensors": int((~finite).sum().item()),
            "global_norm": float(torch.linalg.vector_norm(norms).item()),
            "max_abs_gradient": float(maxima.max().item()),
        }


def _parameter_statistics(model: Any) -> dict[str, Any]:
    parameters = [
        parameter.detach()
        for parameter in model.parameters()
        if parameter.numel()
    ]
    dtype_counts: dict[str, int] = {}
    for parameter in parameters:
        dtype = str(parameter.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if not parameters:
        maximum = 0.0
    else:
        with torch.no_grad():
            maxima = torch.stack(
                [
                    parameter.abs().max().to(dtype=torch.float64)
                    for parameter in parameters
                ]
            )
            maximum = float(maxima.max().item())
    return {
        "parameter_tensors": len(parameters),
        "parameter_dtypes": dtype_counts,
        "max_abs_parameter": maximum,
    }


def _optimizer_configuration(
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    dtype_counts: dict[str, int] = {}
    device_counts: dict[str, int] = {}
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group["params"]
        for parameter in parameters:
            dtype = str(parameter.dtype)
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            device = str(parameter.device)
            device_counts[device] = device_counts.get(device, 0) + 1
        groups.append(
            {
                "group": group_index,
                "parameter_tensors": len(parameters),
                "lr": group["lr"],
                "betas": group["betas"],
                "eps": group["eps"],
                "weight_decay": group["weight_decay"],
                "foreach": group.get("foreach"),
                "fused": group.get("fused"),
            }
        )
    return {
        "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
        "parameter_dtypes": dtype_counts,
        "parameter_devices": device_counts,
        "groups": groups,
        "implementation_selection": (
            "PyTorch automatic dispatch"
            if any(
                group["foreach"] is None or group["fused"] is None for group in groups
            )
            else "explicit foreach/fused options"
        ),
    }


def _optimizer_state_statistics(
    model: Any, optimizer: torch.optim.Optimizer
) -> dict[str, Any]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = {
        id(parameter): group_index
        for group_index, group in enumerate(optimizer.param_groups)
        for parameter in group["params"]
    }
    dtype_counts: dict[str, int] = {}
    tensor_count = nonfinite_count = 0
    first_nonfinite: list[str] = []
    with torch.no_grad():
        for parameter, state in optimizer.state.items():
            for state_name, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                tensor_count += 1
                dtype = str(value.dtype)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                finite = torch.isfinite(value)
                if bool(finite.all()):
                    continue
                nonfinite_count += 1
                if len(first_nonfinite) < 3:
                    bad_elements = int((~finite).sum().item())
                    first_nonfinite.append(
                        f"{names.get(id(parameter), '<unnamed>')}"
                        f".{state_name}(group={groups.get(id(parameter), -1)}, "
                        f"dtype={dtype}, nonfinite_elements={bad_elements})"
                    )
    return {
        "optimizer_state_tensors": tensor_count,
        "optimizer_state_dtypes": dtype_counts,
        "nonfinite_optimizer_state_tensors": nonfinite_count,
        "first_nonfinite_optimizer_state_entries": first_nonfinite,
    }


def _assert_finite_optimizer_state(
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    context: str,
    statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    statistics = statistics or _optimizer_state_statistics(model, optimizer)
    if statistics["nonfinite_optimizer_state_tensors"]:
        raise FloatingPointError(
            "Non-finite optimizer state after optimizer update under "
            f"{context}: {statistics}"
        )
    return statistics


def _assert_finite_model_parameters(
    model: Any,
    *,
    stage: str,
    context: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    parameter_names = _nonfinite_parameter_names(model)
    if parameter_names:
        names = {name: parameter for name, parameter in model.named_parameters()}
        groups = (
            {
                id(parameter): group_index
                for group_index, group in enumerate(optimizer.param_groups)
                for parameter in group["params"]
            }
            if optimizer is not None
            else {}
        )
        first_affected = [
            f"{name}(dtype={names[name].dtype}, "
            f"group={groups.get(id(names[name]), 'unknown')})"
            for name in parameter_names[:3]
        ]
        raise FloatingPointError(
            f"Non-finite model parameters {stage} under {context}: "
            f"{len(parameter_names)} tensors; first affected: "
            f"{', '.join(first_affected)}"
        )


def _prepare_model_for_training(model: torch.nn.Module, device: torch.device) -> None:
    model.to(device=device, dtype=torch.float32)


def _dependency_versions() -> dict[str, str]:
    packages = (
        "torch",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "tokenizers",
        "sentencepiece",
        "safetensors",
    )
    return {package: importlib.metadata.version(package) for package in packages}


def _cuda_driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not versions or len(set(versions)) != 1:
        raise RuntimeError(f"Unable to determine a single CUDA driver version: {versions}")
    return versions[0]


def _parameter_dtype_counts(model: torch.nn.Module) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        dtype = str(parameter.dtype)
        counts[dtype] = counts.get(dtype, 0) + 1
    return counts


def validate_prediction_alignment(
    predictions: pd.DataFrame, expected_record_ids: set[str]
) -> None:
    actual = predictions["record_id"].astype(str)
    if actual.duplicated().any():
        raise RuntimeError("Held-out predictions contain duplicated record IDs")
    if set(actual) != set(expected_record_ids) or len(actual) != len(
        expected_record_ids
    ):
        raise RuntimeError("Prediction-to-record alignment failed")


def initialize_pretrained_components(
    model_path: Path,
    *,
    revision: str | None,
    tokenizer_loader: Callable[..., Any] | None = None,
    model_loader: Callable[..., Any] | None = None,
) -> tuple[Any, Any]:
    """Always construct fresh tokenizer/model objects from the immutable base source."""
    require_local_model_files(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if tokenizer_loader is None or model_loader is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
        model_loader = (
            model_loader or AutoModelForSequenceClassification.from_pretrained
        )
    common = {
        "revision": revision,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    try:
        tokenizer = tokenizer_loader(
            model_path, use_fast=True, fix_mistral_regex=False, **common
        )
    except Exception as exc:
        raise RuntimeError(
            f"stage=loading component=tokenizer model_path={model_path}: {exc}"
        ) from exc
    try:
        model = model_loader(
            model_path,
            num_labels=2,
            label2id={"No": 0, "Yes": 1},
            id2label={0: "No", 1: "Yes"},
            **common,
        )
    except Exception as exc:
        raise RuntimeError(
            f"stage=loading component=model model_path={model_path}: {exc}"
        ) from exc
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError(
            "The pretrained model contains frozen parameters; full fine-tuning is required"
        )
    return tokenizer, model


def load_fold_data(
    config: dict[str, Any], fold: int, *, input_mode: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], Path]:
    frame, dataset_manifest, dataset_dir = load_effective_dataset(config)
    fold_manifest = read_json(dataset_dir / "folds" / f"fold_{fold}.json")
    if int(fold_manifest["fold"]) != fold:
        raise RuntimeError("Fold manifest number mismatch")
    fitting_ids = set(fold_manifest["fitting_record_ids"])
    scoring_ids = set(fold_manifest["scoring_record_ids"])
    fitting = (
        frame.loc[frame["record_id"].astype(str).isin(fitting_ids)]
        .copy()
        .reset_index(drop=True)
    )
    evaluation = (
        frame.loc[frame["record_id"].astype(str).isin(scoring_ids)]
        .copy()
        .reset_index(drop=True)
    )
    if (
        set(fitting["record_id"].astype(str)) != fitting_ids
        or set(evaluation["record_id"].astype(str)) != scoring_ids
    ):
        raise RuntimeError("Fold IDs do not align with the effective dataset")
    if set(fitting["family_id"]) & set(evaluation["family_id"]):
        raise RuntimeError("Fitting and held-out families overlap")
    if set(fitting["family_id"]) & {
        config["partitions"]["tcp_family"],
        config["partitions"]["dns_family"],
    }:
        raise RuntimeError("Reserved TCP/DNS data entered fitting")
    if input_mode == "local_context":
        fitting = apply_context_sanitization(fitting, fold_manifest, role="fitting")
        evaluation = apply_context_sanitization(
            evaluation, fold_manifest, role="evaluation"
        )
    elif input_mode != "target_only":
        raise ValueError(f"Unsupported input mode: {input_mode}")
    return fitting, evaluation, fold_manifest, dataset_manifest, dataset_dir


def _set_determinism(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _predict(
    model: Any, loader: Any, device: Any, *, use_bf16: bool, torch: Any
) -> tuple[np.ndarray, np.ndarray, int, int]:
    model.eval()
    positions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    target_truncated = context_truncated = 0
    with torch.no_grad():
        for batch in loader:
            positions.append(batch.pop("row_position").numpy())
            target_truncated += int(batch.pop("target_truncated").sum())
            context_truncated += int(batch.pop("context_truncated").sum())
            batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with autocast:
                output = model(**inputs).logits
            logits.append(output.detach().float().cpu().numpy())
    return (
        np.concatenate(positions),
        np.concatenate(logits),
        target_truncated,
        context_truncated,
    )


def _run_output(
    config: dict[str, Any], condition: str, fold: int, seed: int, input_mode: str
) -> Path:
    root = resolve_path(config["outputs"]["runs_dir"])
    condition_name = (
        condition if input_mode == "target_only" else f"{condition}_{input_mode}"
    )
    return root / condition_name / f"fold_{fold}" / f"seed_{seed}"


def run_cell(
    *,
    config_path: Path,
    condition: str,
    fold: int,
    seed: int,
    model_path: Path,
    precision: str | None = None,
    input_mode: str | None = None,
    output: Path | None = None,
    diagnostic_numerics: bool = False,
    protocol_path: Path | None = None,
) -> Path:
    config = load_config(config_path)
    protocol: dict[str, Any] | None = None
    protocol_sha256: str | None = None
    if protocol_path is not None:
        if config["experiment"] == "final_annotation_ready_pilot_v1":
            from .experiment2 import load_annotation_ready_protocol

            protocol, protocol_sha256 = load_annotation_ready_protocol(
                protocol_path, config
            )
        else:
            protocol, protocol_sha256 = load_fp32_protocol(protocol_path, config)
    if condition not in config["conditions"]:
        raise ValueError(f"Condition must be one of {sorted(config['conditions'])}")
    if fold not in range(1, int(config["folds"]["count"]) + 1):
        raise ValueError("Fold must be between 1 and 5")
    if seed not in config["training"]["seeds"]:
        raise ValueError(f"Seed must be pre-registered: {config['training']['seeds']}")
    mode = input_mode or config["tokenization"]["input_mode"]
    requested_precision = precision or config["training"]["precision"]
    if requested_precision not in {"auto", "bf16", "fp32"}:
        raise ValueError("precision must be auto, bf16, or fp32")
    if protocol is not None and (requested_precision != "fp32" or mode != "target_only"):
        raise RuntimeError("The frozen FP32 protocol requires fp32 target_only training")
    destination = (output or _run_output(config, condition, fold, seed, mode)).resolve()
    if destination.exists():
        raise FileExistsError(
            f"Run output is immutable and already exists: {destination}"
        )

    fitting, evaluation, fold_manifest, dataset_manifest, dataset_dir = load_fold_data(
        config, fold, input_mode=mode
    )
    if protocol is not None:
        if (
            sha256_file(resolve_path(config["data"]["source_csv"]))
            != protocol["dataset"]["source_sha256"]
            or dataset_manifest["effective_dataset_sha256"]
            != protocol["dataset"]["effective_dataset_sha256"]
            or sha256_file(dataset_dir / "manifest.json")
            != protocol["dataset"]["dataset_manifest_sha256"]
            or sha256_file(
                dataset_dir / "folds" / f"fold_{fold}.json"
            )
            != protocol["dataset"]["fold_manifest_sha256"][str(fold)]
        ):
            raise RuntimeError("Dataset or fold hashes differ from frozen FP32 protocol")
    class_weights = compute_class_weights(fitting["label"].to_numpy(), condition)
    model_path = model_path.resolve()
    require_local_model_files(model_path)
    model_source_hash, model_file_hashes = hash_directory(model_path)
    if protocol is not None and (
        str(model_path) != protocol["model"]["local_path"]
        or model_source_hash != protocol["model"]["source_sha256"]
    ):
        raise RuntimeError("Pretrained model identity differs from frozen FP32 protocol")

    import transformers
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training is required; CUDA is not visible. No training was started."
        )
    _set_determinism(seed, torch)
    tokenizer, model = initialize_pretrained_components(
        model_path, revision=config["model"].get("revision")
    )
    fields = supported_model_fields(model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("The local tokenizer has no pad token")
    max_length = int(config["tokenization"]["max_length"])
    fitting_dataset = ClauseDataset(
        fitting, tokenizer, input_mode=mode, max_length=max_length, model_fields=fields
    )
    evaluation_dataset = ClauseDataset(
        evaluation,
        tokenizer,
        input_mode=mode,
        max_length=max_length,
        model_fields=fields,
    )
    collator = partial(collate_clauses, pad_token_id=int(pad_token_id))
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        fitting_dataset,
        batch_size=int(config["training"]["train_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        evaluation_dataset,
        batch_size=int(config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )
    device = torch.device("cuda")
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if requested_precision == "bf16" and not bf16_supported:
        raise RuntimeError(
            "bf16 was requested but is unsupported; no training was started"
        )
    use_bf16 = requested_precision == "bf16" or (
        requested_precision == "auto" and bf16_supported
    )
    resolved_precision = "bf16" if use_bf16 else "fp32"
    _prepare_model_for_training(model, device)
    parameter_dtypes = _parameter_dtype_counts(model)
    if parameter_dtypes != {"torch.float32": sum(parameter_dtypes.values())}:
        raise RuntimeError(
            f"FP32 training requires FP32 model parameters, got {parameter_dtypes}"
        )
    _assert_finite_model_parameters(
        model,
        stage="after pretrained model loading",
        context=f"precision {resolved_precision}, before training, optimizer step 0",
    )
    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = epochs * optimizer_steps_per_epoch
    warmup_steps = round(total_steps * float(config["training"]["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    loss_weights = (
        None
        if condition == "A1"
        else torch.tensor(
            [class_weights[0], class_weights[1]], dtype=torch.float32, device=device
        )
    )

    run_started = time.perf_counter()
    with immutable_directory(destination) as building:
        write_json(
            building / "run_config.json",
            {
                "base_configuration": public_config(config),
                "resolved": {
                    "condition": condition,
                    "loss": config["conditions"][condition],
                    "fold": fold,
                    "seed": seed,
                    "input_mode": mode,
                    "precision": resolved_precision,
                    "model_path": str(model_path),
                    "diagnostic_numerics": diagnostic_numerics,
                },
            },
        )
        history: list[dict[str, Any]] = []
        total_optimizer_steps = 0
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            rows_seen = optimizer_steps = target_truncated = context_truncated = 0
            for batch_number, batch in enumerate(train_loader, start=1):
                batch.pop("row_position")
                target_truncated += int(batch.pop("target_truncated").sum())
                context_truncated += int(batch.pop("context_truncated").sum())
                labels = batch.pop("labels")
                step_context = _numerical_context(
                    resolved_precision,
                    epoch,
                    batch_number,
                    total_optimizer_steps + 1,
                )
                _validate_training_labels(
                    labels, batch["input_ids"].shape[0], context=step_context
                )
                labels = labels.to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if use_bf16
                    else nullcontext()
                )
                with autocast:
                    logits = model(**inputs).logits
                    loss = (
                        _classification_loss(logits, labels, loss_weights)
                        / accumulation
                    )
                _check_forward_numerics(logits, loss, model, context=step_context)
                _backward_with_diagnostics(loss.backward, context=step_context)
                rows_seen += len(labels)
                loss_sum += float(loss.detach().cpu()) * accumulation * len(labels)
                if batch_number % accumulation == 0 or batch_number == len(
                    train_loader
                ):
                    gradient_statistics = _clip_gradients(
                        model,
                        float(config["training"]["max_grad_norm"]),
                        context=step_context,
                        diagnostic_numerics=diagnostic_numerics,
                    )
                    parameter_statistics = (
                        _parameter_statistics(model) if diagnostic_numerics else None
                    )
                    optimizer_diagnostics = (
                        _optimizer_configuration(optimizer)
                        if diagnostic_numerics
                        else None
                    )
                    try:
                        optimizer.step()
                    except Exception as exc:
                        if not diagnostic_numerics:
                            raise
                        try:
                            optimizer_state = _optimizer_state_statistics(
                                model, optimizer
                            )
                        except Exception as diagnostic_exc:
                            raise RuntimeError(
                                f"stage=optimizer_update; {step_context}: {exc}; "
                                f"optimizer diagnostic collection failed: "
                                f"{diagnostic_exc}"
                            ) from exc
                        diagnostic_detail = {
                            "optimizer": optimizer_diagnostics,
                            "gradients": gradient_statistics,
                            "parameters_before_update": parameter_statistics,
                            "optimizer_state_after_failure": optimizer_state,
                        }
                        raise RuntimeError(
                            f"stage=optimizer_update; {step_context}: {exc}; "
                            f"diagnostics="
                            f"{json.dumps(diagnostic_detail, sort_keys=True)}"
                        ) from exc
                    total_optimizer_steps += 1
                    if diagnostic_numerics:
                        state_statistics: dict[str, Any] | None = None
                        try:
                            state_statistics = _optimizer_state_statistics(
                                model, optimizer
                            )
                            _assert_finite_model_parameters(
                                model,
                                stage="after optimizer update",
                                context=step_context,
                                optimizer=optimizer,
                            )
                            _assert_finite_optimizer_state(
                                model,
                                optimizer,
                                context=step_context,
                                statistics=state_statistics,
                            )
                        except FloatingPointError as exc:
                            try:
                                optimizer_state = (
                                    state_statistics
                                    if state_statistics is not None
                                    else _optimizer_state_statistics(model, optimizer)
                                )
                            except Exception as diagnostic_exc:
                                raise FloatingPointError(
                                    f"{exc}; optimizer diagnostic collection failed: "
                                    f"{diagnostic_exc}"
                                ) from exc
                            diagnostic_detail = {
                                "optimizer": optimizer_diagnostics,
                                "gradients": gradient_statistics,
                                "parameters_before_update": parameter_statistics,
                                "optimizer_state": optimizer_state,
                            }
                            raise FloatingPointError(
                                f"{exc}; optimizer_diagnostics="
                                f"{json.dumps(diagnostic_detail, sort_keys=True)}"
                            ) from exc
                    else:
                        _assert_finite_model_parameters(
                            model,
                            stage="after optimizer update",
                            context=step_context,
                            optimizer=optimizer,
                        )
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": loss_sum / rows_seen,
                    "rows_seen": rows_seen,
                    "optimizer_steps": optimizer_steps,
                    "learning_rate_after_epoch": float(scheduler.get_last_lr()[0]),
                    "target_truncated_rows": target_truncated,
                    "context_truncated_rows": context_truncated,
                    "outer_fold_evaluations": 0,
                }
            )

        checkpoint_dir = building / "checkpoints" / "final_epoch"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        positions, logits, eval_target_truncated, eval_context_truncated = _predict(
            model, eval_loader, device, use_bf16=use_bf16, torch=torch
        )
        if sorted(positions.tolist()) != list(range(len(evaluation))):
            raise RuntimeError(
                "Held-out predictions do not map one-to-one to evaluation rows"
            )
        order = np.argsort(positions)
        logits = logits[order]
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        predictions = evaluation.iloc[positions[order]][
            [
                "item_id",
                "record_id",
                "family_id",
                "document_id",
                "revision",
                "fold",
                "label",
            ]
        ].reset_index(drop=True)
        predictions["logit_no"] = logits[:, 0]
        predictions["logit_yes"] = logits[:, 1]
        predictions["probability_no"] = probabilities[:, 0]
        predictions["probability_yes"] = probabilities[:, 1]
        threshold = float(config["training"]["threshold"])
        predictions["predicted_label_0_5"] = (probabilities[:, 1] >= threshold).astype(
            int
        )
        validate_prediction_alignment(
            predictions, set(fold_manifest["scoring_record_ids"])
        )
        overall, per_family = metrics_for_predictions(predictions, threshold=threshold)
        errors = {
            "false_positive_record_ids": sorted(
                predictions.loc[
                    (predictions["label"] == 0)
                    & (predictions["predicted_label_0_5"] == 1),
                    "record_id",
                ].astype(str)
            ),
            "false_negative_record_ids": sorted(
                predictions.loc[
                    (predictions["label"] == 1)
                    & (predictions["predicted_label_0_5"] == 0),
                    "record_id",
                ].astype(str)
            ),
        }
        (building / "history").mkdir()
        pd.DataFrame(history).to_csv(
            building / "history" / "training_history.csv",
            index=False,
            lineterminator="\n",
        )
        (building / "predictions").mkdir()
        predictions.to_csv(
            building / "predictions" / "held_out_predictions.csv",
            index=False,
            lineterminator="\n",
        )
        (building / "metrics").mkdir()
        write_json(building / "metrics" / "metrics.json", overall)
        pd.DataFrame(per_family).to_csv(
            building / "metrics" / "per_family_metrics.csv",
            index=False,
            lineterminator="\n",
        )
        write_json(building / "metrics" / "error_analysis.json", errors)
        checkpoint_source_hash, checkpoint_hashes = hash_directory(checkpoint_dir)
        write_json(
            building / "checkpoints" / "checkpoint_metadata.json",
            {
                "policy": "final epoch only; no checkpoint selection",
                "epoch": epochs,
                "sha256": checkpoint_source_hash,
                "files": checkpoint_hashes,
            },
        )
        artifact_paths = {
            "run_config": building / "run_config.json",
            "training_history": building / "history" / "training_history.csv",
            "held_out_predictions": building
            / "predictions"
            / "held_out_predictions.csv",
            "metrics": building / "metrics" / "metrics.json",
            "per_family_metrics": building / "metrics" / "per_family_metrics.csv",
            "error_analysis": building / "metrics" / "error_analysis.json",
            "checkpoint_metadata": building
            / "checkpoints"
            / "checkpoint_metadata.json",
        }
        source_hashes = source_file_hashes()
        write_json(
            building / "run_manifest.json",
            {
                "schema_version": 1,
                "created_at": utc_now(),
                "git_commit": git_commit(),
                "config_path": repo_path(Path(config["_config_path"])),
                "config_sha256": config["_config_sha256"],
                "protocol_id": protocol["protocol_id"] if protocol else None,
                "protocol_sha256": protocol_sha256,
                "source_file_hashes": source_hashes,
                "source_files_sha256": sha256_text(stable_json(source_hashes)),
                "original_dataset_sha256": dataset_manifest["original_dataset_sha256"],
                "effective_dataset_sha256": dataset_manifest[
                    "effective_dataset_sha256"
                ],
                "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
                "fold_manifest_sha256": sha256_file(
                    dataset_dir / "folds" / f"fold_{fold}.json"
                ),
                "model_repository": config["model"]["repository"],
                "model_revision": config["model"].get("revision"),
                "model_path": str(model_path),
                "model_source_sha256": model_source_hash,
                "model_file_hashes": model_file_hashes,
                "artifact_hashes": {
                    name: {
                        "path": path.relative_to(building).as_posix(),
                        "sha256": sha256_file(path),
                    }
                    for name, path in artifact_paths.items()
                },
                "fresh_base_initialization": True,
                "resume_checkpoint": None,
                "full_encoder_fine_tuned": True,
                "condition": condition,
                "loss": config["conditions"][condition],
                "actual_class_weights": {
                    str(key): value for key, value in class_weights.items()
                },
                "fitting_positive_labels": int(fitting["label"].sum()),
                "fitting_negative_labels": int(
                    len(fitting) - int(fitting["label"].sum())
                ),
                "held_out_positive_labels": int(evaluation["label"].sum()),
                "held_out_negative_labels": int(
                    len(evaluation) - int(evaluation["label"].sum())
                ),
                "fold": fold,
                "seed": seed,
                "held_out_families": fold_manifest["held_out_families"],
                "fitting_rows": len(fitting),
                "held_out_rows": len(evaluation),
                "fitting_exclusion_count": len(fold_manifest["fitting_exclusions"]),
                "fitting_exclusions": fold_manifest["fitting_exclusions"],
                "context_sanitizations": fold_manifest["context_sanitizations"]
                if mode == "local_context"
                else [],
                "input_mode": mode,
                "max_length": max_length,
                "evaluation_target_truncated_rows": eval_target_truncated,
                "evaluation_context_truncated_rows": eval_context_truncated,
                "epoch_policy": "fixed_5_no_early_stopping",
                "epochs_completed": epochs,
                "outer_fold_evaluations_during_training": 0,
                "outer_fold_evaluations_after_training": 1,
                "threshold_policy": "fixed_0.5_reference_only",
                "diagnostic_numerics": diagnostic_numerics,
                "precision": resolved_precision,
                "parameter_dtypes": parameter_dtypes,
                "forward_autocast": "disabled" if not use_bf16 else "bfloat16",
                "cross_entropy_input_dtype": "float32",
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0),
                "runtime_seconds": time.perf_counter() - run_started,
                "peak_gpu_memory_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "peak_gpu_memory_reserved_mb": torch.cuda.max_memory_reserved(device)
                / (1024**2),
                "python": platform.python_version(),
                "python_executable": str(Path(sys.executable).resolve()),
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "dependency_versions": _dependency_versions(),
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "cuda_driver": _cuda_driver_version(),
                "offline_environment": {
                    "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                    "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                },
            },
        )
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--condition", required=True, choices=("A1", "A2"))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp32"))
    parser.add_argument("--input-mode", choices=("target_only", "local_context"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument(
        "--diagnostic-numerics",
        action="store_true",
        help="Collect optimizer and gradient statistics for numerical failures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = run_cell(
        config_path=args.config,
        condition=args.condition,
        fold=args.fold,
        seed=args.seed,
        model_path=args.model_path,
        precision=args.precision,
        input_mode=args.input_mode,
        output=args.output,
        diagnostic_numerics=args.diagnostic_numerics,
        protocol_path=args.protocol_manifest,
    )
    print(f"Completed immutable run: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
