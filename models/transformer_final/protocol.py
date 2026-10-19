"""Validation for the frozen FP32 Experiment 1 protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import read_json, sha256_file

PROTOCOL_ID = "original_annotation_experiment_1_fp32_v1"


def load_fp32_protocol(
    path: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    protocol = read_json(path)
    if (
        protocol.get("schema_version") != 1
        or protocol.get("protocol_id") != PROTOCOL_ID
    ):
        raise ValueError(f"Unsupported FP32 protocol: {path}")
    if protocol.get("base_config_sha256") != config["_config_sha256"]:
        raise RuntimeError("FP32 protocol does not match the loaded base config")
    training = config["training"]
    tokenization = config["tokenization"]
    fixed_training = protocol["training"]
    dataset = protocol["dataset"]
    model = protocol["model"]
    comparisons = {
        "optimizer": (training["optimizer"], "AdamW"),
        "learning_rate": (training["learning_rate"], fixed_training["learning_rate"]),
        "weight_decay": (training["weight_decay"], fixed_training["weight_decay"]),
        "scheduler": (training["scheduler"], fixed_training["scheduler"]),
        "warmup_ratio": (training["warmup_ratio"], fixed_training["warmup_ratio"]),
        "epochs": (training["epochs"], fixed_training["epochs"]),
        "early_stopping": (
            training["early_stopping"],
            fixed_training["early_stopping"],
        ),
        "train_batch_size": (
            training["train_batch_size"],
            fixed_training["train_batch_size"],
        ),
        "eval_batch_size": (
            training["eval_batch_size"],
            fixed_training["eval_batch_size"],
        ),
        "gradient_accumulation_steps": (
            training["gradient_accumulation_steps"],
            fixed_training["gradient_accumulation_steps"],
        ),
        "max_grad_norm": (
            training["max_grad_norm"],
            fixed_training["gradient_clip_norm"],
        ),
        "seeds": (training["seeds"], fixed_training["seeds"]),
        "fold_count": (config["folds"]["count"], len(fixed_training["folds"])),
        "input_mode": (
            tokenization["input_mode"],
            fixed_training["input_mode"],
        ),
        "max_length": (
            tokenization["max_length"],
            fixed_training["max_length"],
        ),
    }
    changed = [name for name, (actual, expected) in comparisons.items() if actual != expected]
    if changed:
        raise RuntimeError(f"Base config conflicts with frozen FP32 protocol: {changed}")
    if (
        protocol.get("condition_set") != config["conditions"]
        or fixed_training.get("optimizer") != "AdamW"
        or dataset.get("source_sha256") != config["data"]["source_sha256"]
        or dataset.get("fold_source_sha256") != config["folds"]["fold_source_sha256"]
        or dataset.get("reserved_families_excluded")
        != [
            config["partitions"]["tcp_family"],
            config["partitions"]["dns_family"],
        ]
        or model.get("repository") != config["model"]["repository"]
        or model.get("revision") != config["model"].get("revision")
        or fixed_training.get("precision") != "fp32"
        or model.get("parameter_storage_dtype") != "float32"
        or model.get("forward_precision") != "float32_no_autocast"
        or model.get("cross_entropy_input_dtype") != "float32"
        or fixed_training.get("diagnostics_affect_updates") is not False
    ):
        raise RuntimeError("FP32 protocol requirements are incomplete or inconsistent")
    return protocol, sha256_file(path)
