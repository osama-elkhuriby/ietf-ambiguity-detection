"""Configuration loading and protocol invariants for Experiment 1."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import ROOT, read_json, resolve_path, sha256_file

DEFAULT_CONFIG = (
    ROOT
    / "experiments/transformer_final/original_annotation/configs/experiment_v1.json"
)
TRAIN_FAMILIES = (
    "draft-ietf-6man-rfc2460bis",
    "draft-ietf-httpbis-cache-header",
    "draft-ietf-httpbis-http2bis",
    "draft-ietf-httpbis-semantics",
    "draft-ietf-quic-http",
    "draft-ietf-quic-recovery",
    "draft-ietf-quic-tls",
    "draft-ietf-quic-transport",
    "draft-ietf-tls-tls13",
)
TCP_FAMILY = "draft-ietf-tcpm-rfc793bis"
DNS_FAMILY = "draft-ietf-dnsop-svcb-https"
LABEL_MAP = {"No": 0, "Yes": 1}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: dict[str, Any]) -> None:
    _require(config.get("schema_version") == 1, "Unsupported configuration schema")
    data = config.get("data", {})
    experiment = config.get("experiment")
    if experiment == "final_annotation_ready_pilot_v1":
        _require(
            data.get("label_column") == "final_label",
            "The annotation-ready pilot must use its only finalized label column",
        )
        source = str(data.get("source_csv", "")).replace("\\", "/").casefold()
        _require(
            source.endswith("results/annotation/final_annotation_ready.csv"),
            "Unexpected annotation-ready pilot source path",
        )
        _require(
            data.get("source_sha256")
            == "c20959acc2db8345c90ee06a71357aaac161b178b240495e516e4710405b2712",
            "Unexpected annotation-ready pilot source hash",
        )
        _require(
            data.get("expected_source_rows") == 2425
            and data.get("expected_unresolved_item_ids") == ["2161"]
            and data.get("expected_annotated_cohort_omissions") == 121,
            "Annotation-ready cohort contract changed",
        )
        _require(
            data.get("unresolved_exclusion_policy")
            == "Exclude item_id 2161 because final_status is unresolved_blank_manual_verdict and final_source is unresolved; its placeholder final_label is not treated as ground truth.",
            "Annotation-ready unresolved exclusion policy changed",
        )
        _require(
            config.get("partitions", {}).get("train_families") == list(TRAIN_FAMILIES)
            and config["partitions"].get("tcp_family") == TCP_FAMILY
            and config["partitions"].get("dns_family") == DNS_FAMILY
            and config["partitions"].get("sealed_evaluation_enabled") is False,
            "Family partition or sealed-evaluation policy changed",
        )
        _require(
            config.get("folds", {}).get("fold_source_sha256")
            == "c04dc8424ed96486f8521ef57d17e4b503d2c3f41bb116edeeb9535073dd7884"
            and config["folds"].get("count") == 5,
            "Existing family-fold assignment contract changed",
        )
        training = config.get("training", {})
        _require(
            training.get("seeds") == [2718]
            and training.get("precision") == "fp32"
            and training.get("epochs") == 5
            and training.get("early_stopping") is False
            and training.get("learning_rate") == 1e-5
            and training.get("weight_decay") == 0.01
            and training.get("optimizer") == "AdamW"
            and training.get("scheduler") == "linear"
            and training.get("warmup_ratio") == 0.1
            and training.get("train_batch_size") == 8
            and training.get("eval_batch_size") == 16
            and training.get("gradient_accumulation_steps") == 1
            and training.get("max_grad_norm") == 1.0
            and training.get("threshold") == 0.5
            and training.get("diagnostic_numerics") is True,
            "Annotation-ready pilot training contract changed",
        )
        _require(
            config.get("conditions") == {"A2": "weighted_ce"},
            "The annotation-ready pilot must use weighted cross-entropy only",
        )
        _require(
            config.get("tokenization", {}).get("input_mode") == "target_only"
            and config.get("tokenization", {}).get("max_length") == 256
            and config.get("model", {}).get("repository")
            == "microsoft/deberta-v3-base",
            "Annotation-ready model or tokenization protocol changed",
        )
        _require(
            config.get("model", {}).get("offline_only") is True
            and config.get("model", {}).get("fully_fine_tune") is True,
            "The pilot requires offline full-parameter fine-tuning",
        )
        return

    _require(
        experiment == "original_annotation_experiment_1",
        "Wrong experiment",
    )
    _require(
        data.get("label_column") == "final_label", "Experiment 1 must use final_label"
    )
    source = str(data.get("source_csv", "")).replace("\\", "/").casefold()
    _require("reannotated" not in source, "Dataset B is prohibited")
    _require(
        source.endswith("results/annotation/final_annotation (copy 2).csv"),
        "Unexpected Dataset A path",
    )
    _require(
        data.get("source_sha256")
        == "a34df11a87d36dfd14c7d251d96a52a0cac8bb2016db6e4d0475a858f8b382cb",
        "Unexpected Dataset A hash",
    )
    _require(
        tuple(config.get("partitions", {}).get("train_families", ())) == TRAIN_FAMILIES,
        "Training families changed",
    )
    _require(
        config.get("partitions", {}).get("tcp_family") == TCP_FAMILY,
        "TCP reservation changed",
    )
    _require(
        config.get("partitions", {}).get("dns_family") == DNS_FAMILY,
        "DNS reservation changed",
    )
    _require(
        config.get("partitions", {}).get("sealed_evaluation_enabled") is False,
        "DNS evaluation must remain disabled",
    )
    training = config.get("training", {})
    _require(
        training.get("epochs") == 5,
        "Experiment 1 uses a pre-registered five-epoch policy",
    )
    _require(
        training.get("early_stopping") is False,
        "Outer-fold checkpoint selection is prohibited",
    )
    _require(
        training.get("learning_rate") == 1e-5, "Learning rate differs from protocol"
    )
    _require(
        training.get("gradient_accumulation_steps") == 1,
        "Unexpected gradient accumulation",
    )
    _require(
        config.get("tokenization", {}).get("input_mode") == "target_only",
        "Primary experiment must be target_only",
    )
    _require(
        config.get("model", {}).get("repository") == "microsoft/deberta-v3-base",
        "Wrong model repository",
    )
    revision = config.get("model", {}).get("revision")
    _require(
        revision is None or (isinstance(revision, str) and bool(revision.strip())),
        "Model revision must be null for a hash-pinned local directory or a non-empty exact Hugging Face revision",
    )
    _require(
        config.get("conditions") == {"A1": "standard_ce", "A2": "weighted_ce"},
        "Conditions must be A1/A2 only",
    )
    seeds = training.get("seeds")
    _require(seeds == [1729, 2718, 31415], "Seed protocol changed")
    expected_unresolved = data.get("expected_unresolved_item_ids")
    _require(
        expected_unresolved == ["2161"], "Expected unresolved-record contract changed"
    )


def load_config(
    path: Path | str = DEFAULT_CONFIG, *, verify_files: bool = True
) -> dict[str, Any]:
    config_path = resolve_path(path)
    config = read_json(config_path)
    validate_config(config)
    config["_config_path"] = str(config_path)
    config["_config_sha256"] = sha256_file(config_path)
    if verify_files:
        for name in (
            "source_csv",
            "private_key_csv",
            "metadata_csv",
            "canonical_context_csv",
            "fold_source",
        ):
            value = (
                config["data"].get(name)
                if name != "fold_source"
                else config["folds"].get(name)
            )
            resolved = resolve_path(value)
            if not resolved.is_file():
                raise FileNotFoundError(f"Configured {name} is missing: {resolved}")
        hashes = config["data"].get("approved_source_hashes", {})
        for name, expected in hashes.items():
            actual = sha256_file(resolve_path(config["data"][name]))
            if actual != expected:
                raise RuntimeError(
                    f"Configured {name} hash mismatch: expected {expected}, got {actual}"
                )
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Strip runtime helper keys before persisting a resolved configuration."""
    return {key: value for key, value in config.items() if not key.startswith("_")}
