"""Explicit CUDA-only execution of one frozen experiment cell.

Importing this module cannot train.  Training occurs only through main() after a
cell specification is explicitly supplied by a user-invoked CLI train command.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from .core import (
    EXPECTED_MODEL_SHA256,
    _read_json,
    _write_json,
    load_local_config,
    sha256_file,
    utc_now,
    verify_frozen,
)


def focal_loss(logits: Any, labels: Any, class_weights: Any) -> Any:
    """Historical gamma-2 focal loss with FP32 log-softmax and finite guard."""
    import torch
    from torch.nn import functional

    log_probabilities = functional.log_softmax(logits.float(), dim=-1)
    log_pt = log_probabilities.gather(1, labels.reshape(-1, 1)).squeeze(1)
    pt = log_pt.exp()
    alpha = (
        torch.ones_like(pt)
        if class_weights is None
        else class_weights.index_select(0, labels)
    )
    losses = -alpha * (1.0 - pt).pow(2.0) * log_pt
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("stage=loss; non-finite focal loss")
    return losses.mean()


def freeze_upper_layers(model: Any, last_layers: int = 4) -> dict[str, Any]:
    """Apply the historical Partial-4 membership policy."""
    layers = model.deberta.encoder.layer
    if len(layers) != 12:
        raise RuntimeError(f"Expected 12 DeBERTa layers, found {len(layers)}")
    frozen_snapshots: list[tuple[str, Any, Any]] = []
    categories: dict[str, int] = {}
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    for name, parameter in model.named_parameters():
        layer_index = None
        prefix = "deberta.encoder.layer."
        if name.startswith(prefix):
            layer_index = int(name[len(prefix) :].split(".", 1)[0])
        if name.startswith(("classifier.", "pooler.")):
            group, trainable = "task_head_trainable", True
        elif layer_index is not None:
            trainable = layer_index >= 12 - last_layers
            group = "encoder_layer_trainable" if trainable else "encoder_layer_frozen"
        elif name.startswith(("deberta.embeddings.", "deberta.encoder.")):
            group, trainable = "shared_backbone_frozen", False
        else:
            raise RuntimeError(f"Unclassified DeBERTa parameter: {name}")
        parameter.requires_grad_(trainable)
        categories[group] = categories.get(group, 0) + int(parameter.numel())
        if trainable:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
            frozen_snapshots.append(
                (name, parameter, parameter.detach().to(device="cpu").clone())
            )
    if not trainable_names or not frozen_names:
        raise RuntimeError("Partial-4 produced an empty trainable or frozen group")
    return {
        "last_trainable_layers": last_layers,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "frozen_snapshots": frozen_snapshots,
        "parameter_counts_by_policy_group": categories,
        "total_parameters": sum(int(p.numel()) for p in model.parameters()),
    }


def construct_trainable_adamw(
    optimizer_factory: Any,
    parameters: Any,
    expected_trainable: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, list[Any]]:
    supplied = list(parameters)
    expected = list(expected_trainable)
    supplied_ids = [id(value) for value in supplied]
    expected_ids = [id(value) for value in expected]
    if len(supplied_ids) != len(set(supplied_ids)):
        raise RuntimeError("AdamW input contains duplicate parameters")
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Expected trainable parameters contain duplicates")
    if any(not value.requires_grad for value in expected):
        raise RuntimeError("Expected optimizer membership includes a frozen parameter")
    trainable = [value for value in supplied if value.requires_grad]
    if {id(value) for value in trainable} != set(expected_ids):
        raise RuntimeError("AdamW membership does not equal the trainable model membership")
    optimizer = optimizer_factory(trainable, *args, **kwargs)
    optimizer_values = [
        value for group in optimizer.param_groups for value in group["params"]
    ]
    optimizer_ids = [id(value) for value in optimizer_values]
    if len(optimizer_ids) != len(set(optimizer_ids)) or set(optimizer_ids) != set(expected_ids):
        raise RuntimeError("Constructed AdamW membership validation failed")
    return optimizer, trainable


def _validate_spec(spec: dict[str, Any]) -> None:
    frozen = verify_frozen()
    if spec.get("protocol_sha256") != frozen["protocol_sha256"]:
        raise RuntimeError("Cell specification protocol mismatch")
    if spec.get("method") not in {"focal", "partial4"}:
        raise RuntimeError("Unsupported training method")
    if spec.get("kind") not in {"outer", "inner"}:
        raise RuntimeError("Unsupported cell kind")
    if spec.get("seed") != 2718 or spec.get("precision") != "fp32":
        raise RuntimeError("Cell seed or precision differs from frozen protocol")
    if spec.get("input_mode") != "target_only":
        raise RuntimeError("Only target-only inputs are authorized")
    if sha256_file(Path(spec["split_manifest_path"])) != spec["split_manifest_sha256"]:
        raise RuntimeError("Cell split fingerprint mismatch")
    if spec.get("model_source_sha256") != EXPECTED_MODEL_SHA256:
        raise RuntimeError("Cell model fingerprint mismatch")
    if Path(spec["run_path"]).exists():
        raise FileExistsError(f"Immutable run path already exists: {spec['run_path']}")


def execute_cell(spec_path: str | Path) -> Path:
    """Run exactly one explicitly selected cell; never retries or falls back to CPU."""
    spec_path = Path(spec_path).resolve()
    spec = _read_json(spec_path)
    _validate_spec(spec)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    conda_library = Path(os.environ.get("CONDA_PREFIX", Path(os.sys.executable).parent)) / "Library" / "bin"
    os.environ["PATH"] = str(conda_library) + os.pathsep + os.environ.get("PATH", "")

    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback before model loading")
    gpu_name = torch.cuda.get_device_name(0)
    if "RTX 5000 Ada" not in gpu_name:
        raise RuntimeError(f"Expected NVIDIA RTX 5000 Ada GPU, found {gpu_name}")

    from models.transformer_final import train

    config = load_local_config(spec["config_path"])
    dataset_manifest = _read_json(spec["dataset_manifest_path"])
    split_manifest = _read_json(spec["split_manifest_path"])
    frame = pd.read_csv(
        spec["dataset_path"], dtype={"item_id": str, "record_id": str}, keep_default_na=False
    )
    frame["label"] = frame["label"].astype(int)
    frame["fold"] = frame["fold"].astype(int)
    fitting_ids = set(split_manifest["fitting_record_ids"])
    scoring_ids = set(split_manifest["scoring_record_ids"])
    original_load_config = train.load_config
    original_loader = train.load_fold_data
    original_prepare = train._prepare_model_for_training
    original_adamw = torch.optim.AdamW
    original_write_json = train.write_json
    original_loss = train._classification_loss
    captured: dict[str, Any] = {}
    freeze_audit: dict[str, Any] | None = None

    def local_config_loader(path: Path, *, verify_files: bool = True) -> dict[str, Any]:
        del verify_files
        return load_local_config(path)

    def local_fold_loader(
        supplied_config: dict[str, Any], supplied_fold: int, *, input_mode: str
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Path]:
        if Path(supplied_config["_config_path"]).resolve() != Path(spec["config_path"]).resolve():
            raise RuntimeError("Training core requested an unexpected config")
        if supplied_fold != int(spec["train_fold"]):
            raise RuntimeError("Training core requested an unexpected fold")
        if input_mode != "target_only":
            raise RuntimeError("Training core requested non-target input")
        fitting = frame.loc[frame["record_id"].isin(fitting_ids)].copy().reset_index(drop=True)
        scoring = frame.loc[frame["record_id"].isin(scoring_ids)].copy().reset_index(drop=True)
        if set(fitting["record_id"]) != fitting_ids or set(scoring["record_id"]) != scoring_ids:
            raise RuntimeError("Frozen split IDs no longer align with effective data")
        if set(fitting["family_id"]) & set(scoring["family_id"]):
            raise RuntimeError("RFC family overlap in cell")
        forbidden = {config["partitions"]["tcp_family"], config["partitions"]["dns_family"]}
        if (set(fitting["family_id"]) | set(scoring["family_id"])) & forbidden:
            raise RuntimeError("Reserved TCP or sealed DNS entered a cell")
        # The core hashes split_root/folds/fold_N.json in its run manifest.
        split_root = Path(spec["split_manifest_path"]).parent.parent
        return fitting, scoring, split_manifest, dataset_manifest, split_root

    if spec["method"] == "partial4":
        def partial_prepare(model: Any, device: Any) -> None:
            nonlocal freeze_audit
            original_prepare(model, device)
            freeze_audit = freeze_upper_layers(model, 4)
            captured["model"] = model

        train._prepare_model_for_training = partial_prepare

    def guarded_adamw(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        supplied = list(parameters)
        if spec["method"] == "partial4":
            if freeze_audit is None or "model" not in captured:
                raise RuntimeError("Partial-4 freeze was not applied before AdamW")
            expected = [p for p in captured["model"].parameters() if p.requires_grad]
        else:
            expected = supplied
        optimizer, trainable = construct_trainable_adamw(
            original_adamw, supplied, expected, *args, **kwargs
        )
        captured["optimizer_parameter_count"] = len(trainable)
        captured["optimizer_parameter_elements"] = sum(int(p.numel()) for p in trainable)
        return optimizer

    def write_with_metadata(path: Path, value: dict[str, Any]) -> None:
        updated = json.loads(json.dumps(value))
        if path.name == "run_config.json":
            updated["cell_id"] = spec["cell_id"]
            updated["kind"] = spec["kind"]
            updated["outer_fold"] = spec["outer_fold"]
            updated["inner_fold"] = spec["inner_fold"]
            updated["protocol_sha256"] = spec["protocol_sha256"]
            updated["resolved"]["analysis_method"] = spec["method"]
            if spec["method"] == "focal":
                updated["resolved"].update(
                    loss="focal_gamma_2",
                    focal_gamma=2.0,
                    focal_alpha="fitting-side class weights No=1, Yes=negative/positive",
                    loss_reduction="batch_mean",
                )
            else:
                updated["resolved"].update(
                    fine_tuning_policy="partial_4", trainable_last_layers=4
                )
        elif path.name == "run_manifest.json":
            updated.update(
                cell_id=spec["cell_id"],
                cell_kind=spec["kind"],
                analysis_method=spec["method"],
                outer_fold=spec["outer_fold"],
                inner_fold=spec["inner_fold"],
                protocol_sha256=spec["protocol_sha256"],
                split_manifest_sha256=spec["split_manifest_sha256"],
            )
            if spec["method"] == "focal":
                updated.update(
                    loss="focal_gamma_2",
                    focal_gamma=2.0,
                    focal_alpha="fitting-side class weights No=1, Yes=negative/positive",
                    loss_reduction="batch_mean",
                )
            else:
                updated.update(
                    full_encoder_fine_tuned=False,
                    fine_tuning_policy="partial_4",
                    trainable_last_layers=4,
                    optimizer_parameter_count=captured.get("optimizer_parameter_count"),
                    optimizer_parameter_elements=captured.get("optimizer_parameter_elements"),
                )
        original_write_json(path, updated)

    train.load_config = local_config_loader
    train.load_fold_data = local_fold_loader
    train.write_json = write_with_metadata
    train._classification_loss = focal_loss if spec["method"] == "focal" else original_loss
    torch.optim.AdamW = guarded_adamw
    started = time.perf_counter()
    try:
        output = train.run_cell(
            config_path=Path(spec["config_path"]),
            condition="A2",
            fold=int(spec["train_fold"]),
            seed=2718,
            model_path=Path(spec["model_path"]),
            precision="fp32",
            input_mode="target_only",
            output=Path(spec["run_path"]),
            diagnostic_numerics=True,
            protocol_path=None,
        )
    finally:
        train.load_config = original_load_config
        train.load_fold_data = original_loader
        train._prepare_model_for_training = original_prepare
        train.write_json = original_write_json
        train._classification_loss = original_loss
        torch.optim.AdamW = original_adamw

    audit: dict[str, Any] = {
        "schema_version": 1,
        "cell_id": spec["cell_id"],
        "method": spec["method"],
        "elapsed_seconds_wrapper": time.perf_counter() - started,
        "optimizer_parameter_count": captured.get("optimizer_parameter_count"),
        "optimizer_parameter_elements": captured.get("optimizer_parameter_elements"),
        "split_manifest_sha256": spec["split_manifest_sha256"],
        "protocol_sha256": spec["protocol_sha256"],
    }
    if freeze_audit is not None:
        model = captured["model"]
        changed_frozen = [
            name
            for name, parameter, initial in freeze_audit["frozen_snapshots"]
            if not torch.equal(parameter.detach().cpu(), initial)
        ]
        frozen_gradients = [
            name for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        nonfinite_trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and not bool(torch.isfinite(parameter).all())
        ]
        if changed_frozen or frozen_gradients or nonfinite_trainable:
            raise RuntimeError(
                f"Partial-4 post-training audit failed: changed={changed_frozen[:5]}, "
                f"frozen_gradients={frozen_gradients[:5]}, nonfinite={nonfinite_trainable[:5]}"
            )
        audit.update(
            last_trainable_layers=4,
            parameter_counts_by_policy_group=freeze_audit["parameter_counts_by_policy_group"],
            total_parameters=freeze_audit["total_parameters"],
            frozen_parameter_tensors=len(freeze_audit["frozen_names"]),
            frozen_parameters_unchanged=True,
            frozen_parameters_have_no_gradients=True,
            trainable_parameters_finite=True,
            trainable_parameters=captured["optimizer_parameter_elements"],
            trainable_percent=100.0 * captured["optimizer_parameter_elements"] / freeze_audit["total_parameters"],
        )
    _write_json(spec["audit_path"], audit, exclusive=True)
    run_manifest_path = output / "run_manifest.json"
    prediction_path = output / "predictions" / "held_out_predictions.csv"
    completion = {
        "schema_version": 1,
        "completed_at": utc_now(),
        "cell_id": spec["cell_id"],
        "protocol_sha256": spec["protocol_sha256"],
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "predictions_sha256": sha256_file(prediction_path),
        "audit_sha256": sha256_file(Path(spec["audit_path"])),
        "automatic_retry": False,
    }
    _write_json(spec["completion_path"], completion, exclusive=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    args = parser.parse_args()
    output = execute_cell(args.spec)
    print(f"Completed immutable cell: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

