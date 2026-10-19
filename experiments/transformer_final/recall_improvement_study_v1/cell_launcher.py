"""Launch one frozen study cell through the existing guarded training loop."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _freeze_policy(model: Any, last_layers: int) -> dict[str, Any]:
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
            group = "task_head_trainable"
            trainable = True
        elif layer_index is not None:
            group = (
                "encoder_layer_trainable"
                if layer_index >= 12 - last_layers
                else "encoder_layer_frozen"
            )
            trainable = layer_index >= 12 - last_layers
        elif name.startswith(("deberta.embeddings.", "deberta.encoder.")):
            group = "shared_backbone_frozen"
            trainable = last_layers == 12
        else:
            raise RuntimeError(f"Unclassified DeBERTa parameter: {name}")
        parameter.requires_grad_(trainable)
        categories[group] = categories.get(group, 0) + parameter.numel()
        if trainable:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
            frozen_snapshots.append(
                (name, parameter, parameter.detach().to(device="cpu").clone())
            )
    if not trainable_names or not frozen_names:
        raise RuntimeError(
            "Partial fine-tuning policy produced an empty parameter group"
        )
    return {
        "last_trainable_layers": last_layers,
        "trainable_names": trainable_names,
        "frozen_names": frozen_names,
        "frozen_snapshots": frozen_snapshots,
        "parameter_counts_by_policy_group": categories,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }


def _construct_trainable_adamw(
    optimizer_factory: Any,
    parameters: Any,
    expected_trainable: Any,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, list[Any]]:
    parameter_list = list(parameters)
    expected = list(expected_trainable)
    parameter_ids = [id(parameter) for parameter in parameter_list]
    expected_ids = [id(parameter) for parameter in expected]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise RuntimeError("AdamW input contains duplicate parameters")
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Expected trainable parameters contain duplicates")
    if any(not parameter.requires_grad for parameter in expected):
        raise RuntimeError("Expected optimizer parameters include frozen parameters")

    trainable = [
        parameter for parameter in parameter_list if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("AdamW received no trainable parameters")
    if {id(parameter) for parameter in trainable} != set(expected_ids):
        raise RuntimeError(
            "AdamW trainable parameters do not match the expected model parameters"
        )

    optimizer = optimizer_factory(trainable, *args, **kwargs)
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    if (
        len(optimizer_ids) != len(set(optimizer_ids))
        or set(optimizer_ids) != set(expected_ids)
        or any(not parameter.requires_grad for parameter in optimizer_parameters)
    ):
        raise RuntimeError(
            "Constructed AdamW parameter membership does not match the "
            "expected trainable parameters"
        )
    return optimizer, trainable


def _focal_loss(logits: Any, labels: Any, class_weights: Any) -> Any:
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


def execute_cell(spec_path: Path) -> Path:
    conda_library_bin = Path(
        os.environ.get("CONDA_PREFIX", Path(os.sys.executable).parent)
    )
    conda_library_bin = conda_library_bin / "Library" / "bin"
    os.environ["PATH"] = (
        str(conda_library_bin) + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch

    from models.transformer_final import train
    from models.transformer_final.common import read_json
    from models.transformer_final.config import load_config
    from models.transformer_final.dataset import load_effective_dataset

    spec = read_json(spec_path)
    config = load_config(spec["config_path"])
    expected_ids_fit = set(spec.get("fitting_record_ids", []))
    expected_ids_score = set(spec.get("scoring_record_ids", []))
    custom_manifest = spec.get("fold_manifest")
    original_loader = train.load_fold_data
    original_prepare = train._prepare_model_for_training
    original_adamw = torch.optim.AdamW
    original_write_json = train.write_json
    original_classification_loss = train._classification_loss
    captured: dict[str, Any] = {}
    frozen_policy: dict[str, Any] | None = None

    if custom_manifest is not None:
        base_frame, dataset_manifest, dataset_dir = load_effective_dataset(config)

        def load_custom_fold(
            _config: dict[str, Any], _fold: int, *, input_mode: str
        ) -> tuple[Any, Any, dict[str, Any], dict[str, Any], Path]:
            if (
                Path(_config["_config_path"]).resolve()
                != Path(spec["config_path"]).resolve()
            ):
                raise RuntimeError("Nested cell requested an unexpected configuration")
            if int(_fold) != int(custom_manifest["inner_fold"]):
                raise RuntimeError("Nested cell requested an unexpected inner fold")
            fit_ids = set(custom_manifest["fitting_record_ids"])
            score_ids = set(custom_manifest["scoring_record_ids"])
            fitting = (
                base_frame.loc[base_frame["record_id"].astype(str).isin(fit_ids)]
                .copy()
                .reset_index(drop=True)
            )
            evaluation = (
                base_frame.loc[base_frame["record_id"].astype(str).isin(score_ids)]
                .copy()
                .reset_index(drop=True)
            )
            if input_mode != "target_only":
                raise RuntimeError(
                    "Nested calibration cells must use target-only input"
                )
            if (
                set(fitting["record_id"].astype(str)) != fit_ids
                or set(evaluation["record_id"].astype(str)) != score_ids
            ):
                raise RuntimeError("Inner fold IDs no longer align with frozen data")
            return fitting, evaluation, custom_manifest, dataset_manifest, dataset_dir

        train.load_fold_data = load_custom_fold
    elif expected_ids_fit or expected_ids_score:
        raise RuntimeError("Only nested calibration specs may supply record ID lists")

    method = spec["method"]
    if method in {"partial_4", "partial_2"}:
        last_layers = 4 if method == "partial_4" else 2

        def prepare_with_freeze(model: Any, device: Any) -> None:
            nonlocal frozen_policy
            original_prepare(model, device)
            frozen_policy = _freeze_policy(model, last_layers)
            captured["model"] = model

        train._prepare_model_for_training = prepare_with_freeze

    def adamw_trainable_only(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        parameter_list = list(parameters)
        if method in {"partial_4", "partial_2"}:
            if frozen_policy is None or "model" not in captured:
                raise RuntimeError("Partial freeze policy was not applied before AdamW")
            expected_trainable = [
                parameter
                for parameter in captured["model"].parameters()
                if parameter.requires_grad
            ]
        else:
            expected_trainable = parameter_list
        optimizer, trainable = _construct_trainable_adamw(
            original_adamw,
            parameter_list,
            expected_trainable,
            *args,
            **kwargs,
        )
        captured["optimizer_parameter_count"] = len(trainable)
        captured["optimizer_parameter_elements"] = sum(
            parameter.numel() for parameter in trainable
        )
        return optimizer

    torch.optim.AdamW = adamw_trainable_only

    if method in {"focal", "partial_4", "partial_2"}:

        def write_with_study_metadata(path: Path, value: dict[str, Any]) -> None:
            updated = value
            if path.name == "run_config.json":
                updated = json.loads(json.dumps(value))
                resolved = updated["resolved"]
                if method == "focal":
                    resolved["loss"] = "focal_gamma_2"
                    resolved["focal_gamma"] = 2.0
                    resolved["focal_alpha"] = "existing fitting-side class weights"
                if method in {"partial_4", "partial_2"}:
                    resolved["fine_tuning_policy"] = method
                    resolved["trainable_last_layers"] = (
                        4 if method == "partial_4" else 2
                    )
            elif path.name == "run_manifest.json":
                updated = json.loads(json.dumps(value))
                if method == "focal":
                    updated["loss"] = "focal_gamma_2"
                    updated["focal_gamma"] = 2.0
                    updated["focal_alpha"] = "existing fitting-side class weights"
                if method in {"partial_4", "partial_2"}:
                    updated["full_encoder_fine_tuned"] = False
                    updated["fine_tuning_policy"] = method
                    updated["trainable_last_layers"] = 4 if method == "partial_4" else 2
                    updated["optimizer_parameter_count"] = captured.get(
                        "optimizer_parameter_count"
                    )
                    updated["optimizer_parameter_elements"] = captured.get(
                        "optimizer_parameter_elements"
                    )
            original_write_json(path, updated)

        train.write_json = write_with_study_metadata
        if method == "focal":
            train._classification_loss = _focal_loss

    started = time.perf_counter()
    try:
        output = train.run_cell(
            config_path=Path(spec["config_path"]),
            condition="A2",
            fold=int(spec["train_fold"]),
            seed=int(spec["seed"]),
            model_path=Path(spec["model_path"]),
            precision="fp32",
            input_mode=spec["input_mode"],
            output=Path(spec["run_path"]),
            diagnostic_numerics=True,
            protocol_path=None,
        )
    finally:
        train.load_fold_data = original_loader
        train._prepare_model_for_training = original_prepare
        torch.optim.AdamW = original_adamw
        train.write_json = original_write_json
        train._classification_loss = original_classification_loss

    elapsed = time.perf_counter() - started
    audit: dict[str, Any] = {
        "cell_id": spec["cell_id"],
        "method": method,
        "elapsed_seconds": elapsed,
        "optimizer_parameter_count": captured.get("optimizer_parameter_count"),
        "optimizer_parameter_elements": captured.get("optimizer_parameter_elements"),
    }
    if frozen_policy is not None:
        model = captured["model"]
        frozen_differences = [
            name
            for name, parameter, initial in frozen_policy["frozen_snapshots"]
            if not torch.equal(parameter.detach().cpu(), initial)
        ]
        frozen_gradients = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        trainable_nonfinite = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not bool(torch.isfinite(parameter).all())
        ]
        if frozen_differences or frozen_gradients or trainable_nonfinite:
            raise RuntimeError(
                "Partial fine-tuning audit failed: "
                f"changed_frozen={frozen_differences[:5]}, "
                f"frozen_gradients={frozen_gradients[:5]}, "
                f"nonfinite_trainable={trainable_nonfinite[:5]}"
            )
        audit.update(
            {
                "last_trainable_layers": frozen_policy["last_trainable_layers"],
                "parameter_counts_by_policy_group": frozen_policy[
                    "parameter_counts_by_policy_group"
                ],
                "total_parameters": frozen_policy["total_parameters"],
                "frozen_parameter_tensors": len(frozen_policy["frozen_names"]),
                "frozen_parameters_unchanged": True,
                "frozen_parameters_have_no_gradients": True,
                "trainable_parameters_finite": True,
                "dropout_policy": (
                    "model.train() remains enabled; frozen weights are unchanged "
                    "while standard dropout behavior is preserved"
                ),
            }
        )
        total = frozen_policy["total_parameters"]
        audit["trainable_parameters"] = audit["optimizer_parameter_elements"]
        audit["trainable_percent"] = (
            100.0 * audit["optimizer_parameter_elements"] / total
        )

    if expected_ids_score:
        audit["custom_fitting_rows"] = len(expected_ids_fit)
        audit["custom_scoring_rows"] = len(expected_ids_score)
        audit["custom_split_manifest_sha256"] = spec["fold_manifest_sha256"]
    audit_path = Path(spec["audit_path"])
    _write_json(audit_path, audit)
    print(f"Validated treatment audit: {audit_path}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(execute_cell(args.spec.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
