"""Train one leakage-controlled family-grouped transformer fold."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import subprocess
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.experiment import (
    compute_training_weights,
    load_development_split,
    load_frozen_folds,
    load_nested_folds,
)
from models.transformer.modeling import (
    ClauseDataset,
    collate_clauses,
    weighted_cross_entropy,
)

DEFAULT_CONFIG = ROOT / "experiments/transformer/configs/deberta_clause_only.yaml"


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(np.int64)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    pr_auc = float(average_precision_score(labels, probabilities)) if labels.any() else 0.0
    return {
        "threshold": 0.5,
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "f2": float(fbeta_score(labels, predicted, beta=2, zero_division=0)),
        "pr_auc": pr_auc,
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "false_negative_rate": (
            float(false_negative / (false_negative + true_positive))
            if false_negative + true_positive
            else 0.0
        ),
        "confusion_matrix": {
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        },
    }


def _predict(model: Any, loader: Any, device: Any, use_bf16: bool) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    rows: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            positions = batch.pop("row_position")
            batch.pop("loss_weight")
            batch.pop("labels")
            inputs = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
            ):
                output = model(**inputs).logits
            if not torch.isfinite(output).all():
                raise FloatingPointError(
                    "Non-finite prediction logits; rerun with --precision fp32"
                )
            rows.append(positions.numpy())
            logits.append(output.float().cpu().numpy())
    row_array = np.concatenate(rows)
    logit_array = np.concatenate(logits)
    order = np.argsort(row_array)
    return row_array[order], logit_array[order]


def _tiny_balanced(frame: pd.DataFrame, limit_per_class: int) -> pd.DataFrame:
    pieces = [group.head(limit_per_class) for _, group in frame.groupby("label", sort=True)]
    return pd.concat(pieces).sort_index().reset_index(drop=True)


def _checkpoint_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(directory)).replace("\\", "/"): sha256_file(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def run(args: argparse.Namespace) -> Path:
    run_started = time.perf_counter()
    import torch
    import transformers
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DebertaV2Config,
        get_linear_schedule_with_warmup,
    )

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_dir = (ROOT / config["dataset_dir"]).resolve()
    training = config["training"]
    learning_rates = [float(value) for value in training["learning_rates"]]
    learning_rate = args.learning_rate
    if learning_rate is None:
        if len(learning_rates) != 1:
            raise ValueError("Select one configured learning rate with --learning-rate")
        learning_rate = learning_rates[0]
    if learning_rate not in learning_rates:
        raise ValueError("--learning-rate must be one of the values pinned in the config")
    seed = args.seed if args.seed is not None else int(training["screening_seed"])
    _set_seed(seed)

    full_train = load_development_split(dataset_dir, "train")
    dataset_manifest = json.loads(
        (dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    train_info = dataset_manifest["split_files"]["train"]
    fold_assignment = (ROOT / config["selection"]["fold_assignment"]).resolve()
    folds = load_frozen_folds(
        full_train,
        fold_assignment,
        expected_sha256=config["selection"]["fold_assignment_sha256"],
        expected_train_sha256=train_info["sha256"],
    )
    inner_assignment = (
        ROOT / config["selection"]["inner_selection_assignment"]
    ).resolve()
    nested_folds = load_nested_folds(
        full_train,
        folds,
        inner_assignment,
        expected_sha256=config["selection"]["inner_selection_assignment_sha256"],
        expected_train_sha256=train_info["sha256"],
    )
    if args.fold < 1 or args.fold > len(folds):
        raise ValueError(f"--fold must be between 1 and {len(folds)}")
    nested_fold = nested_folds[args.fold - 1]
    fold = nested_fold.outer
    train_frame = full_train.iloc[nested_fold.fitting_indices].reset_index(drop=True)
    selection_frame = full_train.iloc[nested_fold.selection_indices].reset_index(drop=True)
    selection_scoring_record_ids = set(
        full_train.iloc[nested_fold.selection_scoring_indices]["record_id"].astype(str)
    )
    outer_validation_frame = full_train.iloc[fold.validation_indices].reset_index(drop=True)
    outer_scoring_record_ids = set(
        full_train.iloc[fold.scoring_indices]["record_id"].astype(str)
    )
    if args.smoke:
        train_frame = _tiny_balanced(train_frame, 4)
        selection_frame = _tiny_balanced(selection_frame, 2)
        selection_scoring_record_ids = set(selection_frame["record_id"].astype(str))
        outer_validation_frame = _tiny_balanced(outer_validation_frame, 2)
        outer_scoring_record_ids = set(
            outer_validation_frame["record_id"].astype(str)
        )

    weights = compute_training_weights(train_frame, args.weight_strategy)
    repository = config["model"]["repository"]
    revision = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        repository,
        revision=revision,
        local_files_only=not args.allow_download,
        use_fast=True,
    )
    if args.smoke:
        model_config = DebertaV2Config(
            vocab_size=tokenizer.vocab_size,
            hidden_size=48,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=96,
            max_position_embeddings=512,
            num_labels=2,
        )
        model = AutoModelForSequenceClassification.from_config(model_config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            repository,
            revision=revision,
            local_files_only=not args.allow_download,
            num_labels=2,
            dtype=torch.float32,
        )

    max_length = int(config["tokenization"]["max_length"])
    variant = config["variant"]
    train_dataset = ClauseDataset(
        train_frame,
        tokenizer,
        variant=variant,
        max_length=max_length,
        weights=weights,
    )
    selection_dataset = ClauseDataset(
        selection_frame,
        tokenizer,
        variant=variant,
        max_length=max_length,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise RuntimeError("The pinned tokenizer has no padding token")
    collator = partial(collate_clauses, pad_token_id=pad_token_id)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["train_batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    selection_loader = DataLoader(
        selection_dataset,
        batch_size=int(training["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_supported = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if args.precision == "bf16" and not bf16_supported:
        raise RuntimeError("bf16 was requested but is not supported on this device")
    use_bf16 = args.precision == "bf16" or (
        args.precision == "auto" and bf16_supported
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    max_epochs = 1 if args.smoke else int(training["max_epochs"])
    total_steps = max_epochs * len(train_loader)
    warmup_steps = round(total_steps * float(training["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Run output is immutable and already exists: {output_dir}")
    temporary = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    checkpoint_dir = temporary / "checkpoint"
    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    patience = 0
    training_started = time.perf_counter()
    try:
        for epoch in range(1, max_epochs + 1):
            model.train()
            loss_sum = 0.0
            example_count = 0
            for batch in train_loader:
                batch.pop("row_position")
                labels = batch.pop("labels").to(device)
                loss_weight = batch.pop("loss_weight").to(device)
                inputs = {key: value.to(device) for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16
                ):
                    logits = model(**inputs).logits
                    loss = weighted_cross_entropy(logits, labels, loss_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Non-finite training loss; rerun with --precision fp32"
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["max_grad_norm"])
                )
                optimizer.step()
                scheduler.step()
                batch_size = len(labels)
                loss_sum += float(loss.detach().cpu()) * batch_size
                example_count += batch_size

            positions, selection_logits = _predict(
                model, selection_loader, device, use_bf16
            )
            probabilities = torch.softmax(
                torch.from_numpy(selection_logits), dim=1
            ).numpy()[:, 1]
            labels = selection_frame.iloc[positions]["label"].to_numpy(dtype=np.int64)
            record_ids = (
                selection_frame.iloc[positions]["record_id"].astype(str).to_numpy()
            )
            scoring_mask = np.array(
                [
                    record_id in selection_scoring_record_ids
                    for record_id in record_ids
                ],
                dtype=bool,
            )
            selection_metrics = _metrics(
                labels[scoring_mask], probabilities[scoring_mask]
            )
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": loss_sum / example_count,
                    **{
                        f"selection_{key}": value
                        for key, value in selection_metrics.items()
                        if key != "confusion_matrix"
                    },
                }
            )
            score = float(selection_metrics[training["checkpoint_metric"]])
            if score > best_score:
                best_score = score
                best_epoch = epoch
                patience = 0
                model.save_pretrained(checkpoint_dir, safe_serialization=True)
                tokenizer.save_pretrained(checkpoint_dir)
            else:
                patience += 1
                if patience >= int(training["early_stopping_patience"]):
                    break

        training_runtime_seconds = time.perf_counter() - training_started
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        best_model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir, local_files_only=True, dtype=torch.float32
        ).to(device)

        # The outer fold is materialized only after checkpoint selection has finished.
        outer_validation_dataset = ClauseDataset(
            outer_validation_frame,
            tokenizer,
            variant=variant,
            max_length=max_length,
        )
        outer_validation_loader = DataLoader(
            outer_validation_dataset,
            batch_size=int(training["eval_batch_size"]),
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
            pin_memory=torch.cuda.is_available(),
        )
        positions, validation_logits = _predict(
            best_model, outer_validation_loader, device, use_bf16
        )
        probability = torch.softmax(torch.from_numpy(validation_logits), dim=1).numpy()[:, 1]
        predictions = outer_validation_frame.iloc[positions][
            ["record_id", "family_id", "label", "label_tier"]
        ].reset_index(drop=True)
        predictions["logit_0"] = validation_logits[:, 0]
        predictions["logit_1"] = validation_logits[:, 1]
        predictions["ambiguity_probability"] = probability
        predictions["predicted_label_0_5"] = (probability >= 0.5).astype(int)
        predictions["included_in_scoring"] = predictions["record_id"].astype(str).isin(
            outer_scoring_record_ids
        )
        scoring = predictions["included_in_scoring"].to_numpy(dtype=bool)
        final_metrics = _metrics(
            predictions.loc[scoring, "label"].to_numpy(dtype=np.int64), probability[scoring]
        )
        pd.DataFrame(history).to_csv(
            temporary / "training_history.csv", index=False, lineterminator="\n"
        )
        predictions.to_csv(
            temporary / "validation_predictions.csv", index=False, lineterminator="\n"
        )
        _json(temporary / "metrics.json", final_metrics)
        resolved_config = {
            **config,
            "resolved": {
                "learning_rate": learning_rate,
                "seed": seed,
                "fold": args.fold,
                "weight_strategy": args.weight_strategy,
                "requested_precision": args.precision,
                "smoke": args.smoke,
            },
        }
        _json(temporary / "run_config.json", resolved_config)
        total_runtime_seconds = time.perf_counter() - run_started
        run_manifest = {
            "schema_version": 1,
            "git_commit": _git_commit(),
            "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
            "config_sha256": sha256_file(config_path),
            "train_split_sha256": train_info["sha256"],
            "fold_assignment_sha256": sha256_file(fold_assignment),
            "inner_selection_assignment_sha256": sha256_file(inner_assignment),
            "model_repository": repository,
            "model_revision": revision,
            "model_license": config["model"]["license"],
            "fold": args.fold,
            "held_out_families": list(fold.held_out_families),
            "excluded_exact_duplicate_record_ids": list(
                fold.excluded_exact_duplicate_record_ids
            ),
            "inner_selection_family": nested_fold.inner_selection_family,
            "excluded_inner_duplicate_record_ids": list(
                nested_fold.excluded_inner_duplicate_record_ids
            ),
            "training_rows": len(train_frame),
            "inner_selection_rows": len(selection_frame),
            "inner_selection_scored_rows": len(selection_scoring_record_ids),
            "inner_selection_positive_labels": int(
                selection_frame.loc[
                    selection_frame["record_id"].astype(str).isin(
                        selection_scoring_record_ids
                    ),
                    "label",
                ].sum()
            ),
            "checkpoint_selection_partition": "inner_grouped_family",
            "outer_fold_evaluations_during_training": 0,
            "outer_fold_evaluations_after_selection": 1,
            "validation_rows": len(outer_validation_frame),
            "scored_validation_rows": int(scoring.sum()),
            "seed": seed,
            "best_epoch": best_epoch,
            "precision": "bf16" if use_bf16 else "fp32",
            "model_parameter_dtype": str(next(best_model.parameters()).dtype),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "training_runtime_seconds": training_runtime_seconds,
            "total_runtime_seconds": total_runtime_seconds,
            "peak_gpu_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "peak_gpu_memory_reserved_mb": (
                torch.cuda.max_memory_reserved(device) / (1024**2)
                if device.type == "cuda"
                else None
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "platform": platform.platform(),
            "process_id": os.getpid(),
            "checkpoint_hashes": _checkpoint_hashes(checkpoint_dir),
        }
        _json(temporary / "run_manifest.json", run_manifest)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--weight-strategy",
        choices=("unweighted", "class", "confidence", "combined"),
        default="unweighted",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--precision", choices=("auto", "bf16", "fp32"), default="fp32"
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    output = run(parse_args())
    print(f"Run written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
