"""Aggregate a complete 5-fold x 3-seed condition without changing predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common import (
    immutable_directory,
    hash_directory,
    read_json,
    resolve_path,
    sha256_file,
    write_json,
)
from .config import DEFAULT_CONFIG, load_config
from .dataset import load_effective_dataset
from .metrics import binary_metrics
from .protocol import load_fp32_protocol
from .run_validation import validate_completed_run

DEFAULT_PROTOCOL = (
    DEFAULT_CONFIG.parents[0] / "fp32_protocol_v1.json"
)
DEFAULT_RUN_OVERRIDES = (
    DEFAULT_CONFIG.parents[0] / "run_path_overrides_fp32_v1.json"
)

SCALAR_METRICS = (
    "average_precision",
    "roc_auc",
    "precision",
    "recall",
    "f1",
    "f2",
    "mcc",
    "accuracy",
    "positive_prevalence",
    "ap_prevalence_lift",
    "false_positives",
    "false_negatives",
    "positive_labels",
    "negative_labels",
)


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    flattened = {key: metrics.get(key) for key in SCALAR_METRICS}
    flattened.update(
        {
            f"confusion_{key}": value
            for key, value in metrics.get("confusion_matrix", {}).items()
        }
    )
    return flattened


def across_seed_summary(
    per_seed: pd.DataFrame, columns: list[str]
) -> dict[str, dict[str, float]]:
    if per_seed["seed"].duplicated().any() or len(per_seed) < 2:
        raise ValueError("Across-seed aggregation requires at least two unique seeds")
    return {
        column: {
            "mean": float(per_seed[column].mean()),
            "sample_std": float(per_seed[column].std(ddof=1)),
        }
        for column in columns
    }


def aggregate_condition(
    config_path: Path,
    condition: str,
    *,
    protocol_path: Path = DEFAULT_PROTOCOL,
    run_overrides_path: Path = DEFAULT_RUN_OVERRIDES,
    output: Path | None = None,
) -> Path:
    config = load_config(config_path)
    if condition not in config["conditions"]:
        raise ValueError(f"Unknown condition: {condition}")
    protocol, protocol_sha256 = load_fp32_protocol(protocol_path, config)
    override_pairs = json.loads(
        run_overrides_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    overrides = override_pairs.get("overrides")
    if (
        override_pairs.get("schema_version") != 1
        or override_pairs.get("protocol_id") != protocol["protocol_id"]
        or not isinstance(overrides, dict)
        or set(overrides)
        - {f"A1/fold_1/seed_1729"}
    ):
        raise ValueError(f"Invalid run-path override mapping: {run_overrides_path}")
    effective, dataset_manifest, _ = load_effective_dataset(config)
    runs_root = resolve_path(config["outputs"]["runs_dir"])
    source_hash = sha256_file(resolve_path(config["data"]["source_csv"]))
    model_source_hash, model_file_hashes = hash_directory(
        Path(protocol["model"]["local_path"])
    )
    if (
        source_hash != protocol["dataset"]["source_sha256"]
        or model_source_hash != protocol["model"]["source_sha256"]
        or dataset_manifest["effective_dataset_sha256"]
        != protocol["dataset"]["effective_dataset_sha256"]
        or sha256_file(
            resolve_path(config["outputs"]["dataset_dir"]) / "manifest.json"
        )
        != protocol["dataset"]["dataset_manifest_sha256"]
    ):
        raise RuntimeError("Source dataset or pretrained model hash mismatch")
    if overrides != {
        "A1/fold_1/seed_1729": {
            "path": "A1/fold_1/seed_1729_diagnostic_numerics_v2",
            "legacy_protocol_compatible": True,
        }
    }:
        raise ValueError(f"Unexpected run-path override mapping: {run_overrides_path}")
    rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    selected_paths: set[Path] = set()
    for seed in config["training"]["seeds"]:
        for fold in range(1, int(config["folds"]["count"]) + 1):
            cell_key = f"{condition}/fold_{fold}/seed_{seed}"
            override = overrides.get(cell_key)
            if override is not None and (
                not isinstance(override, dict)
                or not isinstance(override.get("path"), str)
                or Path(override["path"]).is_absolute()
                or not isinstance(
                    override.get("legacy_protocol_compatible", False), bool
                )
            ):
                raise ValueError(f"Invalid run-path override for {cell_key}")
            run = (
                runs_root / condition / f"fold_{fold}" / f"seed_{seed}"
                if override is None
                else runs_root / override["path"]
            ).resolve()
            try:
                run.relative_to(runs_root.resolve())
            except ValueError as exc:
                raise ValueError(f"Run path escapes runs root: {run}") from exc
            if run in selected_paths:
                raise RuntimeError(f"Duplicate run path selected for {cell_key}: {run}")
            selected_paths.add(run)
            canonical_run = (
                runs_root / condition / f"fold_{fold}" / f"seed_{seed}"
            ).resolve()
            if override is not None and canonical_run != run and (
                canonical_run / "run_manifest.json"
            ).is_file():
                raise RuntimeError(
                    f"Both canonical and overridden runs exist for {cell_key}"
                )
            frame = validate_completed_run(
                run=run,
                condition=condition,
                fold=fold,
                seed=seed,
                config=config,
                protocol=protocol,
                protocol_sha256=protocol_sha256,
                effective=effective,
                dataset_manifest=dataset_manifest,
                dataset_dir=resolve_path(config["outputs"]["dataset_dir"]),
                model_source_sha256=model_source_hash,
                model_file_hashes=model_file_hashes,
                legacy_protocol_compatible=bool(
                    override and override.get("legacy_protocol_compatible", False)
                ),
            )
            manifest = read_json(run / "run_manifest.json")
            metrics = binary_metrics(
                frame["label"].to_numpy(),
                frame["probability_yes"].to_numpy(),
                threshold=float(config["training"]["threshold"]),
            )
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "fold": fold,
                    "held_out_families": "|".join(manifest["held_out_families"]),
                    "run_path": run.relative_to(runs_root.resolve()).as_posix(),
                    "class_weights": json.dumps(
                        manifest["actual_class_weights"], sort_keys=True
                    ),
                    **_flatten_metrics(metrics),
                    "rows": len(frame),
                }
            )
            frame.insert(0, "condition", condition)
            frame.insert(1, "seed", seed)
            predictions.append(frame)
    all_predictions = pd.concat(predictions, ignore_index=True)
    per_run = pd.DataFrame(rows).sort_values(["seed", "fold"]).reset_index(drop=True)
    expected_cells = len(config["training"]["seeds"]) * int(config["folds"]["count"])
    if (
        len(per_run) != expected_cells
        or per_run.duplicated(["seed", "fold"]).any()
        or set(per_run["seed"]) != set(config["training"]["seeds"])
        or set(per_run["fold"]) != set(range(1, int(config["folds"]["count"]) + 1))
    ):
        raise RuntimeError("Aggregate does not contain each required seed/fold exactly once")
    seed_rows: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    expected_ids = set(effective["record_id"].astype(str))
    for seed, frame in all_predictions.groupby("seed", sort=True):
        if (
            frame["record_id"].duplicated().any()
            or set(frame["record_id"].astype(str)) != expected_ids
        ):
            raise RuntimeError(
                f"Seed {seed} does not contain one OOF prediction per effective record"
            )
        pooled = binary_metrics(
            frame["label"].to_numpy(),
            frame["probability_yes"].to_numpy(),
            threshold=float(config["training"]["threshold"]),
        )
        fold_metrics = per_run.loc[per_run["seed"] == seed]
        seed_rows.append(
            {
                "condition": condition,
                "seed": int(seed),
                "mean_fold_average_precision": float(
                    fold_metrics["average_precision"].mean()
                ),
                "std_fold_average_precision": float(
                    fold_metrics["average_precision"].std(ddof=1)
                ),
                **{
                    f"pooled_{key}": value
                    for key, value in _flatten_metrics(pooled).items()
                },
                "rows": len(frame),
            }
        )
        for family, subset in frame.groupby("family_id", sort=True):
            metric = binary_metrics(
                subset["label"].to_numpy(),
                subset["probability_yes"].to_numpy(),
                threshold=float(config["training"]["threshold"]),
            )
            family_rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "family_id": str(family),
                    **_flatten_metrics(metric),
                    "rows": len(subset),
                }
            )
    per_seed = pd.DataFrame(seed_rows).sort_values("seed").reset_index(drop=True)
    numeric = [
        column
        for column in per_seed.columns
        if column not in {"condition", "seed", "rows"}
    ]
    summary = {
        "schema_version": 1,
        "condition": condition,
        "primary_metric": "average_precision",
        "threshold_policy": "fixed 0.5 only; no threshold optimization",
        "seeds": list(config["training"]["seeds"]),
        "folds": int(config["folds"]["count"]),
        "effective_dataset_sha256": dataset_manifest["effective_dataset_sha256"],
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256,
        "run_cells": len(per_run),
        "across_seed_summary": across_seed_summary(per_seed, numeric),
    }
    destination = (
        output or (resolve_path(config["outputs"]["aggregate_dir"]) / condition)
    ).resolve()
    with immutable_directory(destination) as building:
        per_run.to_csv(
            building / "per_fold_seed_metrics.csv", index=False, lineterminator="\n"
        )
        per_seed.to_csv(
            building / "per_seed_metrics.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame(family_rows).to_csv(
            building / "per_family_seed_metrics.csv", index=False, lineterminator="\n"
        )
        all_predictions.sort_values(["seed", "fold", "record_id"]).to_csv(
            building / "oof_predictions.csv", index=False, lineterminator="\n"
        )
        write_json(building / "summary.json", summary)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--condition", choices=("A1", "A2"), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--run-overrides", type=Path, default=DEFAULT_RUN_OVERRIDES)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = aggregate_condition(
        args.config,
        args.condition,
        protocol_path=args.protocol,
        run_overrides_path=args.run_overrides,
        output=args.output,
    )
    print(f"Aggregate written to {output}")
    return 0


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key in run overrides: {key}")
        result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
