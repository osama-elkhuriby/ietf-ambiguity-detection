"""Prepare an immutable V2 protocol and verify compatible V1 study outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.transformer_final.recall_improvement_study_v1.study_protocol import (
    EXPECTED_BASE_PROTOCOL,
    make_inner_splits,
)
from models.transformer_final.common import (
    read_json,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
    write_json,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.leakage import audit_leakage, build_fold_manifests

REPO_ROOT = Path(__file__).resolve().parents[3]
V1_ROOT = REPO_ROOT / "experiments" / "transformer_final" / "recall_improvement_study_v1"
V2_ROOT = Path(__file__).resolve().parent
V1_PROTOCOL_PATH = V1_ROOT / "protocols" / "recall_improvement_protocol_v1.json"
V1_SPLITS_PATH = V1_ROOT / "protocols" / "nested_inner_splits_v1.json"
V2_PROTOCOL_PATH = V2_ROOT / "protocols" / "recall_improvement_protocol_v2.json"
V2_OUTER_PATH = V2_ROOT / "protocols" / "outer_fold_manifests_v2.json"
V2_SPLITS_PATH = V2_ROOT / "protocols" / "nested_inner_splits_v2.json"
V2_COMPATIBILITY_PATH = V2_ROOT / "V1_TO_V2_COMPATIBILITY.json"
V2_INVENTORY_PATH = V2_ROOT / "EXPERIMENT_INVENTORY.json"
V2_BUDGET_PATH = V2_ROOT / "time_budget.json"
V2_RUNNER_CONFIG_PATH = V2_ROOT / "V2_RUNNER_CONFIG.json"
STALE_POLICY_CHANGE = (
    "remove only fitting-side context sanitizations for records excluded from fitting"
)


def _study_source_hashes() -> dict[str, str]:
    return {
        path.relative_to(REPO_ROOT).as_posix(): sha256_file(path)
        for path in sorted(V2_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _effective_policy(
    policy: list[dict[str, Any]], fitting_ids: set[str]
) -> list[dict[str, Any]]:
    return [
        item
        for item in policy
        if item["role"] != "fitting" or item["record_id"] in fitting_ids
    ]


def _compare_manifest(
    old: dict[str, Any], new: dict[str, Any], *, nested: bool = False
) -> dict[str, Any]:
    fitting_key = "fitting_record_ids"
    scoring_key = "scoring_record_ids"
    fields = [fitting_key, scoring_key, "fitting_exclusions"]
    if not nested:
        fields.extend(
            [
                "candidate_fitting_record_ids",
                "fitting_families",
                "held_out_families",
            ]
        )
    else:
        fields.extend(["fitting_families", "held_out_families"])
    for field in fields:
        if old[field] != new[field]:
            raise RuntimeError(
                f"V1/V2 manifest membership mismatch for {field}: "
                f"fold={old.get('fold', old.get('outer_fold'))}"
            )

    fitting_ids = set(old[fitting_key])
    expected_policy = _effective_policy(old["context_sanitizations"], fitting_ids)
    if new["context_sanitizations"] != expected_policy:
        raise RuntimeError("V2 context policy differs beyond stale fitting entries")

    old_counts = dict(old["counts"])
    new_counts = dict(new["counts"])
    old_counts.pop("context_fields_sanitized", None)
    new_counts.pop("context_fields_sanitized", None)
    if old_counts != new_counts:
        raise RuntimeError("V1/V2 manifest counts differ beyond sanitization count")

    stale = [
        item
        for item in old["context_sanitizations"]
        if item["role"] == "fitting" and item["record_id"] not in fitting_ids
    ]
    if any(item["role"] != "fitting" for item in stale):
        raise RuntimeError("Only stale fitting sanitizations may be removed")
    return {
        "fitting_rows": len(fitting_ids),
        "scoring_rows": len(old[scoring_key]),
        "fitting_exclusion_count": len(old["fitting_exclusions"]),
        "stale_sanitizations_removed": stale,
        "valid_sanitizations_preserved": len(expected_policy),
    }


def _assert_artifacts_intact(run_path: Path, manifest: dict[str, Any]) -> None:
    for name, artifact in manifest["artifact_hashes"].items():
        artifact_path = run_path / artifact["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Missing V1 artifact {name}: {artifact_path}")
        if sha256_file(artifact_path) != artifact["sha256"]:
            raise RuntimeError(f"V1 artifact hash mismatch: {artifact_path}")


def _verify_reusable_run(
    *,
    cell_id: str,
    run_path: Path,
    run_record: dict[str, Any],
    old_protocol: dict[str, Any],
    expected_manifest: dict[str, Any],
    expected_split: dict[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    manifest_path = run_path / "run_manifest.json"
    manifest = read_json(manifest_path)
    expected_fold = int(
        expected_manifest.get(
            "fold",
            expected_split["inner_fold"] if expected_split is not None else -1,
        )
    )
    if (
        manifest.get("condition") != "A2"
        or manifest.get("precision") != "fp32"
        or manifest.get("input_mode") != mode
        or manifest.get("fold") != expected_fold
        or manifest.get("seed") != int(old_protocol["training"]["seed"])
        or manifest.get("epochs_completed") != 5
        or manifest.get("fitting_rows") != len(expected_manifest["fitting_record_ids"])
        or manifest.get("held_out_rows") != len(expected_manifest["scoring_record_ids"])
        or manifest.get("config_sha256") != old_protocol["config_sha256"]
        or manifest.get("effective_dataset_sha256")
        != old_protocol["dataset"]["effective_dataset_sha256"]
        or manifest.get("dataset_manifest_sha256")
        != old_protocol["dataset"]["dataset_manifest_sha256"]
        or manifest.get("fold_manifest_sha256")
        != old_protocol["dataset"]["fold_manifest_sha256"][str(expected_fold)]
        or manifest.get("model_source_sha256") != old_protocol["model"]["source_sha256"]
        or manifest.get("model_file_hashes") != old_protocol["model"]["model_file_hashes"]
        or manifest.get("source_file_hashes")
        != old_protocol["source_snapshot"]["core_files"]
    ):
        raise RuntimeError(f"Incompatible V1 run provenance: {cell_id}")
    if sha256_file(manifest_path) != run_record["run_manifest_sha256"]:
        raise RuntimeError(f"V1 run-manifest hash mismatch: {cell_id}")
    _assert_artifacts_intact(run_path, manifest)

    if manifest.get("fitting_exclusions") != expected_manifest["fitting_exclusions"]:
        raise RuntimeError(f"V1 run exclusions differ from V2 policy: {cell_id}")
    if mode == "local_context" and manifest.get(
        "context_sanitizations"
    ) != expected_manifest["context_sanitizations"]:
        raise RuntimeError(f"V1 run context transformation differs: {cell_id}")

    predictions_path = run_path / manifest["artifact_hashes"][
        "held_out_predictions"
    ]["path"]
    predictions = pd.read_csv(predictions_path)
    if (
        set(predictions["record_id"].astype(str))
        != set(expected_manifest["scoring_record_ids"])
        or predictions["record_id"].astype(str).duplicated().any()
    ):
        raise RuntimeError(f"V1 prediction membership mismatch: {cell_id}")

    audit_sha256 = None
    if expected_split is not None:
        audit_path = V1_ROOT / "audits" / "B_nested_threshold_calibration" / (
            cell_id.split("/", 1)[1] + ".json"
        )
        audit = read_json(audit_path)
        audit_sha256 = sha256_file(audit_path)
        expected_split_hash = sha256_text(stable_json(expected_split))
        if (
            audit.get("custom_split_manifest_sha256") != expected_split_hash
            or audit.get("custom_fitting_rows") != len(expected_split["fitting_record_ids"])
            or audit.get("custom_scoring_rows") != len(expected_split["scoring_record_ids"])
        ):
            raise RuntimeError(f"V1 nested split audit mismatch: {cell_id}")
    return {
        "v1_cell_id": cell_id,
        "v1_run_path": str(run_path),
        "v1_run_manifest_sha256": sha256_file(manifest_path),
        "v1_audit_sha256": audit_sha256,
        "v1_artifact_hashes": manifest["artifact_hashes"],
        "v1_model_file_hashes": manifest["model_file_hashes"],
        "v1_dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "v1_effective_dataset_sha256": manifest["effective_dataset_sha256"],
        "v1_fold_manifest_sha256": manifest["fold_manifest_sha256"],
        "v1_source_file_hashes": manifest["source_file_hashes"],
        "executed_under_protocol": "recall_improvement_study_v1",
        "v2_status": "REUSED_BY_REFERENCE_AFTER_COMPATIBILITY_CHECK",
    }


def _write_protocol_artifacts(
    old_protocol: dict[str, Any],
    old_protocol_sha256: str,
    frame: pd.DataFrame,
    dataset_manifest: dict[str, Any],
    dataset_dir: Path,
    outer_manifests: list[dict[str, Any]],
    nested_splits: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    outer_json = json.dumps(outer_manifests, indent=2, sort_keys=True) + "\n"
    nested_json = json.dumps(nested_splits, indent=2, sort_keys=True) + "\n"
    V2_OUTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    V2_OUTER_PATH.write_bytes(outer_json.encode("utf-8"))
    V2_SPLITS_PATH.write_bytes(nested_json.encode("utf-8"))

    core_hashes = source_file_hashes()
    study_hashes = _study_source_hashes()
    source_snapshot = {
        "core_files": core_hashes,
        "study_files": study_hashes,
        "compatibility_dependencies": {
            "v1_study_protocol.py": sha256_file(
                V1_ROOT / "study_protocol.py"
            ),
            "v1_cell_launcher.py": sha256_file(V1_ROOT / "cell_launcher.py"),
            "v1_study_runner.py": sha256_file(V1_ROOT / "study_runner.py"),
        },
    }
    protocol = json.loads(json.dumps(old_protocol))
    protocol.update(
        {
            "schema_version": 2,
            "study_id": "recall_improvement_study_v2",
            "parent_protocol": {
                "study_id": "recall_improvement_study_v1",
                "path": str(V1_PROTOCOL_PATH),
                "sha256": old_protocol_sha256,
                "source_snapshot_sha256": old_protocol["source_snapshot_sha256"],
                "reuse_policy": (
                    "V1 runs remain V1-provenance artifacts and are referenced, "
                    "never copied or relabeled as V2 executions."
                ),
            },
            "dataset": {
                **old_protocol["dataset"],
                "study_outer_fold_manifests_path": str(V2_OUTER_PATH),
                "study_outer_fold_manifests_sha256": sha256_file(V2_OUTER_PATH),
                "study_nested_splits_path": str(V2_SPLITS_PATH),
                "study_nested_splits_sha256": sha256_file(V2_SPLITS_PATH),
                "source_rows_verified": len(frame),
                "effective_dataset_sha256_verified": dataset_manifest[
                    "effective_dataset_sha256"
                ],
                "dataset_manifest_sha256_verified": sha256_file(
                    dataset_dir / "manifest.json"
                ),
            },
            "inner_cv": {
                **old_protocol["inner_cv"],
                "manifests_path": str(V2_SPLITS_PATH),
                "manifests_sha256": sha256_file(V2_SPLITS_PATH),
            },
            "source_snapshot": source_snapshot,
            "source_snapshot_sha256": sha256_text(stable_json(source_snapshot)),
            "policy_correction": {
                "change": STALE_POLICY_CHANGE,
                "fitting_exclusions_changed": False,
                "scoring_membership_changed": False,
                "valid_context_sanitizations_changed": False,
            },
        }
    )
    protocol_path = V2_PROTOCOL_PATH
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return protocol, sha256_file(protocol_path)


def _make_specs(protocol: dict[str, Any], protocol_sha256: str, manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seed = int(protocol["training"]["seed"])
    config_path = str(Path(protocol["config_path"]).resolve())
    model_path = str(Path(protocol["model"]["local_path"]).resolve())
    for fold in (3, 4, 5):
        manifest = manifests[fold - 1]
        specs.append(
            {
                "cell_id": f"C_local_context/fold_{fold}/seed_{seed}",
                "method": "context",
                "train_fold": fold,
                "seed": seed,
                "config_path": config_path,
                "model_path": model_path,
                "input_mode": "local_context",
                "fold_manifest": manifest,
                "fold_manifest_sha256": sha256_text(stable_json(manifest)),
                "run_path": str(V2_ROOT / "runs" / "C_local_context" / f"fold_{fold}" / f"seed_{seed}"),
                "audit_path": str(V2_ROOT / "audits" / "C_local_context" / f"fold_{fold}_seed_{seed}.json"),
                "log_path": str(V2_ROOT / "logs" / f"fold_{fold}_seed_{seed}"),
                "protocol_sha256": protocol_sha256,
                "study_id": "recall_improvement_study_v2",
            }
        )
    stages = (
        ("G_partial_4", "partial_4", "target_only"),
        ("D_focal_loss", "focal", "target_only"),
        ("G2_partial_2", "partial_2", "target_only"),
    )
    for stage, method, input_mode in stages:
        for fold in range(1, 6):
            manifest = manifests[fold - 1]
            specs.append(
                {
                    "cell_id": f"{stage}/fold_{fold}/seed_{seed}",
                    "method": method,
                    "train_fold": fold,
                    "seed": seed,
                    "config_path": config_path,
                    "model_path": model_path,
                    "input_mode": input_mode,
                    "fold_manifest": manifest,
                    "fold_manifest_sha256": sha256_text(stable_json(manifest)),
                    "run_path": str(V2_ROOT / "runs" / stage / f"fold_{fold}" / f"seed_{seed}"),
                    "audit_path": str(V2_ROOT / "audits" / stage / f"fold_{fold}_seed_{seed}.json"),
                    "log_path": str(V2_ROOT / "logs" / f"{stage}_fold_{fold}_seed_{seed}"),
                    "protocol_sha256": protocol_sha256,
                    "study_id": "recall_improvement_study_v2",
                }
            )
    spec_root = V2_ROOT / "cell_specs"
    spec_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        spec_path = spec_root / f"{sha256_text(spec['cell_id'])}.json"
        write_json(spec_path, spec)
        spec["spec_path"] = str(spec_path)
        spec["spec_sha256"] = sha256_file(spec_path)
    return specs


def prepare_v2() -> Path:
    if any(
        path.exists()
        for path in (
            V2_PROTOCOL_PATH,
            V2_OUTER_PATH,
            V2_SPLITS_PATH,
            V2_COMPATIBILITY_PATH,
            V2_INVENTORY_PATH,
        )
    ):
        raise FileExistsError("V2 study artifacts already exist and are immutable")

    old_protocol = read_json(V1_PROTOCOL_PATH)
    old_protocol_sha256 = sha256_file(V1_PROTOCOL_PATH)
    v1_inventory = read_json(V1_ROOT / "EXPERIMENT_INVENTORY.json")
    if v1_inventory.get("protocol_sha256") != old_protocol_sha256:
        raise RuntimeError("V1 inventory and frozen protocol hashes disagree")
    if old_protocol["baseline_protocol_sha256"] != EXPECTED_BASE_PROTOCOL:
        raise RuntimeError("V1 baseline protocol hash is not the frozen baseline")

    config = load_config(Path(old_protocol["config_path"]))
    frame, dataset_manifest, dataset_dir = load_effective_dataset(config)
    if (
        dataset_manifest["effective_dataset_sha256"]
        != old_protocol["dataset"]["effective_dataset_sha256"]
        or sha256_file(dataset_dir / "manifest.json")
        != old_protocol["dataset"]["dataset_manifest_sha256"]
        or len(frame) != old_protocol["dataset"]["rows"]
    ):
        raise RuntimeError("Current effective dataset differs from frozen V1")

    fold_frame = frame.copy()
    fold_frame["fold"] = fold_frame["fold"].astype(int)
    audit = audit_leakage(fold_frame, near_threshold=0.85)
    if audit["training_blocked"]:
        raise RuntimeError("Leakage audit blocks V2 outer folds")
    outer_manifests = build_fold_manifests(fold_frame, audit)
    if len(outer_manifests) != 5:
        raise RuntimeError("Expected five corrected V2 outer-fold manifests")

    outer_comparisons: dict[str, Any] = {}
    for fold, new in enumerate(outer_manifests, start=1):
        old = read_json(dataset_dir / "folds" / f"fold_{fold}.json")
        if sha256_file(dataset_dir / "folds" / f"fold_{fold}.json") != old_protocol[
            "dataset"
        ]["fold_manifest_sha256"][str(fold)]:
            raise RuntimeError(f"Frozen V1 fold {fold} hash mismatch")
        outer_comparisons[str(fold)] = _compare_manifest(old, new)

    nested_splits = [
        split
        for outer_fold in range(1, 6)
        for split in make_inner_splits(fold_frame, outer_fold, splits=3)
    ]
    v1_nested = json.loads(V1_SPLITS_PATH.read_text(encoding="utf-8"))
    if len(v1_nested) != 15 or len(nested_splits) != 15:
        raise RuntimeError("V1 and V2 must each contain 15 nested splits")
    nested_lookup = {
        (int(item["outer_fold"]), int(item["inner_fold"])): item
        for item in nested_splits
    }
    nested_comparisons: dict[str, Any] = {}
    for old in v1_nested:
        key = (int(old["outer_fold"]), int(old["inner_fold"]))
        new = nested_lookup[key]
        nested_comparisons[f"{key[0]}_{key[1]}"] = _compare_manifest(
            old, new, nested=True
        )

    reusable: list[dict[str, Any]] = []
    for old in v1_nested:
        cell_id = (
            "B_nested_threshold_calibration/"
            f"outer_{old['outer_fold']}_inner_{old['inner_fold']}"
        )
        record = v1_inventory["cells"][cell_id]
        if record.get("status") != "COMPLETE":
            raise RuntimeError(f"Expected completed V1 nested cell: {cell_id}")
        reusable.append(
            _verify_reusable_run(
                cell_id=cell_id,
                run_path=Path(record["run_path"]),
                run_record=record,
                old_protocol=old_protocol,
                expected_manifest=old,
                expected_split=old,
                mode="target_only",
            )
        )
    for fold in (1, 2):
        cell_id = (
            f"C_local_context/fold_{fold}/seed_{old_protocol['training']['seed']}"
        )
        record = v1_inventory["cells"][cell_id]
        if record.get("status") != "COMPLETE":
            raise RuntimeError(f"Expected completed V1 context cell: {cell_id}")
        reusable.append(
            _verify_reusable_run(
                cell_id=cell_id,
                run_path=Path(record["run_path"]),
                run_record=record,
                old_protocol=old_protocol,
                expected_manifest=outer_manifests[fold - 1],
                expected_split=None,
                mode="local_context",
            )
        )

    protocol, protocol_sha256 = _write_protocol_artifacts(
        old_protocol,
        old_protocol_sha256,
        frame,
        dataset_manifest,
        dataset_dir,
        outer_manifests,
        nested_splits,
    )

    compatibility = {
        "schema_version": 1,
        "v1_protocol": {
            "path": str(V1_PROTOCOL_PATH),
            "sha256": old_protocol_sha256,
            "source_snapshot_sha256": old_protocol["source_snapshot_sha256"],
        },
        "v2_protocol": {
            "path": str(V2_PROTOCOL_PATH),
            "sha256": protocol_sha256,
            "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        },
        "scientific_equivalence": {
            "effective_dataset_sha256_unchanged": True,
            "family_assignments_unchanged": True,
            "fitting_membership_unchanged": True,
            "scoring_membership_unchanged": True,
            "fitting_exclusions_and_reasons_unchanged": True,
            "valid_context_transformations_unchanged": True,
            "only_policy_difference": STALE_POLICY_CHANGE,
            "outer_fold_comparison": outer_comparisons,
            "nested_split_comparison": nested_comparisons,
        },
        "reusable_v1_runs": reusable,
        "reused_runs_remain_v1_provenance": True,
        "reused_runs_copied_or_modified": False,
    }
    write_json(V2_COMPATIBILITY_PATH, compatibility)

    specs = _make_specs(protocol, protocol_sha256, outer_manifests)
    remaining_cells = {
        spec["cell_id"]: {
            "status": "PENDING_SEPARATE_GPU_AUTHORIZATION",
            "run_path": spec["run_path"],
            "spec_path": spec["spec_path"],
            "spec_sha256": spec["spec_sha256"],
            "fold_manifest_sha256": spec["fold_manifest_sha256"],
        }
        for spec in specs
    }
    v1_budget = read_json(V1_ROOT / "time_budget.json")
    budget = {
        key: v1_budget[key]
        for key in (
            "authorization_window_start",
            "authorization_window_deadline",
            "budget_seconds",
            "planning_estimates",
            "planning_margin_seconds",
            "observed_baseline_cell_seconds",
        )
        if key in v1_budget
    }
    budget.update(
        {
            "study_id": "recall_improvement_study_v2",
            "execution_authorized": False,
            "actual": {
                "gpu_cells_started": 0,
                "gpu_cells_completed": 0,
                "gpu_cell_seconds": 0.0,
                "cell_ids": {},
            },
            "status": "PREPARED_NOT_AUTHORIZED",
        }
    )
    write_json(V2_BUDGET_PATH, budget)

    inventory = {
        "schema_version": 2,
        "study_id": "recall_improvement_study_v2",
        "protocol_path": str(V2_PROTOCOL_PATH),
        "protocol_sha256": protocol_sha256,
        "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        "compatibility_manifest_path": str(V2_COMPATIBILITY_PATH),
        "compatibility_manifest_sha256": sha256_file(V2_COMPATIBILITY_PATH),
        "overall_status": "PREPARED_NOT_AUTHORIZED",
        "actual": {
            "gpu_cells_started": 0,
            "gpu_cells_completed": 0,
            "gpu_cell_seconds": 0.0,
        },
        "dataset": protocol["dataset"],
        "reused_v1_runs": [
            {
                "cell_id": item["v1_cell_id"],
                "status": item["v2_status"],
                "v1_run_path": item["v1_run_path"],
                "v1_run_manifest_sha256": item["v1_run_manifest_sha256"],
                "executed_under_protocol": item["executed_under_protocol"],
            }
            for item in reusable
        ],
        "pending_v2_cells": remaining_cells,
        "failed_v1_cell_not_retried": (
            "C_local_context/fold_3/seed_2718; its V1 logs and failed inventory "
            "record are preserved. A V2 cell spec is prepared at a distinct path."
        ),
        "stages": {
            "B_nested_threshold_calibration": "REUSED_FROM_V1",
            "C_local_context": "3_PENDING_V2",
            "G_partial_4": "5_PENDING_V2",
            "D_focal_loss": "5_PENDING_V2",
            "G2_partial_2": "5_PENDING_V2",
            "E_error_analysis": "V1_OUTPUTS_PRESERVED_FOR_COMPATIBILITY_REVIEW",
        },
    }
    write_json(V2_INVENTORY_PATH, inventory)
    write_json(
        V2_RUNNER_CONFIG_PATH,
        {
            "study_id": "recall_improvement_study_v2",
            "protocol_path": str(V2_PROTOCOL_PATH),
            "protocol_sha256": protocol_sha256,
            "inventory_path": str(V2_INVENTORY_PATH),
            "budget_path": str(V2_BUDGET_PATH),
            "logs_root": str(V2_ROOT / "logs"),
            "outputs_root": str(V2_ROOT / "runs"),
            "launcher_module": (
                "experiments.transformer_final.recall_improvement_study_v2.cell_launcher"
            ),
            "pending_cell_count": len(specs),
            "gpu_execution_authorized": False,
            "manual_command_template": (
                "$env:RECALL_STUDY_V2_GPU_AUTHORIZED='1'; "
                "python -m experiments.transformer_final."
                "recall_improvement_study_v2.cell_launcher --spec <spec_path>"
            ),
            "manual_launch_is_blocked_until_explicit_authorization": True,
        },
    )
    for directory in ("logs", "runs", "audits"):
        (V2_ROOT / directory).mkdir(parents=True, exist_ok=True)
    return V2_PROTOCOL_PATH


if __name__ == "__main__":
    print(prepare_v2())
