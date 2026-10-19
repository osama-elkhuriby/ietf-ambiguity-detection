"""Validate the package or execute exactly one manually authorized V3 cell."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_BOOTSTRAP))

from models.transformer_final.common import (
    read_json,
    sha256_file,
    sha256_text,
    stable_json,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
V3_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent
V2_ROOT = V3_ROOT.parent / "recall_improvement_study_v2"
V3_PROTOCOL_PATH = (
    V3_ROOT / "protocols" / "recall_improvement_protocol_v3_optimizer_fix.json"
)
V3_INVENTORY_PATH = V3_ROOT / "EXPERIMENT_INVENTORY.json"
V2_INVENTORY_PATH = V2_ROOT / "EXPERIMENT_INVENTORY.json"
V2_PROTOCOL_PATH = V2_ROOT / "protocols" / "recall_improvement_protocol_v2.json"
PACKAGE_MANIFEST_PATH = PACKAGE_ROOT / "PACKAGE_MANIFEST.json"
MANUAL_CONFIG_PATH = PACKAGE_ROOT / "MANUAL_RUNNER_CONFIG.json"
EXPECTED_V3_PROTOCOL_SHA256 = (
    "9502e659c1f246f433ce534fa7165959a6519c6637c71b3be0289f6638e72c51"
)
EXPECTED_V3_SOURCE_SNAPSHOT_SHA256 = (
    "22af9f00506241a4d78918b3b1fc65f1a787281a685c966fc881df99671ee8c4"
)
REQUIRED_ENVIRONMENT_VARIABLE = "RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED"
PENDING = "PENDING_SEPARATE_GPU_AUTHORIZATION"

METHODS = {
    "G_partial_4": "partial_4",
    "D_focal_loss": "focal",
    "G2_partial_2": "partial_2",
}


def _legacy_v3_protocol() -> dict[str, Any]:
    from experiments.transformer_final.recall_improvement_study_v3_optimizer_fix import (
        _protocol,
    )

    protocol = _protocol()
    if (
        sha256_file(V3_PROTOCOL_PATH) != EXPECTED_V3_PROTOCOL_SHA256
        or protocol.get("source_snapshot_sha256")
        != EXPECTED_V3_SOURCE_SNAPSHOT_SHA256
    ):
        raise RuntimeError("Frozen V3 protocol compatibility check failed")
    return protocol


def _scientific_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        key: spec[key]
        for key in (
            "cell_id",
            "method",
            "train_fold",
            "seed",
            "config_path",
            "input_mode",
            "model_path",
            "fold_manifest_sha256",
        )
    }


def _parent_attempt(spec: dict[str, Any]) -> dict[str, Any]:
    value = spec.get("parent_v2_attempt") or spec.get("replaces_attempt")
    if not isinstance(value, dict):
        raise RuntimeError("V3 specification lacks its immutable V2 parent attempt")
    return value


def _load_package() -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(MANUAL_CONFIG_PATH)
    package = read_json(PACKAGE_MANIFEST_PATH)
    script_hashes = package.get("scripts", {})
    scripts_intact = bool(script_hashes) and all(
        (PACKAGE_ROOT / name).is_file()
        and sha256_file(PACKAGE_ROOT / name) == expected
        for name, expected in script_hashes.items()
    )
    if (
        config.get("manual_gpu_execution_authorized") is not True
        or config.get("one_cell_per_process") is not True
        or config.get("automatic_retry") is not False
        or config.get("automatic_multi_cell_execution") is not False
        or config.get("protocol_sha256") != EXPECTED_V3_PROTOCOL_SHA256
        or config.get("source_snapshot_sha256")
        != EXPECTED_V3_SOURCE_SNAPSHOT_SHA256
        or package.get("cell_count") != 15
        or package.get("execution_order") is None
        or package.get("manual_config_sha256") != sha256_file(MANUAL_CONFIG_PATH)
        or not scripts_intact
        or package.get("one_cell_per_process") is not True
        or package.get("automatic_retry") is not False
        or package.get("automatic_multi_cell_execution") is not False
    ):
        raise RuntimeError("Manual execution package integrity check failed")
    return config, package


def _inventory_record_for_spec(
    inventory: dict[str, Any], spec_path: Path
) -> tuple[str, dict[str, Any]]:
    matches = [
        (cell_id, record)
        for cell_id, record in inventory.get("attempts", {}).items()
        if Path(record.get("spec_path", "")).resolve() == spec_path.resolve()
    ]
    if len(matches) != 1:
        raise RuntimeError("Specification is not uniquely registered in V3 inventory")
    return matches[0]


def _validate_static_attempt(
    spec_path: Path,
    *,
    protocol: dict[str, Any],
    inventory: dict[str, Any],
    package: dict[str, Any],
    verify_dataset: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec_path = spec_path.resolve()
    cell_id, record = _inventory_record_for_spec(inventory, spec_path)
    spec = read_json(spec_path)
    if (
        cell_id != spec.get("cell_id")
        or record.get("attempt_id") != spec.get("attempt_id")
        or sha256_file(spec_path) != record.get("spec_sha256")
        or package["specification_sha256"].get(cell_id) != record.get("spec_sha256")
        or spec.get("study_id") != protocol["study_id"]
        or spec.get("protocol_sha256") != EXPECTED_V3_PROTOCOL_SHA256
        or spec.get("parent_protocol_sha256")
        != protocol["parent_protocol"]["sha256"]
        or int(spec.get("seed", -1)) != int(protocol["training"]["seed"])
        or spec.get("input_mode") != "target_only"
    ):
        raise RuntimeError(f"V3 manual specification identity mismatch: {cell_id}")
    stage = cell_id.split("/", 1)[0]
    if (
        METHODS.get(stage) != spec.get("method")
        or int(spec.get("train_fold", 0)) not in range(1, 6)
    ):
        raise RuntimeError(f"Unexpected manual experiment cell: {cell_id}")

    auth_path = Path(record["authorization_path"]).resolve()
    authorization = read_json(auth_path)
    if (
        sha256_file(auth_path) != record.get("authorization_sha256")
        or package["authorization_sha256"].get(cell_id)
        != record.get("authorization_sha256")
        or authorization.get("execution_authorized") is not True
        or authorization.get("one_cell_only") is not True
        or authorization.get("automatic_retry") is not False
        or authorization.get("cell_id") != cell_id
        or authorization.get("attempt_id") != spec["attempt_id"]
        or Path(authorization.get("spec_path", "")).resolve() != spec_path
        or authorization.get("spec_sha256") != record["spec_sha256"]
        or authorization.get("required_environment_variable")
        != REQUIRED_ENVIRONMENT_VARIABLE
        or authorization.get("required_environment_value") != "1"
    ):
        raise PermissionError(f"File authorization gate failed: {cell_id}")

    parent = _parent_attempt(spec)
    parent_path = Path(parent["spec_path"]).resolve()
    parent_spec = read_json(parent_path)
    v2_inventory = read_json(V2_INVENTORY_PATH)
    parent_record = v2_inventory["pending_v2_cells"].get(cell_id)
    if (
        parent_record is None
        or Path(parent_record["spec_path"]).resolve() != parent_path
        or sha256_file(parent_path) != parent["spec_sha256"]
        or parent_record.get("spec_sha256") != parent["spec_sha256"]
        or parent_record.get("status") != parent.get("status")
        or _scientific_spec(spec) != _scientific_spec(parent_spec)
        or sha256_text(stable_json(_scientific_spec(spec)))
        != spec["scientific_spec_sha256"]
    ):
        raise RuntimeError(f"V2 parent or frozen scientific spec changed: {cell_id}")
    if cell_id == "G_partial_4/fold_1/seed_2718":
        failed_log = Path(parent["stderr_log_path"])
        if (
            parent.get("status") != "FAILED_NO_AUTOMATIC_RETRY"
            or not failed_log.is_file()
            or sha256_file(failed_log) != parent.get("stderr_sha256")
        ):
            raise RuntimeError("The original failed Partial-4 Fold 1 attempt changed")

    if verify_dataset:
        parent_protocol = read_json(V2_PROTOCOL_PATH)
        outer_path = Path(
            parent_protocol["dataset"]["study_outer_fold_manifests_path"]
        )
        if (
            sha256_file(outer_path)
            != parent_protocol["dataset"]["study_outer_fold_manifests_sha256"]
        ):
            raise RuntimeError("Frozen V2 outer-fold manifest digest changed")
        manifests = json.loads(outer_path.read_text(encoding="utf-8"))
        matches = [
            item
            for item in manifests
            if int(item["fold"]) == int(spec["train_fold"])
        ]
        if (
            len(matches) != 1
            or sha256_text(stable_json(matches[0]))
            != spec["fold_manifest_sha256"]
        ):
            raise RuntimeError(f"Frozen outer fold changed: {cell_id}")
        verify_spec = dict(spec)
        verify_spec["fold_manifest"] = matches[0]
        from experiments.transformer_final.recall_improvement_study_v2_resume import (
            _verify_dataset_and_fold,
        )

        _verify_dataset_and_fold(verify_spec, parent_protocol)
    return spec, record


def _output_paths(spec: dict[str, Any], record: dict[str, Any]) -> tuple[Path, ...]:
    log_base = Path(spec["log_path"])
    return (
        Path(spec["run_path"]),
        Path(spec["audit_path"]),
        log_base.with_suffix(".stdout.log"),
        log_base.with_suffix(".stderr.log"),
        Path(record["start_marker_path"]),
        Path(record["completion_manifest_path"]),
        Path(record["failure_manifest_path"]),
    )


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _save_inventory(inventory: dict[str, Any]) -> None:
    write_json(V3_INVENTORY_PATH, inventory)


def _validate_completed_output(
    spec: dict[str, Any], record: dict[str, Any], protocol: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = Path(spec["run_path"]).resolve()
    audit_path = Path(spec["audit_path"]).resolve()
    manifest_path = output / "run_manifest.json"
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    parent_protocol = read_json(V2_PROTOCOL_PATH)
    method = spec["method"]
    expected_full = method == "focal"
    if (
        manifest.get("condition") != "A2"
        or int(manifest.get("fold", 0)) != int(spec["train_fold"])
        or int(manifest.get("seed", 0)) != 2718
        or manifest.get("precision") != "fp32"
        or manifest.get("input_mode") != "target_only"
        or manifest.get("epochs_completed") != 5
        or manifest.get("full_encoder_fine_tuned") is not expected_full
        or manifest.get("study_id") != protocol["study_id"]
        or manifest.get("study_protocol_sha256") != EXPECTED_V3_PROTOCOL_SHA256
        or manifest.get("study_source_snapshot_sha256")
        != EXPECTED_V3_SOURCE_SNAPSHOT_SHA256
        or manifest.get("study_cell_id") != spec["cell_id"]
        or manifest.get("study_attempt_id") != spec["attempt_id"]
        or manifest.get("study_fold_manifest_sha256")
        != spec["fold_manifest_sha256"]
        or manifest.get("config_sha256") != parent_protocol["config_sha256"]
        or manifest.get("effective_dataset_sha256")
        != parent_protocol["dataset"]["effective_dataset_sha256"]
        or manifest.get("dataset_manifest_sha256")
        != parent_protocol["dataset"]["dataset_manifest_sha256"]
        or manifest.get("model_source_sha256")
        != parent_protocol["model"]["source_sha256"]
        or manifest.get("model_file_hashes")
        != parent_protocol["model"]["model_file_hashes"]
        or manifest.get("source_file_hashes")
        != parent_protocol["source_snapshot"]["core_files"]
    ):
        raise RuntimeError(f"Completed V3 run provenance failed: {spec['cell_id']}")
    if method == "focal":
        if (
            manifest.get("loss") != "focal_gamma_2"
            or manifest.get("focal_gamma") != 2.0
            or audit.get("optimizer_parameter_count") != 202
            or audit.get("optimizer_parameter_elements") != 184423682
        ):
            raise RuntimeError("Focal-loss output validation failed")
    else:
        layers = 4 if method == "partial_4" else 2
        expected_tensors = 68 if layers == 4 else 36
        expected_elements = 28943618 if layers == 4 else 14767874
        expected_frozen_tensors = 134 if layers == 4 else 166
        expected_groups = (
            {
                "task_head_trainable": 592130,
                "encoder_layer_trainable": 28351488,
                "encoder_layer_frozen": 56702976,
                "shared_backbone_frozen": 98777088,
            }
            if layers == 4
            else {
                "task_head_trainable": 592130,
                "encoder_layer_trainable": 14175744,
                "encoder_layer_frozen": 70878720,
                "shared_backbone_frozen": 98777088,
            }
        )
        if (
            manifest.get("fine_tuning_policy") != method
            or manifest.get("trainable_last_layers") != layers
            or manifest.get("optimizer_parameter_count") != expected_tensors
            or manifest.get("optimizer_parameter_elements") != expected_elements
            or audit.get("last_trainable_layers") != layers
            or audit.get("optimizer_parameter_count") != expected_tensors
            or audit.get("optimizer_parameter_elements") != expected_elements
            or audit.get("frozen_parameter_tensors") != expected_frozen_tensors
            or audit.get("parameter_counts_by_policy_group") != expected_groups
            or audit.get("frozen_parameters_unchanged") is not True
            or audit.get("frozen_parameters_have_no_gradients") is not True
            or audit.get("trainable_parameters_finite") is not True
        ):
            raise RuntimeError(f"Partial-freeze output validation failed: {method}")
    for artifact in manifest.get("artifact_hashes", {}).values():
        artifact_path = (output / artifact["path"]).resolve()
        if (
            output not in artifact_path.parents
            or not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError("Completed run artifact hash validation failed")
    if audit.get("study_attempt_id") != spec["attempt_id"]:
        raise RuntimeError("Treatment audit lacks manual attempt provenance")
    return manifest, audit


def validate_package() -> dict[str, Any]:
    protocol = _legacy_v3_protocol()
    _, package = _load_package()
    inventory = read_json(V3_INVENTORY_PATH)
    if set(package["execution_order"]) != set(inventory.get("attempts", {})):
        raise RuntimeError("Manual package and V3 inventory cell sets differ")
    statuses: dict[str, str] = {}
    verified_folds: set[int] = set()
    for cell_id in package["execution_order"]:
        record = inventory["attempts"][cell_id]
        spec_path = Path(record["spec_path"])
        fold = int(cell_id.split("/fold_", 1)[1].split("/", 1)[0])
        spec, record = _validate_static_attempt(
            spec_path,
            protocol=protocol,
            inventory=inventory,
            package=package,
            verify_dataset=fold not in verified_folds,
        )
        verified_folds.add(fold)
        status = record.get("status")
        paths = _output_paths(spec, record)
        if status == PENDING and any(path.exists() for path in paths):
            raise RuntimeError(f"Pending attempt already has outputs: {cell_id}")
        if status == "COMPLETE":
            manifest, audit = _validate_completed_output(spec, record, protocol)
            completion_path = Path(record["completion_manifest_path"])
            completion = read_json(completion_path)
            if (
                completion.get("run_manifest_sha256")
                != sha256_file(Path(spec["run_path"]) / "run_manifest.json")
                or completion.get("audit_sha256") != sha256_file(Path(spec["audit_path"]))
                or record.get("run_manifest_sha256")
                != completion["run_manifest_sha256"]
                or record.get("audit_sha256") != completion["audit_sha256"]
                or manifest.get("study_attempt_id") != completion.get("attempt_id")
                or audit.get("study_attempt_id") != completion.get("attempt_id")
            ):
                raise RuntimeError(f"Completion evidence changed: {cell_id}")
        if status == "FAILED_NO_AUTOMATIC_RETRY" and not Path(
            record["failure_manifest_path"]
        ).is_file():
            raise RuntimeError(f"Failed attempt lacks failure manifest: {cell_id}")
        if status not in {PENDING, "IN_PROGRESS", "COMPLETE", "FAILED_NO_AUTOMATIC_RETRY"}:
            raise RuntimeError(f"Unknown manual attempt status: {cell_id}: {status}")
        statuses[cell_id] = str(status)
    return {
        "protocol_sha256": sha256_file(V3_PROTOCOL_PATH),
        "source_snapshot_sha256": protocol["source_snapshot_sha256"],
        "cells": len(statuses),
        "statuses": statuses,
        "cuda_used": False,
    }


def _execute_one(
    spec: dict[str, Any],
    record: dict[str, Any],
    inventory: dict[str, Any],
    protocol: dict[str, Any],
) -> Path:
    if os.environ.get(REQUIRED_ENVIRONMENT_VARIABLE) != "1":
        raise PermissionError(
            f"Set {REQUIRED_ENVIRONMENT_VARIABLE}=1 only for the one selected cell"
        )
    expected_python = Path(read_json(V2_PROTOCOL_PATH)["environment"]["python_executable"])
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(
            "Manual training must use the frozen ietf-ambiguity-detection "
            f"Conda interpreter: {expected_python}"
        )
    if record.get("status") != PENDING:
        raise RuntimeError(
            f"Attempt is not pending; retries are forbidden: {spec['cell_id']}"
        )
    paths = _output_paths(spec, record)
    log_base = Path(spec["log_path"])
    log_paths = {
        log_base.with_suffix(".stdout.log").resolve(),
        log_base.with_suffix(".stderr.log").resolve(),
    }
    if any(path.exists() for path in paths if path.resolve() not in log_paths):
        raise FileExistsError(
            f"Attempt output already exists; retries are forbidden: {spec['cell_id']}"
        )
    for path in paths:
        if V3_ROOT.resolve() not in path.resolve().parents:
            raise RuntimeError("Manual attempt output escaped the V3 namespace")

    started_at = datetime.now().astimezone().isoformat()
    marker = Path(record["start_marker_path"])
    _write_exclusive(
        marker,
        {
            "cell_id": spec["cell_id"],
            "attempt_id": spec["attempt_id"],
            "started_at": started_at,
            "spec_sha256": record["spec_sha256"],
            "authorization_sha256": record["authorization_sha256"],
            "automatic_retry": False,
        },
    )
    record.update({"status": "IN_PROGRESS", "started_at": started_at})
    _save_inventory(inventory)
    started = time.perf_counter()
    try:
        from experiments.transformer_final.recall_improvement_study_v2_resume import (
            _verify_cuda_and_gpu,
        )

        gpu_preflight = _verify_cuda_and_gpu()
        from models.transformer_final import train
        from experiments.transformer_final.recall_improvement_study_v1.cell_launcher import (
            execute_cell as execute_v1_cell,
        )

        original_write_json = train.write_json

        def write_with_manual_v3_provenance(
            path: Path, value: dict[str, Any]
        ) -> None:
            updated = value
            if path.name == "run_manifest.json":
                updated = json.loads(json.dumps(value))
                updated.update(
                    {
                        "study_id": protocol["study_id"],
                        "study_protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
                        "study_source_snapshot_sha256": (
                            EXPECTED_V3_SOURCE_SNAPSHOT_SHA256
                        ),
                        "study_cell_id": spec["cell_id"],
                        "study_attempt_id": spec["attempt_id"],
                        "study_fold_manifest_sha256": spec[
                            "fold_manifest_sha256"
                        ],
                        "parent_v2_protocol_sha256": protocol[
                            "parent_protocol"
                        ]["sha256"],
                        "study_execution_provenance": {
                            "execution_profile": "manual_execution_v1",
                            "manual_launcher_sha256": sha256_file(Path(__file__)),
                            "corrected_v1_cell_launcher_sha256": sha256_file(
                                V3_ROOT.parent
                                / "recall_improvement_study_v1"
                                / "cell_launcher.py"
                            ),
                            "authorization_sha256": record[
                                "authorization_sha256"
                            ],
                            "one_cell_per_process": True,
                            "automatic_retry": False,
                            "new_run_executed_under_v3_optimizer_fix": True,
                        },
                    }
                )
            original_write_json(path, updated)

        train.write_json = write_with_manual_v3_provenance
        try:
            output = execute_v1_cell(Path(record["spec_path"]).resolve())
        finally:
            train.write_json = original_write_json

        audit_path = Path(spec["audit_path"])
        audit = read_json(audit_path)
        audit.update(
            {
                "study_id": protocol["study_id"],
                "study_protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
                "study_source_snapshot_sha256": EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
                "study_cell_id": spec["cell_id"],
                "study_attempt_id": spec["attempt_id"],
                "gpu_preflight": gpu_preflight,
                "execution_profile": "manual_execution_v1",
            }
        )
        write_json(audit_path, audit)
        manifest, _ = _validate_completed_output(spec, record, protocol)
        elapsed = time.perf_counter() - started
        run_manifest_hash = sha256_file(output / "run_manifest.json")
        audit_hash = sha256_file(audit_path)
        completed_at = datetime.now().astimezone().isoformat()
        completion = {
            "schema_version": 1,
            "status": "COMPLETE",
            "cell_id": spec["cell_id"],
            "attempt_id": spec["attempt_id"],
            "started_at": started_at,
            "completed_at": completed_at,
            "elapsed_seconds": elapsed,
            "run_path": str(output),
            "run_manifest_sha256": run_manifest_hash,
            "audit_path": str(audit_path),
            "audit_sha256": audit_hash,
            "spec_sha256": record["spec_sha256"],
            "authorization_sha256": record["authorization_sha256"],
            "protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
            "source_snapshot_sha256": EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
            "gpu_preflight": gpu_preflight,
            "epochs_completed": manifest["epochs_completed"],
        }
        _write_exclusive(Path(record["completion_manifest_path"]), completion)
        record.update(
            {
                "status": "COMPLETE",
                "completed_at": completed_at,
                "elapsed_seconds": elapsed,
                "run_path": str(output),
                "run_manifest_sha256": run_manifest_hash,
                "audit_path": str(audit_path),
                "audit_sha256": audit_hash,
                "gpu_preflight": gpu_preflight,
            }
        )
        _save_inventory(inventory)
        return output
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        failed_at = datetime.now().astimezone().isoformat()
        record.update(
            {
                "status": "FAILED_NO_AUTOMATIC_RETRY",
                "failed_at": failed_at,
                "elapsed_seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _save_inventory(inventory)
        failure_path = Path(record["failure_manifest_path"])
        if not failure_path.exists():
            _write_exclusive(
                failure_path,
                {
                    "schema_version": 1,
                    "status": "FAILED_NO_AUTOMATIC_RETRY",
                    "cell_id": spec["cell_id"],
                    "attempt_id": spec["attempt_id"],
                    "started_at": started_at,
                    "failed_at": failed_at,
                    "elapsed_seconds": elapsed,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "run_path": spec["run_path"],
                    "audit_path": spec["audit_path"],
                    "stdout_log": str(Path(spec["log_path"]).with_suffix(".stdout.log")),
                    "stderr_log": str(Path(spec["log_path"]).with_suffix(".stderr.log")),
                    "spec_sha256": record["spec_sha256"],
                    "authorization_sha256": record["authorization_sha256"],
                    "automatic_retry": False,
                },
            )
        raise


def run_spec(spec_path: Path) -> Path:
    protocol = _legacy_v3_protocol()
    _, package = _load_package()
    inventory = read_json(V3_INVENTORY_PATH)
    spec, record = _validate_static_attempt(
        spec_path,
        protocol=protocol,
        inventory=inventory,
        package=package,
        verify_dataset=True,
    )
    if os.environ.get(REQUIRED_ENVIRONMENT_VARIABLE) != "1":
        raise PermissionError(
            f"Set {REQUIRED_ENVIRONMENT_VARIABLE}=1 only for the one selected cell"
        )
    if record.get("status") != PENDING:
        raise RuntimeError(
            f"Attempt is not pending; retries are forbidden: {spec['cell_id']}"
        )
    if any(path.exists() for path in _output_paths(spec, record)):
        raise FileExistsError(
            f"Attempt output already exists; retries are forbidden: {spec['cell_id']}"
        )
    log_base = Path(spec["log_path"])
    stdout_path = log_base.with_suffix(".stdout.log")
    stderr_path = log_base.with_suffix(".stderr.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("x", encoding="utf-8") as stdout_handle,
        stderr_path.open("x", encoding="utf-8") as stderr_handle,
    ):
        try:
            with (
                contextlib.redirect_stdout(stdout_handle),
                contextlib.redirect_stderr(stderr_handle),
            ):
                output = _execute_one(spec, record, inventory, protocol)
        except BaseException:
            traceback.print_exc(file=stderr_handle)
            stderr_handle.flush()
            raise
    print(f"Completed exactly one V3 cell: {output}", flush=True)
    print(f"Cell logs: {stdout_path} ; {stderr_path}", flush=True)
    print(
        f"Completion manifest: {record['completion_manifest_path']}", flush=True
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", type=Path, help="Execute this one immutable cell")
    group.add_argument(
        "--validate-only",
        action="store_true",
        help="Run CPU-only package, protocol, dataset, and artifact checks",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.validate_only:
        print(json.dumps(validate_package(), indent=2, sort_keys=True))
        return 0
    run_spec(args.spec.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
