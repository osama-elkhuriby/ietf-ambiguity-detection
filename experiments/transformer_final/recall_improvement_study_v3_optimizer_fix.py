"""Run one separately authorized optimizer-fix study attempt."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.transformer_final.recall_improvement_study_v2.prepare_v2 import (
    V2_PROTOCOL_PATH,
    _study_source_hashes,
)
from experiments.transformer_final.recall_improvement_study_v2_resume import (
    _verify_cuda_and_gpu,
    _verify_dataset_and_fold,
)
from models.transformer_final.common import (
    read_json,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = REPO_ROOT / "experiments" / "transformer_final" / "recall_improvement_study_v3_optimizer_fix"
V3_PROTOCOL_PATH = V3_ROOT / "protocols" / "recall_improvement_protocol_v3_optimizer_fix.json"
V3_CONFIG_PATH = V3_ROOT / "V3_RUNNER_CONFIG.json"
V3_BUDGET_PATH = V3_ROOT / "time_budget.json"
V3_INVENTORY_PATH = V3_ROOT / "EXPERIMENT_INVENTORY.json"
V1_LAUNCHER_PATH = (
    REPO_ROOT
    / "experiments"
    / "transformer_final"
    / "recall_improvement_study_v1"
    / "cell_launcher.py"
)
FAILED_V2_CELL_ID = "G_partial_4/fold_1/seed_2718"
MINIMUM_CELL_SECONDS = 300


def _v3_config() -> dict[str, Any]:
    return read_json(V3_CONFIG_PATH)


def _v3_budget() -> dict[str, Any]:
    return read_json(V3_BUDGET_PATH)


def _protocol() -> dict[str, Any]:
    protocol = read_json(V3_PROTOCOL_PATH)
    config = _v3_config()
    if (
        protocol.get("study_id") != "recall_improvement_study_v3_optimizer_fix"
        or sha256_file(V3_PROTOCOL_PATH) != config.get("protocol_sha256")
        or sha256_file(V2_PROTOCOL_PATH) != protocol["parent_protocol"]["sha256"]
    ):
        raise RuntimeError("V3 or parent V2 protocol identity/digest mismatch")

    parent = read_json(V2_PROTOCOL_PATH)
    snapshot = protocol["source_snapshot"]
    parent_snapshot = parent["source_snapshot"]
    if (
        source_file_hashes() != parent_snapshot["core_files"]
        or _study_source_hashes() != parent_snapshot["study_files"]
        or sha256_text(stable_json(parent_snapshot))
        != parent["source_snapshot_sha256"]
    ):
        raise RuntimeError("Frozen V2 parent source snapshot mismatch")
    expected_snapshot = {
        "parent_v2_protocol_sha256": sha256_file(V2_PROTOCOL_PATH),
        "parent_v2_source_snapshot_sha256": parent["source_snapshot_sha256"],
        "v1_cell_launcher_sha256": sha256_file(V1_LAUNCHER_PATH),
        "v1_study_protocol_sha256": parent_snapshot["compatibility_dependencies"][
            "v1_study_protocol.py"
        ],
        "v1_study_runner_sha256": parent_snapshot["compatibility_dependencies"][
            "v1_study_runner.py"
        ],
        "v3_runner_sha256": sha256_file(Path(__file__)),
    }
    if (
        snapshot != expected_snapshot
        or sha256_text(stable_json(snapshot))
        != protocol["source_snapshot_sha256"]
    ):
        raise RuntimeError("V3 optimizer-fix source snapshot mismatch")

    scientific_settings = {
        key: parent[key]
        for key in (
            "baseline_protocol_path",
            "baseline_protocol_sha256",
            "config_path",
            "config_sha256",
            "dataset",
            "inner_cv",
            "interventions",
            "model",
            "policy_correction",
            "training",
        )
    }
    if (
        sha256_text(stable_json(scientific_settings))
        != protocol["scientific_settings_sha256"]
    ):
        raise RuntimeError("V3 scientific settings differ from frozen V2")
    return protocol


def _check_authorization() -> dict[str, Any]:
    if os.environ.get("RECALL_STUDY_V3_GPU_AUTHORIZED") != "1":
        raise PermissionError(
            "V3 GPU execution requires separate authorization via "
            "RECALL_STUDY_V3_GPU_AUTHORIZED=1"
        )
    if _v3_config().get("gpu_execution_authorized") is not True:
        raise PermissionError("V3 GPU authorization gate is disabled")
    budget = _v3_budget()
    if budget.get("execution_authorized") is not True:
        raise PermissionError("V3 execution-budget gate is disabled")
    budget_seconds = budget.get("budget_seconds")
    window_start = budget.get("authorization_window_start")
    deadline_value = budget.get("authorization_window_deadline")
    if (
        not isinstance(budget_seconds, int)
        or budget_seconds <= 0
        or not budget.get("authorization_session_id")
        or not window_start
        or not deadline_value
    ):
        raise PermissionError("V3 execution window has not been authorized")
    started = datetime.fromisoformat(window_start)
    deadline = datetime.fromisoformat(deadline_value)
    if (
        started.tzinfo is None
        or deadline.tzinfo is None
        or abs((deadline - started).total_seconds() - budget_seconds) > 1
    ):
        raise RuntimeError("V3 authorization window is malformed")
    if datetime.now(deadline.tzinfo) >= deadline:
        raise TimeoutError("The authorized V3 execution window has expired")
    remaining = (deadline - datetime.now(deadline.tzinfo)).total_seconds()
    if remaining < MINIMUM_CELL_SECONDS:
        raise TimeoutError("Insufficient V3 execution time remains for one cell")
    return budget


def _load_cell_spec(spec_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _protocol()
    config = _v3_config()
    inventory = read_json(V3_INVENTORY_PATH)
    cell = inventory["attempts"].get(FAILED_V2_CELL_ID)
    if (
        Path(config["protocol_path"]).resolve() != V3_PROTOCOL_PATH.resolve()
        or Path(config["inventory_path"]).resolve() != V3_INVENTORY_PATH.resolve()
        or inventory.get("study_id") != protocol["study_id"]
        or inventory.get("protocol_sha256") != sha256_file(V3_PROTOCOL_PATH)
        or cell is None
        or cell.get("status") != "PENDING_SEPARATE_GPU_AUTHORIZATION"
        or Path(cell["spec_path"]).resolve() != spec_path.resolve()
        or sha256_file(spec_path) != cell["spec_sha256"]
    ):
        raise RuntimeError(
            "V3 attempt is not pending or its specification digest differs"
        )
    spec = read_json(spec_path)
    if (
        spec.get("study_id") != protocol["study_id"]
        or spec.get("protocol_sha256") != sha256_file(V3_PROTOCOL_PATH)
        or spec.get("parent_protocol_sha256")
        != protocol["parent_protocol"]["sha256"]
    ):
        raise RuntimeError("V3 cell specification protocol identity mismatch")
    return spec, inventory


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


def _verify_attempt(spec: dict[str, Any], protocol: dict[str, Any]) -> None:
    from experiments.transformer_final.recall_improvement_study_v2.prepare_v2 import (
        V2_INVENTORY_PATH,
    )

    parent = protocol["parent_protocol"]
    failed_spec_path = Path(spec["replaces_attempt"]["spec_path"])
    failed_spec = read_json(failed_spec_path)
    v2_inventory = read_json(V2_INVENTORY_PATH)
    failed_record = v2_inventory["pending_v2_cells"][FAILED_V2_CELL_ID]
    failed_log = Path(spec["replaces_attempt"]["stderr_log_path"])
    if (
        failed_record.get("status") != "FAILED_NO_AUTOMATIC_RETRY"
        or sha256_file(failed_spec_path)
        != spec["replaces_attempt"]["spec_sha256"]
        or not failed_log.is_file()
        or sha256_file(failed_log) != spec["replaces_attempt"]["stderr_sha256"]
        or failed_spec.get("study_id") != "recall_improvement_study_v2"
        or failed_spec.get("protocol_sha256") != parent["sha256"]
    ):
        raise RuntimeError("Original failed V2 attempt or its log changed")

    if (
        _scientific_spec(spec) != _scientific_spec(failed_spec)
        or sha256_text(stable_json(_scientific_spec(spec)))
        != spec["scientific_spec_sha256"]
    ):
        raise RuntimeError("V3 attempt changed the frozen scientific cell spec")
    if (
        spec["method"] != "partial_4"
        or spec["input_mode"] != "target_only"
        or int(spec["train_fold"]) != 1
        or int(spec["seed"]) != int(protocol["training"]["seed"])
    ):
        raise RuntimeError("V3 replacement is not the frozen Partial-4 Fold 1 cell")

    parent_protocol = read_json(V2_PROTOCOL_PATH)
    outer_path = Path(parent_protocol["dataset"]["study_outer_fold_manifests_path"])
    outer_manifests = json.loads(outer_path.read_text(encoding="utf-8"))
    matching = [
        value for value in outer_manifests
        if int(value["fold"]) == int(spec["train_fold"])
    ]
    if (
        sha256_file(outer_path)
        != parent_protocol["dataset"]["study_outer_fold_manifests_sha256"]
        or len(matching) != 1
        or sha256_text(stable_json(matching[0])) != spec["fold_manifest_sha256"]
    ):
        raise RuntimeError("V3 cell fold manifest no longer matches V2")

    verify_spec = dict(spec)
    verify_spec["fold_manifest"] = matching[0]
    _verify_dataset_and_fold(verify_spec, parent_protocol)

    run_path = Path(spec["run_path"]).resolve()
    audit_path = Path(spec["audit_path"]).resolve()
    log_base = Path(spec["log_path"]).resolve()
    output_paths = (
        run_path,
        audit_path,
        log_base.with_suffix(".stdout.log"),
        log_base.with_suffix(".stderr.log"),
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError("V3 attempt output paths already exist; retry is forbidden")
    if any(V3_ROOT.resolve() not in path.parents for path in output_paths):
        raise RuntimeError("V3 attempt outputs must remain in their separate namespace")


def _write_inventory(inventory: dict[str, Any]) -> None:
    write_json(V3_INVENTORY_PATH, inventory)


def _record_budget(
    budget: dict[str, Any], elapsed: float, *, completed: bool
) -> None:
    actual = budget.setdefault(
        "actual",
        {
            "cell_ids": {},
            "gpu_cell_seconds": 0.0,
            "gpu_cells_started": 0,
            "gpu_cells_completed": 0,
        },
    )
    actual["gpu_cell_seconds"] = float(actual.get("gpu_cell_seconds", 0.0)) + elapsed
    actual["gpu_cells_started"] = int(actual.get("gpu_cells_started", 0)) + 1
    actual["cell_ids"][FAILED_V2_CELL_ID] = {
        "completed": completed,
        "elapsed_seconds": elapsed,
    }
    if completed:
        actual["gpu_cells_completed"] = int(
            actual.get("gpu_cells_completed", 0)
        ) + 1


def _validate_completed_output(
    output: Path, audit_path: Path, spec: dict[str, Any], protocol: dict[str, Any]
) -> tuple[str, str]:
    manifest_path = output / "run_manifest.json"
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    if (
        manifest.get("condition") != "A2"
        or manifest.get("fold") != 1
        or manifest.get("seed") != protocol["training"]["seed"]
        or manifest.get("precision") != "fp32"
        or manifest.get("input_mode") != "target_only"
        or manifest.get("epochs_completed") != 5
        or manifest.get("fine_tuning_policy") != "partial_4"
        or manifest.get("trainable_last_layers") != 4
        or manifest.get("study_id") != protocol["study_id"]
        or manifest.get("study_protocol_sha256") != sha256_file(V3_PROTOCOL_PATH)
        or manifest.get("study_source_snapshot_sha256")
        != protocol["source_snapshot_sha256"]
        or manifest.get("study_cell_id") != spec["cell_id"]
        or manifest.get("study_attempt_id") != spec["attempt_id"]
        or manifest.get("full_encoder_fine_tuned") is not False
        or audit.get("last_trainable_layers") != 4
        or audit.get("frozen_parameters_unchanged") is not True
        or audit.get("frozen_parameters_have_no_gradients") is not True
        or audit.get("trainable_parameters_finite") is not True
        or audit.get("optimizer_parameter_count") != 68
        or audit.get("optimizer_parameter_elements") != 28943618
        or audit.get("frozen_parameter_tensors") != 134
        or audit.get("parameter_counts_by_policy_group")
        != {
            "task_head_trainable": 592130,
            "encoder_layer_trainable": 28351488,
            "encoder_layer_frozen": 56702976,
            "shared_backbone_frozen": 98777088,
        }
    ):
        raise RuntimeError("V3 Partial-4 output or freeze audit validation failed")
    for artifact in manifest.get("artifact_hashes", {}).values():
        artifact_path = (output / artifact["path"]).resolve()
        if output.resolve() not in artifact_path.parents or (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            raise RuntimeError("V3 run artifact hash validation failed")
    return sha256_file(manifest_path), sha256_file(audit_path)


def execute_attempt(spec_path: Path) -> Path:
    budget = _check_authorization()
    protocol = _protocol()
    spec, inventory = _load_cell_spec(spec_path.resolve())
    _verify_attempt(spec, protocol)
    gpu_preflight = _verify_cuda_and_gpu()
    started = time.perf_counter()
    cell = inventory["attempts"][FAILED_V2_CELL_ID]
    cell.update(
        {
            "status": "IN_PROGRESS",
            "started_at": datetime.now().astimezone().isoformat(),
            "gpu_preflight": gpu_preflight,
            "authorization_session_id": budget["authorization_session_id"],
        }
    )
    _write_inventory(inventory)

    run_path = Path(spec["run_path"])
    audit_path = Path(spec["audit_path"])
    log_base = Path(spec["log_path"])
    stdout_path = log_base.with_suffix(".stdout.log")
    stderr_path = log_base.with_suffix(".stderr.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    from models.transformer_final import train
    from experiments.transformer_final.recall_improvement_study_v1.cell_launcher import (
        execute_cell as execute_v1_cell,
    )

    original_write_json = train.write_json

    def write_v3_provenance(path: Path, value: dict[str, Any]) -> None:
        updated = value
        if path.name == "run_manifest.json":
            updated = json.loads(json.dumps(value))
            updated.update(
                {
                    "study_id": protocol["study_id"],
                    "study_protocol_sha256": sha256_file(V3_PROTOCOL_PATH),
                    "study_source_snapshot_sha256": protocol[
                        "source_snapshot_sha256"
                    ],
                    "study_cell_id": spec["cell_id"],
                    "study_attempt_id": spec["attempt_id"],
                    "parent_v2_protocol_sha256": protocol["parent_protocol"][
                        "sha256"
                    ],
                    "study_execution_provenance": {
                        "v3_runner_sha256": sha256_file(Path(__file__)),
                        "corrected_v1_cell_launcher_sha256": sha256_file(
                            V1_LAUNCHER_PATH
                        ),
                        "new_run_executed_under_v3_optimizer_fix": True,
                    },
                }
            )
        original_write_json(path, updated)

    train.write_json = write_v3_provenance
    try:
        with (
            stdout_path.open("x", encoding="utf-8") as stdout_handle,
            stderr_path.open("x", encoding="utf-8") as stderr_handle,
            contextlib.redirect_stdout(stdout_handle),
            contextlib.redirect_stderr(stderr_handle),
        ):
            output = execute_v1_cell(spec_path.resolve())
        manifest_hash, audit_hash = _validate_completed_output(
            output, audit_path, spec, protocol
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        with stderr_path.open("a", encoding="utf-8") as stderr_handle:
            stderr_handle.write(traceback.format_exc())
        cell.update(
            {
                "status": "FAILED_NO_AUTOMATIC_RETRY",
                "completed_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": elapsed,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _record_budget(budget, elapsed, completed=False)
        _write_inventory(inventory)
        write_json(V3_BUDGET_PATH, budget)
        raise
    finally:
        train.write_json = original_write_json

    elapsed = time.perf_counter() - started
    cell.update(
        {
            "status": "COMPLETE",
            "completed_at": datetime.now().astimezone().isoformat(),
            "elapsed_seconds": elapsed,
            "run_manifest_sha256": manifest_hash,
            "audit_sha256": audit_hash,
        }
    )
    _record_budget(budget, elapsed, completed=True)
    _write_inventory(inventory)
    write_json(V3_BUDGET_PATH, budget)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(execute_attempt(args.spec.resolve()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
