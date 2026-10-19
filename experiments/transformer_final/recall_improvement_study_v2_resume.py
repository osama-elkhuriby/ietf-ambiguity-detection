"""Resume approved V2 cells without altering the frozen study source snapshot."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import traceback
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from experiments.transformer_final.recall_improvement_study_v2.cell_launcher import (
    _custom_outer_loader,
)
from experiments.transformer_final.recall_improvement_study_v2.prepare_v2 import (
    V2_BUDGET_PATH,
    V2_COMPATIBILITY_PATH,
    V2_INVENTORY_PATH,
    V2_PROTOCOL_PATH,
    V2_ROOT,
    _study_source_hashes,
)
from experiments.transformer_final.recall_improvement_study_v1.study_runner import (
    _gpu_preflight,
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
RUNNER_CONFIG_PATH = V2_ROOT / "V2_RUNNER_CONFIG.json"
AUTHORIZED_ATTEMPT_ID = "C_local_context/fold_3/seed_2718"
VALIDATION_RESERVE_SECONDS = 2700
CELL_ESTIMATES = {
    "C_local_context": 450,
    "G_partial_4": 300,
    "D_focal_loss": 300,
    "G2_partial_2": 300,
}
PRIORITY_STAGES = (
    (
        "C_local_context",
        [
            "C_local_context/fold_3/seed_2718",
            "C_local_context/fold_4/seed_2718",
            "C_local_context/fold_5/seed_2718",
        ],
        True,
    ),
    (
        "G_partial_4",
        [f"G_partial_4/fold_{fold}/seed_2718" for fold in range(1, 6)],
        True,
    ),
    (
        "D_focal_loss",
        [f"D_focal_loss/fold_{fold}/seed_2718" for fold in range(1, 6)],
        False,
    ),
    (
        "G2_partial_2",
        [f"G2_partial_2/fold_{fold}/seed_2718" for fold in range(1, 6)],
        False,
    ),
)


def _now() -> datetime:
    return datetime.now().astimezone()


def _inventory() -> dict[str, Any]:
    return read_json(V2_INVENTORY_PATH)


def _budget() -> dict[str, Any]:
    return read_json(V2_BUDGET_PATH)


def _runner_config() -> dict[str, Any]:
    return read_json(RUNNER_CONFIG_PATH)


def _protocol() -> dict[str, Any]:
    protocol = read_json(V2_PROTOCOL_PATH)
    if (
        protocol.get("study_id") != "recall_improvement_study_v2"
        or sha256_file(V2_PROTOCOL_PATH)
        != _runner_config().get("protocol_sha256")
    ):
        raise RuntimeError("V2 protocol identity or digest mismatch")
    snapshot = protocol["source_snapshot"]
    if (
        source_file_hashes() != snapshot["core_files"]
        or _study_source_hashes() != snapshot["study_files"]
        or sha256_text(stable_json(snapshot))
        != protocol["source_snapshot_sha256"]
    ):
        raise RuntimeError("V2 frozen source snapshot mismatch")
    return protocol


def _check_authorization() -> dict[str, Any]:
    if os.environ.get("RECALL_STUDY_V2_GPU_AUTHORIZED") != "1":
        raise PermissionError(
            "V2 execution requires RECALL_STUDY_V2_GPU_AUTHORIZED=1"
        )
    if not _runner_config().get("gpu_execution_authorized"):
        raise PermissionError("V2 runner GPU authorization is disabled")
    budget = _budget()
    if budget.get("execution_authorized") is not True:
        raise PermissionError("V2 execution-budget authorization is disabled")
    deadline = datetime.fromisoformat(budget["authorization_window_deadline"])
    if deadline.tzinfo is None or _now() >= deadline:
        raise TimeoutError("The authorized V2 execution window has expired")
    return budget


def authorize_new_session() -> None:
    config = _runner_config()
    protocol = _protocol()
    if sha256_file(V2_PROTOCOL_PATH) != config.get("protocol_sha256"):
        raise RuntimeError("Refusing authorization against a changed V2 protocol")
    budget = _budget()
    started = _now()
    deadline = started + timedelta(seconds=21600)
    session_id = f"v2-{started.strftime('%Y%m%dT%H%M%S%z')}"
    history = budget.setdefault("authorization_history", [])
    history.append(
        {
            "session_id": session_id,
            "authorization_granted_at": started.isoformat(),
            "authorization_window_start": started.isoformat(),
            "authorization_window_deadline": deadline.isoformat(),
            "budget_seconds": 21600,
        }
    )
    budget.update(
        {
            "authorization_session_id": session_id,
            "authorization_window_start": started.isoformat(),
            "authorization_window_deadline": deadline.isoformat(),
            "budget_seconds": 21600,
            "execution_authorized": True,
            "status": "AUTHORIZED_NEW_SESSION",
        }
    )
    config["gpu_execution_authorized"] = True
    write_json(V2_BUDGET_PATH, budget)
    write_json(RUNNER_CONFIG_PATH, config)
    print(
        json.dumps(
            {
                "session_id": session_id,
                "authorization_window_start": started.isoformat(),
                "authorization_window_deadline": deadline.isoformat(),
                "budget_seconds": 21600,
                "study_id": protocol["study_id"],
                "protocol_sha256": sha256_file(V2_PROTOCOL_PATH),
            },
            indent=2,
        )
    )


def _attempt_scientific_spec(spec: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(spec)
    for key in (
        "run_path",
        "audit_path",
        "log_path",
        "attempt_id",
        "replaces_attempt",
    ):
        value.pop(key, None)
    return value


def prepare_fold3_attempt() -> str:
    _check_authorization()
    protocol = _protocol()
    inventory = _inventory()
    blocked = inventory["pending_v2_cells"][AUTHORIZED_ATTEMPT_ID]
    if blocked.get("status") != "BLOCKED_GPU_RESOURCE_NO_RETRY":
        raise RuntimeError("The original Fold 3 attempt is not in its preserved blocked state")
    original_spec_path = Path(blocked["spec_path"])
    if sha256_file(original_spec_path) != blocked["spec_sha256"]:
        raise RuntimeError("The original blocked Fold 3 spec digest changed")
    blocker_log = Path(inventory["preflight_blocker"]["log_path"])
    if not blocker_log.is_file():
        raise FileNotFoundError(f"The original blocked-attempt log is missing: {blocker_log}")
    attempts = inventory.setdefault("authorized_attempts", {})
    if attempts:
        existing = attempts.get(AUTHORIZED_ATTEMPT_ID)
        if existing and existing.get("status") == "AUTHORIZED_PENDING":
            return str(existing["attempt_id"])
        raise RuntimeError(
            "A Fold 3 replacement attempt already exists; automatic retries are forbidden"
        )

    original_spec = read_json(original_spec_path)
    stamp = _now().strftime("%Y%m%dT%H%M%S%z")
    attempt_id = f"fold3-local-context-{stamp}"
    attempt_spec_path = (
        V2_ROOT / "cell_specs" / "authorized_attempts" / f"{attempt_id}.json"
    )
    run_path = (
        V2_ROOT
        / "runs"
        / "C_local_context"
        / "fold_3"
        / "seed_2718"
        / f"attempt_{attempt_id}"
    )
    audit_path = (
        V2_ROOT / "audits" / "authorized_attempts" / f"{attempt_id}.json"
    )
    log_path = V2_ROOT / "logs" / "authorized_attempts" / attempt_id
    attempt_spec = deepcopy(original_spec)
    attempt_spec.update(
        {
            "attempt_id": attempt_id,
            "replaces_attempt": {
                "cell_id": AUTHORIZED_ATTEMPT_ID,
                "status": blocked["status"],
                "spec_sha256": blocked["spec_sha256"],
                "log_path": str(blocker_log),
            },
            "run_path": str(run_path),
            "audit_path": str(audit_path),
            "log_path": str(log_path),
        }
    )
    if _attempt_scientific_spec(attempt_spec) != _attempt_scientific_spec(
        original_spec
    ):
        raise RuntimeError("Replacement attempt changed scientific cell settings")
    if (
        attempt_spec["study_id"] != protocol["study_id"]
        or attempt_spec["protocol_sha256"] != sha256_file(V2_PROTOCOL_PATH)
        or attempt_spec["seed"] != protocol["training"]["seed"]
        or attempt_spec["fold_manifest_sha256"]
        != blocked["fold_manifest_sha256"]
    ):
        raise RuntimeError("Replacement attempt does not match frozen V2 provenance")
    if attempt_spec_path.exists() or run_path.exists() or audit_path.exists():
        raise FileExistsError("A replacement-attempt output path already exists")
    attempt_spec_path.parent.mkdir(parents=True, exist_ok=True)
    with attempt_spec_path.open("x", encoding="utf-8") as handle:
        json.dump(attempt_spec, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    attempts[AUTHORIZED_ATTEMPT_ID] = {
        "attempt_id": attempt_id,
        "cell_id": AUTHORIZED_ATTEMPT_ID,
        "status": "AUTHORIZED_PENDING",
        "replaces_status": blocked["status"],
        "replaces_spec_path": str(original_spec_path),
        "replaces_spec_sha256": blocked["spec_sha256"],
        "replaces_log_path": str(blocker_log),
        "spec_path": str(attempt_spec_path),
        "spec_sha256": sha256_file(attempt_spec_path),
        "scientific_spec_sha256": sha256_text(
            stable_json(_attempt_scientific_spec(attempt_spec))
        ),
        "protocol_sha256": sha256_file(V2_PROTOCOL_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "run_path": str(run_path),
        "audit_path": str(audit_path),
        "log_path": str(log_path),
        "created_at": _now().isoformat(),
    }
    write_json(V2_INVENTORY_PATH, inventory)
    print(
        json.dumps(
            {
                "attempt_id": attempt_id,
                "cell_id": AUTHORIZED_ATTEMPT_ID,
                "spec_path": str(attempt_spec_path),
                "run_path": str(run_path),
                "replaces_status": blocked["status"],
                "scientific_spec_sha256": attempts[AUTHORIZED_ATTEMPT_ID][
                    "scientific_spec_sha256"
                ],
            },
            indent=2,
        )
    )
    return attempt_id


def _verify_dataset_and_fold(spec: dict[str, Any], protocol: dict[str, Any]) -> None:
    from models.transformer_final.config import load_config
    from models.transformer_final.dataset import load_effective_dataset

    config_path = Path(spec["config_path"])
    if sha256_file(config_path) != protocol["config_sha256"]:
        raise RuntimeError("V2 configuration hash mismatch")
    config = load_config(config_path)
    frame, dataset_manifest, dataset_dir = load_effective_dataset(config)
    dataset = protocol["dataset"]
    if (
        sha256_file(dataset_dir / "manifest.json")
        != dataset["dataset_manifest_sha256"]
        or dataset_manifest["effective_dataset_sha256"]
        != dataset["effective_dataset_sha256"]
        or sha256_file(dataset_dir / dataset_manifest["dataset_file"])
        != dataset["effective_dataset_sha256"]
        or sha256_file(Path(config["data"]["source_csv"]))
        != dataset["source_sha256"]
        or frame["record_id"].astype(str).duplicated().any()
        or len(frame) != dataset["rows"]
    ):
        raise RuntimeError("V2 dataset identity or integrity check failed")
    audit = read_json(dataset_dir / dataset_manifest["leakage_audit"])
    if audit.get("training_blocked") is not False:
        raise RuntimeError("Dataset leakage audit blocks training")
    outer_path = Path(dataset["study_outer_fold_manifests_path"])
    if sha256_file(outer_path) != dataset["study_outer_fold_manifests_sha256"]:
        raise RuntimeError("V2 outer-fold manifest file digest mismatch")
    fold = int(spec["train_fold"])
    manifests = json.loads(outer_path.read_text(encoding="utf-8"))
    matches = [item for item in manifests if int(item["fold"]) == fold]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one frozen outer manifest for fold {fold}")
    manifest = matches[0]
    baseline_fold_path = dataset_dir / "folds" / f"fold_{fold}.json"
    baseline_manifest = read_json(baseline_fold_path)
    if (
        sha256_file(baseline_fold_path)
        != dataset["fold_manifest_sha256"][str(fold)]
        or set(baseline_manifest["candidate_fitting_record_ids"])
        != set(manifest["candidate_fitting_record_ids"])
        or set(baseline_manifest["fitting_record_ids"])
        != set(manifest["fitting_record_ids"])
        or set(baseline_manifest["scoring_record_ids"])
        != set(manifest["scoring_record_ids"])
        or baseline_manifest["fitting_exclusions"]
        != manifest["fitting_exclusions"]
    ):
        raise RuntimeError("V2 fold differs from the verified dataset fold")
    if (
        spec.get("fold_manifest") != manifest
        or sha256_text(stable_json(manifest)) != spec["fold_manifest_sha256"]
        or int(manifest["fold"]) != fold
    ):
        raise RuntimeError("Cell fold manifest differs from the frozen V2 manifest")
    fitting_ids = set(manifest["fitting_record_ids"])
    candidate_ids = set(manifest["candidate_fitting_record_ids"])
    scoring_ids = set(manifest["scoring_record_ids"])
    frame_ids = set(frame["record_id"].astype(str))
    if (
        candidate_ids & scoring_ids
        or candidate_ids | scoring_ids != frame_ids
        or not fitting_ids <= candidate_ids
        or len(scoring_ids) != len(manifest["scoring_record_ids"])
        or set(manifest["fitting_families"])
        & set(manifest["held_out_families"])
    ):
        raise RuntimeError("V2 fold membership or family-disjointness check failed")
    if any(
        item["role"] not in {"fitting", "evaluation"}
        or item["field"] not in {"section_heading", "previous_clause", "next_clause"}
        or item["record_id"]
        not in (fitting_ids if item["role"] == "fitting" else scoring_ids)
        for item in manifest["context_sanitizations"]
    ):
        raise RuntimeError("V2 context-sanitization leakage policy check failed")
    reserved = set(dataset["reserved_families"])
    if set(manifest["fitting_families"]) & reserved:
        raise RuntimeError("A reserved family appears in V2 fitting data")


def _verify_shared_artifacts(spec: dict[str, Any]) -> dict[str, Any]:
    protocol = _protocol()
    inventory = _inventory()
    compatibility = read_json(V2_COMPATIBILITY_PATH)
    if (
        sha256_file(V2_COMPATIBILITY_PATH)
        != inventory["compatibility_manifest_sha256"]
        or len(compatibility["reusable_v1_runs"]) != 17
        or compatibility["reused_runs_copied_or_modified"] is not False
        or compatibility["reused_runs_remain_v1_provenance"] is not True
    ):
        raise RuntimeError("V1-to-V2 compatibility/reuse gate failed")
    if spec.get("study_id") != protocol["study_id"]:
        raise RuntimeError("Cell specification has the wrong study identity")
    if spec.get("protocol_sha256") != sha256_file(V2_PROTOCOL_PATH):
        raise RuntimeError("Cell specification does not match the V2 protocol")
    _verify_dataset_and_fold(spec, protocol)
    return protocol


def _verify_cuda_and_gpu() -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                "assert torch.cuda.device_count() == 1, 'expected one CUDA device'; "
                "print(torch.cuda.get_device_name(0))"
            ),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    gpu = _gpu_preflight()
    if result.stdout.strip() != gpu["name"]:
        raise RuntimeError("CUDA device does not match the GPU preflight device")
    return gpu


def _record_budget_cell(
    cell_id: str, elapsed_seconds: float, *, completed: bool
) -> None:
    budget = _budget()
    actual = budget.setdefault(
        "actual",
        {"cell_ids": {}, "gpu_cell_seconds": 0.0, "gpu_cells_started": 0,
         "gpu_cells_completed": 0},
    )
    actual.setdefault("cell_ids", {})[cell_id] = {
        "completed": completed,
        "elapsed_seconds": elapsed_seconds,
    }
    actual["gpu_cell_seconds"] = float(actual.get("gpu_cell_seconds", 0.0)) + (
        elapsed_seconds
    )
    actual["gpu_cells_started"] = int(actual.get("gpu_cells_started", 0)) + 1
    if completed:
        actual["gpu_cells_completed"] = int(
            actual.get("gpu_cells_completed", 0)
        ) + 1
    write_json(V2_BUDGET_PATH, budget)


def _set_attempt_status(
    attempt: dict[str, Any],
    status: str,
    *,
    timestamp_key: str,
) -> dict[str, Any]:
    inventory = _inventory()
    record = inventory["authorized_attempts"][AUTHORIZED_ATTEMPT_ID]
    record["status"] = status
    record[timestamp_key] = _now().isoformat()
    if status == "IN_PROGRESS":
        inventory["actual"]["gpu_cells_started"] += 1
    elif status == "COMPLETE":
        inventory["actual"]["gpu_cells_completed"] += 1
    write_json(V2_INVENTORY_PATH, inventory)
    attempt.update(record)
    return inventory


def _execute_fold3_attempt(attempt_id: str) -> Path:
    _check_authorization()
    inventory = _inventory()
    attempt = inventory.get("authorized_attempts", {}).get(AUTHORIZED_ATTEMPT_ID)
    if not attempt or attempt.get("attempt_id") != attempt_id:
        raise RuntimeError("Authorized Fold 3 attempt identity mismatch")
    if attempt.get("status") != "AUTHORIZED_PENDING":
        raise RuntimeError(
            f"Automatic retry is forbidden for attempt status {attempt.get('status')}"
        )
    if sha256_file(Path(__file__)) != attempt["runner_sha256"]:
        raise RuntimeError("Authorized-attempt runner changed after registration")
    original_record = inventory["pending_v2_cells"][AUTHORIZED_ATTEMPT_ID]
    if (
        original_record.get("status") != "BLOCKED_GPU_RESOURCE_NO_RETRY"
        or original_record.get("spec_sha256") != attempt["replaces_spec_sha256"]
        or sha256_file(Path(original_record["spec_path"]))
        != original_record["spec_sha256"]
        or not Path(attempt["replaces_log_path"]).is_file()
    ):
        raise RuntimeError("Original blocked attempt or its log is not preserved")
    spec_path = Path(attempt["spec_path"])
    if sha256_file(spec_path) != attempt["spec_sha256"]:
        raise RuntimeError("Replacement-attempt spec digest mismatch")
    spec = read_json(spec_path)
    original_spec = read_json(Path(original_record["spec_path"]))
    if (
        _attempt_scientific_spec(spec) != _attempt_scientific_spec(original_spec)
        or sha256_text(stable_json(_attempt_scientific_spec(spec)))
        != attempt["scientific_spec_sha256"]
    ):
        raise RuntimeError("Replacement attempt changed frozen scientific settings")
    _verify_shared_artifacts(spec)
    gpu_preflight = _verify_cuda_and_gpu()
    run_path = Path(spec["run_path"])
    audit_path = Path(spec["audit_path"])
    stdout_path = Path(spec["log_path"] + ".stdout.log")
    stderr_path = Path(spec["log_path"] + ".stderr.log")
    if any(path.exists() for path in (run_path, audit_path, stdout_path, stderr_path)):
        raise FileExistsError("Replacement-attempt output or log path already exists")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)

    from models.transformer_final import train

    original_loader = train.load_fold_data
    original_write_json = train.write_json
    v2_launcher_path = (
        V2_ROOT / "cell_launcher.py"
    )
    delegated_v1_launcher_path = (
        V2_ROOT.parent / "recall_improvement_study_v1" / "cell_launcher.py"
    )
    started = _now()
    _set_attempt_status(attempt, "IN_PROGRESS", timestamp_key="started_at")
    elapsed_seconds = 0.0
    try:
        with (
            stdout_path.open("x", encoding="utf-8") as stdout,
            stderr_path.open("x", encoding="utf-8") as stderr,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            try:
                train.load_fold_data = _custom_outer_loader(spec)

                def write_with_attempt_provenance(
                    path: Path, value: dict[str, Any]
                ) -> None:
                    updated = value
                    if path.name == "run_manifest.json":
                        updated = json.loads(json.dumps(value))
                        updated.update(
                            {
                                "study_id": "recall_improvement_study_v2",
                                "study_protocol_sha256": spec["protocol_sha256"],
                                "study_source_snapshot_sha256": _protocol()[
                                    "source_snapshot_sha256"
                                ],
                                "study_cell_id": spec["cell_id"],
                                "study_fold_manifest_sha256": spec[
                                    "fold_manifest_sha256"
                                ],
                                "study_attempt_id": attempt_id,
                                "study_execution_provenance": {
                                    "v2_launcher_sha256": sha256_file(
                                        v2_launcher_path
                                    ),
                                    "authorized_attempt_runner_sha256": sha256_file(
                                        Path(__file__)
                                    ),
                                    "delegated_v1_cell_launcher_sha256": sha256_file(
                                        delegated_v1_launcher_path
                                    ),
                                    "new_run_executed_under_v2": True,
                                    "replaces_blocked_attempt": original_record[
                                        "spec_sha256"
                                    ],
                                },
                            }
                        )
                    original_write_json(path, updated)

                train.write_json = write_with_attempt_provenance
                delegated_spec = dict(spec)
                delegated_spec.pop("fold_manifest", None)
                delegated_spec.pop("fold_manifest_sha256", None)
                with tempfile.TemporaryDirectory(
                    prefix="authorized_attempt_",
                    dir=V2_ROOT / "cell_specs",
                ) as temp_dir:
                    temp_spec = Path(temp_dir) / spec_path.name
                    temp_spec.write_text(
                        json.dumps(delegated_spec, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    from experiments.transformer_final.recall_improvement_study_v1.cell_launcher import (
                        execute_cell,
                    )

                    output = execute_cell(temp_spec)
                manifest = read_json(output / "run_manifest.json")
                if (
                    manifest.get("study_protocol_sha256")
                    != spec["protocol_sha256"]
                    or manifest.get("study_cell_id") != spec["cell_id"]
                    or manifest.get("study_attempt_id") != attempt_id
                    or manifest.get("study_execution_provenance", {}).get(
                        "new_run_executed_under_v2"
                    )
                    is not True
                ):
                    raise RuntimeError("V2 authorized-attempt provenance check failed")
                audit = read_json(audit_path)
                audit.update(
                    {
                        "study_id": "recall_improvement_study_v2",
                        "study_protocol_sha256": spec["protocol_sha256"],
                        "authorized_attempt_id": attempt_id,
                        "replaces_blocked_attempt": original_record["spec_sha256"],
                        "gpu_preflight": gpu_preflight,
                    }
                )
                write_json(audit_path, audit)
                elapsed_seconds = float(audit["elapsed_seconds"])
            except BaseException:
                traceback.print_exc(file=stderr)
                raise
        latest = _inventory()
        latest_attempt = latest["authorized_attempts"][AUTHORIZED_ATTEMPT_ID]
        latest_attempt["status"] = "COMPLETE"
        latest_attempt["completed_at"] = _now().isoformat()
        latest_attempt["run_manifest_sha256"] = sha256_file(
            Path(spec["run_path"]) / "run_manifest.json"
        )
        latest_attempt["elapsed_seconds"] = elapsed_seconds
        latest["actual"]["gpu_cells_completed"] += 1
        write_json(V2_INVENTORY_PATH, latest)
        _record_budget_cell(attempt_id, elapsed_seconds, completed=True)
        print(f"Completed authorized V2 Fold 3 attempt: {output}")
        return output
    except BaseException:
        latest = _inventory()
        latest_attempt = latest["authorized_attempts"][AUTHORIZED_ATTEMPT_ID]
        was_in_progress = latest_attempt.get("status") == "IN_PROGRESS"
        latest_attempt["status"] = "FAILED_NO_AUTOMATIC_RETRY"
        latest_attempt["failed_at"] = _now().isoformat()
        write_json(V2_INVENTORY_PATH, latest)
        wall_seconds = max(0.0, (_now() - started).total_seconds())
        if was_in_progress:
            _record_budget_cell(attempt_id, wall_seconds, completed=False)
        raise
    finally:
        train.load_fold_data = original_loader
        train.write_json = original_write_json


def _cuda_probe() -> None:
    _verify_cuda_and_gpu()


def _run_regular_cell(cell_id: str, spec_path: Path) -> float:
    _check_authorization()
    inventory = _inventory()
    cell = inventory["pending_v2_cells"].get(cell_id)
    if (
        cell is None
        or cell.get("status") != "PENDING_SEPARATE_GPU_AUTHORIZATION"
        or Path(cell["spec_path"]).resolve() != spec_path.resolve()
        or sha256_file(spec_path) != cell["spec_sha256"]
    ):
        raise RuntimeError(f"Cell is not pending or its spec changed: {cell_id}")
    spec = read_json(spec_path)
    protocol = _verify_shared_artifacts(spec)
    if (
        spec.get("study_id") != protocol["study_id"]
        or spec.get("protocol_sha256") != sha256_file(V2_PROTOCOL_PATH)
        or int(spec["seed"]) != int(protocol["training"]["seed"])
    ):
        raise RuntimeError(f"Cell does not match frozen V2 settings: {cell_id}")
    from experiments.transformer_final.recall_improvement_study_v2.cell_launcher import (
        _verify_launch,
    )

    _verify_launch(spec_path, spec)
    _cuda_probe()
    started = _now()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.transformer_final.recall_improvement_study_v2.cell_launcher",
            "--spec",
            str(spec_path),
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
        check=False,
    )
    wall_seconds = max(0.0, (_now() - started).total_seconds())
    if result.returncode != 0:
        _record_budget_cell(cell_id, wall_seconds, completed=False)
        raise subprocess.CalledProcessError(result.returncode, result.args)
    audit = read_json(Path(spec["audit_path"]))
    elapsed = float(audit["elapsed_seconds"])
    _record_budget_cell(cell_id, elapsed, completed=True)
    return elapsed


def _remaining_budget_seconds(budget: dict[str, Any]) -> float:
    deadline = datetime.fromisoformat(budget["authorization_window_deadline"])
    if deadline.tzinfo is None:
        raise RuntimeError("V2 authorization deadline must include a timezone")
    return (deadline - _now()).total_seconds()


def run_study() -> None:
    _check_authorization()
    inventory = _inventory()
    attempt = inventory.get("authorized_attempts", {}).get(AUTHORIZED_ATTEMPT_ID)
    if attempt is None:
        attempt_id = prepare_fold3_attempt()
    elif attempt.get("status") == "AUTHORIZED_PENDING":
        attempt_id = str(attempt["attempt_id"])
    elif attempt.get("status") == "COMPLETE":
        attempt_id = None
    else:
        raise RuntimeError(
            f"Fold 3 replacement attempt is {attempt.get('status')}; no retry will be made"
        )

    for stage, cell_ids, required in PRIORITY_STAGES:
        budget = _check_authorization()
        inventory = _inventory()
        if stage == "C_local_context":
            stage_cells = cell_ids
        else:
            stage_cells = [
                cell_id
                for cell_id in cell_ids
                if inventory["pending_v2_cells"].get(cell_id, {}).get("status")
                == "PENDING_SEPARATE_GPU_AUTHORIZATION"
            ]
        seconds_needed = (
            CELL_ESTIMATES[stage] * len(stage_cells) + VALIDATION_RESERVE_SECONDS
        )
        if _remaining_budget_seconds(budget) < seconds_needed:
            if required:
                raise TimeoutError(
                    f"Insufficient budget for required stage {stage} plus final validation"
                )
            print(f"Skipping optional V2 stage {stage}: insufficient reserved budget")
            continue
        for index, cell_id in enumerate(stage_cells):
            budget = _check_authorization()
            cells_left = len(stage_cells) - index
            time_needed = (
                CELL_ESTIMATES[stage] * cells_left + VALIDATION_RESERVE_SECONDS
            )
            if _remaining_budget_seconds(budget) < time_needed:
                if required:
                    raise TimeoutError(
                        f"Insufficient budget to finish required stage {stage}"
                    )
                print(
                    f"Stopping optional V2 stage {stage}: final-validation reserve "
                    "would be consumed"
                )
                break
            if stage == "C_local_context" and cell_id == AUTHORIZED_ATTEMPT_ID:
                if attempt_id is None:
                    continue
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.transformer_final.recall_improvement_study_v2_resume",
                        "run-attempt",
                        "--attempt-id",
                        attempt_id,
                    ],
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    check=True,
                )
            else:
                current = _inventory()["pending_v2_cells"][cell_id]
                if current.get("status") != "PENDING_SEPARATE_GPU_AUTHORIZATION":
                    raise RuntimeError(
                        f"Unexpected cell state for {cell_id}: {current.get('status')}"
                    )
                _run_regular_cell(cell_id, Path(current["spec_path"]))
    print("All authorized V2 training priorities finished or optional stages skipped.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("authorize", "prepare-fold3", "run-attempt", "run-study"),
    )
    parser.add_argument("--attempt-id")
    args = parser.parse_args()
    if args.action == "authorize":
        authorize_new_session()
    elif args.action == "prepare-fold3":
        print(prepare_fold3_attempt())
    elif args.action == "run-attempt":
        if not args.attempt_id:
            parser.error("run-attempt requires --attempt-id")
        _execute_fold3_attempt(args.attempt_id)
    else:
        run_study()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
