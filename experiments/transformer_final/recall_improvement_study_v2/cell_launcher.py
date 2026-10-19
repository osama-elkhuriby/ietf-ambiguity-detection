"""Launch one V2 cell after protocol, authorization, and GPU preflight checks."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.transformer_final.recall_improvement_study_v2.prepare_v2 import (
    V2_BUDGET_PATH,
    V2_INVENTORY_PATH,
    V2_PROTOCOL_PATH,
    V2_ROOT,
    _study_source_hashes,
)
from models.transformer_final.common import (
    read_json,
    sha256_file,
    sha256_text,
    source_file_hashes,
    stable_json,
)


def _verify_launch(spec_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("RECALL_STUDY_V2_GPU_AUTHORIZED") != "1":
        raise PermissionError(
            "Set RECALL_STUDY_V2_GPU_AUTHORIZED=1 only after separately "
            "authorizing V2 GPU execution."
        )
    inventory = read_json(V2_INVENTORY_PATH)
    cell = inventory["pending_v2_cells"].get(spec["cell_id"])
    if (
        cell is None
        or cell.get("status") != "PENDING_SEPARATE_GPU_AUTHORIZATION"
        or Path(cell["spec_path"]).resolve() != spec_path.resolve()
        or sha256_file(spec_path) != cell["spec_sha256"]
    ):
        raise RuntimeError("V2 cell specification or inventory digest mismatch")
    protocol = read_json(V2_PROTOCOL_PATH)
    if (
        protocol["study_id"] != "recall_improvement_study_v2"
        or sha256_file(V2_PROTOCOL_PATH) != spec["protocol_sha256"]
    ):
        raise RuntimeError("V2 protocol identity or digest mismatch")
    snapshot = protocol["source_snapshot"]
    if (
        source_file_hashes() != snapshot["core_files"]
        or _study_source_hashes() != snapshot["study_files"]
        or sha256_text(stable_json(snapshot)) != protocol["source_snapshot_sha256"]
    ):
        raise RuntimeError("V2 source snapshot mismatch")
    config_path = Path(spec["config_path"])
    if sha256_file(config_path) != protocol["config_sha256"]:
        raise RuntimeError("V2 configuration hash mismatch")
    if (
        spec.get("study_id") != protocol["study_id"]
        or spec.get("protocol_sha256") != sha256_file(V2_PROTOCOL_PATH)
    ):
        raise RuntimeError("Cell specification does not match V2 protocol")

    budget = read_json(V2_BUDGET_PATH)
    if budget.get("execution_authorized") is not True:
        raise PermissionError(
            "V2 budget authorization is disabled; enable it only after explicit "
            "GPU-run authorization."
        )
    deadline = datetime.fromisoformat(budget["authorization_window_deadline"])
    if deadline.tzinfo is None or datetime.now(deadline.tzinfo) >= deadline:
        raise TimeoutError("The authorized study execution window has expired")

    from experiments.transformer_final.recall_improvement_study_v1.study_runner import (
        _gpu_preflight,
    )

    return _gpu_preflight()


def _custom_outer_loader(spec: dict[str, Any]) -> Any:
    import pandas as pd

    from models.transformer_final.dataset import load_effective_dataset
    from models.transformer_final.leakage import apply_context_sanitization

    manifest = spec["fold_manifest"]
    expected_fold = int(spec["train_fold"])

    def load_fold(
        config: dict[str, Any], fold: int, *, input_mode: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], Path]:
        if fold != expected_fold or int(manifest["fold"]) != expected_fold:
            raise RuntimeError("V2 cell requested an unexpected fold")
        frame, dataset_manifest, dataset_dir = load_effective_dataset(config)
        fitting_ids = set(manifest["fitting_record_ids"])
        scoring_ids = set(manifest["scoring_record_ids"])
        fitting = (
            frame.loc[frame["record_id"].astype(str).isin(fitting_ids)]
            .copy()
            .reset_index(drop=True)
        )
        evaluation = (
            frame.loc[frame["record_id"].astype(str).isin(scoring_ids)]
            .copy()
            .reset_index(drop=True)
        )
        if (
            set(fitting["record_id"].astype(str)) != fitting_ids
            or set(evaluation["record_id"].astype(str)) != scoring_ids
        ):
            raise RuntimeError("V2 fold manifest no longer matches the dataset")
        if input_mode == "local_context":
            fitting = apply_context_sanitization(
                fitting, manifest, role="fitting"
            )
            evaluation = apply_context_sanitization(
                evaluation, manifest, role="evaluation"
            )
        elif input_mode != "target_only":
            raise ValueError(f"Unsupported V2 input mode: {input_mode}")
        return fitting, evaluation, manifest, dataset_manifest, dataset_dir

    return load_fold


def execute_cell(spec_path: Path) -> Path:
    spec = read_json(spec_path)
    run_path = Path(spec["run_path"])
    if run_path.exists():
        raise FileExistsError(
            f"V2 cell output already exists; automatic retry is forbidden: "
            f"{spec['cell_id']}"
        )
    gpu_preflight = _verify_launch(spec_path, spec)
    from models.transformer_final import train
    from models.transformer_final.common import write_json

    original_loader = train.load_fold_data
    original_write_json = train.write_json
    delegated_spec = dict(spec)
    if spec["method"] == "context":
        train.load_fold_data = _custom_outer_loader(spec)
        delegated_spec.pop("fold_manifest", None)
        delegated_spec.pop("fold_manifest_sha256", None)
    else:
        delegated_spec.pop("fold_manifest", None)
        delegated_spec.pop("fold_manifest_sha256", None)

    def write_with_v2_provenance(path: Path, value: dict[str, Any]) -> None:
        updated = value
        if path.name == "run_manifest.json":
            updated = json.loads(json.dumps(value))
            updated["study_id"] = "recall_improvement_study_v2"
            updated["study_protocol_sha256"] = spec["protocol_sha256"]
            updated["study_source_snapshot_sha256"] = read_json(
                V2_PROTOCOL_PATH
            )["source_snapshot_sha256"]
            updated["study_cell_id"] = spec["cell_id"]
            updated["study_fold_manifest_sha256"] = spec[
                "fold_manifest_sha256"
            ]
            updated["study_execution_provenance"] = {
                "v2_launcher_sha256": sha256_file(Path(__file__)),
                "delegated_v1_cell_launcher_sha256": sha256_file(
                    V2_ROOT.parent
                    / "recall_improvement_study_v1"
                    / "cell_launcher.py"
                ),
                "new_run_executed_under_v2": True,
            }
        original_write_json(path, updated)

    train.write_json = write_with_v2_provenance
    inventory_path = V2_INVENTORY_PATH
    inventory = read_json(inventory_path)
    cell = inventory["pending_v2_cells"].get(spec["cell_id"])
    if cell is None or cell.get("status") != "PENDING_SEPARATE_GPU_AUTHORIZATION":
        raise RuntimeError(f"V2 inventory does not authorize this cell: {spec['cell_id']}")
    cell["status"] = "IN_PROGRESS"
    cell["started_at"] = datetime.now().astimezone().isoformat()
    inventory["actual"]["gpu_cells_started"] += 1
    write_json(inventory_path, inventory)
    try:
        with tempfile.TemporaryDirectory(
            prefix="launch_", dir=V2_ROOT / "cell_specs"
        ) as temporary_directory:
            temp_spec = Path(temporary_directory) / spec_path.name
            temp_spec.write_text(
                json.dumps(delegated_spec, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            from experiments.transformer_final.recall_improvement_study_v1.cell_launcher import (
                execute_cell as execute_v1_cell,
            )

            output = execute_v1_cell(temp_spec)
        run_manifest = read_json(output / "run_manifest.json")
        if (
            run_manifest.get("study_protocol_sha256") != spec["protocol_sha256"]
            or run_manifest.get("study_cell_id") != spec["cell_id"]
            or run_manifest.get("study_execution_provenance", {}).get(
                "new_run_executed_under_v2"
            )
            is not True
        ):
            raise RuntimeError("V2 run manifest provenance validation failed")
        audit_path = Path(spec["audit_path"])
        audit = read_json(audit_path)
        audit["study_id"] = "recall_improvement_study_v2"
        audit["study_protocol_sha256"] = spec["protocol_sha256"]
        audit["gpu_preflight"] = gpu_preflight
        write_json(audit_path, audit)
        cell["status"] = "COMPLETE"
        cell["completed_at"] = datetime.now().astimezone().isoformat()
        cell["run_manifest_sha256"] = sha256_file(output / "run_manifest.json")
        inventory["actual"]["gpu_cells_completed"] += 1
        write_json(inventory_path, inventory)
        return output
    except BaseException:
        cell["status"] = "FAILED_NO_AUTOMATIC_RETRY"
        cell["failed_at"] = datetime.now().astimezone().isoformat()
        write_json(inventory_path, inventory)
        raise
    finally:
        train.load_fold_data = original_loader
        train.write_json = original_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec.resolve())
    if os.environ.get("RECALL_STUDY_V2_GPU_AUTHORIZED") != "1":
        raise PermissionError(
            "Set RECALL_STUDY_V2_GPU_AUTHORIZED=1 only after separately "
            "authorizing V2 GPU execution."
        )
    if read_json(V2_BUDGET_PATH).get("execution_authorized") is not True:
        raise PermissionError(
            "V2 budget authorization is disabled; enable it only after explicit "
            "GPU-run authorization."
        )
    run_path = Path(spec["run_path"])
    stdout_path = Path(spec["log_path"] + ".stdout.log")
    stderr_path = Path(spec["log_path"] + ".stderr.log")
    if run_path.exists() or stdout_path.exists() or stderr_path.exists():
        raise FileExistsError(
            f"V2 cell output or logs already exist; automatic retry is forbidden: "
            f"{spec['cell_id']}"
        )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        stdout_path.open("x", encoding="utf-8") as stdout,
        stderr_path.open("x", encoding="utf-8") as stderr,
    ):
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                output = execute_cell(args.spec.resolve())
                print(f"Completed V2 cell: {output}", flush=True)
        except BaseException:
            traceback.print_exc(file=stderr)
            raise
    print(f"Cell logs: {stdout_path} ; {stderr_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
