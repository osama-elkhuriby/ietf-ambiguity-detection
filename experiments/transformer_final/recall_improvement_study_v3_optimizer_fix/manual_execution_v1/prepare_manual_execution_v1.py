"""Prepare the immutable one-cell-at-a-time V3 manual execution package."""

from __future__ import annotations

import json
import sys
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
MANUAL_CONFIG_PATH = PACKAGE_ROOT / "MANUAL_RUNNER_CONFIG.json"
PACKAGE_MANIFEST_PATH = PACKAGE_ROOT / "PACKAGE_MANIFEST.json"
AUTHORIZATION_ROOT = PACKAGE_ROOT / "authorizations"
EXPECTED_V3_PROTOCOL_SHA256 = (
    "9502e659c1f246f433ce534fa7165959a6519c6637c71b3be0289f6638e72c51"
)
EXPECTED_V3_SOURCE_SNAPSHOT_SHA256 = (
    "22af9f00506241a4d78918b3b1fc65f1a787281a685c966fc881df99671ee8c4"
)
EXPECTED_FAILED_V2_STDERR_SHA256 = (
    "bf3dd00e4d6e39ce920862e56c04b5c0952e807cd8cc82b6b1a786865d6f8f0b"
)
PREPARED_AT = "2026-10-18T21:23:37+03:00"
REQUIRED_ENVIRONMENT_VARIABLE = "RECALL_STUDY_V3_MANUAL_GPU_AUTHORIZED"

STAGES = (
    ("G_partial_4", "partial_4", "partial4"),
    ("D_focal_loss", "focal", "focal-loss"),
    ("G2_partial_2", "partial_2", "partial2"),
)


def _write_new(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Immutable manual package artifact already exists: {path}")
    write_json(path, value)


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


def prepare() -> Path:
    if PACKAGE_MANIFEST_PATH.exists() or MANUAL_CONFIG_PATH.exists():
        raise FileExistsError("Manual execution V1 is already prepared and immutable")
    protocol = read_json(V3_PROTOCOL_PATH)
    if (
        sha256_file(V3_PROTOCOL_PATH) != EXPECTED_V3_PROTOCOL_SHA256
        or protocol.get("source_snapshot_sha256")
        != EXPECTED_V3_SOURCE_SNAPSHOT_SHA256
    ):
        raise RuntimeError("Frozen V3 protocol identity changed")

    v2_inventory = read_json(V2_INVENTORY_PATH)
    v3_inventory = read_json(V3_INVENTORY_PATH)
    if (
        v3_inventory.get("protocol_sha256") != EXPECTED_V3_PROTOCOL_SHA256
        or v3_inventory.get("study_id")
        != "recall_improvement_study_v3_optimizer_fix"
    ):
        raise RuntimeError("Existing V3 inventory identity changed")

    manual_config = {
        "schema_version": 1,
        "execution_profile": "manual_execution_v1",
        "manual_gpu_execution_authorized": True,
        "required_environment_variable": REQUIRED_ENVIRONMENT_VARIABLE,
        "one_cell_per_process": True,
        "automatic_retry": False,
        "automatic_multi_cell_execution": False,
        "expected_gpu": "NVIDIA RTX 5000 Ada Generation",
        "protocol_path": str(V3_PROTOCOL_PATH.resolve()),
        "protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
        "source_snapshot_sha256": EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
        "prepared_at": PREPARED_AT,
    }
    _write_new(MANUAL_CONFIG_PATH, manual_config)

    order: list[str] = []
    specification_hashes: dict[str, str] = {}
    authorization_hashes: dict[str, str] = {}
    for stage, method, slug in STAGES:
        for fold in range(1, 6):
            cell_id = f"{stage}/fold_{fold}/seed_2718"
            parent_record = v2_inventory["pending_v2_cells"][cell_id]
            parent_spec_path = Path(parent_record["spec_path"]).resolve()
            parent_spec = read_json(parent_spec_path)
            if sha256_file(parent_spec_path) != parent_record["spec_sha256"]:
                raise RuntimeError(f"V2 parent specification changed: {cell_id}")

            if stage == "G_partial_4" and fold == 1:
                spec_path = (
                    V3_ROOT
                    / "cell_specs"
                    / "partial4_fold1_optimizer_fix_attempt.json"
                ).resolve()
                spec = read_json(spec_path)
                attempt_id = spec["attempt_id"]
                if (
                    parent_record.get("status") != "FAILED_NO_AUTOMATIC_RETRY"
                    or sha256_file(
                        Path(spec["replaces_attempt"]["stderr_log_path"])
                    )
                    != EXPECTED_FAILED_V2_STDERR_SHA256
                ):
                    raise RuntimeError("The failed Partial-4 Fold 1 attempt changed")
            else:
                attempt_id = f"manual-v1-{slug}-fold{fold}-seed2718"
                spec_path = (
                    V3_ROOT / "cell_specs" / f"{attempt_id}.json"
                ).resolve()
                run_path = (
                    V3_ROOT
                    / "runs"
                    / stage
                    / f"fold_{fold}"
                    / "seed_2718"
                    / f"attempt_{attempt_id}"
                ).resolve()
                spec = {
                    "attempt_id": attempt_id,
                    "audit_path": str(
                        (
                            V3_ROOT
                            / "audits"
                            / "manual_execution_v1"
                            / f"{attempt_id}.json"
                        ).resolve()
                    ),
                    "cell_id": cell_id,
                    "config_path": parent_spec["config_path"],
                    "execution_profile": "manual_execution_v1",
                    "fold_manifest_sha256": parent_spec["fold_manifest_sha256"],
                    "input_mode": parent_spec["input_mode"],
                    "log_path": str(
                        (
                            V3_ROOT
                            / "logs"
                            / "manual_execution_v1"
                            / attempt_id
                        ).resolve()
                    ),
                    "method": method,
                    "model_path": parent_spec["model_path"],
                    "parent_protocol_sha256": protocol["parent_protocol"]["sha256"],
                    "parent_v2_attempt": {
                        "cell_id": cell_id,
                        "spec_path": str(parent_spec_path),
                        "spec_sha256": sha256_file(parent_spec_path),
                        "status": parent_record["status"],
                    },
                    "protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
                    "run_path": str(run_path),
                    "scientific_spec_sha256": sha256_text(
                        stable_json(_scientific_spec(parent_spec))
                    ),
                    "seed": 2718,
                    "source_snapshot_sha256": EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
                    "study_id": "recall_improvement_study_v3_optimizer_fix",
                    "train_fold": fold,
                }
                _write_new(spec_path, spec)

            spec_sha256 = sha256_file(spec_path)
            if _scientific_spec(spec) != _scientific_spec(parent_spec):
                raise RuntimeError(f"V3 manual spec changed scientific fields: {cell_id}")
            auth_path = (AUTHORIZATION_ROOT / f"{attempt_id}.json").resolve()
            authorization = {
                "schema_version": 1,
                "execution_authorized": True,
                "execution_profile": "manual_execution_v1",
                "cell_id": cell_id,
                "attempt_id": attempt_id,
                "spec_path": str(spec_path),
                "spec_sha256": spec_sha256,
                "required_environment_variable": REQUIRED_ENVIRONMENT_VARIABLE,
                "required_environment_value": "1",
                "one_cell_only": True,
                "automatic_retry": False,
                "authorized_at": PREPARED_AT,
            }
            _write_new(auth_path, authorization)
            auth_sha256 = sha256_file(auth_path)
            record = v3_inventory["attempts"].setdefault(cell_id, {})
            record.update(
                {
                    "attempt_id": attempt_id,
                    "authorization_path": str(auth_path),
                    "authorization_sha256": auth_sha256,
                    "cell_id": cell_id,
                    "completion_manifest_path": str(
                        (
                            V3_ROOT
                            / "completion_manifests"
                            / "manual_execution_v1"
                            / f"{attempt_id}.json"
                        ).resolve()
                    ),
                    "created_at": record.get("created_at", PREPARED_AT),
                    "execution_profile": "manual_execution_v1",
                    "failure_manifest_path": str(
                        (
                            V3_ROOT
                            / "failure_manifests"
                            / "manual_execution_v1"
                            / f"{attempt_id}.json"
                        ).resolve()
                    ),
                    "spec_path": str(spec_path),
                    "spec_sha256": spec_sha256,
                    "start_marker_path": str(
                        (
                            V3_ROOT
                            / "attempt_markers"
                            / "manual_execution_v1"
                            / f"{attempt_id}.started.json"
                        ).resolve()
                    ),
                    "status": record.get(
                        "status", "PENDING_SEPARATE_GPU_AUTHORIZATION"
                    ),
                }
            )
            if record["status"] != "PENDING_SEPARATE_GPU_AUTHORIZATION":
                raise RuntimeError(f"Unexpected existing V3 attempt status: {cell_id}")
            order.append(cell_id)
            specification_hashes[cell_id] = spec_sha256
            authorization_hashes[cell_id] = auth_sha256

    v3_inventory.update(
        {
            "manual_execution_profile": "manual_execution_v1",
            "manual_execution_order": order,
            "manual_execution_prepared_at": PREPARED_AT,
            "automatic_retry": False,
            "automatic_multi_cell_execution": False,
        }
    )
    write_json(V3_INVENTORY_PATH, v3_inventory)

    scripts = {
        name: sha256_file(PACKAGE_ROOT / name)
        for name in (
            "prepare_manual_execution_v1.py",
            "manual_cell_launcher.py",
            "analyze_completed_v1.py",
        )
    }
    package_manifest = {
        "schema_version": 1,
        "execution_profile": "manual_execution_v1",
        "prepared_at": PREPARED_AT,
        "protocol_sha256": EXPECTED_V3_PROTOCOL_SHA256,
        "source_snapshot_sha256": EXPECTED_V3_SOURCE_SNAPSHOT_SHA256,
        "manual_config_sha256": sha256_file(MANUAL_CONFIG_PATH),
        "scripts": scripts,
        "specification_sha256": specification_hashes,
        "authorization_sha256": authorization_hashes,
        "execution_order": order,
        "cell_count": len(order),
        "failed_v2_partial4_fold1_stderr_sha256": (
            EXPECTED_FAILED_V2_STDERR_SHA256
        ),
        "one_cell_per_process": True,
        "automatic_retry": False,
        "automatic_multi_cell_execution": False,
    }
    _write_new(PACKAGE_MANIFEST_PATH, package_manifest)
    return PACKAGE_MANIFEST_PATH


if __name__ == "__main__":
    print(prepare())
