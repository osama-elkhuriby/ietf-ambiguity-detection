"""Run one bounded synthetic Qwen GPU-placement verification against Ollama."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from models.annotation_v2.ollama import (
    ModelAdapter,
    OllamaClient,
    atomic_write_json,
    classify_envelope,
    hardware_snapshot,
    post_probe_block_reasons,
    utc_now,
)

MODEL = "qwen3.5:27b"
EXPECTED_DIGEST = "64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59"
EXPECTED_VERSION = "0.40.4"
BASE_URL = "http://127.0.0.1:11435"
SYNTHETIC_FIXTURE = {
    "task_scope": "Synthetic hardware/schema probe; no corpus data.",
    "document_id": "synthetic-v2",
    "revision": "00",
    "section_id": "1",
    "section_heading": "Synthetic",
    "context_variant": "local",
    "coverage_warning": "synthetic probe only",
    "retrieval_status": [],
    "omitted_for_budget": [],
    "context_items": [],
    "target_clause": "An endpoint MAY select either explicitly permitted option.",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_server(client: OllamaClient, log_path: Path, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    version: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        try:
            version = client.request("/api/version", timeout=2)
        except (ConnectionError, RuntimeError, TimeoutError):
            time.sleep(0.25)
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "inference compute" in log_text:
            return version
        time.sleep(0.25)
    raise TimeoutError("standalone server did not complete GPU discovery within 120 seconds")


def summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid_ram = [
        item["ram"]["available_bytes"]
        for item in samples
        if isinstance(item.get("ram", {}).get("available_bytes"), int)
    ]
    valid_paging = [
        item["paging"]["used_bytes"]
        for item in samples
        if isinstance(item.get("paging", {}).get("used_bytes"), int)
    ]
    valid_gpu = [
        item["gpu"]["used_mib"]
        for item in samples
        if isinstance(item.get("gpu", {}).get("used_mib"), int)
    ]
    valid_utilization = [
        item["gpu"]["utilization_percent"]
        for item in samples
        if isinstance(item.get("gpu", {}).get("utilization_percent"), int)
    ]
    return {
        "sample_count": len(samples),
        "minimum_available_ram_bytes": min(valid_ram) if valid_ram else None,
        "maximum_paging_used_bytes": max(valid_paging) if valid_paging else None,
        "maximum_gpu_used_mib": max(valid_gpu) if valid_gpu else None,
        "maximum_gpu_utilization_percent": max(valid_utilization) if valid_utilization else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.runtime.resolve()
    executable = runtime / "ollama.exe"
    if not executable.is_file():
        raise FileNotFoundError(executable)
    if args.output.exists():
        raise FileExistsError(f"diagnostic namespace already exists: {args.output}")
    args.output.mkdir(parents=True)

    config = load_json(args.config)
    settings = config["model_settings"][MODEL]
    limits = config["limits"]
    adapter = ModelAdapter(model=MODEL, **settings)
    request_payload = adapter.request(SYNTHETIC_FIXTURE)
    atomic_write_json(args.output / "request.json", request_payload, exclusive=True)

    model_store_setting = os.environ.get("OLLAMA_MODELS")
    model_store = Path(model_store_setting or Path.home() / ".ollama" / "models").resolve()
    server_log_path = args.output / "server.log"
    monitor_samples: list[dict[str, Any]] = []
    monitor_stop = threading.Event()
    server: subprocess.Popen[str] | None = None
    inference_requests = 0
    unload_control_requests = 0
    result: dict[str, Any] = {
        "diagnostic_version": "gpu-placement-v1",
        "started_at": utc_now(),
        "runtime": str(runtime),
        "base_url": BASE_URL,
        "model": MODEL,
        "expected_digest": EXPECTED_DIGEST,
        "model_store": str(model_store),
        "model_store_from_environment": model_store_setting is not None,
        "inference_request_limit": 1,
        "retry_limit": 0,
        "settings": settings,
        "limits": {
            "ram_reserve_gib": limits["ram_reserve_gib"],
            "heavy_paging_delta_gib": limits["heavy_paging_delta_gib"],
            "projected_pilot_runtime_ceiling_seconds": limits[
                "projected_pilot_runtime_ceiling_seconds"
            ],
        },
    }

    def monitor() -> None:
        while not monitor_stop.is_set():
            monitor_samples.append(hardware_snapshot())
            monitor_stop.wait(1.0)

    try:
        environment = os.environ.copy()
        environment["OLLAMA_HOST"] = "127.0.0.1:11435"
        with server_log_path.open("x", encoding="utf-8") as server_log:
            server = subprocess.Popen(
                [str(executable), "serve"],
                cwd=runtime,
                env=environment,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

            client = OllamaClient(BASE_URL)
            version_envelope = wait_for_server(client, server_log_path)
            result["version_envelope"] = version_envelope
            if version_envelope.get("version") != EXPECTED_VERSION:
                raise RuntimeError(f"unexpected standalone version: {version_envelope}")

            discovery_log = server_log_path.read_text(encoding="utf-8", errors="replace")
            result["cuda_discovery_verified"] = (
                "library=CUDA" in discovery_log
                and "total_vram=\"0 B\"" not in discovery_log
                and "total_vram" in discovery_log
            )
            if not result["cuda_discovery_verified"]:
                raise RuntimeError("standalone startup did not report CUDA with nonzero VRAM")

            initial_ps = client.request("/api/ps")
            result["api_ps_before"] = initial_ps
            if initial_ps.get("models"):
                raise RuntimeError("a model was resident before the bounded request")

            tags = client.tags()
            tag = tags.get(MODEL)
            if tag is None:
                raise RuntimeError(f"existing model tag is unavailable in {model_store}")
            result["model_tag"] = tag
            actual_digest = str(tag.get("digest", ""))
            result["actual_digest"] = actual_digest
            if actual_digest != EXPECTED_DIGEST:
                raise RuntimeError(
                    f"model digest mismatch: expected {EXPECTED_DIGEST}, got {actual_digest}"
                )

            shown = client.show(MODEL)
            atomic_write_json(args.output / "model_show.json", shown, exclusive=True)

            before = hardware_snapshot()
            result["hardware_before"] = before
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            started = time.monotonic()
            inference_requests += 1
            try:
                envelope = client.request("/api/chat", request_payload, adapter.timeout_seconds)
            finally:
                wall_seconds = time.monotonic() - started
                monitor_stop.set()
                monitor_thread.join(timeout=20)

            after_resident = hardware_snapshot()
            resident_ps = client.request("/api/ps")
            result["hardware_after_resident"] = after_resident
            result["api_ps_resident"] = resident_ps
            result["wall_seconds"] = wall_seconds
            atomic_write_json(args.output / "response_envelope.json", envelope, exclusive=True)

            operational, annotation, errors = classify_envelope(
                envelope, SYNTHETIC_FIXTURE["target_clause"]
            )
            result["structural_validation"] = {
                "operational_status": operational,
                "annotation": annotation,
                "errors": errors,
            }
            result["timings_ns"] = {
                "total_duration": envelope.get("total_duration"),
                "load_duration": envelope.get("load_duration"),
                "prompt_eval_duration": envelope.get("prompt_eval_duration"),
                "eval_duration": envelope.get("eval_duration"),
            }
            result["token_counts"] = {
                "prompt_eval_count": envelope.get("prompt_eval_count"),
                "eval_count": envelope.get("eval_count"),
            }
            result["monitor_summary"] = summarize_samples(monitor_samples)
            result["frozen_gate_reasons"] = post_probe_block_reasons(
                before, after_resident, wall_seconds, limits
            )
            result["frozen_gates_passed"] = not result["frozen_gate_reasons"]
            result["resident_gpu_placement_verified"] = any(
                int(item.get("size_vram", 0)) > 0 for item in resident_ps.get("models", [])
            )

            client.unload(MODEL)
            unload_control_requests += 1
            unload_deadline = time.monotonic() + 30
            while time.monotonic() < unload_deadline:
                unloaded_ps = client.request("/api/ps")
                if not unloaded_ps.get("models"):
                    break
                time.sleep(0.25)
            else:
                raise TimeoutError("Qwen remained resident after explicit unload")
            result["api_ps_after_unload"] = unloaded_ps
            result["hardware_after_unload"] = hardware_snapshot()
            result["status"] = "VALID" if operational == "VALID" else operational
    except (KeyError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        result["status"] = "FAILED"
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        monitor_stop.set()
        result["inference_request_count"] = inference_requests
        result["unload_control_request_count"] = unload_control_requests
        result["monitor_samples"] = monitor_samples
        result["completed_at"] = utc_now()
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        result["candidate_server_exit_code"] = server.returncode if server is not None else None
        atomic_write_json(args.output / "result.json", result, exclusive=True)

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
