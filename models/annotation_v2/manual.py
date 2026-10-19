"""Manual-only commands for standalone annotation-v2 execution.

Nothing in this module starts or stops Ollama. Live commands are deliberately
single-purpose so an operator must invoke each expensive stage explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .cli import DEFAULT_EXPERIMENT_ROOT
from .data import file_sha256
from .ollama import (
    ModelAdapter,
    OllamaClient,
    atomic_write_json,
    classify_envelope,
    hardware_snapshot,
    post_probe_block_reasons,
    utc_now,
)

STANDALONE_VERSION = "0.40.4"
ALLOWED_LOCAL_MODELS = ("qwen3.5:27b", "gemma4:31b")
EXPECTED_DIGESTS = {
    "qwen3.5:27b": "64ae5b1d651fb2709d5928ccb8dcfb4bb2cc9c2c8001ac34497e36a6cf905b59",
    "gemma4:31b": "17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97",
}
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _monitor_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def values(section: str, field: str) -> list[int]:
        return [
            int(sample[section][field])
            for sample in samples
            if isinstance(sample.get(section, {}).get(field), int)
        ]

    ram = values("ram", "available_bytes")
    paging = values("paging", "used_bytes")
    gpu = values("gpu", "used_mib")
    utilization = values("gpu", "utilization_percent")
    return {
        "sample_count": len(samples),
        "minimum_available_ram_bytes": min(ram) if ram else None,
        "maximum_paging_used_bytes": max(paging) if paging else None,
        "maximum_gpu_used_mib": max(gpu) if gpu else None,
        "maximum_gpu_utilization_percent": max(utilization) if utilization else None,
    }


def verification_failures(result: dict[str, Any], model: str) -> list[str]:
    """Return deterministic eligibility failures for a saved synthetic result."""
    failures: list[str] = []
    if result.get("model") != model:
        failures.append("model identity mismatch")
    if result.get("version_envelope", {}).get("version") != STANDALONE_VERSION:
        failures.append("standalone version mismatch")
    if result.get("actual_digest") != EXPECTED_DIGESTS[model]:
        failures.append("model digest mismatch")
    if result.get("inference_request_count") != 1:
        failures.append("verification did not use exactly one inference request")
    if result.get("retry_limit") != 0:
        failures.append("verification retry limit was not zero")
    if result.get("status") != "VALID":
        failures.append("verification status is not VALID")
    validation = result.get("structural_validation", {})
    if validation.get("operational_status") != "VALID" or validation.get("errors"):
        failures.append("response failed structural validation")
    if validation.get("annotation", {}).get("label") != "CLEAR":
        failures.append("clear-discretion fixture was not labeled CLEAR")
    if not result.get("resident_gpu_placement_verified"):
        failures.append("resident GPU placement was not verified")
    if not result.get("frozen_gates_passed"):
        failures.append("frozen resource gates did not pass")
    if result.get("api_ps_after_unload", {}).get("models"):
        failures.append("model remained resident after verification unload")
    return failures


def _allocation_evidence(server_log: Path) -> list[str]:
    if not server_log.is_file():
        return []
    needles = (
        "inference compute",
        "total_vram",
        "offloaded",
        "CUDA0 model buffer",
        "CUDA_Host model buffer",
        "CUDA0 KV buffer",
        "CUDA0 compute buffer",
        "prompt eval time",
        "eval time",
        "total time",
    )
    return [
        line
        for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(needle in line for needle in needles)
    ]


def verify_gpu(
    *,
    model: str,
    base_url: str,
    server_log: Path,
    output: Path,
    config: dict[str, Any],
    client: OllamaClient | None = None,
    snapshot: Callable[[], dict[str, Any]] = hardware_snapshot,
    sample_interval: float = 1.0,
) -> dict[str, Any]:
    """Execute exactly one synthetic request against an already-running server."""
    if model not in ALLOWED_LOCAL_MODELS:
        raise PermissionError(f"local verification is not allowed for {model}")
    if output.exists():
        raise FileExistsError(f"diagnostic namespace already exists: {output}")
    if not server_log.is_file():
        raise FileNotFoundError(f"standalone server log is missing: {server_log}")
    output.mkdir(parents=True)

    active_client = client or OllamaClient(base_url)
    settings = config["model_settings"][model]
    limits = config["limits"]
    adapter = ModelAdapter(model=model, **settings)
    request_payload = adapter.request(SYNTHETIC_FIXTURE)
    atomic_write_json(output / "request.json", request_payload, exclusive=True)

    model_store_setting = os.environ.get("OLLAMA_MODELS")
    model_store = Path(model_store_setting or Path.home() / ".ollama" / "models").resolve()
    samples: list[dict[str, Any]] = []
    stop_monitor = threading.Event()
    inference_requests = 0
    unload_controls = 0
    attempted = False
    result: dict[str, Any] = {
        "diagnostic_version": "manual-gpu-placement-v1",
        "started_at": utc_now(),
        "base_url": base_url,
        "model": model,
        "expected_digest": EXPECTED_DIGESTS[model],
        "model_store": str(model_store),
        "model_store_from_environment": model_store_setting is not None,
        "server_log": str(server_log.resolve()),
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
        while not stop_monitor.is_set():
            samples.append(snapshot())
            stop_monitor.wait(sample_interval)

    try:
        version = active_client.request("/api/version")
        result["version_envelope"] = version
        if version.get("version") != STANDALONE_VERSION:
            raise RuntimeError(f"expected Ollama {STANDALONE_VERSION}, got {version}")
        startup_log = server_log.read_text(encoding="utf-8", errors="replace")
        if "library=CUDA" not in startup_log or "total_vram=\"0 B\"" in startup_log:
            raise RuntimeError("server log does not prove CUDA discovery with nonzero VRAM")
        before_ps = active_client.request("/api/ps")
        result["api_ps_before"] = before_ps
        if before_ps.get("models"):
            raise RuntimeError("a model is already resident; unload it before verification")

        tags = active_client.tags()
        tag = tags.get(model)
        if tag is None:
            raise RuntimeError(f"model is absent from existing store: {model_store}")
        result["model_tag"] = tag
        result["actual_digest"] = str(tag.get("digest", ""))
        if result["actual_digest"] != EXPECTED_DIGESTS[model]:
            raise RuntimeError(
                f"digest mismatch: expected {EXPECTED_DIGESTS[model]}, "
                f"got {result['actual_digest']}"
            )
        atomic_write_json(output / "model_show.json", active_client.show(model), exclusive=True)

        before = snapshot()
        result["hardware_before"] = before
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        started = time.monotonic()
        attempted = True
        inference_requests = 1
        try:
            envelope = active_client.request("/api/chat", request_payload, adapter.timeout_seconds)
        finally:
            wall_seconds = time.monotonic() - started
            stop_monitor.set()
            monitor_thread.join(timeout=20)

        after = snapshot()
        resident_ps = active_client.request("/api/ps")
        atomic_write_json(output / "response_envelope.json", envelope, exclusive=True)
        operational, annotation, errors = classify_envelope(
            envelope, SYNTHETIC_FIXTURE["target_clause"]
        )
        result.update(
            {
                "hardware_after_resident": after,
                "api_ps_resident": resident_ps,
                "wall_seconds": wall_seconds,
                "structural_validation": {
                    "operational_status": operational,
                    "annotation": annotation,
                    "errors": errors,
                },
                "timings_ns": {
                    "total_duration": envelope.get("total_duration"),
                    "load_duration": envelope.get("load_duration"),
                    "prompt_eval_duration": envelope.get("prompt_eval_duration"),
                    "eval_duration": envelope.get("eval_duration"),
                },
                "token_counts": {
                    "prompt_eval_count": envelope.get("prompt_eval_count"),
                    "eval_count": envelope.get("eval_count"),
                },
                "monitor_summary": _monitor_summary(samples),
                "frozen_gate_reasons": post_probe_block_reasons(
                    before, after, wall_seconds, limits
                ),
                "resident_gpu_placement_verified": any(
                    item.get("digest") == EXPECTED_DIGESTS[model]
                    and int(item.get("size_vram", 0)) > 0
                    for item in resident_ps.get("models", [])
                ),
            }
        )
        result["frozen_gates_passed"] = not result["frozen_gate_reasons"]
        result["allocation_log_evidence"] = _allocation_evidence(server_log)
        semantic_adherence = (
            operational == "VALID" and annotation is not None and annotation.get("label") == "CLEAR"
        )
        result["synthetic_clear_adherence"] = semantic_adherence
        result["status"] = "VALID" if semantic_adherence else "FAILED"
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        result["status"] = "FAILED"
        result["failure"] = f"{type(exc).__name__}: {exc}"
    finally:
        stop_monitor.set()
        if attempted:
            try:
                active_client.unload(model)
                unload_controls = 1
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    after_unload_ps = active_client.request("/api/ps")
                    if not after_unload_ps.get("models"):
                        break
                    time.sleep(0.25)
                else:
                    raise TimeoutError("model remained resident after unload")
                result["api_ps_after_unload"] = after_unload_ps
                result["hardware_after_unload"] = snapshot()
            except (OSError, RuntimeError) as exc:
                result["unload_failure"] = f"{type(exc).__name__}: {exc}"
                result["status"] = "FAILED"
        result["inference_request_count"] = inference_requests
        result["unload_control_request_count"] = unload_controls
        result["monitor_samples"] = samples
        result["completed_at"] = utc_now()
        atomic_write_json(output / "result.json", result, exclusive=True)
    return result


def inspect_verification(path: Path) -> dict[str, Any]:
    result = _load_json(path)
    model = str(result.get("model", ""))
    failures = verification_failures(result, model) if model in EXPECTED_DIGESTS else [
        "unsupported model identity"
    ]
    timings = result.get("timings_ns", {})
    summary = result.get("monitor_summary", {})
    resident = (result.get("api_ps_resident", {}).get("models") or [{}])[0]
    return {
        "result_path": str(path.resolve()),
        "model": model,
        "status": result.get("status"),
        "eligible": not failures,
        "eligibility_failures": failures,
        "digest": result.get("actual_digest"),
        "semantic_label": result.get("structural_validation", {})
        .get("annotation", {})
        .get("label"),
        "gpu_size_vram_bytes": resident.get("size_vram"),
        "maximum_gpu_used_mib": summary.get("maximum_gpu_used_mib"),
        "minimum_available_ram_bytes": summary.get("minimum_available_ram_bytes"),
        "load_seconds": (timings.get("load_duration") or 0) / 1e9,
        "prompt_eval_seconds": (timings.get("prompt_eval_duration") or 0) / 1e9,
        "generation_seconds": (timings.get("eval_duration") or 0) / 1e9,
        "wall_seconds": result.get("wall_seconds"),
        "frozen_gate_reasons": result.get("frozen_gate_reasons", []),
    }


def assemble_panel(
    *,
    output_root: Path,
    qwen_result: Path,
    gemma_result: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Assemble an annotation eligibility gate entirely from saved evidence."""
    output_root.mkdir(parents=True, exist_ok=True)
    eligibility_path = output_root / "manual_eligibility_report.json"
    final_path = output_root / "preflight_report_final.json"
    if eligibility_path.exists() or final_path.exists():
        raise FileExistsError("manual panel evidence has already been assembled")

    evidence_paths = {
        "qwen3.5:27b": qwen_result.resolve(),
        "gemma4:31b": gemma_result.resolve(),
    }
    models: list[dict[str, Any]] = []
    all_valid = True
    for model in ALLOWED_LOCAL_MODELS:
        path = evidence_paths[model]
        result = _load_json(path)
        failures = verification_failures(result, model)
        all_valid = all_valid and not failures
        models.append(
            {
                "model": model,
                "status": "VALID" if not failures else "MODEL_BLOCKED_PREFLIGHT",
                "model_digest": EXPECTED_DIGESTS[model],
                "evidence_path": str(path),
                "evidence_sha256": file_sha256(path),
                "eligibility_failures": failures,
            }
        )
    models.append(
        {
            "model": "gpt-oss:120b",
            "status": "MODEL_BLOCKED_PREFLIGHT",
            "model_digest": "714d86958b6e870a9320e78126ac1ce3477727a1cd9bf7194b8ace0d84c62fd8",
            "eligibility_failures": [
                "locally blocked by the frozen feasibility decision; no fallback or substitution"
            ],
        }
    )
    eligibility = {
        "created_at": utc_now(),
        "experiment_id": config["experiment_id"],
        "standalone_version": STANDALONE_VERSION,
        "manual_evidence_assembly": True,
        "models": models,
        "eligible_models": [entry["model"] for entry in models if entry["status"] == "VALID"],
        "new_inference_requests": 0,
    }
    atomic_write_json(eligibility_path, eligibility, exclusive=True)
    if not all_valid:
        raise RuntimeError(
            f"manual panel is not eligible; inspect {eligibility_path} before continuing"
        )

    final_report = {
        **eligibility,
        "ollama_version": STANDALONE_VERSION,
        "inference_request_count": 2,
        "request_accounting": {
            "qwen_saved_verification": 1,
            "gemma_manual_verification": 1,
            "gpt_oss_120b": 0,
            "assembly_requests": 0,
        },
        "notes": [
            "Qwen evidence was reused and not regenerated.",
            "Gemma evidence must be produced by one manually executed synthetic request.",
            "GPT-OSS-120B remains locally blocked and cannot be substituted.",
            "This eligibility gate does not establish annotation accuracy.",
        ],
    }
    atomic_write_json(final_path, final_report, exclusive=True)
    return final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "configs/requested_panel_v2.json",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    verify = sub.add_parser("verify-gpu")
    verify.add_argument("--model", choices=("gemma4:31b",), required=True)
    verify.add_argument("--base-url", default="http://127.0.0.1:11435")
    verify.add_argument("--server-log", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)

    inspect = sub.add_parser("inspect-verification")
    inspect.add_argument("--result", type=Path, required=True)

    assemble = sub.add_parser("assemble-panel")
    assemble.add_argument("--output-root", type=Path, required=True)
    assemble.add_argument("--qwen-result", type=Path, required=True)
    assemble.add_argument("--gemma-result", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_json(args.config.resolve())
    if args.command == "verify-gpu":
        result = verify_gpu(
            model=args.model,
            base_url=args.base_url,
            server_log=args.server_log.resolve(),
            output=args.output.resolve(),
            config=config,
        )
        printable = inspect_verification(args.output.resolve() / "result.json")
        print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") == "VALID" else 1
    if args.command == "inspect-verification":
        result = inspect_verification(args.result.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result["eligible"] else 1
    result = assemble_panel(
        output_root=args.output_root.resolve(),
        qwen_result=args.qwen_result.resolve(),
        gemma_result=args.gemma_result.resolve(),
        config=config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
