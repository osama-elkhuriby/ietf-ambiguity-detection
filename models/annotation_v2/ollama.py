"""Ollama transport, model adapters, hardware gates, and append-only execution."""

from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .constants import (
    ADAPTER_VERSION,
    OUTPUT_SCHEMA,
    RUNNER_VERSION,
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
)
from .data import canonical_json, render_user_payload, sha256_text
from .schema import validate_annotation


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def safe_model_name(name: str) -> str:
    return name.replace(":", "__").replace("/", "_")


def atomic_write_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"append-only artifact already exists: {path}")
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if exclusive and path.exists():
        temp.unlink(missing_ok=True)
        raise FileExistsError(f"append-only artifact already exists: {path}")
    temp.replace(path)


@dataclass(frozen=True)
class ModelAdapter:
    model: str
    reasoning_setting: bool | str
    temperature: float
    seed: int
    num_ctx: int
    num_predict: int
    timeout_seconds: int

    def request(self, user_payload: dict[str, Any], validation_error: str = "") -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False, indent=2),
            },
        ]
        if validation_error:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous independent attempt failed structural validation: "
                        f"{validation_error[:500]}. Start again from the original bounded "
                        "input and return one corrected JSON object only."
                    ),
                }
            )
        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": self.reasoning_setting,
            "format": OUTPUT_SCHEMA,
            "keep_alive": "5m",
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def request(
        self, path: str, payload: dict[str, Any] | None = None, timeout: int = 30
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=None if payload is None else canonical_json(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:
            raise TimeoutError(f"Ollama request exceeded {timeout}s") from exc
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError(f"Ollama request exceeded {timeout}s") from exc
            raise ConnectionError(f"Ollama transport error: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON envelope") from exc

    def tags(self) -> dict[str, dict[str, Any]]:
        response = self.request("/api/tags")
        return {
            str(item.get("name") or item.get("model")): item
            for item in response.get("models", [])
        }

    def show(self, model: str) -> dict[str, Any]:
        return self.request("/api/show", {"model": model})

    def unload(self, model: str) -> None:
        self.request(
            "/api/generate",
            {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=60,
        )


def _nvidia_snapshot() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        values = [part.strip() for part in completed.stdout.strip().split(",")]
        return {
            "name": values[0],
            "total_mib": int(values[1]),
            "free_mib": int(values[2]),
            "used_mib": int(values[3]),
            "utilization_percent": int(values[4]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        return {"error": str(exc)}


def hardware_snapshot() -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        ram = {
            "total_bytes": memory.total,
            "available_bytes": memory.available,
            "percent_used": memory.percent,
        }
        paging = {
            "total_bytes": swap.total,
            "used_bytes": swap.used,
            "percent_used": swap.percent,
        }
    except ImportError:
        ram = {"error": "psutil unavailable"}
        paging = {"error": "psutil unavailable"}
    return {"captured_at": utc_now(), "ram": ram, "paging": paging, "gpu": _nvidia_snapshot()}


def assess_feasibility(
    model: str,
    tag_metadata: dict[str, Any],
    hardware: dict[str, Any],
    limits: dict[str, Any],
) -> dict[str, Any]:
    size = int(tag_metadata.get("size", 0))
    ram_available = int(hardware.get("ram", {}).get("available_bytes", 0))
    vram_free = int(hardware.get("gpu", {}).get("free_mib", 0)) * 1024 * 1024
    reserve = int(float(limits["ram_reserve_gib"]) * 1024**3)
    staging = int(size * float(limits["weight_staging_fraction"]))
    buffers = int(float(limits["runtime_buffer_gib"]) * 1024**3)
    context_cache = int(float(limits["context_cache_gib"]) * 1024**3)
    combined_needed = size + staging + buffers + context_cache + reserve
    combined_available = ram_available + vram_free
    host_needed = staging + buffers + context_cache + reserve
    reasons: list[str] = []
    if not ram_available or not vram_free:
        reasons.append("RAM or VRAM monitoring unavailable")
    if host_needed > ram_available:
        reasons.append("available RAM cannot preserve reserve plus staging/runtime allowance")
    if combined_needed > combined_available:
        reasons.append("combined available RAM/VRAM is below conservative peak estimate")
    if size > vram_free and (size - vram_free) > max(0, ram_available - reserve):
        reasons.append("offloaded weight remainder exceeds RAM after reserve")
    return {
        "model": model,
        "model_size_bytes": size,
        "assumptions": {
            "weight_residency_bytes": size,
            "possible_host_staging_bytes": staging,
            "runtime_buffer_bytes": buffers,
            "context_cache_bytes": context_cache,
            "os_application_reserve_bytes": reserve,
            "peak_combined_estimate_bytes": combined_needed,
            "available_combined_bytes": combined_available,
            "full_host_weight_copy_not_assumed": True,
            "uncertainty": "Ollama allocation and driver staging can exceed this estimate.",
        },
        "feasible_for_bounded_probe": not reasons,
        "block_reasons": reasons,
    }


def post_probe_block_reasons(
    before: dict[str, Any],
    after: dict[str, Any],
    wall_seconds: float,
    limits: dict[str, Any],
) -> list[str]:
    """Apply frozen post-load safety gates to measured probe observations."""
    reasons: list[str] = []
    reserve = int(float(limits["ram_reserve_gib"]) * 1024**3)
    available_after = int(after.get("ram", {}).get("available_bytes", 0))
    if not available_after:
        reasons.append("post-probe RAM monitoring unavailable")
    elif available_after < reserve:
        reasons.append("post-probe available RAM fell below frozen reserve")
    paging_delta = int(after.get("paging", {}).get("used_bytes", 0)) - int(
        before.get("paging", {}).get("used_bytes", 0)
    )
    paging_limit = int(float(limits["heavy_paging_delta_gib"]) * 1024**3)
    if paging_delta > paging_limit:
        reasons.append("heavy paging threshold exceeded")
    projected = wall_seconds * 24
    if projected > float(limits["projected_pilot_runtime_ceiling_seconds"]):
        reasons.append("projected pilot runtime exceeds frozen ceiling")
    return reasons


def cache_identity(
    *,
    model_digest: str,
    adapter: ModelAdapter,
    context_variant: str,
    user_payload: dict[str, Any],
    retrieval_manifest_hash: str,
    template_hash: str,
    ollama_version: str,
) -> dict[str, Any]:
    identity = {
        "model_digest": model_digest,
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "rendered_user_sha256": sha256_text(canonical_json(user_payload)),
        "output_schema_sha256": sha256_text(canonical_json(OUTPUT_SCHEMA)),
        "schema_version": SCHEMA_VERSION,
        "context_variant": context_variant,
        "full_input_fingerprint": sha256_text(canonical_json(user_payload)),
        "retrieval_manifest_sha256": retrieval_manifest_hash,
        "template_sha256": template_hash,
        "ollama_version": ollama_version,
        "adapter_version": ADAPTER_VERSION,
        "runner_version": RUNNER_VERSION,
        "reasoning_setting": adapter.reasoning_setting,
        "temperature": adapter.temperature,
        "seed": adapter.seed,
        "num_ctx": adapter.num_ctx,
        "num_predict": adapter.num_predict,
    }
    return {**identity, "cache_key": sha256_text(canonical_json(identity))}


def classify_envelope(
    envelope: dict[str, Any], target: str
) -> tuple[str, dict[str, Any] | None, list[str]]:
    done_reason = str(envelope.get("done_reason", ""))
    if done_reason in {"length", "max_tokens"}:
        return "TOKEN_EXHAUSTED", None, [f"done_reason={done_reason}"]
    message = envelope.get("message") or {}
    content = str(message.get("content", "")).strip()
    if not content:
        if str(message.get("refusal", "")).strip():
            return "REFUSAL", None, ["model refused"]
        return "MALFORMED_OUTPUT", None, ["empty final content"]
    try:
        annotation = json.loads(content)
    except json.JSONDecodeError as exc:
        return "MALFORMED_OUTPUT", None, [f"invalid JSON: {exc.msg}"]
    errors = validate_annotation(annotation, target)
    if errors:
        return "INVALID_OUTPUT", None, errors
    return "VALID", annotation, []


def execute_cell(
    *,
    client: OllamaClient,
    adapter: ModelAdapter,
    model_digest: str,
    model_metadata: dict[str, Any],
    bundle: dict[str, Any],
    output_root: Path,
    experiment_id: str,
    ollama_version: str,
    max_retries: int,
) -> dict[str, Any]:
    user_payload = render_user_payload(bundle)
    identity = cache_identity(
        model_digest=model_digest,
        adapter=adapter,
        context_variant=bundle["variant"],
        user_payload=user_payload,
        retrieval_manifest_hash=sha256_text(canonical_json({
            "items": bundle["items"], "retrieval": bundle["retrieval"], "omitted": bundle["omitted"]
        })),
        template_hash=sha256_text(str(model_metadata.get("template", ""))),
        ollama_version=ollama_version,
    )
    cell_dir = output_root / "cells" / safe_model_name(adapter.model) / bundle["variant"] / bundle["record_id"]
    accepted_path = cell_dir / "accepted.json"
    if accepted_path.is_file():
        accepted = json.loads(accepted_path.read_text(encoding="utf-8"))
        if accepted.get("cache_identity", {}).get("cache_key") == identity["cache_key"]:
            return accepted

    last_error = ""
    result: dict[str, Any] | None = None
    for attempt_number in range(1, max_retries + 2):
        request_payload = adapter.request(user_payload, last_error)
        request_sha = sha256_text(canonical_json(request_payload))
        attempt_path = cell_dir / "attempts" / f"attempt-{attempt_number:03d}.json"
        if attempt_path.exists():
            existing = json.loads(attempt_path.read_text(encoding="utf-8"))
            if existing.get("request_sha256") != request_sha:
                raise RuntimeError(f"append-only attempt identity conflict: {attempt_path}")
            operational = existing["operational_status"]
            annotation = existing.get("annotation")
            errors = existing.get("validation_errors", [])
        else:
            started = time.monotonic()
            before = hardware_snapshot()
            try:
                envelope = client.request("/api/chat", request_payload, adapter.timeout_seconds)
                operational, annotation, errors = classify_envelope(envelope, bundle["target_clause"])
            except TimeoutError as exc:
                envelope, operational, annotation, errors = {}, "TIMEOUT", None, [str(exc)]
            except ConnectionError as exc:
                envelope, operational, annotation, errors = {}, "TRANSPORT_ERROR", None, [str(exc)]
            except RuntimeError as exc:
                envelope, operational, annotation, errors = {}, "TRANSPORT_ERROR", None, [str(exc)]
            after = hardware_snapshot()
            attempt = {
                "experiment_id": experiment_id,
                "record_id": bundle["record_id"],
                "model": adapter.model,
                "model_digest": model_digest,
                "context_variant": bundle["variant"],
                "attempt_number": attempt_number,
                "request": request_payload,
                "request_sha256": request_sha,
                "response_envelope": envelope,
                "api_reasoning": (envelope.get("message") or {}).get("thinking", ""),
                "annotation": annotation,
                "semantic_label": annotation.get("label") if annotation else None,
                "operational_status": operational,
                "validation_errors": errors,
                "done_reason": envelope.get("done_reason"),
                "prompt_eval_count": envelope.get("prompt_eval_count"),
                "eval_count": envelope.get("eval_count"),
                "load_duration_ns": envelope.get("load_duration"),
                "prompt_eval_duration_ns": envelope.get("prompt_eval_duration"),
                "eval_duration_ns": envelope.get("eval_duration"),
                "total_duration_ns": envelope.get("total_duration"),
                "wall_seconds": round(time.monotonic() - started, 3),
                "hardware_before": before,
                "hardware_after": after,
                "cache_identity": identity,
                "created_at": utc_now(),
            }
            atomic_write_json(attempt_path, attempt, exclusive=True)
        result = {
            "experiment_id": experiment_id,
            "record_id": bundle["record_id"],
            "family_id": bundle["family_id"],
            "document_id": bundle["document_id"],
            "revision": bundle["revision"],
            "model": adapter.model,
            "model_digest": model_digest,
            "context_variant": bundle["variant"],
            "semantic_label": annotation.get("label") if annotation else None,
            "operational_status": operational,
            "annotation": annotation,
            "accepted_attempt": (
                attempt_path.relative_to(output_root).as_posix() if operational == "VALID" else None
            ),
            "attempt_count": attempt_number,
            "cache_identity": identity,
        }
        if operational == "VALID":
            break
        last_error = "; ".join(errors)
    assert result is not None
    atomic_write_json(accepted_path, result)
    return result
