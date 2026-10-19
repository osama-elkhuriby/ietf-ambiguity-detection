from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from models.annotation_v2.cli import validate_manual_eligibility
from models.annotation_v2.manual import (
    EXPECTED_DIGESTS,
    STANDALONE_VERSION,
    assemble_panel,
    verification_failures,
    verify_gpu,
)


def _valid_annotation() -> dict[str, str]:
    return {
        "label": "CLEAR",
        "ambiguity_subtype": "none",
        "evidence_span": "",
        "interpretation_a": "",
        "interpretation_b": "",
        "missing_information": "",
        "justification": "Both choices are explicitly permitted by the target.",
        "context_issue": "none",
    }


def _saved_result(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "version_envelope": {"version": STANDALONE_VERSION},
        "actual_digest": EXPECTED_DIGESTS[model],
        "inference_request_count": 1,
        "retry_limit": 0,
        "status": "VALID",
        "structural_validation": {
            "operational_status": "VALID",
            "annotation": _valid_annotation(),
            "errors": [],
        },
        "resident_gpu_placement_verified": True,
        "frozen_gates_passed": True,
        "api_ps_after_unload": {"models": []},
    }


class _FakeClient:
    def __init__(self) -> None:
        self.resident = False
        self.chat_requests = 0
        self.unloads = 0

    def request(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: int = 30,
    ) -> dict[str, Any]:
        del payload, timeout
        if path == "/api/version":
            return {"version": STANDALONE_VERSION}
        if path == "/api/ps":
            models = []
            if self.resident:
                models = [{
                    "model": "gemma4:31b",
                    "digest": EXPECTED_DIGESTS["gemma4:31b"],
                    "size_vram": 20_000_000_000,
                }]
            return {"models": models}
        if path == "/api/chat":
            self.chat_requests += 1
            self.resident = True
            return {
                "done": True,
                "done_reason": "stop",
                "message": {"content": json.dumps(_valid_annotation())},
                "total_duration": 2_000_000_000,
                "load_duration": 1_000_000_000,
                "prompt_eval_duration": 400_000_000,
                "eval_duration": 600_000_000,
                "prompt_eval_count": 100,
                "eval_count": 50,
            }
        raise AssertionError(f"unexpected endpoint: {path}")

    def tags(self) -> dict[str, dict[str, Any]]:
        return {
            "gemma4:31b": {
                "name": "gemma4:31b",
                "digest": EXPECTED_DIGESTS["gemma4:31b"],
            }
        }

    def show(self, model: str) -> dict[str, Any]:
        assert model == "gemma4:31b"
        return {"model": model, "details": {"quantization_level": "Q4_K_M"}}

    def unload(self, model: str) -> None:
        assert model == "gemma4:31b"
        self.unloads += 1
        self.resident = False


def _snapshot() -> dict[str, Any]:
    return {
        "captured_at": "2026-10-14T00:00:00+00:00",
        "ram": {"available_bytes": 32 * 1024**3},
        "paging": {"used_bytes": 0},
        "gpu": {"used_mib": 20_000, "utilization_percent": 90},
    }


def test_manual_gpu_verification_is_one_request_no_retry_and_unloads(tmp_path: Path) -> None:
    client = _FakeClient()
    server_log = tmp_path / "server.log"
    server_log.write_text(
        'msg="inference compute" library=CUDA total_vram="32 GiB"\n'
        "offloaded 65/65 layers to GPU\n",
        encoding="utf-8",
    )
    config = {
        "model_settings": {
            "gemma4:31b": {
                "num_ctx": 16384,
                "num_predict": 768,
                "reasoning_setting": False,
                "seed": 20261014,
                "temperature": 0.0,
                "timeout_seconds": 300,
            }
        },
        "limits": {
            "ram_reserve_gib": 8.0,
            "heavy_paging_delta_gib": 1.0,
            "projected_pilot_runtime_ceiling_seconds": 21600,
        },
    }

    result = verify_gpu(
        model="gemma4:31b",
        base_url="http://127.0.0.1:11435",
        server_log=server_log,
        output=tmp_path / "gemma-verification",
        config=config,
        client=client,
        snapshot=_snapshot,
        sample_interval=0.001,
    )

    assert result["status"] == "VALID"
    assert result["inference_request_count"] == 1
    assert result["retry_limit"] == 0
    assert result["resident_gpu_placement_verified"] is True
    assert result["frozen_gates_passed"] is True
    assert result["api_ps_after_unload"] == {"models": []}
    assert client.chat_requests == 1
    assert client.unloads == 1
    assert verification_failures(result, "gemma4:31b") == []


def test_verification_gate_rejects_digest_resource_or_semantic_failure() -> None:
    result = _saved_result("gemma4:31b")
    result["actual_digest"] = "wrong"
    result["frozen_gates_passed"] = False
    result["structural_validation"]["annotation"]["label"] = "AMBIGUOUS"

    failures = verification_failures(result, "gemma4:31b")

    assert "model digest mismatch" in failures
    assert "frozen resource gates did not pass" in failures
    assert "clear-discretion fixture was not labeled CLEAR" in failures


def test_saved_evidence_assembly_keeps_gpt_oss_blocked(tmp_path: Path) -> None:
    qwen_path = tmp_path / "qwen.json"
    gemma_path = tmp_path / "gemma.json"
    qwen_path.write_text(json.dumps(_saved_result("qwen3.5:27b")), encoding="utf-8")
    gemma_path.write_text(json.dumps(_saved_result("gemma4:31b")), encoding="utf-8")

    report = assemble_panel(
        output_root=tmp_path / "manual-run",
        qwen_result=qwen_path,
        gemma_result=gemma_path,
        config={"experiment_id": "test-experiment"},
    )

    by_model = {item["model"]: item for item in report["models"]}
    assert report["manual_evidence_assembly"] is True
    assert report["new_inference_requests"] == 0
    assert report["eligible_models"] == ["qwen3.5:27b", "gemma4:31b"]
    assert by_model["gpt-oss:120b"]["status"] == "MODEL_BLOCKED_PREFLIGHT"
    assert report["request_accounting"]["gpt_oss_120b"] == 0


def test_manual_pilot_gate_requires_exact_models_version_and_digests() -> None:
    report = {
        "manual_evidence_assembly": True,
        "ollama_version": STANDALONE_VERSION,
        "models": [
            {
                "model": model,
                "status": "VALID",
                "model_digest": EXPECTED_DIGESTS[model],
            }
            for model in ("qwen3.5:27b", "gemma4:31b")
        ] + [{"model": "gpt-oss:120b", "status": "MODEL_BLOCKED_PREFLIGHT"}],
    }
    tags = {
        model: {"digest": EXPECTED_DIGESTS[model]}
        for model in ("qwen3.5:27b", "gemma4:31b")
    }

    assert validate_manual_eligibility(report, tags, STANDALONE_VERSION) == set(tags)

    with pytest.raises(RuntimeError, match="server version mismatch"):
        validate_manual_eligibility(report, tags, "0.0.0")
    bad_tags = {**tags, "gemma4:31b": {"digest": "wrong"}}
    with pytest.raises(RuntimeError, match="model digest mismatch"):
        validate_manual_eligibility(report, bad_tags, STANDALONE_VERSION)
    incomplete = {**report, "models": report["models"][1:]}
    with pytest.raises(PermissionError, match="requires exactly"):
        validate_manual_eligibility(incomplete, tags, STANDALONE_VERSION)
