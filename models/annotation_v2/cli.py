"""Command-line interface for the separately versioned annotation-v2 experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .aggregation import conservative_aggregate, v1_compatible_comparison
from .constants import (
    ADAPTER_VERSION,
    AGGREGATION_VERSION,
    CONTEXT_BUILDER_VERSION,
    EXPERIMENT_ID,
    OUTPUT_SCHEMA,
    RUBRIC_VERSION,
    RUNNER_VERSION,
    SCHEMA_VERSION,
    SYNTHETIC_EXAMPLES,
    SYSTEM_PROMPT,
)
from .data import (
    ContextBuilder,
    file_sha256,
    load_allowed_families,
    load_development_rows,
    load_v1_votes_for_allowed_rows,
    render_user_payload,
    select_review_sample,
    sha256_text,
)
from .ollama import (
    ModelAdapter,
    OllamaClient,
    assess_feasibility,
    atomic_write_json,
    classify_envelope,
    execute_cell,
    hardware_snapshot,
    post_probe_block_reasons,
    safe_model_name,
    utc_now,
)
from .schema import validate_annotation

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = ROOT / "experiments/annotation_v2"
DEFAULT_OUTPUT_ROOT = ROOT / "corpus/processed/llm_annotations/v2/requested-panel"
FOLDS_PATH = ROOT / "experiments/transformer/configs/folds_v1.json"
GROUND_TRUTH_ROOT = ROOT / "corpus/processed/ground_truth"
ALIGNMENT_ROOT = ROOT / "corpus/processed/alignments"
V1_RESPONSE_ROOT = ROOT / "corpus/processed/llm_annotations/v1/responses"

REQUESTED_MODELS = ("qwen3.5:27b", "gemma4:31b", "gpt-oss:120b")
MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "qwen3.5:27b": {"reasoning_setting": False, "timeout_seconds": 300},
    "gemma4:31b": {"reasoning_setting": False, "timeout_seconds": 300},
    "gpt-oss:120b": {"reasoning_setting": "low", "timeout_seconds": 600},
}
LIMITS = {
    "num_ctx": 16384,
    "num_predict": 768,
    "temperature": 0.0,
    "seed": 20261014,
    "max_retries_per_cell": 1,
    "pilot_first_attempt_ceiling": 72,
    "pilot_total_request_ceiling": 144,
    "preflight_request_ceiling_per_model": 3,
    "preflight_total_request_ceiling": 9,
    "ram_reserve_gib": 8.0,
    "weight_staging_fraction": 0.25,
    "runtime_buffer_gib": 2.0,
    "context_cache_gib": 1.0,
    "heavy_paging_delta_gib": 1.0,
    "probe_duration_seconds": 600,
    "projected_pilot_runtime_ceiling_seconds": 21600,
    "context_char_limit": 24000,
}


def _config_path(experiment_root: Path) -> Path:
    return experiment_root / "configs/requested_panel_v2.json"


def _load_config(experiment_root: Path) -> dict[str, Any]:
    path = _config_path(experiment_root)
    if not path.is_file():
        raise FileNotFoundError(f"freeze the experiment first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _v1_consensus(votes: dict[str, dict[str, Any]]) -> str:
    labels = [
        item.get("label")
        for item in votes.values()
        if item.get("status") == "valid" and item.get("label") in {"yes", "no"}
    ]
    if labels.count("yes") >= 2:
        return "yes"
    if labels.count("no") >= 2:
        return "no"
    return "uncertain"


def _write_rubric(experiment_root: Path) -> None:
    examples = "\n".join(
        f"- `{item['case']}` — {item['target']} → **{item['expected']}**: {item['reason']}"
        for item in SYNTHETIC_EXAMPLES
    )
    text = f"""# Annotation rubric v2

Version: `{RUBRIC_VERSION}`

This experiment changes both the annotator panel and the semantic rubric. It is not
directly equivalent to v1 and does not replace any v1 artifact. All judgments are
conditional on a bounded supplied same-version context bundle; the pipeline does not
claim exhaustive review of a complete RFC or draft.

## Boundary from v1

V1 allowed a concrete missing constraint to support ambiguity while also directing
annotators to abstain when external context was missing. V2 resolves that conflict:
missing excerpt content, failed retrieval, and unresolved references do not establish
specification-level ambiguity. U-2 requires positive source support within the reviewed
relevant scope that a material constraint remains unresolved. If decisive evidence or
coverage is unavailable, the label is `INSUFFICIENT_CONTEXT`.

Contradictions remain ambiguity subtypes I-1/I-2/I-3 when demonstrated by supplied
evidence. Under-specification remains U-1/U-2/U-3/U-4 only when it creates materially
different required or permitted behaviour and is source-supported. Intentional
discretion is CLEAR. Non-normative explanation is outside the candidate-generation
decision; v2 preserves the frozen first-stage normativity classification and never
re-infers it from keywords.

## Exact system prompt

```text
{SYSTEM_PROMPT}
```

## Synthetic worked cases

{examples}

Structural validation checks schema conformance only. It does not establish semantic
correctness or model understanding.
"""
    path = experiment_root / "RUBRIC_V2.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    atomic_write_json(experiment_root / "schemas/annotation_response_v2.json", OUTPUT_SCHEMA)


def freeze(experiment_root: Path) -> dict[str, Any]:
    if _config_path(experiment_root).exists():
        raise FileExistsError("experiment is already frozen; refusing to replace it")
    allowed = load_allowed_families(FOLDS_PATH)
    rows, sources = load_development_rows(GROUND_TRUTH_ROOT, allowed)
    votes = load_v1_votes_for_allowed_rows(rows, V1_RESPONSE_ROOT)
    sampling = select_review_sample(rows, votes, LIMITS["seed"])
    consensus_counts = Counter(_v1_consensus(votes[row["record_id"]]) for row in rows)
    if len(rows) != 2312 or consensus_counts != Counter({"no": 1858, "yes": 395, "uncertain": 59}):
        raise RuntimeError(
            f"development cohort differs from audited freeze: rows={len(rows)}, "
            f"consensus={dict(consensus_counts)}"
        )

    allowlist = {
        "schema_version": 1,
        "seed": LIMITS["seed"],
        "ordered_record_ids": sampling["ordered_sample_ids"],
    }
    pilot = {
        "schema_version": 1,
        "seed": LIMITS["seed"],
        "selection": "first four frozen random plus first eight frozen targeted",
        "ordered_record_ids": sampling["ordered_pilot_ids"],
    }
    private = {
        "warning": "PRIVATE: contains sampling strata and historical v1 vote patterns.",
        **sampling,
    }
    atomic_write_json(experiment_root / "samples/review_sample_100_v1.json", allowlist)
    atomic_write_json(experiment_root / "samples/pilot_sample_12_v1.json", pilot)
    atomic_write_json(experiment_root / "private/review_sample_mapping_v1.json", private)
    _write_rubric(experiment_root)

    config = {
        "experiment_id": EXPERIMENT_ID,
        "frozen_at": utc_now(),
        "requested_panel": list(REQUESTED_MODELS),
        "development_governance": {
            "policy": "deny-by-default; only exact family IDs below may be opened",
            "allowed_family_ids": list(allowed),
            "excluded_family_ids": [
                "draft-ietf-tcpm-rfc793bis",
                "draft-ietf-dnsop-svcb-https",
            ],
            "excluded_roles": ["TCP validation", "sealed DNS test", "all non-allowlisted lineages"],
            "folds_path": FOLDS_PATH.relative_to(ROOT).as_posix(),
            "folds_sha256": file_sha256(FOLDS_PATH),
            "source_files": sources,
            "candidate_rows": len(rows),
            "definitive_v1_records": consensus_counts["yes"] + consensus_counts["no"],
            "consensus_uncertain_v1_records": consensus_counts["uncertain"],
            "count_reconciliation": (
                "2312 governed candidates = 2253 definitive v1 majority labels + "
                "59 v1 consensus-uncertain records."
            ),
        },
        "versions": {
            "rubric": RUBRIC_VERSION,
            "schema": SCHEMA_VERSION,
            "context_builder": CONTEXT_BUILDER_VERSION,
            "adapter": ADAPTER_VERSION,
            "aggregation": AGGREGATION_VERSION,
            "runner": RUNNER_VERSION,
        },
        "model_settings": {
            model: {
                **MODEL_SETTINGS[model],
                "temperature": LIMITS["temperature"],
                "seed": LIMITS["seed"],
                "num_ctx": LIMITS["num_ctx"],
                "num_predict": LIMITS["num_predict"],
            }
            for model in REQUESTED_MODELS
        },
        "limits": LIMITS,
        "sampling": {
            "seed": LIMITS["seed"],
            "random_first": 40,
            "targeted_after_exclusion": 60,
            "requested_quotas": sampling["requested_quotas"],
            "achieved_quotas": sampling["achieved_quotas"],
            "shortages_before_topup": sampling["shortages_before_topup"],
            "topup_count": sampling["topup_count"],
            "review_allowlist_path": "samples/review_sample_100_v1.json",
            "review_allowlist_sha256": file_sha256(experiment_root / "samples/review_sample_100_v1.json"),
            "pilot_allowlist_path": "samples/pilot_sample_12_v1.json",
            "pilot_allowlist_sha256": file_sha256(experiment_root / "samples/pilot_sample_12_v1.json"),
            "private_mapping_path": "private/review_sample_mapping_v1.json",
            "private_mapping_sha256": file_sha256(experiment_root / "private/review_sample_mapping_v1.json"),
        },
        "fallback": {
            "requested_panel_substitution_allowed": False,
            "separate_experiment_id_if_requested": "ietf-ambiguity-annotation-v2-gpt-oss-20b-fallback",
        },
        "v1_immutability": "No v1 prompt, response, vote, consensus, target, split, or result is modified.",
    }
    atomic_write_json(_config_path(experiment_root), config, exclusive=True)
    return config


def _rows_index(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    allowed = tuple(config["development_governance"]["allowed_family_ids"])
    rows, _ = load_development_rows(GROUND_TRUTH_ROOT, allowed)
    return {row["record_id"]: row for row in rows}


def _sample_ids(experiment_root: Path, filename: str) -> list[str]:
    payload = json.loads((experiment_root / "samples" / filename).read_text(encoding="utf-8"))
    return list(payload["ordered_record_ids"])


def dry_run(experiment_root: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(experiment_root)
    rows = _rows_index(config)
    pilot_ids = _sample_ids(experiment_root, "pilot_sample_12_v1.json")
    builder = ContextBuilder(
        ALIGNMENT_ROOT,
        config["development_governance"]["allowed_family_ids"],
        config["limits"]["context_char_limit"],
    )
    manifests: list[dict[str, Any]] = []
    rendered_requests: list[dict[str, Any]] = []
    for record_id in pilot_ids:
        for variant in ("local", "resolved"):
            bundle = builder.build(rows[record_id], variant)
            path = output_root / "frozen_inputs" / variant / f"{record_id}.json"
            atomic_write_json(path, bundle)
            manifests.append({
                "record_id": record_id,
                "variant": variant,
                "path": path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(path),
                "items": len(bundle["items"]),
                "retrieval_events": len(bundle["retrieval"]),
                "omitted": len(bundle["omitted"]),
            })
            for model in config["requested_panel"]:
                adapter = ModelAdapter(model=model, **config["model_settings"][model])
                request_path = (
                    output_root
                    / "rendered_requests"
                    / safe_model_name(model)
                    / variant
                    / f"{record_id}.json"
                )
                atomic_write_json(
                    request_path,
                    adapter.request(render_user_payload(bundle)),
                )
                rendered_requests.append(
                    {
                        "record_id": record_id,
                        "model": model,
                        "variant": variant,
                        "path": request_path.relative_to(output_root).as_posix(),
                        "sha256": file_sha256(request_path),
                    }
                )

    fixture_results = []
    for item in SYNTHETIC_EXAMPLES:
        label = item["expected"]
        annotation = {
            "label": label,
            "ambiguity_subtype": "other" if label == "AMBIGUOUS" else "none",
            "evidence_span": item["target"] if label == "AMBIGUOUS" else "",
            "interpretation_a": "reading A" if label == "AMBIGUOUS" else "",
            "interpretation_b": "reading B" if label == "AMBIGUOUS" else "",
            "missing_information": "needed source context" if label == "INSUFFICIENT_CONTEXT" else "",
            "justification": item["reason"],
            "context_issue": "missing_definition" if label == "INSUFFICIENT_CONTEXT" else "none",
        }
        fixture_results.append({"case": item["case"], "errors": validate_annotation(annotation, item["target"])})
    if any(item["errors"] for item in fixture_results):
        raise RuntimeError(f"synthetic dry-run validation failed: {fixture_results}")
    report = {
        "completed_at": utc_now(),
        "experiment_id": config["experiment_id"],
        "pilot_records": len(pilot_ids),
        "context_cells": len(manifests),
        "context_manifest": manifests,
        "rendered_request_count": len(rendered_requests),
        "rendered_request_manifest": rendered_requests,
        "synthetic_schema_cases": fixture_results,
        "network_requests": 0,
    }
    atomic_write_json(output_root / "dry_run_report.json", report)
    return report


def build_review_packet(experiment_root: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(experiment_root)
    rows = _rows_index(config)
    sample_ids = _sample_ids(experiment_root, "review_sample_100_v1.json")
    private = json.loads((experiment_root / "private/review_sample_mapping_v1.json").read_text(encoding="utf-8"))
    private_by_id = {item["record_id"]: item for item in private["private_selection"]}
    builder = ContextBuilder(
        ALIGNMENT_ROOT,
        config["development_governance"]["allowed_family_ids"],
        config["limits"]["context_char_limit"],
    )
    blinded: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for index, record_id in enumerate(sample_ids, start=1):
        item_id = f"V2R{index:03d}"
        bundle = builder.build(rows[record_id], "resolved")
        blinded.append({
            "review_item_id": item_id,
            "document_id": bundle["document_id"],
            "revision": bundle["revision"],
            "section_id": bundle["section_id"],
            "section_heading": bundle["section_heading"],
            "target_clause": bundle["target_clause"],
            "context_items": bundle["items"],
            "retrieval_status": bundle["retrieval"],
            "unresolved_context_warnings": [
                event for event in bundle["retrieval"] if event["status"] != "retrieved"
            ] + bundle["omitted"],
            "review_fields": {
                "label": "",
                "ambiguity_subtype": "",
                "evidence_span": "",
                "interpretation_a": "",
                "interpretation_b": "",
                "missing_information": "",
                "justification": "",
                "context_issue": "",
            },
        })
        mapping.append({"review_item_id": item_id, **private_by_id[record_id]})
    packet_path = experiment_root / "review/blinded_review_packet_v1.json"
    map_path = experiment_root / "private/blinded_review_mapping_v1.json"
    atomic_write_json(packet_path, {"items": blinded, "note": "One resolved bundle per item; no model data included."})
    atomic_write_json(map_path, {"items": mapping, "warning": "PRIVATE sampling/identity map"})
    report = {
        "items": len(blinded),
        "packet_path": packet_path.relative_to(ROOT).as_posix(),
        "packet_sha256": file_sha256(packet_path),
        "private_mapping_path": map_path.relative_to(ROOT).as_posix(),
        "private_mapping_sha256": file_sha256(map_path),
        "interpretation": "A single-reviewer result is a limited expert reference, not reproducible gold truth.",
    }
    atomic_write_json(output_root / "review_packet_manifest.json", report)
    return report


def _ollama_version() -> str:
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=15, check=True)
        return result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def _metadata_subset(tag: dict[str, Any], shown: dict[str, Any]) -> dict[str, Any]:
    details = tag.get("details") or shown.get("details") or {}
    return {
        "name": tag.get("name") or tag.get("model"),
        "digest": tag.get("digest"),
        "size": tag.get("size"),
        "modified_at": tag.get("modified_at"),
        "format": details.get("format"),
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "capabilities": shown.get("capabilities"),
        "model_info": shown.get("model_info"),
        "template_sha256": sha256_text(str(shown.get("template", ""))),
        "parameters": shown.get("parameters"),
    }


def preflight(
    experiment_root: Path,
    output_root: Path,
    base_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    config = _load_config(experiment_root)
    client = OllamaClient(base_url)
    tags = client.tags()
    version = _ollama_version()
    initial_hardware = hardware_snapshot()
    request_count = 0
    models: list[dict[str, Any]] = []
    probe_payload = {
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
    for model in config["requested_panel"]:
        tag = tags.get(model)
        if tag is None:
            models.append({"model": model, "status": "MODEL_BLOCKED_PREFLIGHT", "reasons": ["model tag not installed"]})
            continue
        shown = client.show(model)
        metadata = _metadata_subset(tag, shown)
        feasibility = assess_feasibility(model, tag, initial_hardware, config["limits"])
        entry: dict[str, Any] = {"model": model, "metadata": metadata, "feasibility": feasibility}
        if not feasibility["feasible_for_bounded_probe"]:
            entry["status"] = "MODEL_BLOCKED_PREFLIGHT"
            entry["probe_requests"] = 0
            models.append(entry)
            continue
        settings = config["model_settings"][model]
        adapter = ModelAdapter(model=model, **settings)
        before = hardware_snapshot()
        started = time.monotonic()
        try:
            envelope = client.request("/api/chat", adapter.request(probe_payload), adapter.timeout_seconds)
            request_count += 1
            operational, annotation, errors = classify_envelope(envelope, probe_payload["target_clause"])
        except TimeoutError as exc:
            request_count += 1
            envelope, operational, annotation, errors = {}, "TIMEOUT", None, [str(exc)]
        except (ConnectionError, RuntimeError) as exc:
            request_count += 1
            envelope, operational, annotation, errors = {}, "TRANSPORT_ERROR", None, [str(exc)]
        after = hardware_snapshot()
        wall = round(time.monotonic() - started, 3)
        paging_delta = int(after.get("paging", {}).get("used_bytes", 0)) - int(before.get("paging", {}).get("used_bytes", 0))
        projected = wall * 24
        post_block_reasons = post_probe_block_reasons(
            before, after, wall, config["limits"]
        )
        entry.update({
            "status": operational if operational == "VALID" else "MODEL_BLOCKED_PREFLIGHT",
            "probe_requests": 1,
            "probe_semantic_label": annotation.get("label") if annotation else None,
            "probe_validation_errors": errors,
            "done_reason": envelope.get("done_reason"),
            "prompt_eval_count": envelope.get("prompt_eval_count"),
            "eval_count": envelope.get("eval_count"),
            "wall_seconds": wall,
            "load_duration_ns": envelope.get("load_duration"),
            "eval_duration_ns": envelope.get("eval_duration"),
            "hardware_before": before,
            "hardware_after": after,
            "paging_delta_bytes": paging_delta,
            "heavy_paging_detected": "heavy paging threshold exceeded" in post_block_reasons,
            "projected_24_cell_pilot_seconds": projected,
        })
        if post_block_reasons:
            entry["status"] = "MODEL_BLOCKED_PREFLIGHT"
            entry.setdefault("block_reasons", []).extend(post_block_reasons)
        try:
            client.unload(model)
        except (ConnectionError, RuntimeError, TimeoutError) as exc:
            entry["unload_warning"] = str(exc)
        models.append(entry)
    if request_count > config["limits"]["preflight_total_request_ceiling"]:
        raise RuntimeError("preflight inference request ceiling exceeded")
    report = {
        "experiment_id": config["experiment_id"],
        "completed_at": utc_now(),
        "platform": platform.platform(),
        "ollama_version": version,
        "initial_hardware": initial_hardware,
        "models": models,
        "inference_request_count": request_count,
        "request_ceiling": config["limits"]["preflight_total_request_ceiling"],
        "notes": [
            "Metadata calls and unload control calls are reported but are not inference probes.",
            "Feasibility includes weight residency, possible staging, runtime buffers, context cache, and reserve.",
            "No risky model was loaded when the frozen gate failed.",
        ],
    }
    atomic_write_json(output_root / "preflight_report.json", report)
    return report


def audit_existing_preflight(output_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Correctly apply frozen gates to preserved measurements without new inference."""
    source_path = output_root / "preflight_report.json"
    if not source_path.is_file():
        raise FileNotFoundError("no preflight report to audit")
    report = json.loads(source_path.read_text(encoding="utf-8"))
    for entry in report["models"]:
        if entry.get("probe_requests", 0) == 0:
            continue
        reasons = post_probe_block_reasons(
            entry["hardware_before"],
            entry["hardware_after"],
            float(entry["wall_seconds"]),
            config["limits"],
        )
        entry["post_probe_gate_reasons"] = reasons
        if reasons:
            entry["status"] = "MODEL_BLOCKED_PREFLIGHT"
            existing = list(entry.get("block_reasons", []))
            entry["block_reasons"] = list(dict.fromkeys([*existing, *reasons]))
    report["policy_audit"] = {
        "audited_at": utc_now(),
        "source_report_sha256": file_sha256(source_path),
        "new_inference_requests": 0,
        "note": (
            "Applied the already-frozen post-probe RAM reserve, paging, and runtime "
            "gates. The initial report omitted the RAM-reserve decision check."
        ),
    }
    final_path = output_root / "preflight_report_final.json"
    atomic_write_json(final_path, report, exclusive=True)
    return report


def _write_nonrun_cell(
    output_root: Path,
    model: str,
    variant: str,
    row: dict[str, str],
    status: str,
    reason: str,
) -> dict[str, Any]:
    result = {
        "experiment_id": EXPERIMENT_ID,
        "record_id": row["record_id"],
        "family_id": row["family_id"],
        "document_id": row["document_id"],
        "revision": row["revision"],
        "model": model,
        "context_variant": variant,
        "semantic_label": None,
        "operational_status": status,
        "annotation": None,
        "accepted_attempt": None,
        "attempt_count": 0,
        "reason": reason,
    }
    path = output_root / "cells" / safe_model_name(model) / variant / row["record_id"] / "accepted.json"
    if not path.exists():
        atomic_write_json(path, result)
    return result


def validate_manual_eligibility(
    preflight_report: dict[str, Any],
    tags: dict[str, dict[str, Any]],
    active_server_version: str,
) -> set[str]:
    """Validate the saved-evidence gate before any eligible-only annotation."""
    if not preflight_report.get("manual_evidence_assembly"):
        raise PermissionError(
            "--eligible-only requires a manually assembled evidence report"
        )
    preflight_by_model = {item["model"]: item for item in preflight_report["models"]}
    eligible_models = {
        model
        for model, entry in preflight_by_model.items()
        if entry.get("status") == "VALID"
    }
    expected_eligible = {"qwen3.5:27b", "gemma4:31b"}
    if eligible_models != expected_eligible:
        raise PermissionError(
            "manual pilot requires exactly the verified local models; "
            f"expected={sorted(expected_eligible)}, actual={sorted(eligible_models)}"
        )
    if preflight_by_model.get("gpt-oss:120b", {}).get("status") == "VALID":
        raise PermissionError("GPT-OSS-120B must remain blocked for local execution")
    evidence_version = str(preflight_report.get("ollama_version", ""))
    if active_server_version != evidence_version:
        raise RuntimeError(
            f"server version mismatch: evidence={evidence_version}, "
            f"active={active_server_version}"
        )
    for model in eligible_models:
        expected_digest = str(preflight_by_model[model].get("model_digest", ""))
        actual_digest = str(tags.get(model, {}).get("digest", ""))
        if not expected_digest or actual_digest != expected_digest:
            raise RuntimeError(
                f"model digest mismatch for {model}: "
                f"evidence={expected_digest}, active={actual_digest}"
            )
    return eligible_models


def annotate_scope(
    experiment_root: Path,
    output_root: Path,
    scope: str,
    confirm_full: bool,
    base_url: str = "http://127.0.0.1:11434",
    eligible_only: bool = False,
) -> dict[str, Any]:
    config = _load_config(experiment_root)
    if scope == "full" and not confirm_full:
        raise PermissionError("full annotation requires explicit --confirm-full")
    preflight_path = output_root / "preflight_report_final.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("run bounded preflight before annotation")
    preflight_report = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight_by_model = {item["model"]: item for item in preflight_report["models"]}
    rows = _rows_index(config)
    ids = (
        _sample_ids(experiment_root, "pilot_sample_12_v1.json")
        if scope == "pilot"
        else sorted(rows)
    )
    builder = ContextBuilder(
        ALIGNMENT_ROOT,
        config["development_governance"]["allowed_family_ids"],
        config["limits"]["context_char_limit"],
    )
    client = OllamaClient(base_url)
    tags = client.tags()
    version = preflight_report["ollama_version"]
    if eligible_only:
        server_version = str(client.request("/api/version").get("version", ""))
        validate_manual_eligibility(preflight_report, tags, server_version)
    results: list[dict[str, Any]] = []
    live_requests_before = sum(
        1 for path in (output_root / "cells").glob("**/attempts/attempt-*.json")
    ) if (output_root / "cells").exists() else 0
    for model in config["requested_panel"]:
        preflight_model = preflight_by_model.get(model, {})
        feasible = preflight_model.get("status") == "VALID"
        if not feasible:
            for record_id in ids:
                for variant in ("local", "resolved"):
                    results.append(_write_nonrun_cell(
                        output_root, model, variant, rows[record_id],
                        "MODEL_BLOCKED_PREFLIGHT", "model did not pass frozen preflight gate",
                    ))
            continue
        tag = tags.get(model)
        if not tag:
            for record_id in ids:
                for variant in ("local", "resolved"):
                    results.append(_write_nonrun_cell(
                        output_root, model, variant, rows[record_id],
                        "NOT_RUN_DEPENDENCY", "installed model tag disappeared after preflight",
                    ))
            continue
        shown = client.show(model)
        settings = config["model_settings"][model]
        adapter = ModelAdapter(model=model, **settings)
        for record_id in ids:
            for variant in ("local", "resolved"):
                current_requests = sum(
                    1 for path in (output_root / "cells").glob("**/attempts/attempt-*.json")
                ) if (output_root / "cells").exists() else 0
                new_requests = current_requests - live_requests_before
                ceiling = config["limits"]["pilot_total_request_ceiling"]
                if scope == "pilot" and new_requests >= ceiling:
                    results.append(_write_nonrun_cell(
                        output_root, model, variant, rows[record_id],
                        "NOT_RUN_BUDGET", "absolute pilot request ceiling reached",
                    ))
                    continue
                frozen_path = output_root / "frozen_inputs" / variant / f"{record_id}.json"
                if eligible_only and not frozen_path.is_file():
                    raise FileNotFoundError(
                        f"eligible-only execution requires frozen input: {frozen_path}"
                    )
                bundle = (
                    json.loads(frozen_path.read_text(encoding="utf-8"))
                    if frozen_path.is_file()
                    else builder.build(rows[record_id], variant)
                )
                result = execute_cell(
                    client=client,
                    adapter=adapter,
                    model_digest=str(tag.get("digest", "")),
                    model_metadata=shown,
                    bundle=bundle,
                    output_root=output_root,
                    experiment_id=config["experiment_id"],
                    ollama_version=version,
                    max_retries=config["limits"]["max_retries_per_cell"],
                )
                results.append(result)
        try:
            client.unload(model)
        except (ConnectionError, RuntimeError, TimeoutError):
            pass
    report = summarize_results(experiment_root, output_root, ids, rows)
    atomic_write_json(output_root / f"{scope}_run_report.json", report)
    return report


def _contingency(left: list[str], right: list[str]) -> dict[str, int]:
    table = Counter(zip(left, right, strict=True))
    return {f"{a}|{b}": table[(a, b)] for a in ("AMBIGUOUS", "CLEAR") for b in ("AMBIGUOUS", "CLEAR")}


def summarize_results(
    experiment_root: Path,
    output_root: Path,
    ids: list[str],
    rows: dict[str, dict[str, str]],
) -> dict[str, Any]:
    config = _load_config(experiment_root)
    votes = load_v1_votes_for_allowed_rows((rows[rid] for rid in ids), V1_RESPONSE_ROOT)
    all_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    for model in config["requested_panel"]:
        for variant in ("local", "resolved"):
            for record_id in ids:
                path = output_root / "cells" / safe_model_name(model) / variant / record_id / "accepted.json"
                if path.is_file():
                    all_results[(record_id, variant, model)] = json.loads(path.read_text(encoding="utf-8"))
    operational = Counter(item["operational_status"] for item in all_results.values())
    labels_by_model_variant: dict[str, Counter[str]] = defaultdict(Counter)
    for (_, variant, model), item in all_results.items():
        if item["operational_status"] == "VALID":
            labels_by_model_variant[f"{model}|{variant}"][str(item["semantic_label"])] += 1

    aggregates: list[dict[str, Any]] = []
    transitions: Counter[str] = Counter()
    v1_transitions: Counter[str] = Counter()
    for record_id in ids:
        per_variant: dict[str, dict[str, Any]] = {}
        for variant in ("local", "resolved"):
            judgments = [all_results.get((record_id, variant, model), {
                "operational_status": "NOT_RUN_DEPENDENCY", "semantic_label": None
            }) for model in config["requested_panel"]]
            decision = conservative_aggregate(judgments)
            comparison = v1_compatible_comparison(judgments)
            per_variant[variant] = decision
            aggregates.append({
                "record_id": record_id,
                "variant": variant,
                **decision,
                "v1_compatible_comparison": comparison,
            })
            v1_label = _v1_consensus(votes[record_id])
            v1_transitions[f"{v1_label}->{decision['semantic_label'] or decision['aggregation_status']}"] += 1
        transitions[
            f"{per_variant['local']['semantic_label'] or per_variant['local']['aggregation_status']}"
            f"->{per_variant['resolved']['semantic_label'] or per_variant['resolved']['aggregation_status']}"
        ] += 1

    pairwise: dict[str, Any] = {}
    for variant in ("local", "resolved"):
        for i, first in enumerate(config["requested_panel"]):
            for second in config["requested_panel"][i + 1 :]:
                left: list[str] = []
                right: list[str] = []
                for record_id in ids:
                    a = all_results.get((record_id, variant, first), {})
                    b = all_results.get((record_id, variant, second), {})
                    if a.get("operational_status") == b.get("operational_status") == "VALID" and a.get("semantic_label") in {"AMBIGUOUS", "CLEAR"} and b.get("semantic_label") in {"AMBIGUOUS", "CLEAR"}:
                        left.append(a["semantic_label"])
                        right.append(b["semantic_label"])
                pairwise[f"{variant}|{first}|{second}"] = {
                    "matched_valid_binary_n": len(left),
                    "contingency": _contingency(left, right) if left else {},
                    "raw_agreement": sum(a == b for a, b in zip(left, right, strict=True)) / len(left) if left else None,
                }

    attempts = list((output_root / "cells").glob("**/attempts/attempt-*.json")) if (output_root / "cells").exists() else []
    runtimes = []
    for path in attempts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        runtimes.append(float(payload.get("wall_seconds", 0)))
    report = {
        "experiment_id": config["experiment_id"],
        "created_at": utc_now(),
        "records": len(ids),
        "expected_cells": len(ids) * 2 * len(config["requested_panel"]),
        "materialized_cells": len(all_results),
        "operational_status_counts": dict(operational),
        "semantic_counts_by_model_variant": {key: dict(value) for key, value in labels_by_model_variant.items()},
        "aggregates": aggregates,
        "local_to_resolved_transitions": dict(transitions),
        "historical_v1_to_v2_transitions": dict(v1_transitions),
        "pairwise_agreement": pairwise,
        "attempt_requests": len(attempts),
        "total_attempt_wall_seconds": round(sum(runtimes), 3),
        "mean_attempt_wall_seconds": round(sum(runtimes) / len(runtimes), 3) if runtimes else None,
        "interpretation_limits": [
            "Models and rubric both changed from v1; transitions do not isolate either cause.",
            "Agreement and adherence do not establish label validity or human accuracy.",
            "Targeted sample rates are not population estimates.",
        ],
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("dry-run")
    sub.add_parser("review-packet")
    sub.add_parser("preflight")
    sub.add_parser("audit-preflight")
    annotate = sub.add_parser("annotate")
    annotate.add_argument("--scope", choices=("pilot", "full"), required=True)
    annotate.add_argument("--confirm-full", action="store_true")
    annotate.add_argument("--eligible-only", action="store_true")
    sub.add_parser("report-pilot")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    output_root = args.output_root.resolve()
    if args.command == "freeze":
        result = freeze(experiment_root)
    elif args.command == "dry-run":
        result = dry_run(experiment_root, output_root)
    elif args.command == "review-packet":
        result = build_review_packet(experiment_root, output_root)
    elif args.command == "preflight":
        result = preflight(experiment_root, output_root, args.base_url)
    elif args.command == "audit-preflight":
        result = audit_existing_preflight(output_root, _load_config(experiment_root))
    elif args.command == "annotate":
        result = annotate_scope(
            experiment_root,
            output_root,
            args.scope,
            args.confirm_full,
            args.base_url,
            args.eligible_only,
        )
    else:
        config = _load_config(experiment_root)
        rows = _rows_index(config)
        ids = _sample_ids(experiment_root, "pilot_sample_12_v1.json")
        result = summarize_results(experiment_root, output_root, ids, rows)
        atomic_write_json(output_root / "pilot_run_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
