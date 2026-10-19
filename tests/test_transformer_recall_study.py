from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from experiments.transformer_final.recall_improvement_study_v1 import study_runner
from experiments.transformer_final.recall_improvement_study_v1.cell_launcher import (
    _focal_loss,
    _freeze_policy,
)
from experiments.transformer_final.recall_improvement_study_v1.study_analysis import (
    _choose_threshold,
    _deployed_metric_row,
)
from experiments.transformer_final.recall_improvement_study_v1.study_protocol import (
    DEFAULT_EXPERIMENT_CONFIG,
    make_inner_splits,
)
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.leakage import audit_leakage, build_fold_manifests
from experiments.transformer_final.recall_improvement_study_v1.study_runner import (
    _cells_for_stage,
    _is_known_windows_graphics_process,
    _parse_nvidia_smi_process_types,
    _record_cell_time,
    _reset_time_skipped_stages,
)


class _DebertaLike(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deberta = nn.Module()
        self.deberta.embeddings = nn.Embedding(8, 4)
        self.deberta.encoder = nn.Module()
        self.deberta.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])
        self.deberta.encoder.LayerNorm = nn.LayerNorm(4)
        self.pooler = nn.Linear(4, 4)
        self.classifier = nn.Linear(4, 2)


def test_focal_loss_is_finite_and_backpropagates_on_cpu() -> None:
    logits = torch.tensor([[2.0, -1.0], [-0.5, 0.5]], requires_grad=True)
    labels = torch.tensor([0, 1])
    class_weights = torch.tensor([1.0, 3.0])

    loss = _focal_loss(logits, labels, class_weights)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_partial_freezing_policy_keeps_requested_layers_and_head_trainable() -> None:
    model = _DebertaLike()

    policy = _freeze_policy(model, last_layers=4)

    assert len(model.deberta.encoder.layer) == 12
    assert all(
        not value.requires_grad for value in model.deberta.embeddings.parameters()
    )
    assert all(
        all(parameter.requires_grad == (index >= 8) for parameter in layer.parameters())
        for index, layer in enumerate(model.deberta.encoder.layer)
    )
    assert all(parameter.requires_grad for parameter in model.pooler.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert (
        sum(policy["parameter_counts_by_policy_group"].values())
        == policy["total_parameters"]
    )
    assert len(policy["frozen_snapshots"]) == len(policy["frozen_names"])


def test_partial_two_freezes_first_ten_layers_and_shared_backbone() -> None:
    model = _DebertaLike()

    _freeze_policy(model, last_layers=2)

    assert all(
        all(
            parameter.requires_grad == (index >= 10) for parameter in layer.parameters()
        )
        for index, layer in enumerate(model.deberta.encoder.layer)
    )
    assert all(
        not value.requires_grad
        for value in model.deberta.encoder.LayerNorm.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.pooler.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_nested_splits_are_deterministic_and_family_disjoint() -> None:
    rows = []
    for family in range(9):
        for record in range(3):
            rows.append(
                {
                    "record_id": f"record-{family}-{record}",
                    "family_id": f"family-{family}",
                    "document_id": f"document-{family}-{record}",
                    "revision": str(record),
                    "lineage_ids": "",
                    "target_clause": (
                        f"Unique family {family} clause {record} applies to "
                        f"special marker {family * 101 + record}."
                    ),
                    "section_heading": "",
                    "previous_clause": "",
                    "next_clause": "",
                    "fold": family % 4 + 1,
                    "label": (family + record) % 2,
                }
            )
    frame = pd.DataFrame(rows)

    first = make_inner_splits(frame, outer_fold=5)
    second = make_inner_splits(frame, outer_fold=5)

    assert first == second
    assert len(first) == 3
    scored = []
    for split in first:
        fitting = set(split["fitting_record_ids"])
        scoring = set(split["scoring_record_ids"])
        assert fitting.isdisjoint(scoring)
        assert set(split["fitting_families"]).isdisjoint(split["held_out_families"])
        for item in split["context_sanitizations"]:
            role_ids = fitting if item["role"] == "fitting" else scoring
            assert item["record_id"] in role_ids
        scored.extend(scoring)
    assert len(scored) == len(set(scored)) == len(frame)


def test_effective_dataset_outer_and_nested_context_policies_are_in_scope() -> None:
    config = load_config(DEFAULT_EXPERIMENT_CONFIG)
    frame, _, _ = load_effective_dataset(config)
    audit = audit_leakage(frame, near_threshold=0.85)
    assert not audit["training_blocked"]

    outer_manifests = build_fold_manifests(frame, audit)
    assert len(outer_manifests) == 5
    for manifest in outer_manifests:
        fitting_ids = set(manifest["fitting_record_ids"])
        scoring_ids = set(manifest["scoring_record_ids"])
        for item in manifest["context_sanitizations"]:
            role_ids = fitting_ids if item["role"] == "fitting" else scoring_ids
            assert item["record_id"] in role_ids

    nested_count = 0
    for outer_fold in range(1, 6):
        splits = make_inner_splits(frame, outer_fold, splits=3)
        assert len(splits) == 3
        for split in splits:
            fitting_ids = set(split["fitting_record_ids"])
            scoring_ids = set(split["scoring_record_ids"])
            for item in split["context_sanitizations"]:
                role_ids = fitting_ids if item["role"] == "fitting" else scoring_ids
                assert item["record_id"] in role_ids
            nested_count += 1
    assert nested_count == 15


def test_v2_launcher_requires_explicit_gpu_authorization_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from experiments.transformer_final.recall_improvement_study_v2.cell_launcher import (
        _verify_launch,
    )

    monkeypatch.delenv("RECALL_STUDY_V2_GPU_AUTHORIZED", raising=False)
    with pytest.raises(PermissionError, match="separately authorizing"):
        _verify_launch(Path("unused"), {})


def test_nested_threshold_selection_obeys_precision_then_recall_policy() -> None:
    frame = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "probability_yes": [0.9, 0.4, 0.3, 0.2],
        }
    )

    selected = _choose_threshold(frame, [0.1, 0.3, 0.5])

    assert selected["threshold"] == 0.3
    assert selected["precision"] >= 0.2
    assert selected["recall"] == 1.0
    assert selected["selection_policy"] == ("maximum_recall_at_precision_at_least_0.20")


def test_nested_threshold_uses_declared_f2_fallback() -> None:
    frame = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "probability_yes": [0.1, 0.05, 0.8, 0.7],
        }
    )

    selected = _choose_threshold(frame, [0.5, 0.9])

    assert selected["precision"] < 0.2
    assert selected["selection_policy"] == (
        "fallback_maximum_f2_then_precision_then_threshold"
    )


def test_fold_specific_threshold_metrics_use_deployed_predictions() -> None:
    frame = pd.DataFrame(
        {
            "label": [1, 1, 0, 0],
            "probability_yes": [0.9, 0.4, 0.3, 0.2],
            "predicted_label_nested_threshold": [1, 1, 1, 0],
        }
    )

    metrics = _deployed_metric_row(frame)

    assert metrics["threshold"] == "fold_specific"
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == 2 / 3
    assert metrics["recall"] == 1.0
    assert metrics["specificity"] == 0.5


def test_runner_builds_expected_stage_cells_without_duplicate_paths() -> None:
    protocol = {
        "training": {"seed": 2718},
        "config_path": r"D:\study\config.json",
        "model": {"local_path": r"D:\models\deberta"},
    }
    splits = [
        {
            "outer_fold": outer,
            "inner_fold": inner,
            "fitting_record_ids": [f"fit-{outer}-{inner}"],
            "scoring_record_ids": [f"score-{outer}-{inner}"],
        }
        for outer in range(1, 6)
        for inner in range(1, 4)
    ]

    nested = _cells_for_stage("B_nested_threshold_calibration", protocol, splits)
    context = _cells_for_stage("C_local_context", protocol, splits)
    partial = _cells_for_stage("G_partial_4", protocol, splits)

    assert len(nested) == 15
    assert len({cell["run_path"] for cell in nested}) == 15
    assert all(cell["input_mode"] == "target_only" for cell in nested)
    assert len(context) == len(partial) == 5
    assert all(cell["input_mode"] == "local_context" for cell in context)
    assert all(cell["method"] == "partial_4" for cell in partial)


def test_windows_gpu_preflight_distinguishes_compute_from_desktop_graphics() -> None:
    output = """
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|    0   N/A  N/A       14184    C+G   C:\\Windows\\System32\\dwm.exe   N/A   |
|    0   N/A  N/A       24568      C   C:\\Python\\python.exe           4096MiB |
"""

    process_types = _parse_nvidia_smi_process_types(output)

    assert process_types == {14184: "C+G", 24568: "C"}
    assert _is_known_windows_graphics_process(r"C:\Windows\System32\dwm.exe")
    assert _is_known_windows_graphics_process("Code.exe")
    assert _is_known_windows_graphics_process("msedge.exe")
    assert _is_known_windows_graphics_process("AnyDesk.exe")
    assert _is_known_windows_graphics_process("Taskmgr")
    assert not _is_known_windows_graphics_process("python.exe")


def test_runner_reopens_stages_skipped_by_an_expired_budget() -> None:
    inventory = {
        "stages": {
            "pending": {"status": "NOT_STARTED"},
            "skipped": {"status": "SKIPPED_TIME_BUDGET"},
            "complete": {"status": "COMPLETE"},
        }
    }

    _reset_time_skipped_stages(inventory)

    assert inventory["stages"]["pending"]["status"] == "NOT_STARTED"
    assert inventory["stages"]["skipped"]["status"] == "NOT_STARTED"
    assert inventory["stages"]["complete"]["status"] == "COMPLETE"


def test_reused_completed_cell_records_elapsed_time_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(study_runner, "_save_budget", lambda budget: None)
    budget = {
        "actual": {
            "cell_ids": {},
            "gpu_cell_seconds": 0.0,
            "gpu_cells_completed": 0,
            "seconds_by_stage": {},
        }
    }
    spec = {"cell_id": "stage/fold_1/seed_1"}
    record = {"elapsed_seconds": 12.5}

    _record_cell_time(budget, spec, record, completed=True)
    _record_cell_time(budget, spec, record, completed=True)

    assert budget["actual"]["gpu_cell_seconds"] == 12.5
    assert budget["actual"]["gpu_cells_completed"] == 1
    assert budget["actual"]["seconds_by_stage"] == {"stage": 12.5}
    assert budget["actual"]["cell_ids"]["stage/fold_1/seed_1"]["completed"]
