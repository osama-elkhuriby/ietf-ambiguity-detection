from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import torch

from models.transformer_final.aggregate import (
    DEFAULT_RUN_OVERRIDES,
    across_seed_summary,
)
from models.transformer_final.common import immutable_directory
from models.transformer_final.compare import compare
from models.transformer_final.config import (
    DEFAULT_CONFIG,
    DNS_FAMILY,
    LABEL_MAP,
    TCP_FAMILY,
    TRAIN_FAMILIES,
    load_config,
    public_config,
    validate_config,
)
from models.transformer_final.dataset import (
    assemble_dataset,
    load_effective_dataset,
    verify_dataset,
)
from models.transformer_final.leakage import (
    apply_context_sanitization,
    audit_leakage,
    build_fold_manifests,
)
from models.transformer_final.metrics import binary_metrics
from models.transformer_final.modeling import collate_clauses
from models.transformer_final.protocol import load_fp32_protocol
from models.transformer_final.experiment_runner import _cell_directory
from models.transformer_final.experiment2 import (
    DEFAULT_EXPERIMENT_CONFIG,
    audit_annotation_ready_source,
)
from models.transformer_final.experiment2_analysis import (
    _distribution_diagnostics,
    _plot_artifacts,
    _threshold_metrics,
    _threshold_table,
    _top_k_table,
)
from models.transformer_final.tokenization import (
    audit_lengths,
    encode_example,
    serialize_local_context,
)
from models.transformer_final.train import (
    _assert_finite_optimizer_state,
    _assert_finite_model_parameters,
    _backward_with_diagnostics,
    _check_forward_numerics,
    _classification_loss,
    _clip_gradients,
    _gradient_statistics,
    _numerical_context,
    _optimizer_configuration,
    _optimizer_state_statistics,
    _parameter_statistics,
    _prepare_model_for_training,
    _validate_training_labels,
    compute_class_weights,
    initialize_pretrained_components,
    validate_prediction_alignment,
)


class FakeTokenizer:
    model_input_names: ClassVar[list[str]] = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, text: str, **kwargs):
        add_special_tokens = bool(kwargs["add_special_tokens"])
        self.calls.append((text, add_special_tokens))
        values = [ord(character) % 97 + 3 for character in text]
        input_ids = [101, *values, 102] if add_special_tokens else values
        special_mask = (
            [1, *([0] * len(values)), 1] if add_special_tokens else [0] * len(values)
        )
        result = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "special_tokens_mask": special_mask,
        }
        if kwargs.get("return_token_type_ids"):
            result["token_type_ids"] = [0] * len(input_ids)
        return result


def _synthetic_sources():
    families = [*TRAIN_FAMILIES, TCP_FAMILY, DNS_FAMILY]
    source_rows = []
    key_rows = []
    metadata_rows = []
    context_rows = []
    for index, family in enumerate(families, start=1):
        item = str(index)
        record = f"record-{index}"
        document = f"{family}-01"
        target = f"Clause {index} MUST be deterministic."
        source_rows.append(
            {
                "item_id": item,
                "document": document,
                "target_clause": target,
                "final_label": "Yes" if index % 2 == 0 else "No",
                "final_source": "unresolved" if index == 1 else "model_agreement",
                "final_status": "unresolved_blank_manual_verdict"
                if index == 1
                else "resolved",
            }
        )
        key_rows.append({"item_id": item, "record_id": record, "family_id": family})
        metadata_rows.append(
            {
                "record_id": record,
                "family_id": family,
                "document_id": document,
                "revision": "01",
                "clause_ids": f"clause-{index}",
                "clause_text": target,
                "section_ids": "1",
                "lineage_ids": f"lineage-{index}",
            }
        )
        context_rows.append(
            {
                "record_id": record,
                "family_id": family,
                "target_clause": target,
                "section_heading": "Section",
                "previous_clause": "Previous",
                "next_clause": "Next",
                "context_source_ids": "{}",
                "context_resolution_status": "both",
                "alignment_source_path": "approved.json",
                "source_hashes": "{}",
            }
        )
    folds = {family: (index % 5) + 1 for index, family in enumerate(TRAIN_FAMILIES)}
    return tuple(
        pd.DataFrame(rows)
        for rows in (source_rows, key_rows, metadata_rows, context_rows)
    ) + (folds,)


def test_label_mapping_and_unresolved_exclusion_and_reserved_partitions():
    source, key, metadata, context, folds = _synthetic_sources()
    effective, report = assemble_dataset(
        source,
        key,
        metadata,
        context,
        expected_unresolved_item_ids={"1"},
        family_to_fold=folds,
    )
    assert LABEL_MAP == {"No": 0, "Yes": 1}
    assert "1" not in set(effective["item_id"])
    assert report["unresolved_exclusions"][0]["item_id"] == "1"
    assert not set(effective["family_id"]) & {TCP_FAMILY, DNS_FAMILY}
    assert set(effective["label"]) <= {0, 1}


def test_dataset_a_only_configuration_rejects_dataset_b_and_wrong_label():
    config = load_config(DEFAULT_CONFIG, verify_files=False)
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    bad = copy.deepcopy(config)
    bad["data"]["source_csv"] = "results/annotation/final_annotation_reannotated.csv"
    with pytest.raises(ValueError, match="Dataset B"):
        validate_config(bad)
    bad = copy.deepcopy(config)
    bad["data"]["label_column"] = "revised_label"
    with pytest.raises(ValueError, match="final_label"):
        validate_config(bad)


def test_annotation_ready_pilot_config_freezes_one_seed_weighted_fp32():
    config = load_config(DEFAULT_EXPERIMENT_CONFIG)

    assert config["experiment"] == "final_annotation_ready_pilot_v1"
    assert config["data"]["label_column"] == "final_label"
    assert config["conditions"] == {"A2": "weighted_ce"}
    assert config["training"]["seeds"] == [2718]
    assert config["training"]["precision"] == "fp32"
    assert config["partitions"]["sealed_evaluation_enabled"] is False


def test_annotation_ready_source_authority_and_explicit_unresolved_exclusion():
    config = load_config(DEFAULT_EXPERIMENT_CONFIG)
    audit = audit_annotation_ready_source(config)

    assert audit["rows"] == 2425
    assert audit["resolved_label_counts"] == {"No": 2276, "Yes": 148}
    assert audit["unresolved_item_ids"] == ["2161"]
    assert audit["excluded_unresolved_count"] == 1
    assert audit["duplicate_item_ids"] == 0
    assert audit["blank_target_clauses"] == 0
    assert len(audit["canonical_family_counts"]) == 11
    assert "only binary Yes/No outcome column" in audit["label_authority"]


def test_threshold_metrics_report_all_confusion_and_operating_metrics():
    frame = pd.DataFrame(
        {
            "label": [1, 0, 1, 0],
            "probability_yes": [0.9, 0.5, 0.4, 0.1],
            "fold": [1, 1, 2, 2],
        }
    )

    at_half = _threshold_metrics(frame, 0.5)
    at_point_four = _threshold_metrics(frame, 0.4)

    assert {
        key: at_half[key]
        for key in (
            "true_positives",
            "false_positives",
            "true_negatives",
            "false_negatives",
            "predicted_yes",
            "predicted_no",
        )
    } == {
        "true_positives": 1,
        "false_positives": 1,
        "true_negatives": 1,
        "false_negatives": 1,
        "predicted_yes": 2,
        "predicted_no": 2,
    }
    assert at_half["specificity"] == pytest.approx(0.5)
    assert at_half["balanced_accuracy"] == pytest.approx(0.5)
    assert at_half["mcc"] == pytest.approx(0.0)
    assert at_point_four["predicted_yes"] == 3
    assert at_point_four["true_positives"] == 2


def test_pilot_threshold_tables_include_every_fixed_threshold_and_fold():
    frame = pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(6)],
            "label": [1, 0, 1, 0, 0, 1],
            "probability_yes": [0.9, 0.8, 0.5, 0.4, 0.02, 0.001],
            "fold": [1, 1, 2, 2, 3, 3],
        }
    )

    overall = _threshold_table(frame)
    by_fold = pd.concat(
        [_threshold_table(frame, fold=fold) for fold in (1, 2, 3)],
        ignore_index=True,
    )
    distribution = _distribution_diagnostics(frame)
    top_k = _top_k_table(frame)

    assert len(overall) == 15
    assert len(by_fold) == 45
    assert (overall["threshold"] == 0.5).sum() == 1
    assert overall.iloc[-1]["predicted_yes"] == 3
    assert distribution["actual_yes_score_distribution"]["count"] == 3
    assert len(top_k) == 5
    assert top_k.iloc[0]["true_ambiguous_retrieved"] == 3


def test_pilot_generates_all_seven_required_plots(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 0, 1],
            "probability_yes": [0.9, 0.8, 0.5, 0.4, 0.02, 0.001],
            "fold": [1, 1, 2, 2, 3, 3],
        }
    )
    output = tmp_path / "plots"

    _plot_artifacts(frame, _threshold_table(frame), output)

    assert {path.name for path in output.glob("*.png")} == {
        "precision_recall_curve.png",
        "roc_curve.png",
        "probability_histogram.png",
        "predicted_yes_vs_threshold.png",
        "precision_recall_vs_threshold.png",
        "f1_f2_vs_threshold.png",
        "per_fold_average_precision.png",
    }


def test_fp32_protocol_pins_training_precision_and_existing_configuration():
    config = load_config(DEFAULT_CONFIG)
    protocol_path = DEFAULT_CONFIG.parent / "fp32_protocol_v1.json"
    protocol, protocol_hash = load_fp32_protocol(protocol_path, config)

    assert protocol["training"]["precision"] == "fp32"
    assert protocol["model"]["parameter_storage_dtype"] == "float32"
    assert protocol["model"]["forward_precision"] == "float32_no_autocast"
    assert protocol_hash == hashlib.sha256(protocol_path.read_bytes()).hexdigest()

    changed = copy.deepcopy(config)
    changed["training"]["learning_rate"] = 2e-5
    with pytest.raises(RuntimeError, match="conflicts with frozen FP32 protocol"):
        load_fp32_protocol(protocol_path, changed)


def test_fp32_run_override_maps_only_the_completed_legacy_cell(tmp_path: Path):
    mapping = json.loads(DEFAULT_RUN_OVERRIDES.read_text(encoding="utf-8"))
    runs_root = tmp_path / "runs"

    path = _cell_directory(runs_root, "A1", 1, 1729, mapping["overrides"])

    assert path == (
        runs_root / "A1" / "fold_1" / "seed_1729_diagnostic_numerics_v2"
    ).resolve()
    assert set(mapping["overrides"]) == {"A1/fold_1/seed_1729"}


def test_paired_comparison_writes_full_fp32_metric_tables(tmp_path: Path):
    config = public_config(load_config(DEFAULT_CONFIG))
    aggregate_root = tmp_path / "aggregates"
    config["outputs"]["aggregate_dir"] = str(aggregate_root)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    fold_metrics = (
        "average_precision",
        "roc_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "mcc",
        "accuracy",
        "ap_prevalence_lift",
        "false_positives",
        "false_negatives",
    )
    for condition, offset in (("A1", 0.0), ("A2", 0.1)):
        (aggregate_root / condition).mkdir(parents=True)
        rows = []
        for seed in (1729, 2718, 31415):
            for fold in range(1, 6):
                row = {
                    "seed": seed,
                    "fold": fold,
                    "positive_labels": fold,
                    "negative_labels": 100 - fold,
                    "class_weights": '{"0": 1.0, "1": 2.0}',
                }
                row.update({metric: offset + 0.01 for metric in fold_metrics})
                rows.append(row)
        pd.DataFrame(rows).to_csv(
            aggregate_root / condition / "per_fold_seed_metrics.csv", index=False
        )

        seeds = []
        for seed in (1729, 2718, 31415):
            row = {"seed": seed, "mean_fold_average_precision": offset + 0.01}
            row.update(
                {f"pooled_{metric}": offset + 0.01 for metric in fold_metrics}
            )
            seeds.append(row)
        pd.DataFrame(seeds).to_csv(
            aggregate_root / condition / "per_seed_metrics.csv", index=False
        )

        families = []
        for seed in (1729, 2718, 31415):
            for family_id in TRAIN_FAMILIES:
                families.append(
                    {
                        "seed": seed,
                        "family_id": family_id,
                        "average_precision": offset + 0.01,
                        "positive_labels": 1,
                    }
                )
        pd.DataFrame(families).to_csv(
            aggregate_root / condition / "per_family_seed_metrics.csv", index=False
        )

    result = compare(config_path, output=tmp_path / "comparison")

    summary = json.loads((result / "comparison_summary.json").read_text())
    assert summary["paired_cells"] == 15
    assert summary["paired_seeds"] == 3
    assert summary["paired_seed_family_cells"] == 27
    assert summary["fold_seed_differences_A2_minus_A1"]["average_precision"][
        "mean"
    ] == pytest.approx(0.1)
    assert (result / "paired_pooled_seed_comparison.csv").is_file()
    assert (result / "paired_family_seed_comparison.csv").is_file()


def test_strict_identifier_and_target_joins_fail_closed():
    source, key, metadata, context, folds = _synthetic_sources()
    duplicate_key = pd.concat([key, key.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="duplicate item_id"):
        assemble_dataset(
            source,
            duplicate_key,
            metadata,
            context,
            expected_unresolved_item_ids={"1"},
            family_to_fold=folds,
        )
    metadata.loc[0, "clause_text"] = "different"
    with pytest.raises(RuntimeError, match="Target text disagrees"):
        assemble_dataset(
            source,
            key,
            metadata,
            context,
            expected_unresolved_item_ids={"1"},
            family_to_fold=folds,
        )


def _leakage_frame() -> pd.DataFrame:
    rows = [
        ("r1", "f1", 1, "a b c d e f", "lin-x", "", "", ""),
        ("r2", "f2", 2, "a b c d e z", "lin-y", "", "a b c d e f", ""),
        ("r3", "f3", 3, "An endpoint MUST stop", "lin-shared", "", "", ""),
        ("r4", "f4", 4, "An endpoint SHOULD stop", "lin-shared", "", "", ""),
        ("r5", "f5", 5, "unique target", "lin-5", "", "", ""),
    ]
    return pd.DataFrame(
        [
            {
                "record_id": record,
                "family_id": family,
                "document_id": f"{family}-01",
                "revision": "01",
                "fold": fold,
                "target_clause": target,
                "lineage_ids": lineage,
                "section_heading": section,
                "previous_clause": previous,
                "next_clause": following,
                "label": fold % 2,
            }
            for record, family, fold, target, lineage, section, previous, following in rows
        ]
    )


def test_leakage_exclusions_are_label_blind_and_context_is_sanitized():
    frame = _leakage_frame()
    audit = audit_leakage(frame, near_threshold=0.4)
    assert audit["counts"]["near_cross_fold_targets"] >= 1
    assert audit["counts"]["cross_fold_lineage_conflicts"] == 1
    assert audit["counts"]["cross_fold_target_context_overlaps"] == 1
    folds = build_fold_manifests(frame, audit)
    fold1 = folds[0]
    excluded = {item["record_id"] for item in fold1["fitting_exclusions"]}
    assert "r2" in excluded
    assert all(
        item["record_id"] != "r2" or item["role"] != "fitting"
        for item in fold1["context_sanitizations"]
    )
    assert set(fold1["scoring_record_ids"]) == {"r1"}
    assert "r2" not in fold1["fitting_record_ids"]

    sanitized = apply_context_sanitization(
        frame.loc[
            frame["record_id"].astype(str).isin(fold1["fitting_record_ids"])
        ].reset_index(drop=True),
        fold1,
        role="fitting",
    )
    assert set(sanitized["record_id"].astype(str)) == set(fold1["fitting_record_ids"])

    fold2 = folds[1]
    evaluation = frame.loc[
        frame["record_id"].astype(str).isin(fold2["scoring_record_ids"])
    ].reset_index(drop=True)
    sanitized_evaluation = apply_context_sanitization(
        evaluation, fold2, role="evaluation"
    )
    assert (
        sanitized_evaluation.loc[
            sanitized_evaluation["record_id"] == "r2", "previous_clause"
        ].item()
        == ""
    )


def test_context_policy_rejects_missing_non_excluded_records():
    with pytest.raises(RuntimeError, match="references a missing fitting record"):
        apply_context_sanitization(
            pd.DataFrame(
                [
                    {
                        "record_id": "kept",
                        "previous_clause": "context",
                    }
                ]
            ),
            {
                "context_sanitizations": [
                    {
                        "record_id": "missing",
                        "field": "previous_clause",
                        "role": "fitting",
                    }
                ]
            },
            role="fitting",
        )


def test_retained_fitting_context_and_evaluation_context_are_sanitized():
    frame = _leakage_frame()
    frame.loc[frame["record_id"] == "r2", "target_clause"] = "unique clause for r2"
    manifests = build_fold_manifests(
        frame, audit_leakage(frame, near_threshold=0.4)
    )

    fold1 = manifests[0]
    assert "r2" in fold1["fitting_record_ids"]
    assert any(
        item["record_id"] == "r2" and item["role"] == "fitting"
        for item in fold1["context_sanitizations"]
    )
    fitting = frame.loc[
        frame["record_id"].astype(str).isin(fold1["fitting_record_ids"])
    ].reset_index(drop=True)
    sanitized_fitting = apply_context_sanitization(
        fitting, fold1, role="fitting"
    )
    assert (
        sanitized_fitting.loc[
            sanitized_fitting["record_id"] == "r2", "previous_clause"
        ].item()
        == ""
    )

    fold2 = manifests[1]
    evaluation = frame.loc[
        frame["record_id"].astype(str).isin(fold2["scoring_record_ids"])
    ].reset_index(drop=True)
    sanitized_evaluation = apply_context_sanitization(
        evaluation, fold2, role="evaluation"
    )
    assert (
        sanitized_evaluation.loc[
            sanitized_evaluation["record_id"] == "r2", "previous_clause"
        ].item()
        == ""
    )


def test_sanitization_ids_match_retained_role_membership():
    frame = _leakage_frame()
    manifests = build_fold_manifests(
        frame, audit_leakage(frame, near_threshold=0.4)
    )
    for manifest in manifests:
        fitting_ids = set(manifest["fitting_record_ids"])
        scoring_ids = set(manifest["scoring_record_ids"])
        for item in manifest["context_sanitizations"]:
            role_ids = fitting_ids if item["role"] == "fitting" else scoring_ids
            assert item["record_id"] in role_ids


def test_manifest_fix_preserves_scoring_and_exclusion_reasons():
    frame = _leakage_frame()
    manifest = build_fold_manifests(
        frame, audit_leakage(frame, near_threshold=0.4)
    )[0]
    exclusion = next(
        item for item in manifest["fitting_exclusions"] if item["record_id"] == "r2"
    )
    assert exclusion["reasons"]
    assert manifest["scoring_record_ids"] == ["r1"]
    assert set(manifest["fitting_record_ids"]) == {"r3", "r4", "r5"}


def test_real_manifests_have_family_disjointness_unique_scoring_and_no_reserved_fit():
    config = load_config(DEFAULT_CONFIG)
    frame, _, dataset_dir = load_effective_dataset(config)
    verify_dataset(dataset_dir)
    assert not set(frame["family_id"]) & {TCP_FAMILY, DNS_FAMILY}
    scoring = []
    for fold in range(1, 6):
        manifest = __import__("json").loads(
            (dataset_dir / "folds" / f"fold_{fold}.json").read_text(encoding="utf-8")
        )
        assert not set(manifest["held_out_families"]) & set(
            manifest["fitting_families"]
        )
        assert (
            manifest["selection_policy"]["checkpoint_selection"]
            == "none_final_epoch_only"
        )
        assert manifest["selection_policy"]["outer_labels_used_for_selection"] is False
        scoring.extend(manifest["scoring_record_ids"])
    assert len(scoring) == len(set(scoring)) == len(frame)


def test_fitting_only_class_weights():
    fitting = np.array([0, 0, 0, 1])
    assert compute_class_weights(fitting, "A1") == {0: 1.0, 1: 1.0}
    assert compute_class_weights(fitting, "A2") == {0: 1.0, 1: 3.0}
    assert compute_class_weights(np.array([0, 0, 1, 1]), "A2")[1] == 1.0


def test_bfloat16_logits_use_float32_cross_entropy_and_finite_gradients():
    logits = torch.tensor(
        [[0.0, -1000.0], [0.0, 0.125]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    labels = torch.tensor([1, 1], dtype=torch.long)

    loss = _classification_loss(logits, labels, loss_weights=None)

    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(500.3163, abs=0.001)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_float32_logits_have_finite_cross_entropy_and_gradients():
    logits = torch.tensor([[0.25, -0.5], [-0.75, 0.5]], requires_grad=True)
    labels = torch.tensor([0, 1], dtype=torch.long)

    loss = _classification_loss(logits, labels, loss_weights=None)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_float32_cross_entropy_does_not_mask_nonfinite_logits():
    logits = torch.tensor([[float("nan"), 0.0]], dtype=torch.bfloat16)
    labels = torch.tensor([0], dtype=torch.long)

    loss = _classification_loss(logits, labels, loss_weights=None)

    assert not torch.isfinite(loss)


def test_backward_diagnostics_preserve_context_cause_and_prevent_update():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    parameter.grad = torch.tensor([1.0])
    original_error = ValueError("synthetic backward failure")
    context = _numerical_context("fp32", 2, 3, 7)

    def fail_backward() -> None:
        raise original_error

    with pytest.raises(
        RuntimeError,
        match=(
            r"stage=backward; precision fp32, epoch 2, batch 3, "
            r"optimizer step 7: synthetic backward failure"
        ),
    ) as captured:
        _backward_with_diagnostics(fail_backward, context=context)
        optimizer.step()

    assert captured.value.__cause__ is original_error
    assert parameter.item() == pytest.approx(1.0)


def test_successful_backward_diagnostic_invokes_original_operation_once():
    value = torch.tensor(2.0, requires_grad=True)
    context = _numerical_context("bf16", 1, 1, 1)

    _backward_with_diagnostics(value.backward, context=context)

    assert value.grad.item() == pytest.approx(1.0)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_forward_diagnostics_identify_nonfinite_logits_and_parameters(bad_value):
    model = torch.nn.Linear(1, 1)
    logits = torch.tensor([[bad_value, 0.0]])
    context = _numerical_context("bf16", 2, 3, 7)

    with pytest.raises(
        FloatingPointError,
        match=(
            r"Non-finite model logits under precision bf16, epoch 2, batch 3, "
            r"optimizer step 7"
        ),
    ):
        _check_forward_numerics(
            logits, torch.tensor(float("nan")), model, context=context
        )


def test_forward_diagnostics_preserve_nonfinite_cross_entropy_guard():
    model = torch.nn.Linear(1, 1)
    logits = torch.tensor([[0.25, -0.5]])
    context = _numerical_context("fp32", 1, 4, 4)

    with pytest.raises(
        FloatingPointError,
        match=(
            r"Non-finite cross-entropy loss under precision fp32, epoch 1, "
            r"batch 4, optimizer step 4"
        ),
    ):
        _check_forward_numerics(
            logits, torch.tensor(float("nan")), model, context=context
        )


def test_training_labels_are_checked_before_device_transfer():
    context = _numerical_context("fp32", 1, 1, 1)
    _validate_training_labels(torch.tensor([0, 1]), 2, context=context)

    with pytest.raises(ValueError, match="must be binary 0/1"):
        _validate_training_labels(torch.tensor([0, 2]), 2, context=context)
    with pytest.raises(ValueError, match="do not align"):
        _validate_training_labels(torch.tensor([0]), 2, context=context)


def test_nonfinite_gradients_report_clipping_context():
    model = torch.nn.Linear(1, 1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("inf"))
    context = _numerical_context("bf16", 3, 5, 12)
    stats = _gradient_statistics(model)
    assert stats["parameters_with_gradients"] == 2
    assert stats["nonfinite_gradient_tensors"] == 2

    with pytest.raises(
        FloatingPointError,
        match=(
            r"Gradient validation/clipping failed under precision bf16, epoch 3, "
            r"batch 5, optimizer step 12"
        ),
    ):
        _clip_gradients(model, 1.0, context=context)


@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("gradient_value", [0.25, 1e-30, 1e18])
def test_cpu_adamw_diagnostic_statistics_and_update(foreach, gradient_value):
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.01,
        foreach=foreach,
    )
    parameter.grad = torch.full_like(parameter, gradient_value)
    before_parameter = parameter.detach().clone()

    config = _optimizer_configuration(optimizer)
    before_stats = _gradient_statistics(model)
    assert config["optimizer"].endswith("AdamW")
    assert config["parameter_dtypes"] == {"torch.float32": 1}
    assert config["parameter_devices"] == {"cpu": 1}
    assert config["groups"][0]["betas"] == (0.9, 0.999)
    assert config["groups"][0]["eps"] == pytest.approx(1e-8)
    assert config["groups"][0]["foreach"] is foreach
    assert _parameter_statistics(model)["parameter_tensors"] == 1
    assert optimizer.state == {}

    clip_stats = _clip_gradients(
        model,
        1.0,
        context="synthetic CPU AdamW update",
        diagnostic_numerics=True,
    )

    assert clip_stats is not None
    assert clip_stats["before_clip"] == before_stats
    assert clip_stats["clip_grad_norm_returned"] == pytest.approx(
        before_stats["global_norm"]
    )
    assert clip_stats["after_clip"]["nonfinite_gradient_tensors"] == 0
    assert clip_stats["after_clip"]["global_norm"] <= max(
        1.0, before_stats["global_norm"] + 1e-12
    )
    optimizer.step()

    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, before_parameter)
    state = optimizer.state[parameter]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
    assert all(
        not isinstance(value, torch.Tensor) or bool(torch.isfinite(value).all())
        for value in state.values()
    )
    state_stats = _assert_finite_optimizer_state(
        model, optimizer, context="synthetic CPU AdamW update"
    )
    assert state_stats["optimizer_state_tensors"] == 3
    assert state_stats["nonfinite_optimizer_state_tensors"] == 0
    assert state_stats["optimizer_state_dtypes"] == {"torch.float32": 3}


def test_fp32_optimizer_parameters_prevent_half_precision_adamw_corruption():
    model = torch.nn.Linear(1, 1, bias=False).half()
    with torch.no_grad():
        model.weight.fill_(1.0)

    _prepare_model_for_training(model, torch.device("cpu"))

    assert model.weight.dtype == torch.float32
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.01)
    model.weight.grad = torch.full_like(model.weight, 1e-3)
    optimizer.step()

    assert torch.isfinite(model.weight).all()
    state = optimizer.state[model.weight]
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert torch.isfinite(state["exp_avg_sq"]).all()


def test_extreme_finite_gradient_norm_overflow_aborts_before_adamw_update():
    parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    parameter.grad = torch.full_like(parameter, 1e20)
    before_parameter = parameter.detach().clone()
    statistics = _gradient_statistics(model)
    assert statistics["nonfinite_gradient_tensors"] == 0
    assert torch.isfinite(torch.tensor(statistics["global_norm"]))

    with pytest.raises(
        FloatingPointError,
        match=r"Gradient validation/clipping failed.*nonfinite_gradient_tensors': 0",
    ):
        _clip_gradients(
            model,
            1.0,
            context="synthetic CPU extreme-gradient test",
            diagnostic_numerics=True,
        )
        optimizer.step()

    assert optimizer.state == {}
    assert torch.equal(parameter, before_parameter)


def test_cpu_adamw_nonfinite_optimizer_state_is_identified():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    parameter.grad = torch.tensor([0.25])
    optimizer.step()
    optimizer.state[parameter]["exp_avg_sq"].fill_(float("nan"))

    statistics = _optimizer_state_statistics(model, optimizer)

    assert statistics["nonfinite_optimizer_state_tensors"] == 1
    assert statistics["first_nonfinite_optimizer_state_entries"] == [
        "weight.exp_avg_sq(group=0, dtype=torch.float32, nonfinite_elements=1)"
    ]
    with pytest.raises(FloatingPointError, match="Non-finite optimizer state"):
        _assert_finite_optimizer_state(
            model, optimizer, context="synthetic CPU optimizer state test"
        )


def test_cpu_adamw_post_update_parameter_corruption_is_summarized():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    model = torch.nn.Module()
    model.register_parameter("weight", parameter)
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    parameter.grad = torch.tensor([0.25])
    optimizer.step()
    with torch.no_grad():
        parameter.fill_(float("inf"))

    with pytest.raises(
        FloatingPointError,
        match=r"1 tensors; first affected: weight\(dtype=torch.float32, group=0\)",
    ):
        _assert_finite_model_parameters(
            model,
            stage="after optimizer update",
            context="synthetic CPU update",
            optimizer=optimizer,
        )


def test_parameter_corruption_after_synthetic_optimizer_update_is_reported():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    optimizer.step()
    context = _numerical_context("fp32", 1, 1, 1)

    with pytest.raises(
        FloatingPointError,
        match=r"Non-finite model parameters after optimizer update.*weight",
    ):
        _assert_finite_model_parameters(
            model, stage="after optimizer update", context=context
        )


def test_deterministic_target_preserving_preprocessing_and_dynamic_padding():
    tokenizer = FakeTokenizer()
    first = encode_example(
        tokenizer,
        target="abcdefghijklmnopqrstuvwxyz",
        input_mode="target_only",
        max_length=12,
    )
    second = encode_example(
        tokenizer,
        target="abcdefghijklmnopqrstuvwxyz",
        input_mode="target_only",
        max_length=12,
    )
    assert first == second
    assert first.target_truncated and len(first.input_ids) == 12
    assert first.input_ids[0] == 101 and first.input_ids[-1] == 102
    assert first.attention_mask == [1] * 12
    assert first.token_type_ids == [0] * 12
    assert first.input_ids[1:6] == [ord(character) % 97 + 3 for character in "abcde"]
    assert first.input_ids[-6:-1] == [ord(character) % 97 + 3 for character in "vwxyz"]
    without_context = encode_example(
        tokenizer,
        target="same target",
        section="ignored section",
        previous="ignored previous",
        following="ignored next",
        input_mode="target_only",
        max_length=256,
    )
    plain = encode_example(
        tokenizer, target="same target", input_mode="target_only", max_length=256
    )
    assert without_context == plain
    assert all("ignored" not in text for text, _ in tokenizer.calls)
    serialized = serialize_local_context("T", "S", "P", "N")
    assert serialized == "[TARGET]\nT\n\n[SECTION]\nS\n\n[PREVIOUS]\nP\n\n[NEXT]\nN"
    batch = collate_clauses(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "token_type_ids": [0, 0],
                "labels": 0,
                "row_position": 0,
                "target_truncated": False,
                "context_truncated": False,
            },
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "token_type_ids": [0, 0, 0],
                "labels": 1,
                "row_position": 1,
                "target_truncated": False,
                "context_truncated": False,
            },
        ],
        pad_token_id=0,
    )
    assert tuple(batch["input_ids"].shape) == (2, 3)


def test_token_length_audit_uses_shared_encoding_and_reports_overflow():
    tokenizer = FakeTokenizer()
    frame = pd.DataFrame(
        {
            "target_clause": ["short", "abcdefghijklmnopqrstuvwxyz"],
            "section_heading": ["", ""],
            "previous_clause": ["", ""],
            "next_clause": ["", ""],
        }
    )
    report = audit_lengths(tokenizer, frame, max_length=12, input_mode="target_only")
    assert report["rows"] == 2
    assert report["target_truncated_rows"] == 1
    assert report["context_truncated_rows"] == 0
    assert report["target_with_special_tokens"]["max"] == 28
    assert report["encoded_length"]["max"] == 12


def test_optional_token_type_ids_are_not_synthesized():
    class NoTokenTypes(FakeTokenizer):
        model_input_names: ClassVar[list[str]] = ["input_ids", "attention_mask"]

    encoded = encode_example(
        NoTokenTypes(), target="target", input_mode="target_only", max_length=20
    )
    assert encoded.token_type_ids is None


def test_offline_fresh_initialization_does_not_reuse_fold_checkpoint(tmp_path: Path):
    for name in (
        "config.json",
        "tokenizer_config.json",
        "model.safetensors",
        "spm.model",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")
    tokenizer_calls = []
    model_calls = []

    def tokenizer_loader(path, **kwargs):
        tokenizer_calls.append((path, kwargs))
        return object()

    class FakeModel:
        def parameters(self):
            return [SimpleNamespace(requires_grad=True)]

    def model_loader(path, **kwargs):
        model_calls.append((path, kwargs))
        return FakeModel()

    _, first = initialize_pretrained_components(
        tmp_path,
        revision="abc",
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    _, second = initialize_pretrained_components(
        tmp_path,
        revision="abc",
        tokenizer_loader=tokenizer_loader,
        model_loader=model_loader,
    )
    assert first is not second and len(model_calls) == len(tokenizer_calls) == 2
    assert all(
        call[1]["local_files_only"] is True for call in model_calls + tokenizer_calls
    )
    assert all(
        call[1]["num_labels"] == 2
        and call[1]["label2id"] == {"No": 0, "Yes": 1}
        and call[1]["id2label"] == {0: "No", 1: "Yes"}
        for call in model_calls
    )
    assert all(call[1]["fix_mistral_regex"] is False for call in tokenizer_calls)
    assert all("fix_mistral_regex" not in call[1] for call in model_calls)
    assert all("checkpoint" not in str(call[0]).casefold() for call in model_calls)


@pytest.mark.parametrize("failure_component", ["tokenizer", "model"])
def test_pretrained_loading_diagnostics_preserve_component_and_cause(
    tmp_path: Path, failure_component: str
):
    for name in (
        "config.json",
        "tokenizer_config.json",
        "model.safetensors",
        "spm.model",
    ):
        (tmp_path / name).write_text("x", encoding="utf-8")

    original_error = ValueError(f"synthetic {failure_component} failure")

    def tokenizer_loader(_path, **_kwargs):
        assert _path == tmp_path
        assert _kwargs["local_files_only"] is True
        if failure_component == "tokenizer":
            raise original_error
        return object()

    class FakeModel:
        def parameters(self):
            return [SimpleNamespace(requires_grad=True)]

    def model_loader(_path, **_kwargs):
        assert _path == tmp_path
        assert _kwargs["local_files_only"] is True
        if failure_component == "model":
            raise original_error
        return FakeModel()

    with pytest.raises(
        RuntimeError,
        match=(
            rf"stage=loading component={failure_component} model_path=.*"
            rf"synthetic {failure_component} failure"
        ),
    ) as captured:
        initialize_pretrained_components(
            tmp_path,
            revision="abc",
            tokenizer_loader=tokenizer_loader,
            model_loader=model_loader,
        )

    assert captured.value.__cause__ is original_error


def test_prediction_alignment_and_metrics_with_undefined_handling():
    predictions = pd.DataFrame({"record_id": ["a", "b"]})
    validate_prediction_alignment(predictions, {"a", "b"})
    with pytest.raises(RuntimeError, match="duplicated"):
        validate_prediction_alignment(pd.DataFrame({"record_id": ["a", "a"]}), {"a"})
    metric = binary_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert metric["average_precision"] == pytest.approx(5 / 6)
    assert metric["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    undefined = binary_metrics(np.array([0, 0]), np.array([0.1, 0.2]))
    assert undefined["average_precision"] is None and undefined["roc_auc"] is None


def test_across_seed_metric_aggregation_uses_sample_standard_deviation():
    per_seed = pd.DataFrame(
        {"seed": [1, 2, 3], "pooled_average_precision": [0.1, 0.2, 0.3]}
    )
    summary = across_seed_summary(per_seed, ["pooled_average_precision"])
    assert summary["pooled_average_precision"]["mean"] == pytest.approx(0.2)
    assert summary["pooled_average_precision"]["sample_std"] == pytest.approx(0.1)


def test_immutable_outputs_never_overwrite(tmp_path: Path):
    destination = tmp_path / "run"
    with immutable_directory(destination) as building:
        (building / "complete.txt").write_text("first", encoding="utf-8")
    with (
        pytest.raises(FileExistsError, match="already exists"),
        immutable_directory(destination),
    ):
        pass
    assert (destination / "complete.txt").read_text(encoding="utf-8") == "first"
