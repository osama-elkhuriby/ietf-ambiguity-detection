from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.transformer_final.final_annotation_661_v1 import analysis, core, runner


def test_authoritative_csv_identity_schema_and_labels() -> None:
    config = core.load_local_config()
    source = core._resolve_repo_path(config["data"]["source_csv"])
    assert core.sha256_file(source) == core.EXPECTED_DATASET_SHA256
    frame = pd.read_csv(source, dtype=str, keep_default_na=False)
    assert frame.columns.tolist() == core.REQUIRED_SOURCE_COLUMNS
    assert len(frame) == 2406
    assert frame["final_label"].value_counts().to_dict() == {"No": 2247, "Yes": 159}
    assert not frame["item_id"].duplicated().any()
    assert not frame["target_clause"].str.strip().eq("").any()


def test_effective_cohort_governance_and_label_normalization() -> None:
    frame, audit = core.build_effective_dataset(core.load_local_config())
    assert (len(frame), int(frame.label.sum()), int((frame.label == 0).sum())) == (2125, 136, 1989)
    assert set(frame.label) == {0, 1}
    assert set(frame.family_id) == set(core.load_local_config()["partitions"]["train_families"])
    assert not set(frame.family_id) & {
        "draft-ietf-tcpm-rfc793bis",
        "draft-ietf-dnsop-svcb-https",
    }
    assert audit["reserved_label_access_for_feasibility"] is False
    assert audit["unresolved_item_ids"] == ["2161"]


def test_path_regression_accepts_string_and_path(tmp_path: Path) -> None:
    assert core.as_path(str(tmp_path)) == tmp_path.resolve()
    assert core.as_path(tmp_path) == tmp_path.resolve()
    payload = tmp_path / "value.json"
    payload.write_text('{"ok": true}\n', encoding="utf-8")
    assert core._read_json(str(payload)) == {"ok": True}


def test_threshold_selection_and_tie_breaking() -> None:
    labels = [1, 0, 1, 0]
    scores = [0.9, 0.8, 0.7, 0.1]
    selected = analysis.select_precision_threshold(labels, scores, 0.5)
    assert selected["feasible"] is True
    assert selected["recall"] == 1.0
    assert selected["threshold"] == 0.7
    assert selected["precision"] == pytest.approx(2 / 3)


def test_threshold_infeasibility_has_no_fallback() -> None:
    selected = analysis.select_precision_threshold([1, 0, 0], [0.1, 0.9, 0.8], 0.8)
    assert selected["feasible"] is False
    assert np.isnan(selected["threshold"])
    assert "no non-empty" in selected["reason"]


def test_ecdf_mapping_is_fitting_side_and_deterministic() -> None:
    mapped = analysis.empirical_cdf([0.1, 0.2, 0.2, 0.9], [0.05, 0.2, 0.5, 1.0])
    assert mapped.tolist() == [0.0, 0.75, 0.75, 1.0]


def test_ensemble_id_alignment_rejects_duplicates() -> None:
    left = pd.DataFrame({"record_id": ["a", "b"], "label": [0, 1], "score": [0.1, 0.8]})
    right = pd.DataFrame({"record_id": ["a", "b"], "label": [0, 1], "score": [0.2, 0.7]})
    joined = left.merge(right, on=["record_id", "label"], validate="one_to_one")
    assert joined.record_id.tolist() == ["a", "b"]
    with pytest.raises(pd.errors.MergeError):
        left.merge(pd.concat([right, right.iloc[[0]]]), on=["record_id", "label"], validate="one_to_one")


def test_top_k_ordering_precision_and_recall() -> None:
    frame = pd.DataFrame(
        {
            "record_id": ["b", "a", "c", "d"],
            "label": [1, 0, 1, 0],
            "yes_score": [0.8, 0.8, 0.7, 0.1],
        }
    )
    selected = analysis.stable_top_k(frame, "yes_score", 2)
    assert selected.record_id.tolist() == ["a", "b"]
    row = analysis.top_k_rows("focal", frame, scope="synthetic")[0]
    assert row["reviewed"] == 4
    assert row["real_yes"] == 2
    assert row["precision_at_k"] == 0.5
    assert row["recall_at_k"] == 1.0


def _error_fixture() -> pd.DataFrame:
    rows = []
    for item_id, label, score, predicted in [
        ("1", 0, 0.9, 1),
        ("2", 0, 0.8, 1),
        ("3", 1, 0.1, 0),
        ("4", 1, 0.2, 0),
    ]:
        rows.append(
            {
                "item_id": item_id,
                "record_id": f"r{item_id}",
                "family_id": "family",
                "outer_fold": 1,
                "document_id": "doc",
                "revision": "1",
                "clause_ids": "[]",
                "section_ids": "[]",
                "lineage_ids": "[]",
                "section_heading": "heading",
                "previous_clause": "previous",
                "target_clause": "full target text ✓",
                "next_clause": "next",
                "context_source_ids": "[]",
                "context_resolution_status": "resolved",
                "alignment_source_path": "source",
                "source_hashes": "{}",
                "final_source": "model_agreement",
                "final_status": "resolved",
                "label": label,
                "yes_score": score,
                "predicted_label": predicted,
                "decision_threshold": 0.5,
                "threshold_policy": "nested_precision_0.30",
            }
        )
    return pd.DataFrame(rows)


def test_false_positive_and_false_negative_selection() -> None:
    frame = _error_fixture()
    fp = analysis._error_list(frame, "focal", "false_positive")
    fn = analysis._error_list(frame, "focal", "false_negative")
    assert fp.item_id.tolist() == ["1", "2"]
    assert fn.item_id.tolist() == ["3", "4"]
    assert set(fp.actual_label) == {"No"} and set(fp.predicted_label) == {"Yes"}
    assert set(fn.actual_label) == {"Yes"} and set(fn.predicted_label) == {"No"}
    assert fp.columns.tolist() == analysis.ERROR_COLUMNS


def test_threshold_independent_disagreement_ordering() -> None:
    frame = _error_fixture()
    no_high = analysis._disagreement_list(frame, "focal", 0)
    yes_low = analysis._disagreement_list(frame, "focal", 1)
    assert no_high.item_id.tolist() == ["1", "2"]
    assert yes_low.item_id.tolist() == ["3", "4"]


def test_complete_untruncated_text_and_utf8_csv(tmp_path: Path) -> None:
    frame = _error_fixture()
    report = analysis._error_list(frame, "focal", "false_positive")
    path = tmp_path / "errors.csv"
    report.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    raw = path.read_bytes()
    assert "full target text ✓".encode() in raw
    reread = pd.read_csv(path, keep_default_na=False)
    assert reread.loc[0, "target_clause"] == "full target text ✓"


class FakeParameter:
    def __init__(self, size: int = 1) -> None:
        self.size = size
        self.requires_grad = True

    def numel(self) -> int:
        return self.size

    def requires_grad_(self, value: bool) -> FakeParameter:
        self.requires_grad = value
        return self

    def detach(self) -> FakeParameter:
        return self

    def to(self, **_: object) -> FakeParameter:
        return self

    def clone(self) -> FakeParameter:
        return FakeParameter(self.size)


class FakeEncoder:
    def __init__(self) -> None:
        self.layer = [object() for _ in range(12)]


class FakeDeberta:
    def __init__(self) -> None:
        self.encoder = FakeEncoder()


class FakeModel:
    def __init__(self) -> None:
        self.deberta = FakeDeberta()
        self._values = [("deberta.embeddings.word_embeddings.weight", FakeParameter(10))]
        self._values.extend(
            (f"deberta.encoder.layer.{index}.attention.weight", FakeParameter(2))
            for index in range(12)
        )
        self._values.extend(
            [("pooler.dense.weight", FakeParameter(3)), ("classifier.weight", FakeParameter(4))]
        )

    def named_parameters(self):
        return iter(self._values)

    def parameters(self):
        return (value for _, value in self._values)


def test_partial4_trainable_membership_on_synthetic_structure() -> None:
    model = FakeModel()
    audit = runner.freeze_upper_layers(model, 4)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    frozen = [name for name, value in model.named_parameters() if not value.requires_grad]
    assert all(f"layer.{index}." not in "|".join(trainable) for index in range(8))
    assert all(any(f"layer.{index}." in name for name in trainable) for index in range(8, 12))
    assert "pooler.dense.weight" in trainable and "classifier.weight" in trainable
    assert "deberta.embeddings.word_embeddings.weight" in frozen
    assert audit["last_trainable_layers"] == 4


def test_focal_loss_configuration_and_finiteness() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, -1.0], [-1.0, 2.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1], dtype=torch.long)
    weights = torch.tensor([1.0, 4.0], dtype=torch.float32)
    loss = runner.focal_loss(logits, labels, weights)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert core.load_local_config()["methods"]["focal"]["gamma"] == 2.0


def test_metric_undefined_values_are_nan() -> None:
    metrics = analysis.metric_row([1, 1], [0.2, 0.8], [0, 0])
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["precision"])
    assert np.isnan(metrics["mcc"])


def test_no_training_on_import_and_training_is_explicit() -> None:
    source = (core.EXPERIMENT_ROOT / "cli.py").read_text(encoding="utf-8")
    assert "train-outer" in source and "train-inner" in source
    assert "if __name__ == \"__main__\"" in source
    runner_source = (core.EXPERIMENT_ROOT / "runner.py").read_text(encoding="utf-8")
    assert "CUDA is unavailable; refusing CPU fallback before model loading" in runner_source
    assert "automatic_retry" in runner_source


def test_prepared_fold_feasibility_and_protocol_if_frozen() -> None:
    if not core.PROTOCOL_PATH.exists():
        pytest.skip("Preparation freeze occurs after pre-freeze unit tests")
    result = core.preflight()
    assert result["nested_feasible"] is True
    counts = pd.read_csv(core.EXPERIMENT_ROOT / "FOLD_CLASS_COUNTS.csv")
    assert (counts.scope == "outer").sum() == 5
    assert (counts.scope == "inner").sum() == 15
    assert counts[["training_yes", "training_no", "scoring_yes", "scoring_no"]].min().min() > 0

