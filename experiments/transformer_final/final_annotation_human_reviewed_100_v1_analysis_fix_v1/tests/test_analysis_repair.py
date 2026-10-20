from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.transformer_final.final_annotation_human_reviewed_100_v1 import (
    analysis as parent_analysis,
)
from experiments.transformer_final.final_annotation_human_reviewed_100_v1_analysis_fix_v1 import (
    analysis as repair,
)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["1", "2", "3"],
            "record_id": ["r1", "r2", "r3"],
            "family_id": ["family-a", "family-a", "family-b"],
            "document_id": ["doc-a", "doc-a", "doc-b"],
            "revision": [1, 1, 2],
            "label": [0, 1, 0],
            "fold": [1, 1, 2],
            "target_clause": ["a", "b", "c"],
        }
    )


def _predictions() -> pd.DataFrame:
    return _dataset()[repair.IDENTITY_COLUMNS + ["fold", "label"]].copy()


def test_canonical_metadata_merge_preserves_family_and_grouping() -> None:
    merged = repair.merge_metadata(_predictions(), _dataset())
    assert (
        "family_id" in merged
        and "family_id_x" not in merged
        and "family_id_y" not in merged
    )
    assert merged.groupby("family_id").size().to_dict() == {
        "family-a": 2,
        "family-b": 1,
    }
    assert merged["target_clause"].tolist() == ["a", "b", "c"]


def test_corrupted_family_is_rejected() -> None:
    prediction = _predictions()
    prediction.loc[0, "family_id"] = "corrupt-family"
    with pytest.raises(RuntimeError, match="disagree"):
        repair.merge_metadata(prediction, _dataset())


def test_unmatched_record_is_rejected() -> None:
    prediction = _predictions()
    prediction.loc[0, "record_id"] = "missing"
    with pytest.raises(RuntimeError, match="disagree"):
        repair.merge_metadata(prediction, _dataset())


def test_duplicate_metadata_identity_is_rejected() -> None:
    dataset = pd.concat([_dataset(), _dataset().iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="not unique"):
        repair.merge_metadata(_predictions(), dataset)


@pytest.mark.parametrize(
    ("column", "value", "message"), [("label", 1, "labels"), ("fold", 5, "folds")]
)
def test_label_and_fold_integrity_is_preserved(
    column: str, value: int, message: str
) -> None:
    prediction = _predictions()
    prediction.loc[0, column] = value
    with pytest.raises(RuntimeError, match=message):
        repair.merge_metadata(prediction, _dataset())


def test_ecdf_and_threshold_protocol_are_parent_implementations() -> None:
    assert repair.empirical_cdf is parent_analysis.empirical_cdf
    assert (
        repair.select_precision_threshold is parent_analysis.select_precision_threshold
    )
    mapped = repair.empirical_cdf([0.1, 0.2, 0.2, 0.9], [0.05, 0.2, 0.5, 1.0])
    assert mapped.tolist() == [0.0, 0.75, 0.75, 1.0]
    selected = repair.select_precision_threshold(
        [1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1], 0.5
    )
    assert selected["threshold"] == 0.7
    assert selected["recall"] == 1.0
    assert selected["precision"] == pytest.approx(2 / 3)


def test_analysis_adapter_writes_only_to_repair_output_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "repair-output"
    monkeypatch.setattr(repair, "OUTPUT_ROOT", output)
    monkeypatch.setattr(repair, "verify_parent_artifacts", lambda: {"status": "PASS"})
    original_merge = parent_analysis._merge_metadata
    original_csv = parent_analysis._write_csv
    original_text = parent_analysis._write_text
    original_validation = parent_analysis.validate_analysis

    def fake_parent_analyze(
        *, bootstrap_replicates: int | None = None
    ) -> dict[str, object]:
        assert bootstrap_replicates == 7
        assert parent_analysis._merge_metadata is repair.merge_metadata
        assert parent_analysis._write_csv is repair._write_output_csv
        assert parent_analysis._write_text is repair._write_output_text
        assert parent_analysis.validate_analysis is repair.validate_analysis
        parent_analysis._write_csv("probe.csv", pd.DataFrame({"value": [1]}))
        parent_analysis._write_text(repair.PARENT_ROOT / "probe.md", "isolated\n")
        return {"status": "PASS"}

    monkeypatch.setattr(parent_analysis, "analyze", fake_parent_analyze)
    result = repair.analyze(bootstrap_replicates=7)
    assert result["status"] == "PASS"
    assert (output / "probe.csv").is_file() and (output / "probe.md").is_file()
    assert not (repair.PARENT_ROOT / "probe.csv").exists()
    assert not (repair.PARENT_ROOT / "probe.md").exists()
    assert parent_analysis._merge_metadata is original_merge
    assert parent_analysis._write_csv is original_csv
    assert parent_analysis._write_text is original_text
    assert parent_analysis.validate_analysis is original_validation


def test_no_training_surface_or_model_deserialization() -> None:
    cli_source = (repair.REPAIR_ROOT / "cli.py").read_text(encoding="utf-8")
    assert "train-outer" not in cli_source and "train-inner" not in cli_source
    source = Path(repair.__file__).read_text(encoding="utf-8")
    assert "torch.load" not in source and "from_pretrained" not in source
    assert np.isfinite([0.0, 1.0]).all()
