from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from models.llm.final_annotation_dataset import (
    DNS_FAMILY,
    TCP_FAMILY,
    TRAIN_FAMILIES,
    build_final_annotation_dataset,
    verify_final_annotation_dataset,
)


def _row(item_id: int, family: str, revision: int, label: str) -> dict[str, str]:
    return {
        "item_id": str(item_id),
        "document": f"{family}-{revision:02d}",
        "section_heading": "Test section",
        "previous_clause": f"Previous context {item_id}.",
        "target_clause": f"Target requirement {item_id} MUST be implemented.",
        "next_clause": f"Following context {item_id}.",
        "final_label": label,
        "final_source": "model_agreement",
        "final_status": "resolved",
    }


def test_builder_excludes_unresolved_and_physically_separates_reservations(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, str]] = []
    item_id = 1
    for family in TRAIN_FAMILIES:
        for offset in range(12):
            rows.append(_row(item_id, family, 1, "Yes" if offset < 6 else "No"))
            item_id += 1
    for family in (TCP_FAMILY, DNS_FAMILY):
        rows.append(_row(item_id, family, 1, "Yes"))
        item_id += 1
        rows.append(_row(item_id, family, 1, "No"))
        item_id += 1
    unresolved = _row(2161, TRAIN_FAMILIES[0], 1, "No")
    unresolved["final_source"] = "unresolved"
    unresolved["final_status"] = "unresolved_blank_manual_verdict"
    rows.append(unresolved)
    source = tmp_path / "final_annotation_ready.csv"
    pd.DataFrame(rows).to_csv(source, index=False, lineterminator="\n")
    output = tmp_path / "prepared"

    manifest = build_final_annotation_dataset(
        source=source,
        output_root=output,
        expected_source_sha256=None,
    )

    assert manifest["resolved_rows"] == len(rows) - 1
    assert manifest["excluded_unresolved_item_ids"] == ["2161"]
    assert manifest["partition_counts"]["development_train"]["rows"] == 108
    assert manifest["partition_counts"]["reserved_tcp_validation"]["rows"] == 2
    assert manifest["partition_counts"]["sealed_dns_test"]["rows"] == 2
    assert manifest["cross_partition_audit"]["all_clear"] is True
    train = pd.read_csv(output / "development/train.csv")
    assert set(train["split"]) == {"train"}
    assert set(train["family_id"]) == set(TRAIN_FAMILIES)
    assert "2161" not in set(train["source_item_id"].astype(str))
    verify_final_annotation_dataset(output)


def test_frozen_in_context_config_matches_prepared_dataset() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = (
        root / "corpus/processed/llm_final_annotation_ready/v1/manifest.json"
    )
    if not manifest_path.is_file():
        pytest.skip("Generated final-annotation dataset has not been built")
    config = yaml.safe_load(
        (
            root
            / "experiments/llm/configs/qwen3_32b_final_annotation_ready_in_context.yaml"
        ).read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert config["source_annotation_sha256"] == manifest["source_sha256"]
    assert (
        config["development_train_sha256"]
        == manifest["artifacts"]["development_train_sha256"]
    )
    assert (
        config["selection"]["fold_assignment_sha256"]
        == manifest["artifacts"]["fold_assignment_sha256"]
    )
    assert config["parameter_updates"] is False
