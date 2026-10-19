"""Rebuild analysis from completed pilot runs without re-running training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from models.transformer_final.common import read_json, resolve_path, sha256_file
from models.transformer_final.config import load_config
from models.transformer_final.dataset import load_effective_dataset
from models.transformer_final.experiment2 import (
    DEFAULT_EXPERIMENT_CONFIG,
    DEFAULT_PROTOCOL,
    load_annotation_ready_protocol,
)
from models.transformer_final import experiment2_analysis
from models.transformer_final.experiment2_runner import (
    _atomic_json,
    _inventory_path,
)


def rebuild_analysis() -> Path:
    config = load_config(DEFAULT_EXPERIMENT_CONFIG)
    protocol, _ = load_annotation_ready_protocol(DEFAULT_PROTOCOL, config)
    effective, _, _ = load_effective_dataset(config)
    inventory_path = _inventory_path(config)
    inventory = read_json(inventory_path)
    expected_cells = {
        f"fold_{fold}/seed_{protocol['training']['seed']}"
        for fold in protocol["training"]["folds"]
    }
    if set(inventory["cells"]) != expected_cells or any(
        entry.get("status") != "complete" for entry in inventory["cells"].values()
    ):
        raise RuntimeError("Analysis recovery requires exactly five validated cells")

    analysis_path = resolve_path(config["outputs"]["aggregate_dir"])
    if analysis_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable analysis: {analysis_path}")
    target_text = effective[["record_id", "target_clause"]]
    original_error_examples = experiment2_analysis._error_examples

    def error_examples_with_target_text(frame: pd.DataFrame) -> pd.DataFrame:
        if "target_clause" in frame:
            return original_error_examples(frame)
        enriched = frame.merge(
            target_text,
            on="record_id",
            how="left",
            validate="one_to_one",
            indicator=True,
            sort=False,
        )
        if enriched["_merge"].ne("both").any():
            raise RuntimeError("Cannot align error examples to frozen target clauses")
        return original_error_examples(enriched.drop(columns="_merge"))

    original_write_report = experiment2_analysis._write_report
    adapter_hash = sha256_file(Path(__file__).resolve())

    def write_report_with_recovery_provenance(
        **kwargs: Any,
    ) -> None:
        threshold_table = kwargs["threshold_pooled"].copy()
        threshold_table["tn"] = threshold_table["true_negatives"]
        kwargs["threshold_pooled"] = threshold_table
        original_write_report(**kwargs)
        report_path = kwargs["destination"] / "research_report.md"
        with report_path.open("a", encoding="utf-8") as report:
            report.write(
                "\n## Analysis-only recovery\n\n"
                "The validated held-out prediction artifacts intentionally omit "
                "clause text. For error-example reporting only, `target_clause` "
                "was joined by `record_id` from the protocol-pinned effective "
                "dataset; no training, predictions, metrics, or run artifacts "
                "were changed. Recovery adapter SHA-256: "
                f"`{adapter_hash}`.\n"
            )

    experiment2_analysis._error_examples = error_examples_with_target_text
    experiment2_analysis._write_report = write_report_with_recovery_provenance
    try:
        result = experiment2_analysis.create_experiment2_analysis(
            config_path=DEFAULT_EXPERIMENT_CONFIG,
            protocol_path=DEFAULT_PROTOCOL,
            inventory_path=inventory_path,
            analysis_path=analysis_path,
        )
    finally:
        experiment2_analysis._error_examples = original_error_examples
        experiment2_analysis._write_report = original_write_report

    inventory["analysis"] = str(result)
    inventory["analysis_recovery"] = {
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": adapter_hash,
        "reason": "Held-out prediction CSVs omit target_clause; joined for error examples only.",
    }
    _atomic_json(inventory_path, inventory)
    print(f"Experiment 2 analysis complete: {result}")
    return result


if __name__ == "__main__":
    rebuild_analysis()
