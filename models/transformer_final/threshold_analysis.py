"""Exploratory post-hoc threshold tables; never an unbiased outer-fold estimate."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import immutable_directory, write_json
from .metrics import binary_metrics


def analyze(predictions_path: Path, *, output: Path) -> Path:
    frame = pd.read_csv(predictions_path)
    required = {"label", "probability_yes"}
    if not required.issubset(frame):
        raise ValueError(
            f"Predictions are missing columns: {sorted(required - set(frame))}"
        )
    rows = []
    for threshold in np.linspace(0.05, 0.95, 19):
        metric = binary_metrics(
            frame["label"].to_numpy(),
            frame["probability_yes"].to_numpy(),
            threshold=float(threshold),
        )
        rows.append(
            {
                key: value
                for key, value in metric.items()
                if key != "confusion_matrix" and key != "undefined_metric_warnings"
            }
        )
    with immutable_directory(output.resolve()) as building:
        pd.DataFrame(rows).to_csv(
            building / "exploratory_thresholds.csv", index=False, lineterminator="\n"
        )
        write_json(
            building / "WARNING.json",
            {
                "status": "exploratory_only",
                "warning": "Thresholds examined on reported outer-fold predictions must not be presented as unbiased performance or selected for the same evaluation.",
                "recommended_future_use": "Select/calibrate only within separately nested development predictions, then evaluate once on untouched data.",
            },
        )
    return output.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        f"Exploratory threshold analysis written to {analyze(args.predictions, output=args.output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
