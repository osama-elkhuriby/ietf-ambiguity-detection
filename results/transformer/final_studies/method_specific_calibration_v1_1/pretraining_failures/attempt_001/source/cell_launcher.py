"""Execute exactly one new V1.1 GPU cell; reused cells are rejected."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.transformer_final.recall_improvement_study_v1.cell_launcher import (
    execute_cell,
)
from models.transformer_final.common import read_json, sha256_file, write_json


def run(spec_path: Path) -> Path:
    spec_path = spec_path.resolve()
    spec = read_json(spec_path)
    if (
        spec.get("study_id") != "method_specific_calibration_v1_1"
        or spec.get("reuse_by_reference") is not False
        or spec.get("automatic_retry") is not False
        or spec.get("fresh_base_initialization") is not True
        or spec.get("method") not in {"focal", "partial_4"}
    ):
        raise RuntimeError("Invalid or non-executable V1.1 cell specification")
    output = execute_cell(spec_path)
    manifest_path = output / "run_manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "calibration_study_id": spec["study_id"],
            "calibration_protocol_sha256": spec["protocol_sha256"],
            "calibration_cell_id": spec["cell_id"],
            "calibration_attempt_id": spec["attempt_id"],
            "calibration_spec_sha256": sha256_file(spec_path),
            "calibration_outer_fold": int(spec["outer_fold"]),
            "calibration_inner_fold": int(spec["inner_fold"]),
            "calibration_automatic_retry": False,
            "calibration_reuse_by_reference": False,
            "compatibility_fix_version": "V1.1",
        }
    )
    write_json(manifest_path, manifest)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

