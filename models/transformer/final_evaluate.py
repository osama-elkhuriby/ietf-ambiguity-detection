"""One-shot evaluator for the structurally isolated DNS silver test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.classical.feature_pipeline import sha256_file
from models.transformer.dataset import ROOT
from models.transformer.modeling import ClauseDataset, collate_clauses
from models.transformer.train import _metrics, _predict

DEFAULT_SEALED_DIR = (
    ROOT / "corpus/processed/transformer_partitioned/v1/sealed_test"
)
DEFAULT_LOCK = ROOT / "experiments/transformer/final_dns.lock.json"


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(file_path)))
    return digest.hexdigest()


def acquire_final_lock(lock_path: Path, payload: dict[str, Any]) -> None:
    """Claim the final evaluation exactly once using exclusive file creation."""

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with lock_path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    except FileExistsError as error:
        raise RuntimeError(
            f"Final evaluation lock already exists; refusing a second run: {lock_path}"
        ) from error


def load_sealed_test(sealed_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the sealed rows; this function is intentionally confined to this module."""

    manifest_path = sealed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_role") != "sealed_final_test_only":
        raise RuntimeError("Artifact is not marked for sealed final evaluation")
    info = manifest["sealed_file"]
    path = sealed_dir / info["path"]
    if sha256_file(path) != info["sha256"]:
        raise RuntimeError("Sealed test hash does not match its manifest")
    frame = pd.read_csv(path).fillna("")
    if len(frame) != int(info["rows"]) or set(frame["split"]) != {"test"}:
        raise RuntimeError("Sealed test contents violate the manifest")
    if frame["record_id"].duplicated().any():
        raise RuntimeError("Sealed test contains duplicate record IDs")
    return frame, manifest


def run(args: argparse.Namespace) -> Path:
    if not args.final:
        raise PermissionError("Final evaluation requires the explicit --final flag")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    checkpoint = args.checkpoint.resolve()
    calibration_path = args.calibration.resolve()
    run_config_path = args.run_config.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Final output already exists: {output_dir}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    required_calibration = {"temperature", "negative_threshold", "positive_threshold"}
    missing = required_calibration - set(calibration)
    if missing:
        raise ValueError(f"Calibration artifact is missing: {sorted(missing)}")
    temperature = float(calibration["temperature"])
    negative_threshold = float(calibration["negative_threshold"])
    positive_threshold = float(calibration["positive_threshold"])
    if temperature <= 0 or not 0 <= negative_threshold < positive_threshold <= 1:
        raise ValueError("Calibration temperature or thresholds are invalid")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    checkpoint_hash = _directory_hash(checkpoint)
    sealed_manifest_path = args.sealed_dir.resolve() / "manifest.json"
    sealed_manifest_hash = sha256_file(sealed_manifest_path)
    acquire_final_lock(
        args.lock.resolve(),
        {
            "schema_version": 1,
            "claimed_at": datetime.now(UTC).isoformat(),
            "purpose": "single DNS silver-label final evaluation",
            "sealed_manifest_sha256": sealed_manifest_hash,
            "checkpoint_sha256": checkpoint_hash,
            "calibration_sha256": sha256_file(calibration_path),
            "run_config_sha256": sha256_file(run_config_path),
        },
    )

    frame, sealed_manifest = load_sealed_test(args.sealed_dir.resolve())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32
    )
    dataset = ClauseDataset(
        frame,
        tokenizer,
        variant=run_config["variant"],
        max_length=int(run_config["tokenization"]["max_length"]),
    )
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Final checkpoint tokenizer has no padding token")
    loader = DataLoader(
        dataset,
        batch_size=int(run_config["training"]["eval_batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=partial(
            collate_clauses, pad_token_id=int(tokenizer.pad_token_id)
        ),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    positions, logits = _predict(model, loader, device, use_bf16=False)
    uncalibrated = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[:, 1]
    calibrated = torch.softmax(
        torch.from_numpy(logits) / temperature, dim=1
    ).numpy()[:, 1]
    status = np.where(
        calibrated < negative_threshold,
        "clear",
        np.where(calibrated >= positive_threshold, "ambiguous", "abstain"),
    )
    predictions = frame.iloc[positions][
        ["record_id", "family_id", "label", "label_tier"]
    ].reset_index(drop=True)
    predictions["logit_0"] = logits[:, 0]
    predictions["logit_1"] = logits[:, 1]
    predictions["uncalibrated_probability"] = uncalibrated
    predictions["calibrated_probability"] = calibrated
    predictions["abstention_status"] = status

    temporary = output_dir.parent / f".{output_dir.name}.building-{uuid.uuid4().hex}"
    try:
        temporary.mkdir(parents=True)
        predictions.to_csv(
            temporary / "test_predictions.csv", index=False, lineterminator="\n"
        )
        _json(
            temporary / "metrics_at_0_5.json",
            _metrics(frame.iloc[positions]["label"].to_numpy(dtype=np.int64), calibrated),
        )
        _json(
            temporary / "run_manifest.json",
            {
                "schema_version": 1,
                "claim_lock_sha256": sha256_file(args.lock.resolve()),
                "sealed_manifest_sha256": sealed_manifest_hash,
                "sealed_file_sha256": sealed_manifest["sealed_file"]["sha256"],
                "checkpoint_sha256": checkpoint_hash,
                "calibration_sha256": sha256_file(calibration_path),
                "description": "agreement with LLM-consensus silver labels",
                "rows": len(frame),
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sealed-dir", type=Path, default=DEFAULT_SEALED_DIR)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--final", action="store_true")
    return parser.parse_args()


def main() -> int:
    output = run(parse_args())
    print(f"Final evaluation written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
