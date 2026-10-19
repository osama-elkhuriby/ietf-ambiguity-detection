"""Offline preflight checks that never initialize a tokenizer or model."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .common import hash_directory, immutable_directory, resolve_path, write_json
from .config import DEFAULT_CONFIG, load_config


def require_local_model_files(path: Path) -> list[str]:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Local pretrained model directory is missing: {path}. "
            "Download microsoft/deberta-v3-base yourself, then rerun; network fallback is disabled."
        )
    names = {file.name for file in path.iterdir() if file.is_file()}
    missing = [
        name for name in ("config.json", "tokenizer_config.json") if name not in names
    ]
    if not ({"model.safetensors", "pytorch_model.bin"} & names) and not any(
        name.startswith("model-") and name.endswith(".safetensors") for name in names
    ):
        missing.append("model.safetensors or pytorch_model.bin")
    if not ({"spm.model", "tokenizer.json", "sentencepiece.bpe.model"} & names):
        missing.append("spm.model/tokenizer.json/sentencepiece.bpe.model")
    if missing:
        raise RuntimeError(
            f"Local model directory is incomplete ({path}): missing {', '.join(missing)}"
        )
    return sorted(names)


def dependency_versions() -> dict[str, str]:
    packages = (
        "torch",
        "transformers",
        "tokenizers",
        "sentencepiece",
        "pandas",
        "numpy",
        "scikit-learn",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(
                f"Required dependency is not installed: {package}"
            ) from error
    return versions


def build_preflight_report(
    config: dict[str, Any], model_path: Path, *, require_cuda: bool
) -> dict[str, Any]:
    require_local_model_files(model_path)
    model_source_hash, model_files = hash_directory(model_path)
    torch = importlib.import_module("torch")
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is required for the requested training preflight but is not visible to PyTorch"
        )
    return {
        "schema_version": 1,
        "configuration_valid": True,
        "config_sha256": config["_config_sha256"],
        "offline_only": True,
        "model_repository": config["model"]["repository"],
        "model_revision": config["model"].get("revision"),
        "model_path": str(model_path),
        "model_source_sha256": model_source_hash,
        "model_file_hashes": model_files,
        "dependencies": dependency_versions(),
        "cuda": {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "devices": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if cuda_available
            else [],
            "bf16_supported": bool(torch.cuda.is_bf16_supported())
            if cuda_available
            else False,
        },
        "loads_pretrained_weights": False,
        "loads_tokenizer": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_path(args.model_path or config["model"]["default_local_path"])
    report = build_preflight_report(config, model_path, require_cuda=args.require_cuda)
    with immutable_directory(args.output.resolve()) as building:
        write_json(building / "preflight_report.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "configuration_valid",
                    "offline_only",
                    "model_source_sha256",
                    "dependencies",
                    "cuda",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
