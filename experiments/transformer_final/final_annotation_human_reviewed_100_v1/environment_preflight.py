"""Read-only CUDA, dependency, local-model, and numerical-guard preflight.

This module never loads the model or tokenizer and cannot start training.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from .core import ROOT, _resolve_repo_path, _verify_static_sources, load_local_config


PACKAGES = (
    "torch",
    "transformers",
    "numpy",
    "pandas",
    "scikit-learn",
    "tokenizers",
    "sentencepiece",
    "safetensors",
)


def collect() -> dict[str, object]:
    import torch

    config = load_local_config()
    static = _verify_static_sources(config, hash_model=True)
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout.strip()
    fields = [value.strip() for value in query.splitlines()[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError(f"Unexpected nvidia-smi response: {query}")
    train_source = (ROOT / "models/transformer_final/train.py").read_text(
        encoding="utf-8"
    )
    required_guards = {
        "fp32_cross_entropy": "cross_entropy(logits.float()" in train_source,
        "finite_logits_and_loss": "_check_forward_numerics" in train_source,
        "finite_gradient_clipping": "error_if_nonfinite=True" in train_source,
        "finite_optimizer_state": "_assert_finite_optimizer_state" in train_source,
        "finite_model_parameters": "_assert_finite_model_parameters" in train_source,
        "batch_step_diagnostics": "batch_number" in train_source
        and "optimizer step" in train_source,
        "fresh_base_initialization": "initialize_pretrained_components" in train_source,
        "offline_local_only": "local_files_only" in train_source
        and 'TRANSFORMERS_OFFLINE' in train_source,
    }
    cuda_available = bool(torch.cuda.is_available())
    result: dict[str, object] = {
        "status": "PASS"
        if cuda_available and all(required_guards.values()) and "RTX 5000 Ada" in fields[0]
        else "BLOCKED",
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "dependencies": {name: importlib.metadata.version(name) for name in PACKAGES},
        "cuda_available": cuda_available,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None,
        "gpu": {
            "name": fields[0],
            "memory_total_mib": int(fields[1]),
            "memory_free_mib": int(fields[2]),
            "driver_version": fields[3],
        },
        "model_path": static["model_path"],
        "model_sha256": static["model_source_sha256"],
        "model_loaded": False,
        "training_executed": False,
        "numerical_guards": required_guards,
        "precision": config["training"]["precision"],
        "precision_policy": "Preserve validated FP32; no evidence-based compatibility reason to change it.",
    }
    return result


def main() -> int:
    print(json.dumps(collect(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
