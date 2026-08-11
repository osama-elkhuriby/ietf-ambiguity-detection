"""Verify the project environment without downloading data or model weights."""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PACKAGE_IMPORTS = {
    "requests": "requests",
    "beautifulsoup4": "bs4",
    "lxml": "lxml",
    "python-dotenv": "dotenv",
    "PyYAML": "yaml",
    "pydantic": "pydantic",
    "tenacity": "tenacity",
    "tqdm": "tqdm",
    "rich": "rich",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "pyarrow": "pyarrow",
    "joblib": "joblib",
    "spaCy": "spacy",
    "NLTK": "nltk",
    "RapidFuzz": "rapidfuzz",
    "NetworkX": "networkx",
    "sentence-transformers": "sentence_transformers",
    "scikit-learn": "sklearn",
    "imbalanced-learn": "imblearn",
    "XGBoost": "xgboost",
    "Optuna": "optuna",
    "torch": "torch",
    "Transformers": "transformers",
    "Datasets": "datasets",
    "Evaluate": "evaluate",
    "Accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
    "huggingface-hub": "huggingface_hub",
    "Statsmodels": "statsmodels",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "MLflow": "mlflow",
    "JupyterLab": "jupyterlab",
    "ipykernel": "ipykernel",
    "ietfdata": "ietfdata",
    "Krippendorff": "krippendorff",
    "litellm": "litellm",
    "tiktoken": "tiktoken",
    "xml2rfc": "xml2rfc",
    "pytest": "pytest",
}


def installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    matplotlib_config = Path(tempfile.gettempdir()) / "ietf-ambiguity-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))

    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    failures: list[str] = []

    for distribution, module_name in PACKAGE_IMPORTS.items():
        try:
            # Importing LiteLLM can initialize tokenizers and contact remote model
            # metadata endpoints. Presence is enough for this offline check.
            if module_name == "litellm":
                if importlib.util.find_spec(module_name) is None:
                    raise ModuleNotFoundError(module_name)
            else:
                importlib.import_module(module_name)
            print(f"OK      {distribution} {installed_version(distribution)}")
        except Exception as exc:  # Import errors can include missing native libraries.
            failures.append(distribution)
            print(f"MISSING {distribution}: {type(exc).__name__}: {exc}")

    try:
        import spacy

        spacy.load("en_core_web_sm")
        print("OK      spaCy model en_core_web_sm")
    except Exception as exc:
        failures.append("en_core_web_sm")
        print(f"MISSING spaCy model en_core_web_sm: {type(exc).__name__}: {exc}")

    try:
        import torch

        print(f"PyTorch device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception:
        pass

    for command in ("conda-lock", "jupyter-lab", "mlflow", "mypy", "pre-commit", "ruff"):
        path = shutil.which(command)
        if path is None:
            executable_dir = Path(sys.executable).parent
            candidates = (
                executable_dir / command,
                executable_dir / f"{command}.exe",
                executable_dir / "Scripts" / command,
                executable_dir / "Scripts" / f"{command}.exe",
            )
            path = next((str(candidate) for candidate in candidates if candidate.is_file()), None)
        if path:
            print(f"OK      command {command}")
        else:
            failures.append(command)
            print(f"MISSING command {command}")

    if failures:
        print("\nEnvironment check failed: " + ", ".join(failures))
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
