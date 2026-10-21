"""Version the frozen single-seed transformer package for selected new seeds.

This is preparation only: it copies source/configuration files, changes the
experiment identity and one registered training seed, and does not prepare data,
load a model, or train. The generated package must still pass its own preflight
and prepare commands before any explicitly user-authorized GPU command is run.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ID = "final_annotation_human_reviewed_100_v1"
SOURCE_DIR = ROOT / "experiments/transformer_final" / SOURCE_ID
ALLOWED_SEEDS = (1729, 31415)
SOURCE_FILES = (
    "__init__.py",
    "analysis.py",
    "cli.py",
    "core.py",
    "environment_preflight.py",
    "runner.py",
)


def render_seed(seed: int) -> tuple[str, dict[str, str], dict[str, object]]:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Seed must be one of {ALLOWED_SEEDS}")
    destination_id = f"final_annotation_human_reviewed_100_seed{seed}_v1"
    rendered: dict[str, str] = {}
    for name in SOURCE_FILES:
        text = (SOURCE_DIR / name).read_text(encoding="utf-8")
        text = text.replace(SOURCE_ID, destination_id)
        if name in {"core.py", "runner.py"}:
            text = text.replace("2718", str(seed))
        ast.parse(text, filename=f"{destination_id}/{name}")
        rendered[name] = text

    config = json.loads((SOURCE_DIR / "config.json").read_text(encoding="utf-8"))
    config["experiment"] = destination_id
    config["description"] = (
        f"Independent seed-{seed} repetition of the frozen human-reviewed protocol; "
        "same cohort, folds, methods, and nested analysis"
    )
    config["training"]["seeds"] = [seed]
    for key in ("dataset_dir", "runs_dir", "audit_dir"):
        config["outputs"][key] = config["outputs"][key].replace(SOURCE_ID, destination_id)
    if config["training"]["seeds"] != [seed]:
        raise RuntimeError("Rendered seed configuration is inconsistent")
    return destination_id, rendered, config


def prepare_seed(seed: int) -> dict[str, object]:
    destination_id, rendered, config = render_seed(seed)
    destination = SOURCE_DIR.parent / destination_id
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite experiment package: {destination}")
    destination.mkdir(parents=True)
    for name, text in rendered.items():
        (destination / name).write_text(text, encoding="utf-8", newline="\n")
    (destination / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "seed": seed,
        "experiment_id": destination_id,
        "path": str(destination.resolve()),
        "next_safe_command": f"python -m experiments.transformer_final.{destination_id}.cli preflight",
        "training_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", choices=ALLOWED_SEEDS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Render and syntax-check packages in memory without writing files.",
    )
    args = parser.parse_args()
    seeds = args.seed or list(ALLOWED_SEEDS)
    if args.validate_only:
        results = [
            {
                "seed": seed,
                "experiment_id": render_seed(seed)[0],
                "validated_in_memory": True,
                "files_written": False,
            }
            for seed in seeds
        ]
    else:
        results = [prepare_seed(seed) for seed in seeds]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
