# Project scripts

Scripts here support environment checks, experiment launch, annotation preparation,
and focused analysis. Run commands from the repository root unless a guide explicitly
says otherwise.

## Layout

| Path | Purpose |
| --- | --- |
| `check_environment.py` | Offline check of the selected project environment and available compute device. |
| `python_runtime.ps1` | Shared Windows interpreter discovery for experiment launchers. |
| `run_llm_in_context*.ps1` | Staged V1--V4 local LLM workflow launchers. |
| `run_transformer_factorial.ps1` | Transformer factorial experiment launcher. |
| `verify_qwen_standalone_gpu.py` | GPU preflight for standalone Qwen inference. |
| [`annotation/`](annotation/) | Resumable annotation launcher and annotation-sheet preparation utility. |
| [`analysis/`](analysis/) | Local annotation-vote and per-model target analyses. |

## Common commands

```powershell
python scripts/check_environment.py
.\scripts\annotation\run_llm_annotation.cmd --limit 60
python -m scripts.annotation.make_annotation_file --check 5
```

The LLM annotation and analysis helpers read private local data and may write outputs
under `results/annotation/`. Do not commit those outputs. See
[`../corpus/LLM_ANNOTATION.md`](../corpus/LLM_ANNOTATION.md) before running annotation
workflows.

Keep experiment-specific training and evaluation code in `models/` or
`experiments/`, not in this launcher/helper directory. A script that requires a
specific model, hardware configuration, or data snapshot should point to its owning
experiment guide rather than duplicating the protocol here.
