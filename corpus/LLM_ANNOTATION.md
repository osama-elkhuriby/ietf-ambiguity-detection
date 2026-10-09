# Local Ollama consensus annotation

The pipeline reads lineage-deduplicated `reviewer_a.csv` files, asks three
local Ollama models to classify each target independently, validates every
response, derives a conservative consensus, and exports training-ready rows.
It never modifies the source review CSVs.

Input discovery is automatic: every
`ground_truth/<family>/review/all_candidates_v1/reviewer_a.csv` file is
included. With the current corpus this covers all 11 draft families and 2,618
lineage-deduplicated candidates.

Consensus uses a two-of-three majority:

- Three matching `yes` or `no` votes produce a HIGH label.
- Any two matching `yes` or `no` votes produce a MEDIUM label, including when
  the third model disagrees or abstains with `uncertain`.
- Invalid model responses do not vote. Two valid matching votes still form a
  MEDIUM majority; otherwise missing/invalid responses and votes without a
  strict `yes` or `no` majority remain `UNCERTAIN`.

## One-command run

Start Ollama and make sure these models are installed:

```powershell
ollama list
```

Before a large run, load `qwen3:8b` once and run `ollama ps` in another
terminal. The `PROCESSOR` column should report GPU use. Stop and repair the
Ollama GPU setup if it reports `100% CPU`; the 20B--32B models are impractical
for a full corpus on CPU. See Ollama's official
[hardware support](https://docs.ollama.com/gpu) and
[troubleshooting](https://docs.ollama.com/troubleshooting) pages.

Run a 60-record stratified pilot first:

```powershell
.\run_llm_annotation.cmd --limit 60
```

Inspect `corpus/processed/llm_annotations/v1/consensus.csv`. Then continue to
the complete dataset with the same output directory:

```powershell
.\run_llm_annotation.cmd
```

Completed responses are reused, so an interrupted run can be resumed by
running the same command. The existing 1,236 completed records will be reused;
only newly discovered candidates require inference. Use `--retry-invalid` to
ask Ollama to retry saved invalid/error responses.

## Outputs

- `responses/<model>/`: atomic, resumable raw response records.
- `model_predictions/<model>.csv`: validated per-model decisions.
- `consensus.csv`: all selected records and consensus/audit fields.
- `training_dataset.csv`: only HIGH and MEDIUM consensus records.
- `run_manifest.json`: source hashes, model digests, settings, and counts.

`training_dataset.csv` deliberately omits weak candidate labels, errata, later
revision evidence, and rationales from its feature columns. The default split
holds out complete document families: TCPM for validation and HTTP/2bis for
test. HIGH rows have weight 1.0 and MEDIUM rows have weight 0.5.

The result is an **LLM-consensus silver dataset**, not human-validated ground
truth. It must not be presented as a gold benchmark.
