# Annotation v2 manual execution package

Status: **PREPARED_NOT_EXECUTED**

Codex prepared the inputs and validation code but did not start or stop Ollama,
load a model, send an inference request, run annotation, or train a classifier.
The commands below are intentionally separate. Run them one block at a time and
check the stated gate before continuing.

## Prepared, non-live artifacts

- Frozen experiment: `experiments/annotation_v2/configs/requested_panel_v2.json`
- Frozen pilot allowlist: `experiments/annotation_v2/samples/pilot_sample_12_v1.json`
- New output namespace: `corpus/processed/llm_annotations/v2/manual-standalone-v0404`
- Prepared dry run: 12 records, 24 context cells, 72 rendered panel requests,
  zero network requests.
- Reused Qwen result: `corpus/processed/llm_annotations/v2/diagnostics/standalone-v0404-qwen-gpu-verification-20261014/result.json`
- Machine-readable package metadata and hashes: `package_manifest.json`

The rendered GPT-OSS requests are inert audit artifacts from the frozen panel.
The live `--eligible-only` gate permits exactly Qwen and Gemma and keeps
GPT-OSS-120B blocked. No fallback is configured.

## 0. Preparation checks (safe; no model request)

Open **Terminal B**, then keep it open for all control commands:

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Repo = "D:\RFC\ietf-ambiguity-detection"
$Python = "$Repo\.venv\Scripts\python.exe"
$Experiment = "$Repo\experiments\annotation_v2"
$Config = "$Experiment\configs\requested_panel_v2.json"
$Runtime = "C:\Users\mosta\AppData\Local\OllamaStandalone\v0.40.4\runtime"
$BaseUrl = "http://127.0.0.1:11435"
$RunRoot = "$Repo\corpus\processed\llm_annotations\v2\manual-standalone-v0404"
$ServerLog = "$RunRoot\server-11435.log"
$GemmaDiag = "$RunRoot\diagnostics\gemma4-31b-gpu-verification"
$QwenResult = "$Repo\corpus\processed\llm_annotations\v2\diagnostics\standalone-v0404-qwen-gpu-verification-20261014\result.json"
Set-Location -LiteralPath $Repo
```

Verify the executable, saved evidence, hashes, and unused candidate port:

```powershell
if (-not (Test-Path -LiteralPath "$Runtime\ollama.exe")) { throw "Standalone Ollama executable is missing." }
if (-not (Test-Path -LiteralPath $QwenResult)) { throw "Saved Qwen verification is missing." }
if ((Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue)) { throw "Port 11435 is already in use." }
(Get-FileHash -LiteralPath $Config -Algorithm SHA256).Hash.ToLower()
(Get-FileHash -LiteralPath "$Experiment\samples\pilot_sample_12_v1.json" -Algorithm SHA256).Hash.ToLower()
(Get-FileHash -LiteralPath $QwenResult -Algorithm SHA256).Hash.ToLower()
& $Python -m models.annotation_v2.manual inspect-verification --result $QwenResult
```

Expected hashes, in order:

- `c40da4868872b8f9690e0bcfd482efa744030d70f04c0bb102ca05f4e4bf9925`
- `2955788ef80924967e1593918cf804120eb76d8ef4d4e6ef41d3201535334883`
- `260144c848738608b23401b71c8a3171baf8dff97040e1be3d3b80b97b883d9e`

The Qwen inspection must show `eligible: true`, version 0.40.4 evidence,
the frozen digest, GPU residency, no gate failures, and successful unload.

## 1. Start the standalone server and check it

### RUN MANUALLY — NOT EXECUTED BY CODEX

Open **Terminal A**. This is a foreground server; keep this terminal open:

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Runtime = "C:\Users\mosta\AppData\Local\OllamaStandalone\v0.40.4\runtime"
$RunRoot = "D:\RFC\ietf-ambiguity-detection\corpus\processed\llm_annotations\v2\manual-standalone-v0404"
$ServerLog = "$RunRoot\server-11435.log"
$env:OLLAMA_HOST = "127.0.0.1:11435"
Set-Location -LiteralPath $Runtime
& "$Runtime\ollama.exe" serve 2>&1 | Tee-Object -FilePath $ServerLog -Append
```

Back in **Terminal B**, check the server without loading anything:

```powershell
$Version = (Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/version").version
if ($Version -ne "0.40.4") { throw "Wrong Ollama version: $Version" }
$Resident = @((Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/ps").models)
if ($Resident.Count -ne 0) { throw "A model is already resident on port 11435." }
Select-String -LiteralPath $ServerLog -Pattern 'library=CUDA','total_vram=' | Select-Object -Last 10
```

Stop here if the endpoint is not 0.40.4, `/api/ps` is nonempty, or the new
startup log does not show `library=CUDA` and nonzero VRAM.

## 2. One bounded synthetic Gemma verification

### RUN MANUALLY — NOT EXECUTED BY CODEX

This command permits exactly one synthetic request, no retry, no corpus text,
and unloads Gemma afterward. It uses the frozen 16,384-token context, 768-token
generation cap, temperature 0, seed 20261014, `think=false`, and 300-second
timeout.

```powershell
& $Python -m models.annotation_v2.manual --config $Config verify-gpu --model "gemma4:31b" --base-url $BaseUrl --server-log $ServerLog --output $GemmaDiag
```

Expected files:

- `$GemmaDiag\request.json`
- `$GemmaDiag\model_show.json`
- `$GemmaDiag\response_envelope.json`
- `$GemmaDiag\result.json`

Do not rerun this command into the same directory; the diagnostic namespace is
append-only.

## 3. Inspect the saved Gemma feasibility result

This command is read-only and makes no server request:

```powershell
& $Python -m models.annotation_v2.manual inspect-verification --result "$GemmaDiag\result.json"
```

Continue only if it prints `eligible: true`, digest
`17ba34c06c801ba22f6047bab21ced9429c6d43e0e42ecf339b0f3e5a61e8d97`,
semantic label `CLEAR`, positive GPU VRAM placement, and an empty
`frozen_gate_reasons` list.

Assemble the saved Qwen and Gemma evidence. This performs no inference:

```powershell
& $Python -m models.annotation_v2.manual --config $Config assemble-panel --output-root $RunRoot --qwen-result $QwenResult --gemma-result "$GemmaDiag\result.json"
```

Expected output: `$RunRoot\preflight_report_final.json`, with exactly Qwen and
Gemma marked `VALID`, GPT-OSS-120B marked `MODEL_BLOCKED_PREFLIGHT`, and
`manual_evidence_assembly: true`.

## 4. Run only eligible frozen pilot cells

### RUN MANUALLY — NOT EXECUTED BY CODEX

This is the expensive pilot. It uses the 24 already frozen input bundles,
runs one model at a time, unloads between models, and writes attempts
append-only. There are 48 eligible cells and at most one retry per cell, so the
eligible-model maximum is 96 requests. GPT-OSS-120B receives only explicit
blocked/non-run cell records.

```powershell
& $Python -m models.annotation_v2.cli --experiment-root $Experiment --output-root $RunRoot --base-url $BaseUrl annotate --scope pilot --eligible-only
```

Before the first annotation request, the command rechecks server version and
the exact Qwen/Gemma digests. It refuses missing frozen inputs, an incomplete
eligible set, or a valid GPT-OSS-120B entry.

## 5. Resume an interrupted pilot

### RUN MANUALLY — NOT EXECUTED BY CODEX

Run the identical command:

```powershell
& $Python -m models.annotation_v2.cli --experiment-root $Experiment --output-root $RunRoot --base-url $BaseUrl annotate --scope pilot --eligible-only
```

Accepted judgments with the same cache identity are reused. Existing attempts
are append-only and are not repeated. A cache-identity conflict causes an error
instead of silently crossing configurations.

## 6. Generate the final saved-results report

This step performs no model request:

```powershell
& $Python -m models.annotation_v2.cli --experiment-root $Experiment --output-root $RunRoot report-pilot
```

Expected report: `$RunRoot\pilot_run_report.json`. Because GPT-OSS-120B stays
blocked, conservative three-model aggregation remains
`REVIEW_REQUIRED_INCOMPLETE_PANEL`; two-model agreement must not be described
as unanimous three-model consensus.

## 7. Unload and stop the manually started server

### RUN MANUALLY — NOT EXECUTED BY CODEX

In **Terminal B**, unload only an allowed model that is actually resident:

```powershell
$Allowed = @("qwen3.5:27b", "gemma4:31b")
$Resident = @((Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/ps").models)
$Unexpected = @($Resident | Where-Object { $_.model -notin $Allowed })
if ($Unexpected.Count -ne 0) { throw "Unexpected resident model; stop and inspect before unloading." }
foreach ($Item in $Resident) {
    $Body = @{ model = $Item.model; prompt = ""; stream = $false; keep_alive = 0 } | ConvertTo-Json -Compress
    Invoke-RestMethod -Method Post -Uri "$BaseUrl/api/generate" -ContentType "application/json" -Body $Body | Out-Null
}
if (@((Invoke-RestMethod -Method Get -Uri "$BaseUrl/api/ps").models).Count -ne 0) { throw "Model unload did not complete." }
```

Then press **Ctrl+C once in Terminal A** to stop only the foreground 0.40.4
server. Confirm port 11435 is no longer listening:

```powershell
if (Get-NetTCPConnection -LocalPort 11435 -State Listen -ErrorAction SilentlyContinue) { throw "Standalone server is still listening on 11435." }
```

Do not stop or modify the original service on port 11434.

## Mandatory stop conditions

Stop without starting or resuming the pilot if any of these occurs:

- Any frozen config, allowlist, or saved-Qwen hash differs from this package.
- Port 11435 resolves to anything other than Ollama 0.40.4.
- CUDA discovery/nonzero VRAM is absent or a model is resident before Gemma.
- Gemma tag/digest differs, its single request times out or is invalid, the
  clear-discretion fixture is not `CLEAR`, GPU placement is not verified, a
  frozen RAM/paging/runtime gate fails, or unload fails.
- The assembled eligible set is not exactly Qwen plus Gemma, or GPT-OSS-120B is
  not blocked.
- The pilot reports a missing frozen input, digest/version mismatch,
  append-only identity conflict, unexpected model, or resource failure.
- Any command would require a model download, fallback, full-corpus scope,
  classifier training, threshold work, or changing frozen governance.

Do not use `preflight`, `--scope full`, or `--confirm-full` for this package.
