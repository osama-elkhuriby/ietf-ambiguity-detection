param(
    [ValidateSet("audit", "smoke", "all")]
    [string]$Stage = "all",
    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot
. (Join-Path $PSScriptRoot "python_runtime.ps1")
$Python = Resolve-ProjectPython `
    -ProjectRoot $ProjectRoot `
    -RequestedPath $PythonPath `
    -RequiredModules @("numpy", "pandas", "yaml", "sklearn")
Write-Host "Python runtime: $Python"

$Config = Join-Path $ProjectRoot "experiments\llm\configs\qwen3_32b_final_annotation_ready_in_context.yaml"
$DatasetRoot = Join-Path $ProjectRoot "corpus\processed\llm_final_annotation_ready\v1"
$RunRoot = Join-Path $ProjectRoot "experiments\llm\results\qwen3-32b-final-annotation-ready-v1"
$SmokeRoot = Join-Path $ProjectRoot "experiments\llm\results\qwen3-32b-final-annotation-ready-smoke-v1"

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Preflight: verify final_annotation_ready dataset and grouped-fold hashes"
if (-not (Test-Path -LiteralPath (Join-Path $DatasetRoot "manifest.json") -PathType Leaf)) {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.final_annotation_dataset",
        "--output", $DatasetRoot
    )
}
Invoke-CheckedPython -Arguments @(
    "-m", "models.llm.final_annotation_dataset",
    "--output", $DatasetRoot,
    "--verify"
)

$Audit = Join-Path $RunRoot "prompt-audit.json"
if (-not (Test-Path -LiteralPath $Audit -PathType Leaf)) {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.in_context_evaluate", "audit",
        "--config", $Config,
        "--output", $Audit
    )
}
$AuditReport = Get-Content -Raw -LiteralPath $Audit | ConvertFrom-Json
$ConfigHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Config).Hash.ToLowerInvariant()
if ([string]$AuditReport.config_sha256 -ne $ConfigHash) {
    throw "Prompt audit targets a different configuration. Use a new versioned output."
}
$Evaluator = Join-Path $ProjectRoot "models\llm\in_context_evaluate.py"
$EvaluatorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Evaluator).Hash.ToLowerInvariant()
if ([string]$AuditReport.evaluator_sha256 -ne $EvaluatorHash) {
    throw "Prompt audit targets different evaluator code. Use a new versioned output."
}
if ($Stage -eq "audit") {
    Write-Host "Prompt audit complete: $Audit"
    return
}

if ($Stage -eq "smoke") {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.in_context_evaluate", "run",
        "--config", $Config,
        "--run-root", $SmokeRoot,
        "--smoke"
    )
    Write-Host "Smoke evaluation complete: $SmokeRoot"
    return
}

Invoke-CheckedPython -Arguments @(
    "-m", "models.llm.in_context_evaluate", "run",
    "--config", $Config,
    "--run-root", $RunRoot
)

$Analysis = Join-Path $RunRoot "analysis-final"
$AnalysisManifest = Join-Path $Analysis "run_manifest.json"
if (Test-Path -LiteralPath $AnalysisManifest -PathType Leaf) {
    Write-Host "SKIP completed analysis: $Analysis"
} elseif (Test-Path -LiteralPath $Analysis) {
    throw "Refusing incomplete analysis directory: $Analysis"
} else {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.in_context_evaluate", "analyze",
        "--config", $Config,
        "--run-root", $RunRoot,
        "--output", $Analysis
    )
}

Write-Host "Qwen3 32B zero/few-shot evaluation complete."
Write-Host "Summary: $(Join-Path $Analysis 'analysis.json')"
Write-Host "Metrics: $(Join-Path $Analysis 'pooled_metrics.csv')"
Write-Host "Confusion matrices: $Analysis"
