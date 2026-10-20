param(
    [ValidateSet("audit", "smoke", "run", "analyze", "all")]
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

$Config = Join-Path $ProjectRoot "experiments\llm\configs\qwen3_32b_final_annotation_ready_in_context_v2.yaml"
$DatasetRoot = Join-Path $ProjectRoot "corpus\processed\llm_final_annotation_ready\v1"
$RunRoot = Join-Path $ProjectRoot "results\experiments\llm\qwen3-32b-final-annotation-ready-v2"
$SmokeRoot = Join-Path $ProjectRoot "results\experiments\llm\qwen3-32b-final-annotation-ready-v2-smoke"
$Evaluator = Join-Path $ProjectRoot "models\llm\in_context_evaluate_v2.py"
$SharedEvaluator = Join-Path $ProjectRoot "models\llm\in_context_evaluate.py"
$Prompting = Join-Path $ProjectRoot "models\llm\prompting.py"
$OllamaClient = Join-Path $ProjectRoot "models\llm\ollama_client.py"

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
        "-m", "models.llm.in_context_evaluate_v2", "audit",
        "--config", $Config,
        "--output", $Audit
    )
}
$AuditReport = Get-Content -Raw -LiteralPath $Audit | ConvertFrom-Json
$ConfigHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Config).Hash.ToLowerInvariant()
if ([string]$AuditReport.config_sha256 -ne $ConfigHash) {
    throw "V2 prompt audit targets a different configuration. Use a new versioned output."
}
$EvaluatorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Evaluator).Hash.ToLowerInvariant()
if ([string]$AuditReport.code_hashes.evaluator_v2 -ne $EvaluatorHash) {
    throw "V2 prompt audit targets different evaluator code. Use a new versioned output."
}
$SharedEvaluatorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SharedEvaluator).Hash.ToLowerInvariant()
$PromptingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Prompting).Hash.ToLowerInvariant()
$OllamaClientHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $OllamaClient).Hash.ToLowerInvariant()
if ([string]$AuditReport.code_hashes.v1_shared_helpers -ne $SharedEvaluatorHash -or
    [string]$AuditReport.code_hashes.prompting -ne $PromptingHash -or
    [string]$AuditReport.code_hashes.ollama_client -ne $OllamaClientHash) {
    throw "V2 prompt audit targets different shared code. Use a new versioned output."
}
if ($Stage -eq "audit") {
    Write-Host "V2 prompt audit complete: $Audit"
    return
}

if ($Stage -eq "smoke") {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.in_context_evaluate_v2", "run",
        "--config", $Config,
        "--run-root", $SmokeRoot,
        "--smoke"
    )
    Write-Host "V2 smoke evaluation complete: $SmokeRoot"
    return
}

if ($Stage -in @("run", "all")) {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.in_context_evaluate_v2", "run",
        "--config", $Config,
        "--run-root", $RunRoot
    )
    if ($Stage -eq "run") {
        Write-Host "V2 inference complete: $RunRoot"
        return
    }
}

$Analysis = Join-Path $RunRoot "analysis-final"
$AnalysisManifest = Join-Path $Analysis "run_manifest.json"
if (Test-Path -LiteralPath $AnalysisManifest -PathType Leaf) {
    Write-Host "SKIP completed V2 analysis: $Analysis"
} elseif (Test-Path -LiteralPath $Analysis) {
    throw "Refusing incomplete V2 analysis directory: $Analysis"
} else {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.in_context_evaluate_v2", "analyze",
        "--config", $Config,
        "--run-root", $RunRoot,
        "--output", $Analysis
    )
}

Write-Host "Qwen3 32B V2 comparison experiment complete."
Write-Host "Summary: $(Join-Path $Analysis 'analysis.json')"
Write-Host "V1/V2 comparison: $(Join-Path $Analysis 'v1_v2_comparison.csv')"
Write-Host "Cross-fitted thresholds: $(Join-Path $Analysis 'cross_fitted_threshold_metrics.csv')"
