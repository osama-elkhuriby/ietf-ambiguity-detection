param(
    [ValidateSet("audit", "run", "all")]
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
    -RequiredModules @("numpy", "pandas", "scipy", "yaml", "sklearn")
Write-Host "Python runtime: $Python"

$Config = Join-Path $ProjectRoot "experiments\llm\configs\qwen3_32b_final_annotation_ready_hybrid_v3.yaml"
$RunRoot = Join-Path $ProjectRoot "experiments\llm\results\qwen3-32b-final-annotation-ready-v3"
$Audit = Join-Path $RunRoot "protocol-audit.json"
$Results = Join-Path $RunRoot "analysis-final"
$Evaluator = Join-Path $ProjectRoot "models\llm\hybrid_evaluate_v3.py"

function Invoke-CheckedPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Preflight: verify frozen development data and V1 zero-shot scores"
$DatasetRoot = Join-Path $ProjectRoot "corpus\processed\llm_final_annotation_ready\v1"
Invoke-CheckedPython -Arguments @(
    "-m", "models.llm.final_annotation_dataset",
    "--output", $DatasetRoot,
    "--verify"
)

if (Test-Path -LiteralPath $Audit -PathType Leaf) {
    $AuditReport = Get-Content -Raw -LiteralPath $Audit | ConvertFrom-Json
    $ConfigHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Config).Hash.ToLowerInvariant()
    $EvaluatorHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Evaluator).Hash.ToLowerInvariant()
    if ([string]$AuditReport.config_sha256 -ne $ConfigHash -or
        [string]$AuditReport.evaluator_sha256 -ne $EvaluatorHash) {
        throw "V3 audit targets different code or configuration. Use a new versioned output."
    }
    Write-Host "SKIP completed V3 protocol audit: $Audit"
} else {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.hybrid_evaluate_v3", "audit",
        "--config", $Config,
        "--output", $Audit
    )
}

if ($Stage -eq "audit") {
    Write-Host "V3 protocol audit complete: $Audit"
    return
}

if (Test-Path -LiteralPath (Join-Path $Results "run_manifest.json") -PathType Leaf) {
    Write-Host "SKIP completed V3 analysis: $Results"
} elseif (Test-Path -LiteralPath $Results) {
    throw "Refusing incomplete V3 analysis directory: $Results"
} else {
    Invoke-CheckedPython -Arguments @(
        "-m", "models.llm.hybrid_evaluate_v3", "run",
        "--config", $Config,
        "--output", $Results
    )
}

Write-Host "Qwen3 32B hybrid V3 experiment complete."
Write-Host "Summary: $(Join-Path $Results 'analysis.json')"
Write-Host "V1/V2/V3 comparison: $(Join-Path $Results 'v1_v2_v3_comparison.csv')"
Write-Host "Visual comparison: $(Join-Path $Results 'v1_v2_v3_comparison.svg')"
