$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$ResultRoot = Join-Path $PSScriptRoot "..\results\experiments\transformer\factorial-v1"
$Seeds = @(1729, 2718, 31415)
$Conditions = @(
    @{ Name = "soft-target-only"; Target = "soft"; Variant = "target_only" },
    @{ Name = "hard-local-context"; Target = "hard"; Variant = "local_context" },
    @{ Name = "soft-local-context"; Target = "soft"; Variant = "local_context" }
)

foreach ($Condition in $Conditions) {
    foreach ($Seed in $Seeds) {
        foreach ($Fold in 1..5) {
            $Output = Join-Path $ResultRoot "$($Condition.Name)\fold-$Fold-seed-$Seed"
            if (Test-Path -LiteralPath $Output) {
                Write-Host "Skipping completed $Output"
                continue
            }
            Write-Host "Starting $($Condition.Name), seed $Seed, fold $Fold"
            & $Python -m models.transformer.train `
                --output $Output `
                --fold $Fold `
                --learning-rate 1e-5 `
                --seed $Seed `
                --weight-strategy class `
                --target-regime $Condition.Target `
                --variant $Condition.Variant `
                --precision fp32
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed for $($Condition.Name), seed $Seed, fold $Fold"
            }
        }
    }
}
