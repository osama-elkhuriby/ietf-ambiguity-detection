function Resolve-ProjectPython {
    param(
        [Parameter(Mandatory = $true)][string]$ProjectRoot,
        [string]$RequestedPath = "",
        [Parameter(Mandatory = $true)][string[]]$RequiredModules
    )

    $Candidates = [System.Collections.Generic.List[string]]::new()
    if ($RequestedPath) {
        $Candidates.Add($RequestedPath)
    } else {
        if ($env:CONDA_PREFIX) {
            $Candidates.Add((Join-Path $env:CONDA_PREFIX "python.exe"))
        }
        if ($env:USERPROFILE) {
            $Candidates.Add(
                (Join-Path $env:USERPROFILE ".conda\envs\ietf-ambiguity-detection\python.exe")
            )
        }
        $Candidates.Add((Join-Path $ProjectRoot ".venv\Scripts\python.exe"))
        $Command = Get-Command python -ErrorAction SilentlyContinue
        if ($Command) {
            $Candidates.Add($Command.Source)
        }
    }

    $Attempted = [System.Collections.Generic.List[string]]::new()
    $Seen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    $Probe = @"
import sys
for module in sys.argv[1:]:
    __import__(module)
"@
    foreach ($Candidate in $Candidates) {
        if (-not $Candidate -or -not $Seen.Add($Candidate)) {
            continue
        }
        $Attempted.Add($Candidate)
        if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
            continue
        }
        $OriginalPath = $env:PATH
        $OriginalCondaPrefix = $env:CONDA_PREFIX
        $Prefix = Split-Path -Parent $Candidate
        $CondaLibraryBin = Join-Path $Prefix "Library\bin"
        if (Test-Path -LiteralPath $CondaLibraryBin -PathType Container) {
            $RuntimePaths = @(
                $Prefix,
                (Join-Path $Prefix "Library\mingw-w64\bin"),
                (Join-Path $Prefix "Library\usr\bin"),
                $CondaLibraryBin,
                (Join-Path $Prefix "Scripts"),
                (Join-Path $Prefix "bin"),
                $OriginalPath
            )
            $env:CONDA_PREFIX = $Prefix
            $env:PATH = $RuntimePaths -join ";"
        }
        $Succeeded = $false
        try {
            & $Candidate -c $Probe @RequiredModules *> $null
            if ($LASTEXITCODE -eq 0) {
                $Succeeded = $true
            }
        } catch {
            $Succeeded = $false
        }
        if ($Succeeded) {
            return (Resolve-Path -LiteralPath $Candidate).Path
        }
        $env:PATH = $OriginalPath
        if ($null -eq $OriginalCondaPrefix) {
            Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
        } else {
            $env:CONDA_PREFIX = $OriginalCondaPrefix
        }
    }
    $List = $Attempted -join "; "
    throw (
        "No runnable Python interpreter has the required modules " +
        "($($RequiredModules -join ', ')). Attempted: $List. " +
        "Pass a working interpreter with -PythonPath."
    )
}
