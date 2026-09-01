<#
.SYNOPSIS
Creates the neutral-residual model_3248.pt used to start V25 nominal adaptation.

.DESCRIPTION
Runs checkpoint surgery only; it does not launch Isaac Sim, training, or a
robot. The script preserves V24 actor/critic features, makes the two bounded-
Beta output halves identical, zeros the stale critic head, clears Adam moments,
and writes a SHA-256 provenance sidecar beside the generated checkpoint.

The generated file deliberately remains named model_3248.pt. Pass it explicitly
to the V25 nominal phase so the existing curriculum offset starts at zero while
the interrupted-run fallback can continue to use iteration 3248 as its base.

.EXAMPLE
& .\simulation\isaac\rl\parallel_walking\prepare_v25_bootstrap.ps1

.EXAMPLE
& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 `
  -CommandSet robust-straight-low-stance-external-rear-payload `
  -Iterations 350 -NumEnvs 128 -Seed 2501 -V25Phase nominal `
  -Checkpoint .\simulation\isaac\models\parallel-walking-v25-neutral-bootstrap\model_3248.pt
#>

[CmdletBinding()]
param(
    [string]$Source = "simulation\isaac\models\parallel-walking-v24-padded-feet-forward-bias\model_3248.pt",
    [string]$Destination = "simulation\isaac\models\parallel-walking-v25-neutral-bootstrap\model_3248.pt",
    [ValidateRange(0.000001, 1.0)]
    [double]$LearningRate = 0.000075,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaacPython = "C:\isaacsim\python.bat"
$bootstrapScript = Join-Path $PSScriptRoot "bootstrap_v25_from_v24.py"

if (-not (Test-Path -LiteralPath $isaacPython -PathType Leaf)) {
    throw "Isaac Sim Python was not found at $isaacPython"
}
if (-not (Test-Path -LiteralPath $bootstrapScript -PathType Leaf)) {
    throw "Bootstrap script was not found at $bootstrapScript"
}

function ConvertTo-RepositoryPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

$sourcePath = ConvertTo-RepositoryPath -Path $Source
$destinationPath = ConvertTo-RepositoryPath -Path $Destination
if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
    throw "Selected V24 checkpoint was not found at $sourcePath"
}

$arguments = @(
    $bootstrapScript,
    "--source", $sourcePath,
    "--destination", $destinationPath,
    "--learning-rate", $LearningRate.ToString(
        [System.Globalization.CultureInfo]::InvariantCulture
    )
)
if ($Force) {
    $arguments += "--force"
}

Push-Location $repoRoot
try {
    & $isaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V25 checkpoint bootstrap exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Use the generated seed for nominal V25 adaptation:"
Write-Host (
    "& .\simulation\isaac\rl\parallel_walking\train_walking_headless.ps1 " +
    "-CommandSet robust-straight-low-stance-external-rear-payload " +
    "-Iterations 350 -NumEnvs 128 -Seed 2501 -V25Phase nominal " +
    "-Checkpoint `"$destinationPath`""
)
