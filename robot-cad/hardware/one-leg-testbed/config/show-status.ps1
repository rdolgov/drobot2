[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 4)]
    [int]$Leg,

    [string]$Port = "COM4"
)

$ErrorActionPreference = "Stop"
$testbedRoot = Split-Path -Parent $PSScriptRoot
$cli = Join-Path $testbedRoot ".venv\Scripts\drobot-leg.exe"
$configPath = Join-Path $PSScriptRoot "leg-$Leg.toml"
$calibrationPath = Join-Path $PSScriptRoot "calibration-leg-$Leg.json"

foreach ($requiredPath in @($cli, $configPath, $calibrationPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

Write-Host "Reading Leg $Leg telemetry on $Port. This does not enable torque."
& $cli --config $configPath --port $Port --calibration $calibrationPath status
exit $LASTEXITCODE
