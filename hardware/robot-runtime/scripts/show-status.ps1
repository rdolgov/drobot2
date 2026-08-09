[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 4)]
    [int]$Leg,

    [string]$Port = "COM4"
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_common.ps1")
$configPath = Join-Path $ServoConfigRoot "leg-$Leg.toml"
$calibrationPath = Join-Path $ServoConfigRoot "calibration-leg-$Leg.json"

foreach ($requiredPath in @($DrobotLegCli, $configPath, $calibrationPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

Write-Host "Reading Leg $Leg telemetry on $Port. This does not enable torque."
& $DrobotLegCli --config $configPath --port $Port --calibration $calibrationPath status
exit $LASTEXITCODE
