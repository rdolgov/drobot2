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

$idStart = (($Leg - 1) * 3) + 1
$idEnd = $idStart + 2

Write-Host "Scanning for Leg $Leg IDs $idStart-$idEnd on $Port."
& $DrobotLegCli --config $configPath --port $Port scan --id-start $idStart --id-end $idEnd
if ($LASTEXITCODE -ne 0) {
    throw "Servo scan failed; no configuration was written."
}

Write-Warning "Support the leg and keep the physical servo-power cutoff ready."
$confirmation = Read-Host "Type CONFIGURE to apply position mode and safe limits"
if ($confirmation -cne "CONFIGURE") {
    Write-Host "Cancelled; no configuration was written."
    exit 1
}

& $DrobotLegCli --config $configPath --port $Port configure --yes
if ($LASTEXITCODE -ne 0) {
    throw "Motor configuration failed."
}

Write-Host "Configuration finished. Verifying telemetry and torque state."
& $DrobotLegCli --config $configPath --port $Port --calibration $calibrationPath status
exit $LASTEXITCODE
