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

$idStart = (($Leg - 1) * 3) + 1
$idEnd = $idStart + 2

Write-Host "Scanning for Leg $Leg IDs $idStart-$idEnd on $Port."
& $cli --config $configPath --port $Port scan --id-start $idStart --id-end $idEnd
if ($LASTEXITCODE -ne 0) {
    throw "Servo scan failed; no configuration was written."
}

Write-Warning "Support the leg and keep the physical servo-power cutoff ready."
$confirmation = Read-Host "Type CONFIGURE to apply position mode and safe limits"
if ($confirmation -cne "CONFIGURE") {
    Write-Host "Cancelled; no configuration was written."
    exit 1
}

& $cli --config $configPath --port $Port configure --yes
if ($LASTEXITCODE -ne 0) {
    throw "Motor configuration failed."
}

Write-Host "Configuration finished. Verifying telemetry and torque state."
& $cli --config $configPath --port $Port --calibration $calibrationPath status
exit $LASTEXITCODE
