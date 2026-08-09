[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 4)]
    [int]$Leg,

    [string]$Port = "COM4",

    [switch]$SetServoMiddle,

    [ValidateSet(1, 2, 3)]
    [int[]]$MiddleMotor = @(1, 2, 3)
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "_common.ps1")
$configPath = Join-Path $ServoConfigRoot "leg-$Leg.toml"
$calibrationPath = Join-Path $ServoConfigRoot "calibration-leg-$Leg.json"

foreach ($requiredPath in @($DrobotLegCli, $configPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

Write-Host "Checking Leg $Leg on $Port. This status read does not enable torque."
& $DrobotLegCli --config $configPath --port $Port --calibration $calibrationPath status
if ($LASTEXITCODE -ne 0) {
    throw "Telemetry check failed; calibration was not started."
}

Write-Warning "Support the leg at neutral and keep hands clear of every linkage."
Write-Warning "The capture step disables torque, so an unsupported leg can fall."
$confirmation = Read-Host "Type CALIBRATE to continue"
if ($confirmation -cne "CALIBRATE") {
    Write-Host "Cancelled; calibration was not changed."
    exit 1
}

if ($SetServoMiddle) {
    $motorList = $MiddleMotor -join ", "
    Write-Warning "Persistent mode will rewrite servo position correction for motor selector(s): $motorList."
    Write-Warning "Use this only when neutral is near raw 0/4095 or the servo reference was lost."
    $middleConfirmation = Read-Host "Type CENTER-PERSISTENT to confirm the servo-memory write"
    if ($middleConfirmation -cne "CENTER-PERSISTENT") {
        Write-Host "Cancelled before any persistent midpoint write."
        exit 1
    }

    foreach ($motor in $MiddleMotor) {
        & $DrobotLegCli --config $configPath --port $Port set-middle --motor $motor --yes
        if ($LASTEXITCODE -ne 0) {
            throw "Persistent midpoint update failed for motor selector $motor."
        }
    }
}

Write-Host "Capturing the supported physical neutral into $calibrationPath"
& $DrobotLegCli --config $configPath --port $Port --calibration $calibrationPath capture-centers
if ($LASTEXITCODE -ne 0) {
    throw "Neutral-center capture failed."
}

Write-Host "Calibration saved. Verifying near-zero angles and torque OFF."
& $DrobotLegCli --config $configPath --port $Port --calibration $calibrationPath status
exit $LASTEXITCODE
