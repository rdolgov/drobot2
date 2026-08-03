[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 4)]
    [int]$Leg,

    [string]$Port = "COM4",

    [ValidateRange(1, 65535)]
    [int]$HttpPort = 8765,

    [ValidateRange(1.0, 90.0)]
    [double]$RampRate = 30.0,

    [switch]$Demo,

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$testbedRoot = Split-Path -Parent $PSScriptRoot
$webCli = Join-Path $testbedRoot ".venv\Scripts\drobot-leg-web.exe"
$configPath = Join-Path $PSScriptRoot "leg-$Leg.toml"
$calibrationPath = Join-Path $PSScriptRoot "calibration-leg-$Leg.json"

foreach ($requiredPath in @($webCli, $configPath, $calibrationPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

$webArguments = @(
    "--config", $configPath,
    "--calibration", $calibrationPath,
    "--http-port", $HttpPort,
    "--ramp-rate", $RampRate
)

if ($Demo) {
    $webArguments += "--demo"
    Write-Host "Starting the Leg $Leg simulated controller; no serial port will be opened."
}
else {
    $webArguments += @("--port", $Port)
    Write-Warning "Support Leg $Leg, clear its full range, and keep the physical cutoff ready."
    Write-Host "The server starts disarmed and binds only to 127.0.0.1:$HttpPort."
}

if ($NoBrowser) {
    $webArguments += "--no-browser"
}

& $webCli @webArguments
exit $LASTEXITCODE
