[CmdletBinding()]
param(
    [string]$Port = "COM4",

    [ValidateRange(1, 65535)]
    [int]$HttpPort = 8766,

    [switch]$Demo,

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$testAppsRoot = $PSScriptRoot
$hardwareRoot = Split-Path -Parent $testAppsRoot
$webCli = Join-Path $hardwareRoot "one-leg-testbed\.venv\Scripts\drobot-four-leg-web.exe"
$manifest = Join-Path $testAppsRoot "config\four-leg.toml"

foreach ($requiredPath in @($webCli, $manifest)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath. Run .\install-test-apps.ps1 first."
    }
}

$arguments = @(
    "--manifest", $manifest,
    "--http-port", $HttpPort
)

if ($Demo) {
    $arguments += "--demo"
    Write-Host "Starting the simulated 12-servo dashboard. COM4 will not be opened."
}
else {
    Write-Warning "Support the complete robot with every foot clear of the floor."
    Write-Warning "Confirm separate fused leg power, shared data/common ground, and physical cutoff."
    $confirmation = Read-Host "Type CONNECT-12 to open $Port and verify IDs 1-12"
    if ($confirmation -cne "CONNECT-12") {
        Write-Host "Cancelled before opening the serial bus."
        exit 1
    }
    $arguments += @("--port", $Port)
}

if ($NoBrowser) {
    $arguments += "--no-browser"
}

& $webCli @arguments
exit $LASTEXITCODE
