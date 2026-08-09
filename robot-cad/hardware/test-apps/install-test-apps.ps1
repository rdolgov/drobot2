[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$testAppsRoot = $PSScriptRoot
$hardwareRoot = Split-Path -Parent $testAppsRoot
$oneLegRoot = Join-Path $hardwareRoot "one-leg-testbed"
$venvRoot = Join-Path $oneLegRoot ".venv"
$python = Join-Path $venvRoot "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Host "Creating the shared hardware-test virtual environment."
    py -3.11 -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Python virtual-environment creation failed."
    }
}

Write-Host "Installing the shared ST3215 transport and one-leg controller."
& $python -m pip install -e "${oneLegRoot}[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "One-leg testbed installation failed."
}

Write-Host "Installing the Drobot hardware test applications."
& $python -m pip install -e "${testAppsRoot}[dev]"
if ($LASTEXITCODE -ne 0) {
    throw "Hardware test-app installation failed."
}

Write-Host "Installed. Start the simulated dashboard with:"
Write-Host ".\start-four-leg-web.ps1 -Demo"
