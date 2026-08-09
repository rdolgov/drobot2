$RobotRuntimeRoot = Split-Path -Parent $PSScriptRoot
$HardwareRoot = Split-Path -Parent $RobotRuntimeRoot
$OneLegTestbedRoot = Join-Path $HardwareRoot "test-apps\one-leg-testbed"
$ServoConfigRoot = Join-Path $RobotRuntimeRoot "servos"
$DrobotLegCli = Join-Path $OneLegTestbedRoot ".venv\Scripts\drobot-leg.exe"
$DrobotLegWebCli = Join-Path $OneLegTestbedRoot ".venv\Scripts\drobot-leg-web.exe"
