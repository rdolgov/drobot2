[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$cadSkillRoot = $env:ROBOT_CAD_SKILL_ROOT
if (-not $cadSkillRoot) {
    $cacheRoot = Join-Path $env:USERPROFILE ".codex\plugins\cache\text-to-cad\cad"
    $cadSkillRoot = Get-ChildItem -LiteralPath $cacheRoot -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "skills\cad" } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "scripts\step") } |
        Select-Object -First 1
}

$stepTool = Join-Path $cadSkillRoot "scripts\step"
if (-not $cadSkillRoot -or -not (Test-Path -LiteralPath $stepTool)) {
    throw "Could not find the installed CAD skill. Set ROBOT_CAD_SKILL_ROOT."
}

$cadpySource = Join-Path $cadSkillRoot "scripts\packages\cadpy\src"
$previousPythonPath = $env:PYTHONPATH
$pythonPathParts = @($projectRoot, $cadpySource)
if ($previousPythonPath) {
    $pythonPathParts += $previousPythonPath
}
$env:PYTHONPATH = $pythonPathParts -join [IO.Path]::PathSeparator

$targets = @(
    "robot_cad/parts/st3215_motor_bay.py=exports/step/st3215_motor_bay.step",
    "robot_cad/parts/upper_arm.py=exports/step/upper_arm.step",
    "robot_cad/parts/st3215_servo_output_fork.py=exports/step/st3215_servo_output_fork.step",
    "robot_cad/parts/st3215_hip.py=exports/step/st3215_hip.step",
    "robot_cad/parts/st3215_hip_body_mount.py=exports/step/st3215_hip_body_mount.step",
    "robot_cad/parts/quadruped_body.py=exports/step/quadruped_body_base.step",
    "robot_cad/parts/quadruped_body_lid.py=exports/step/quadruped_body_lid.step",
    "robot_cad/parts/quadruped_electronics_tray.py=exports/step/quadruped_electronics_tray.step",
    "robot_cad/assembly/hip_orientation_preview.py=exports/step/hip_orientation_preview.step",
    "robot_cad/assembly/st3215_motor_bay_fit_preview.py=exports/step/st3215_motor_bay_fit_preview.step",
    "robot_cad/assembly/upper_arm_st3215_fit_preview.py=exports/step/upper_arm_st3215_fit_preview.step",
    "robot_cad/assembly/robot_arm.py=exports/step/robot_arm.step",
    "robot_cad/assembly/robot_leg.py=exports/step/robot_leg.step",
    "robot_cad/assembly/quadruped_robot.py=exports/step/quadruped_robot.step"
)
$arguments = @($stepTool) + $targets
if ($Force) {
    $arguments += "--force"
}

Push-Location $projectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "CAD generation failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/st3215_motor_bay.py=exports/step/st3215_motor_bay.step" `
        --stl "exports/stl/st3215_motor_bay.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "ST3215 motor-bay STL export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/st3215_hip.py=exports/step/st3215_hip.step" `
        --stl "../stl/st3215_hip.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "ST3215 hip STL export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/upper_arm.py=exports/step/upper_arm.step" `
        --stl "exports/stl/upper_arm.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "Upper-arm STL export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/quadruped_body.py=exports/step/quadruped_body_base.step" `
        --stl "../stl/quadruped_body_base.stl" `
        --3mf "../3mf/quadruped_body_base.3mf"
    if ($LASTEXITCODE -ne 0) {
        throw "Quadruped body-base mesh export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/quadruped_body_lid.py=exports/step/quadruped_body_lid.step" `
        --stl "../stl/quadruped_body_lid.stl" `
        --3mf "../3mf/quadruped_body_lid.3mf"
    if ($LASTEXITCODE -ne 0) {
        throw "Quadruped body-lid mesh export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/quadruped_electronics_tray.py=exports/step/quadruped_electronics_tray.step" `
        --stl "../stl/quadruped_electronics_tray.stl" `
        --3mf "../3mf/quadruped_electronics_tray.3mf"
    if ($LASTEXITCODE -ne 0) {
        throw "Quadruped electronics-tray mesh export failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
