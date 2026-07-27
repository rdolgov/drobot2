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
    "robot_cad/parts/st3215_servo_visual.py=exports/step/st3215_servo_visual.step",
    "robot_cad/parts/upper_arm.py=exports/step/upper_arm.step",
    "robot_cad/parts/st3215_servo_output_fork.py=exports/step/st3215_servo_output_fork.step",
    "robot_cad/parts/st3215_hip.py=exports/step/st3215_hip.step",
    "robot_cad/parts/st3215_hip_body_mount.py=exports/step/st3215_hip_body_mount.step",
    "robot_cad/parts/quadruped_body.py=exports/step/quadruped_body_base.step",
    "robot_cad/parts/quadruped_body_lid.py=exports/step/quadruped_body_lid.step",
    "robot_cad/parts/quadruped_electronics_tray.py=exports/step/quadruped_electronics_tray.step",
    "robot_cad/parts/lekiwi_12v_battery_reference.py=exports/step/lekiwi_12v_battery_reference.step",
    "robot_cad/parts/waveshare_bus_servo_adapter_a.py=exports/step/waveshare_bus_servo_adapter_a.step",
    "robot_cad/parts/adafruit_bno085.py=exports/step/adafruit_bno085_stemma_qt.step",
    "robot_cad/assembly/hip_orientation_preview.py=exports/step/hip_orientation_preview.step",
    "robot_cad/assembly/st3215_motor_bay_fit_preview.py=exports/step/st3215_motor_bay_fit_preview.step",
    "robot_cad/assembly/upper_arm_st3215_fit_preview.py=exports/step/upper_arm_st3215_fit_preview.step",
    "robot_cad/assembly/robot_arm.py=exports/step/robot_arm.step",
    "robot_cad/assembly/robot_leg.py=exports/step/robot_leg.step",
    "robot_cad/assembly/lekiwi_camera_body_fit_preview.py=exports/step/lekiwi_camera_body_fit_preview.step",
    "robot_cad/assembly/quadruped_imu_tray_fit_preview.py=exports/step/quadruped_imu_tray_fit_preview.step",
    "robot_cad/assembly/quadruped_body_hardware_fit_preview.py=exports/step/quadruped_body_hardware_fit_preview.step",
    "robot_cad/assembly/quadruped_robot.py=exports/step/quadruped_robot_fusion360.step"
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

    # The XCAF writer can fail when force-overwriting the large existing
    # quadruped file on Windows. Generate to a fresh Fusion-specific target,
    # then make the historical filename a byte-identical compatibility copy.
    Copy-Item `
        -LiteralPath "exports/step/quadruped_robot_fusion360.step" `
        -Destination "exports/step/quadruped_robot.step" `
        -Force
    if (Test-Path -LiteralPath "exports/step/.quadruped_robot_fusion360.step.glb") {
        Copy-Item `
            -LiteralPath "exports/step/.quadruped_robot_fusion360.step.glb" `
            -Destination "exports/step/.quadruped_robot.step.glb" `
            -Force
    }

    & $python $stepTool `
        "robot_cad/parts/st3215_motor_bay.py=exports/step/st3215_motor_bay.step" `
        --stl "../stl/st3215_motor_bay.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "ST3215 motor-bay STL export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/st3215_servo_visual.py=exports/step/st3215_servo_visual.step" `
        --stl "../stl/st3215_servo_visual.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "Exact ST3215 visual STL export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/st3215_hip.py=exports/step/st3215_hip.step" `
        --stl "../stl/st3215_hip.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "ST3215 hip STL export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/upper_arm.py=exports/step/upper_arm.step" `
        --stl "../stl/upper_arm.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "Upper-arm STL export failed with exit code $LASTEXITCODE."
    }

    & $python $stepTool `
        "robot_cad/parts/st3215_hip_body_mount.py=exports/step/st3215_hip_body_mount.step" `
        --stl "../stl/st3215_hip_body_mount.stl"
    if ($LASTEXITCODE -ne 0) {
        throw "Body-side hip-mount STL export failed with exit code $LASTEXITCODE."
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
