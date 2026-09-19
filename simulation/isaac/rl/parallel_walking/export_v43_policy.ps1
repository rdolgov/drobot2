[CmdletBinding()]
param(
    [string]$Checkpoint = "",
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")

$commandSet = "foot-warp-trot-residual-run-stop-external-rear-payload"
$context = Get-WalkingContext -CommandSet $commandSet
if (-not $Checkpoint) {
    $Checkpoint = Join-Path $context.RepoRoot (
        "simulation\isaac\models\parallel-running-v43-foot-warp-trot-residual" +
        "\model_399.pt"
    )
}
if (-not $Output) {
    $Output = Join-Path $context.RepoRoot (
        "onboard\models\parallel-running-v43-foot-warp-trot-residual" +
        "\model_399.onnx"
    )
}

$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$highReference = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot (
        "references\v43_v40_low_foot_warp_duty0p46_" +
        "scale1p20_lift0p030.json"
    )
)).Path
$lowReference = (Resolve-Path -LiteralPath (
    Join-Path $PSScriptRoot (
        "references\v43_v40_low_foot_warp_duty0p46_" +
        "scale1p00_lift0p030.json"
    )
)).Path
$exporter = Join-Path $PSScriptRoot "export_policy_onnx.py"

# Keep every trained/deployed behavior-setting value explicit here.  The V43
# exporter also checks this contract and refuses to create a subtly different
# package if a table, task, action semantic, cadence, or actuator limit drifts.
$arguments = @(
    $exporter,
    "--checkpoint", $checkpointPath,
    "--output", $Output,
    "--training-task", $context.Task,
    "--training-profile", $commandSet,
    "--gait-clock-mode", "speed_scaled",
    "--integrated-gait-phase",
    "--gait-park-at-standstill",
    "--gait-period-s", "0.625",
    "--gait-standstill-deadband-m-s", "0.004",
    "--gait-speed-min-m-s", "0.10",
    "--gait-speed-max-m-s", "0.14",
    "--gait-frequency-min-hz", "1.15",
    "--gait-frequency-max-hz", "1.60",
    "--gait-stride-scale-min", "1.0",
    "--gait-pattern", "diagonal_trot",
    "--gait-duty-factor", "0.46",
    "--gait-phase-offsets", "0.0", "0.5", "0.5", "0.0",
    "--action-mode", "gait_residual",
    "--action-history-mode", "applied_target",
    "--residual-action-scale", "0.04",
    "--gait-residual-stop-action-scale", "0.0",
    "--residual-action-scales",
        "0.04", "0.04", "0.04", "0.04",
        "0.12", "0.12", "0.12", "0.12",
        "0.12", "0.12", "0.12", "0.12",
    "--reference-phase-table-path", $highReference,
    "--reference-low-phase-table-path", $lowReference,
    "--reference-blend-speed-min-m-s", "0.10",
    "--reference-blend-speed-max-m-s", "0.14",
    "--reference-start-ramp-s", "1.20",
    "--heading-hold-enabled",
    "--heading-hold-kp-s", "0.75",
    "--heading-hold-max-correction-rad-s", "0.12",
    "--target-velocity-limit-rad-s", "3.14159265358979",
    "--target-velocity-limits-rad-s",
        "1.0471975511966", "1.0471975511966",
        "1.0471975511966", "1.0471975511966",
        "3.14159265358979", "3.14159265358979",
        "3.14159265358979", "3.14159265358979",
        "3.14159265358979", "3.14159265358979",
        "3.14159265358979", "3.14159265358979",
    "--max-target-step-deg", "2.0",
    "--startup-ramp-rate-deg-s", "45.0",
    "--startup-settle-s", "0.5",
    "--startup-position-tolerance-deg", "5.0",
    "--forward-speed-min-m-s", "0.10",
    "--forward-speed-max-m-s", "0.14",
    "--recommended-forward-speed-m-s", "0.12",
    "--command-observation-scales", "10.0", "10.0", "10.0"
)

Push-Location $context.RepoRoot
try {
    & $context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V43 ONNX export exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
