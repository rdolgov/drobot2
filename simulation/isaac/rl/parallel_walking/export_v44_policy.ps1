[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Checkpoint,
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Output,
    [string]$TrainingProvenance = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")
$commandSet = "flat-contact-hardware-matched-run-stop-external-rear-payload"
$context = Get-WalkingContext -CommandSet $commandSet
$checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).Path
$highReference = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "references\v44_flat_contact_high.json")).Path
$lowReference = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "references\v44_flat_contact_low.json")).Path

# Deliberately no selected checkpoint or output default: no V44 model exists
# until this separate task is trained and explicitly qualified. The exporter
# checks saved training provenance, table hashes, and every deployed setting.
$arguments = @(
    (Join-Path $PSScriptRoot "export_policy_onnx.py"),
    "--checkpoint", $checkpointPath,
    "--output", $Output,
    "--training-task", $context.Task,
    "--training-profile", $commandSet,
    "--gait-clock-mode", "speed_scaled",
    "--integrated-gait-phase",
    "--gait-park-at-standstill",
    "--gait-period-s", "2.2222222222222223",
    "--gait-standstill-deadband-m-s", "0.004",
    "--gait-speed-min-m-s", "0.03",
    "--gait-speed-max-m-s", "0.05",
    "--gait-frequency-min-hz", "0.35",
    "--gait-frequency-max-hz", "0.45",
    "--gait-stride-scale-min", "1.0",
    "--gait-pattern", "diagonal_trot",
    "--gait-duty-factor", "0.55",
    "--gait-phase-offsets", "0.0", "0.5", "0.5", "0.0",
    "--action-mode", "gait_residual",
    "--action-history-mode", "applied_target",
    "--residual-action-scale", "0.06",
    "--gait-residual-stop-action-scale", "0.0",
    "--residual-action-scales",
        "0.06", "0.06", "0.06", "0.06",
        "0.20", "0.20", "0.20", "0.20",
        "0.20", "0.20", "0.20", "0.20",
    "--reference-phase-table-path", $highReference,
    "--reference-low-phase-table-path", $lowReference,
    "--reference-blend-speed-min-m-s", "0.03",
    "--reference-blend-speed-max-m-s", "0.05",
    "--reference-start-ramp-s", "1.50",
    "--heading-hold-enabled",
    "--heading-hold-kp-s", "0.75",
    "--heading-hold-max-correction-rad-s", "0.12",
    "--target-velocity-limit-rad-s", "1.5707963267948966",
    "--target-velocity-limits-rad-s",
        "1.0471975511966", "1.0471975511966",
        "1.0471975511966", "1.0471975511966",
        "1.5707963267948966", "1.5707963267948966",
        "1.5707963267948966", "1.5707963267948966",
        "1.5707963267948966", "1.5707963267948966",
        "1.5707963267948966", "1.5707963267948966",
    "--max-target-step-deg", "2.0",
    "--startup-ramp-rate-deg-s", "45.0",
    "--startup-settle-s", "0.5",
    "--startup-position-tolerance-deg", "5.0",
    "--forward-speed-min-m-s", "0.03",
    "--forward-speed-max-m-s", "0.05",
    "--recommended-forward-speed-m-s", "0.04",
    "--command-observation-scales", "10.0", "10.0", "10.0"
)
if ($TrainingProvenance) {
    $arguments += @("--training-provenance", (Resolve-Path -LiteralPath $TrainingProvenance).Path)
}

Push-Location $context.RepoRoot
try {
    & $context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V44 ONNX export exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
