[CmdletBinding()]
param(
    [ValidateSet("forward", "smooth-payload", "external-rear-payload", "low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload", "higher-speed-straight-crawl-external-rear-payload", "padded-feet-forward-bias-external-rear-payload", "robust-straight-low-stance-external-rear-payload", "balanced-four-leg-straight-crawl-external-rear-payload", "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload", "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload", "schedule-matched-support-straight-crawl-external-rear-payload", "symmetry-gated-robust-straight-crawl-external-rear-payload", "fast-run-stop-external-rear-payload", "hardware-matched-run-stop-external-rear-payload", "applied-target-run-stop-external-rear-payload", "integrated-target-velocity-run-stop-external-rear-payload", "command-scaled-high-speed-run-stop-external-rear-payload", "flat-sole-high-speed-run-stop-external-rear-payload", "flat-sole-residual-run-stop-external-rear-payload", "straight-low-abduction-long-run-stop-external-rear-payload", "reference-driven-straight-run-stop-external-rear-payload", "speed-faithful-straight-run-stop-external-rear-payload", "phase-normalized-speed-run-stop-external-rear-payload", "learned-cycle-residual-run-stop-external-rear-payload", "foot-warp-trot-residual-run-stop-external-rear-payload", "flat-contact-hardware-matched-run-stop-external-rear-payload")]
    [string]$CommandSet = "forward",
    [string]$Checkpoint = "",
    [string]$Output = "",
    [ValidateRange(10, 300)]
    [int]$Seconds = 30,
    [ValidateRange(1, 20)]
    [int]$Episodes = 3,
    [int]$Seed = 4401,
    [ValidateSet("task", "nominal", "randomized")]
    [string]$DomainMode = "task",
    [ValidateRange(0.001, 1.0)]
    [Nullable[double]]$ForwardSpeed = $null,
    [Nullable[double]]$ReferenceWeightShiftForwardM = $null,
    [Nullable[double]]$ReferenceRearWeightShiftForwardM = $null,
    [Nullable[double]]$ReferenceWeightShiftLateralM = $null,
    [Nullable[double]]$ReferenceDiagonalSupportLateralShiftM = $null,
    [Nullable[double]]$ReferenceStrideM = $null,
    [Nullable[double]]$ReferenceLiftM = $null,
    [Nullable[double]]$ReferenceStanceForeAftM = $null,
    [Nullable[double]]$ReferenceStanceDownM = $null,
    [Nullable[double]]$ReferenceForwardBodyPitchRad = $null,
    [Nullable[double]]$ReferenceStanceCenterOffsetM = $null,
    [Nullable[double]]$ReferenceFrequencyHz = $null,
    [Nullable[double]]$ReferencePhaseTableScale = $null,
    [string]$ReferencePhaseTablePath = "",
    [Nullable[double]]$ReferenceDutyFactor = $null,
    [Nullable[double]]$ActuatorEffortScale = $null,
    [Nullable[double]]$TargetVelocityScale = $null,
    [Nullable[double]]$IntegratedActionFrequencyGain = $null,
    [Nullable[double]]$IntegratedActionAbductionFrequencyGain = $null,
    [switch]$ZeroPolicyActions,
    [switch]$FlatSoleReference,
    [switch]$DisableRearPayload,
    [string]$PhaseReferenceOutput = "",
    [ValidateRange(32, 2048)]
    [int]$PhaseReferenceBins = 256,
    [ValidateRange(0.0, 300.0)]
    [double]$PhaseReferenceStartSeconds = 3.0,
    [ValidateRange(0.25, 2.0)]
    [double]$PhaseReferenceScaleAboutNeutral = 1.0
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")
$context = Get-WalkingContext -CommandSet $CommandSet
$lowSpeedProfiles = @(
    "low-speed-external-rear-payload",
    "low-speed-crawl-external-rear-payload",
    "higher-speed-straight-crawl-external-rear-payload",
    "padded-feet-forward-bias-external-rear-payload",
    "robust-straight-low-stance-external-rear-payload",
    "balanced-four-leg-straight-crawl-external-rear-payload",
    "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload",
    "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload",
    "schedule-matched-support-straight-crawl-external-rear-payload",
    "symmetry-gated-robust-straight-crawl-external-rear-payload"
)
if ($null -eq $ForwardSpeed) {
    $ForwardSpeed = if ($CommandSet -eq "flat-contact-hardware-matched-run-stop-external-rear-payload") {
        0.04
    }
    elseif ($CommandSet -in $lowSpeedProfiles) {
        0.015
    }
    elseif ($CommandSet -in @("hardware-matched-run-stop-external-rear-payload", "applied-target-run-stop-external-rear-payload", "integrated-target-velocity-run-stop-external-rear-payload")) {
        0.06
    }
    else {
        0.15
    }
}
$source = Resolve-WalkingCheckpoint -Context $context -Checkpoint $Checkpoint
if ($null -eq $source) {
    throw "No $CommandSet walking checkpoint exists yet."
}

Push-Location $context.RepoRoot
try {
    $arguments = @(
        (Join-Path $PSScriptRoot "evaluate_sustained_walking.py"),
        "--checkpoint", $source,
        "--task", $context.Task,
        "--forward-speed", "$ForwardSpeed",
        "--seconds", "$Seconds",
        "--episodes", "$Episodes",
        "--seed", "$Seed",
        "--domain-mode", $DomainMode
    )
    if ($null -ne $ReferenceWeightShiftForwardM) {
        $arguments += @(
            "--reference-weight-shift-forward-m",
            "$ReferenceWeightShiftForwardM"
        )
    }
    if ($null -ne $ReferenceRearWeightShiftForwardM) {
        $arguments += @(
            "--reference-rear-weight-shift-forward-m",
            "$ReferenceRearWeightShiftForwardM"
        )
    }
    if ($null -ne $ReferenceWeightShiftLateralM) {
        $arguments += @(
            "--reference-weight-shift-lateral-m",
            "$ReferenceWeightShiftLateralM"
        )
    }
    if ($null -ne $ReferenceDiagonalSupportLateralShiftM) {
        $arguments += @(
            "--reference-diagonal-support-lateral-shift-m",
            "$ReferenceDiagonalSupportLateralShiftM"
        )
    }
    if ($null -ne $ReferenceStrideM) {
        $arguments += @("--reference-stride-m", "$ReferenceStrideM")
    }
    if ($null -ne $ReferenceLiftM) {
        $arguments += @("--reference-lift-m", "$ReferenceLiftM")
    }
    if ($null -ne $ReferenceStanceForeAftM) {
        $arguments += @(
            "--reference-stance-fore-aft-m",
            "$ReferenceStanceForeAftM"
        )
    }
    if ($null -ne $ReferenceStanceDownM) {
        $arguments += @("--reference-stance-down-m", "$ReferenceStanceDownM")
    }
    if ($null -ne $ReferenceForwardBodyPitchRad) {
        $arguments += @(
            "--reference-forward-body-pitch-rad",
            "$ReferenceForwardBodyPitchRad"
        )
    }
    if ($null -ne $ReferenceStanceCenterOffsetM) {
        $arguments += @(
            "--reference-stance-center-offset-m",
            "$ReferenceStanceCenterOffsetM"
        )
    }
    if ($null -ne $ReferenceFrequencyHz) {
        $arguments += @("--reference-frequency-hz", "$ReferenceFrequencyHz")
    }
    if ($null -ne $ReferencePhaseTableScale) {
        $arguments += @(
            "--reference-phase-table-scale",
            "$ReferencePhaseTableScale"
        )
    }
    if ($ReferencePhaseTablePath) {
        $arguments += @(
            "--reference-phase-table-path",
            $ReferencePhaseTablePath
        )
    }
    if ($null -ne $ReferenceDutyFactor) {
        $arguments += @("--reference-duty-factor", "$ReferenceDutyFactor")
    }
    if ($null -ne $ActuatorEffortScale) {
        $arguments += @("--actuator-effort-scale", "$ActuatorEffortScale")
    }
    if ($null -ne $TargetVelocityScale) {
        $arguments += @("--target-velocity-scale", "$TargetVelocityScale")
    }
    if ($null -ne $IntegratedActionFrequencyGain) {
        $arguments += @(
            "--integrated-action-frequency-gain",
            "$IntegratedActionFrequencyGain"
        )
    }
    if ($null -ne $IntegratedActionAbductionFrequencyGain) {
        $arguments += @(
            "--integrated-action-abduction-frequency-gain",
            "$IntegratedActionAbductionFrequencyGain"
        )
    }
    if ($ZeroPolicyActions) {
        $arguments += "--zero-policy-actions"
    }
    if ($FlatSoleReference) {
        $arguments += "--flat-sole-reference"
    }
    if ($DisableRearPayload) {
        $arguments += "--disable-rear-payload"
    }
    if ($PhaseReferenceOutput) {
        $arguments += @(
            "--phase-reference-output", $PhaseReferenceOutput,
            "--phase-reference-bins", "$PhaseReferenceBins",
            "--phase-reference-start-seconds", "$PhaseReferenceStartSeconds",
            "--phase-reference-scale-about-neutral", "$PhaseReferenceScaleAboutNeutral"
        )
    }
    if ($Output) {
        $arguments += @("--output", $Output)
    }
    & $context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Sustained walking evaluation exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
