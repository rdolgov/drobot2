[CmdletBinding()]
param(
    [ValidateSet("forward", "smooth-payload", "external-rear-payload", "low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload", "higher-speed-straight-crawl-external-rear-payload", "padded-feet-forward-bias-external-rear-payload", "robust-straight-low-stance-external-rear-payload", "balanced-four-leg-straight-crawl-external-rear-payload", "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload", "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload", "schedule-matched-support-straight-crawl-external-rear-payload", "symmetry-gated-robust-straight-crawl-external-rear-payload")]
    [string]$CommandSet = "forward",
    [string]$Checkpoint = "",
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
    [Nullable[double]]$ReferenceStrideM = $null,
    [Nullable[double]]$ReferenceLiftM = $null,
    [Nullable[double]]$ReferenceStanceForeAftM = $null,
    [Nullable[double]]$ReferenceStanceDownM = $null,
    [Nullable[double]]$ReferenceForwardBodyPitchRad = $null,
    [Nullable[double]]$ReferenceStanceCenterOffsetM = $null,
    [Nullable[double]]$ActuatorEffortScale = $null,
    [Nullable[double]]$TargetVelocityScale = $null,
    [switch]$ZeroPolicyActions,
    [switch]$DisableRearPayload
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
    $ForwardSpeed = if ($CommandSet -in $lowSpeedProfiles) { 0.015 } else { 0.15 }
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
    if ($null -ne $ActuatorEffortScale) {
        $arguments += @("--actuator-effort-scale", "$ActuatorEffortScale")
    }
    if ($null -ne $TargetVelocityScale) {
        $arguments += @("--target-velocity-scale", "$TargetVelocityScale")
    }
    if ($ZeroPolicyActions) {
        $arguments += "--zero-policy-actions"
    }
    if ($DisableRearPayload) {
        $arguments += "--disable-rear-payload"
    }
    & $context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Sustained walking evaluation exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
