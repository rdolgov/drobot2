[CmdletBinding()]
param(
    [ValidateSet("fast-run-stop-external-rear-payload", "hardware-matched-run-stop-external-rear-payload", "applied-target-run-stop-external-rear-payload", "integrated-target-velocity-run-stop-external-rear-payload", "command-scaled-high-speed-run-stop-external-rear-payload", "flat-sole-high-speed-run-stop-external-rear-payload", "flat-sole-residual-run-stop-external-rear-payload", "straight-low-abduction-long-run-stop-external-rear-payload", "reference-driven-straight-run-stop-external-rear-payload", "speed-faithful-straight-run-stop-external-rear-payload", "phase-normalized-speed-run-stop-external-rear-payload", "learned-cycle-residual-run-stop-external-rear-payload", "foot-warp-trot-residual-run-stop-external-rear-payload", "flat-contact-hardware-matched-run-stop-external-rear-payload")]
    [string]$CommandSet = "integrated-target-velocity-run-stop-external-rear-payload",
    [string]$Checkpoint = "",
    [ValidateRange(0.02, 0.30)]
    [double]$Speed = 0.06,
    [ValidateRange(1.0, 60.0)]
    [double]$RunSeconds = 4.0,
    [ValidateRange(0.0, 5.0)]
    [double]$RampSeconds = 0.60,
    [ValidateRange(1.0, 30.0)]
    [double]$HoldSeconds = 3.0,
    [ValidateRange(1, 100)]
    [int]$Episodes = 5,
    [int]$Seed = 3101,
    [ValidateSet("task", "nominal", "randomized")]
    [string]$DomainMode = "nominal",
    [switch]$FixedStartPhaseZero,
    [Nullable[double]]$RampStartPhaseCycles = $null,
    [ValidateRange(0.0, 0.25)]
    [double]$RampStartPhaseToleranceCycles = 0.0,
    [ValidateRange(0.0, 5.0)]
    [double]$MaximumPhaseAlignmentWaitSeconds = 0.0,
    [string]$Output = "",
    [string]$EnvOverridesJson = "",
    [string]$PhysxStepDiagnosticJsonl = ""
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")
$context = Get-WalkingContext -CommandSet $CommandSet
if ($CommandSet -eq "flat-contact-hardware-matched-run-stop-external-rear-payload") {
    if (-not $PSBoundParameters.ContainsKey("Speed")) { $Speed = 0.04 }
    if (-not $PSBoundParameters.ContainsKey("RampSeconds")) { $RampSeconds = 0.90 }
    if (-not $PSBoundParameters.ContainsKey("HoldSeconds")) { $HoldSeconds = 2.0 }
    if (-not $PSBoundParameters.ContainsKey("RampStartPhaseCycles")) { $RampStartPhaseCycles = 0.05 }
    if (-not $PSBoundParameters.ContainsKey("RampStartPhaseToleranceCycles")) { $RampStartPhaseToleranceCycles = 0.03 }
    if (-not $PSBoundParameters.ContainsKey("MaximumPhaseAlignmentWaitSeconds")) { $MaximumPhaseAlignmentWaitSeconds = 0.50 }
}
$source = Resolve-WalkingCheckpoint -Context $context -Checkpoint $Checkpoint
if ($null -eq $source) {
    throw "No $CommandSet checkpoint exists yet."
}

$arguments = @(
    (Join-Path $PSScriptRoot "evaluate_run_stop.py"),
    "--checkpoint", $source,
    "--task", $context.Task,
    "--speed", "$Speed",
    "--run-seconds", "$RunSeconds",
    "--ramp-seconds", "$RampSeconds",
    "--hold-seconds", "$HoldSeconds",
    "--episodes", "$Episodes",
    "--seed", "$Seed",
    "--domain-mode", $DomainMode
)
if ($FixedStartPhaseZero) {
    $arguments += "--fixed-start-phase-zero"
}
if ($null -ne $RampStartPhaseCycles) {
    $arguments += @(
        "--ramp-start-phase-cycles", "$RampStartPhaseCycles",
        "--ramp-start-phase-tolerance-cycles", "$RampStartPhaseToleranceCycles",
        "--maximum-phase-alignment-wait-s", "$MaximumPhaseAlignmentWaitSeconds"
    )
}
if ($Output) {
    $arguments += @("--output", $Output)
}
if ($EnvOverridesJson) {
    $arguments += @("--env-overrides-json", $EnvOverridesJson)
}
if ($PhysxStepDiagnosticJsonl) {
    $arguments += @("--physx-step-diagnostic-jsonl", $PhysxStepDiagnosticJsonl)
}

Push-Location $context.RepoRoot
try {
    & $context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Run/stop evaluation exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
