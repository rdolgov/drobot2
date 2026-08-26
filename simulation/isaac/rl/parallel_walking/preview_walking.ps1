[CmdletBinding()]
param(
    [ValidateSet("forward", "backward", "left", "right", "stop")]
    [string]$Command = "forward",
    [ValidateSet("forward", "directional", "smooth-payload", "external-rear-payload", "low-speed-external-rear-payload")]
    [string]$CommandSet = "forward",
    [string]$Checkpoint = "",
    [int]$Seed = 1701,
    [Nullable[double]]$ForwardSpeed = $null,
    [ValidateRange(0, 600)]
    [int]$RecordSeconds = 0,
    [switch]$NoTimeLimit
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")
$context = Get-WalkingContext -CommandSet $CommandSet
$source = Resolve-WalkingCheckpoint -Context $context -Checkpoint $Checkpoint
if ($null -eq $source) {
    throw "No $CommandSet walking checkpoint exists yet. Run train_walking_visible.ps1 or train_walking_headless.ps1 first."
}
if ($CommandSet -eq "forward" -and $Command -ne "forward" -and $Command -ne "stop") {
    Write-Warning "A forward-only checkpoint has not learned $Command. Continue it with -CommandSet directional before judging that command."
}

$arguments = @(
    $context.PlayScript,
    "--motion_command", $Command,
    "--rl_library", "rsl_rl",
    "--task", $context.Task,
    "--checkpoint", $source,
    "--num_envs", "1",
    "--seed", "$Seed",
    "--visualizer", "kit",
    "--max_visible_envs", "1"
)
if ($null -ne $ForwardSpeed) {
    $arguments += @("--forward_speed", "$ForwardSpeed")
}
if ($RecordSeconds -gt 0) {
    # Playback renders at the 60 Hz policy/control rate.  Using 30 here made a
    # requested 30-second review clip contain only 15 seconds of footage.
    $arguments += @("--video", "--video_length", "$($RecordSeconds * 60)")
}
if ($NoTimeLimit) {
    # Keep fall termination active, but do not reset a healthy robot merely
    # because the training horizon elapsed.
    $arguments += "env.disable_time_limit=true"
}

Push-Location $context.RepoRoot
try {
    $horizon = if ($NoTimeLimit) { "until fall or window close" } else { "curriculum episode horizon" }
    Write-Host "Previewing command '$Command' ($horizon) with: $source"
    & $context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab playback exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
