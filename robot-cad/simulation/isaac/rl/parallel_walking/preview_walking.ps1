[CmdletBinding()]
param(
    [ValidateSet("forward", "backward", "left", "right", "stop")]
    [string]$Command = "forward",
    [ValidateSet("forward", "directional")]
    [string]$CommandSet = "forward",
    [string]$Checkpoint = "",
    [int]$Seed = 1701,
    [ValidateRange(0, 600)]
    [int]$RecordSeconds = 0
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
if ($RecordSeconds -gt 0) {
    $arguments += @("--video", "--video_length", "$($RecordSeconds * 30)")
}

Push-Location $context.RepoRoot
try {
    Write-Host "Previewing command '$Command' with: $source"
    & $context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab playback exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
