[CmdletBinding()]
param(
    [ValidateSet("forward", "directional")]
    [string]$CommandSet = "forward",
    [ValidateRange(1, 100000)]
    [int]$Iterations = 20,
    [ValidateRange(1, 64)]
    [int]$NumEnvs = 5,
    [int]$Seed = 1701,
    [string]$Checkpoint = "",
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")
$context = Get-WalkingContext -CommandSet $CommandSet

Push-Location $context.RepoRoot
try {
    Invoke-WalkingTraining `
        -Context $context `
        -EnvironmentCount $NumEnvs `
        -IterationCount $Iterations `
        -Seed $Seed `
        -RunName "manual-visible" `
        -Visualizer "kit" `
        -Checkpoint $Checkpoint `
        -Fresh:$Fresh
}
finally {
    Pop-Location
}
