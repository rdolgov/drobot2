[CmdletBinding()]
param(
    [ValidateSet("forward", "directional", "smooth-payload", "external-rear-payload", "low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload", "higher-speed-straight-crawl-external-rear-payload", "padded-feet-forward-bias-external-rear-payload", "robust-straight-low-stance-external-rear-payload", "balanced-four-leg-straight-crawl-external-rear-payload", "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload", "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload", "schedule-matched-support-straight-crawl-external-rear-payload", "symmetry-gated-robust-straight-crawl-external-rear-payload")]
    [string]$CommandSet = "forward",
    [ValidateRange(1, 100000)]
    [int]$Iterations = 20,
    [ValidateRange(1, 64)]
    [int]$NumEnvs = 5,
    [int]$Seed = 1701,
    [string]$Checkpoint = "",
    [ValidateSet("auto", "nominal", "robust")]
    [string]$V25Phase = "auto",
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
        -V25Phase $V25Phase `
        -Fresh:$Fresh
}
finally {
    Pop-Location
}
