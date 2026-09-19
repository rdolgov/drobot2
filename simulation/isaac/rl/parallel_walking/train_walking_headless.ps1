[CmdletBinding()]
param(
    [ValidateSet("forward", "directional", "smooth-payload", "external-rear-payload", "low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload", "higher-speed-straight-crawl-external-rear-payload", "padded-feet-forward-bias-external-rear-payload", "robust-straight-low-stance-external-rear-payload", "balanced-four-leg-straight-crawl-external-rear-payload", "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload", "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload", "schedule-matched-support-straight-crawl-external-rear-payload", "symmetry-gated-robust-straight-crawl-external-rear-payload", "fast-run-stop-external-rear-payload", "hardware-matched-run-stop-external-rear-payload", "applied-target-run-stop-external-rear-payload", "integrated-target-velocity-run-stop-external-rear-payload", "command-scaled-high-speed-run-stop-external-rear-payload", "flat-sole-high-speed-run-stop-external-rear-payload", "flat-sole-residual-run-stop-external-rear-payload", "straight-low-abduction-long-run-stop-external-rear-payload", "reference-driven-straight-run-stop-external-rear-payload", "speed-faithful-straight-run-stop-external-rear-payload", "phase-normalized-speed-run-stop-external-rear-payload", "learned-cycle-residual-run-stop-external-rear-payload", "foot-warp-trot-residual-run-stop-external-rear-payload", "flat-contact-hardware-matched-run-stop-external-rear-payload")]
    [string]$CommandSet = "forward", # V44 remains opt-in.
    [ValidateRange(1, 100000)]
    [int]$Iterations = 500,
    [ValidateRange(1, 8192)]
    [int]$NumEnvs = 128,
    [int]$Seed = 1701,
    [string]$Checkpoint = "",
    [ValidateSet("auto", "nominal", "bridge", "robust", "stress")]
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
        -RunName "manual-headless" `
        -Visualizer "none" `
        -Checkpoint $Checkpoint `
        -V25Phase $V25Phase `
        -Fresh:$Fresh
}
finally {
    Pop-Location
}
