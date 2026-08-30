[CmdletBinding()]
param(
    [ValidateSet("forward", "smooth-payload", "external-rear-payload", "low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload", "higher-speed-straight-crawl-external-rear-payload")]
    [string]$CommandSet = "forward",
    [string]$Checkpoint = "",
    [ValidateRange(10, 300)]
    [int]$Seconds = 30,
    [ValidateRange(1, 20)]
    [int]$Episodes = 3,
    [ValidateRange(0.001, 1.0)]
    [double]$ForwardSpeed = 0.15
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")
$context = Get-WalkingContext -CommandSet $CommandSet
$source = Resolve-WalkingCheckpoint -Context $context -Checkpoint $Checkpoint
if ($null -eq $source) {
    throw "No $CommandSet walking checkpoint exists yet."
}

Push-Location $context.RepoRoot
try {
    & $context.IsaacPython `
        (Join-Path $PSScriptRoot "evaluate_sustained_walking.py") `
        --checkpoint $source `
        --task $context.Task `
        --forward-speed $ForwardSpeed `
        --seconds $Seconds `
        --episodes $Episodes
    if ($LASTEXITCODE -ne 0) {
        throw "Sustained walking evaluation exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
