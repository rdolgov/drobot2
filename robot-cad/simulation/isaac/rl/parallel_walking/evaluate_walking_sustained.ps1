[CmdletBinding()]
param(
    [string]$Checkpoint = "",
    [ValidateRange(10, 300)]
    [int]$Seconds = 30,
    [ValidateRange(1, 20)]
    [int]$Episodes = 3
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "walking_workflow_common.ps1")
$context = Get-WalkingContext -CommandSet forward
$source = Resolve-WalkingCheckpoint -Context $context -Checkpoint $Checkpoint
if ($null -eq $source) {
    throw "No V16 forward walking checkpoint exists yet."
}

Push-Location $context.RepoRoot
try {
    & $context.IsaacPython `
        (Join-Path $PSScriptRoot "evaluate_sustained_walking.py") `
        --checkpoint $source `
        --seconds $Seconds `
        --episodes $Episodes
    if ($LASTEXITCODE -ne 0) {
        throw "Sustained walking evaluation exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
