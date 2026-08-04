[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("test", "train-visible", "train-headless")]
    [string]$Mode,
    [int]$Iterations = 0,
    [int]$NumEnvs = 0,
    [int]$Seed = 1481,
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
$isaacPython = "C:\isaacsim\python.bat"
$task = "Drobot-Pure-Stairs-Yaw90-Neutral-Foot-Lift7p5-PersistentBias-CEM-Robust-Hip-Direct"
$playScript = Join-Path $PSScriptRoot "play_pure_parallel_stairs.py"
$trainScript = Join-Path $PSScriptRoot "train_pure_parallel_stairs.py"
$packagedCheckpoint = Join-Path $repoRoot "simulation\isaac\models\ppo-stairs-pure-cem-7p5cm-seed1441\model.pt"
$experimentName = "drobot_pure_stairs_yaw90_neutral_foot_lift7p5_persistent_bias_cem_robust_hip_180x250_direct"
$experimentRoot = Join-Path $repoRoot "logs\rsl_rl\$experimentName"
$bootstrapRun = Join-Path $experimentRoot "_workflow-bootstrap"
$bootstrapCheckpoint = Join-Path $bootstrapRun "model_0.pt"

if (-not (Test-Path -LiteralPath $isaacPython -PathType Leaf)) {
    throw "Isaac Sim Python was not found at $isaacPython"
}

function Resolve-WorkflowCheckpoint {
    param([switch]$PreferWorkflowRun)

    if ($Checkpoint) {
        return (Resolve-Path -LiteralPath $Checkpoint).Path
    }
    if ($PreferWorkflowRun -and (Test-Path -LiteralPath $experimentRoot -PathType Container)) {
        $workflowRuns = Get-ChildItem -LiteralPath $experimentRoot -Directory |
            Where-Object { $_.Name -match "_manual-(visible|headless)$" } |
            Sort-Object LastWriteTime -Descending
        foreach ($run in $workflowRuns) {
            $candidate = Get-ChildItem -LiteralPath $run.FullName -Filter "model_*.pt" -File |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
            if ($null -ne $candidate) {
                return $candidate.FullName
            }
        }
    }
    return (Resolve-Path -LiteralPath $packagedCheckpoint).Path
}

function Invoke-WorkflowTraining {
    param(
        [Parameter(Mandatory = $true)][string]$SourceCheckpoint,
        [Parameter(Mandatory = $true)][int]$EnvironmentCount,
        [Parameter(Mandatory = $true)][int]$IterationCount,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)][string]$Visualizer
    )

    New-Item -ItemType Directory -Force -Path $bootstrapRun | Out-Null
    Copy-Item -LiteralPath $SourceCheckpoint -Destination $bootstrapCheckpoint -Force
    Write-Host "Starting from: $SourceCheckpoint"
    Write-Host "Robots: $EnvironmentCount  PPO iterations: $IterationCount  Visualizer: $Visualizer"

    $arguments = @(
        $trainScript,
        "--rl_library", "rsl_rl",
        "--task", $task,
        "--num_envs", "$EnvironmentCount",
        "--seed", "$Seed",
        "--max_iterations", "$IterationCount",
        "--resume",
        "--load_run", "^_workflow-bootstrap$",
        "--checkpoint", "^model_0[.]pt$",
        "--run_name", $RunName,
        "--visualizer", $Visualizer
    )
    if ($Visualizer -eq "kit") {
        $arguments += @("--max_visible_envs", "$EnvironmentCount")
        # RSL-RL's persistent-state generator partitions by environment.  One
        # minibatch keeps an arbitrary small visible count (including five)
        # valid; the 128-robot headless run retains the configured eight.
        $arguments += "agent.algorithm.num_mini_batches=1"
    }
    & $isaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab training exited with code $LASTEXITCODE"
    }
}

Push-Location $repoRoot
try {
    switch ($Mode) {
        "test" {
            $source = Resolve-WorkflowCheckpoint -PreferWorkflowRun
            Write-Host "Opening one isolated robot with: $source"
            Write-Host "Close the Isaac Sim window when you finish inspecting it."
            & $isaacPython $playScript `
                --viewer_env_index 0 `
                --hide_other_robots `
                --neutral_hold_steps 30 `
                --rl_library rsl_rl `
                --task $task `
                --checkpoint $source `
                --num_envs 1 `
                --seed $Seed `
                --visualizer kit
            if ($LASTEXITCODE -ne 0) {
                throw "Isaac Lab playback exited with code $LASTEXITCODE"
            }
        }
        "train-visible" {
            $visibleIterations = if ($Iterations -gt 0) { $Iterations } else { 5 }
            $visibleEnvs = if ($NumEnvs -gt 0) { $NumEnvs } else { 5 }
            $source = Resolve-WorkflowCheckpoint -PreferWorkflowRun
            Invoke-WorkflowTraining `
                -SourceCheckpoint $source `
                -EnvironmentCount $visibleEnvs `
                -IterationCount $visibleIterations `
                -RunName "manual-visible" `
                -Visualizer "kit"
        }
        "train-headless" {
            $headlessIterations = if ($Iterations -gt 0) { $Iterations } else { 500 }
            $headlessEnvs = if ($NumEnvs -gt 0) { $NumEnvs } else { 128 }
            $source = Resolve-WorkflowCheckpoint -PreferWorkflowRun
            Invoke-WorkflowTraining `
                -SourceCheckpoint $source `
                -EnvironmentCount $headlessEnvs `
                -IterationCount $headlessIterations `
                -RunName "manual-headless" `
                -Visualizer "none"
        }
    }
}
finally {
    Pop-Location
}
