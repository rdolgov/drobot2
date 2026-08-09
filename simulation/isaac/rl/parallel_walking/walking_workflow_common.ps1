$ErrorActionPreference = "Stop"

function Get-WalkingContext {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("forward", "directional")]
        [string]$CommandSet
    )

    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
    $isaacPython = "C:\isaacsim\python.bat"
    if (-not (Test-Path -LiteralPath $isaacPython -PathType Leaf)) {
        throw "Isaac Sim Python was not found at $isaacPython"
    }
    $suffix = if ($CommandSet -eq "forward") { "forward" } else { "directional" }
    $task = if ($CommandSet -eq "forward") {
        "Drobot-Commanded-Walk-Forward-Direct"
    }
    else {
        "Drobot-Commanded-Walk-Directional-Direct"
    }
    return [PSCustomObject]@{
        RepoRoot = $repoRoot
        IsaacPython = $isaacPython
        Task = $task
        ExperimentName = "drobot_commanded_walk_${suffix}_v16_sustained_beta_direct"
        ExperimentRoot = Join-Path $repoRoot "logs\rsl_rl\drobot_commanded_walk_${suffix}_v16_sustained_beta_direct"
        BundledCheckpoint = if ($CommandSet -eq "forward") {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v16\model_250.pt"
        }
        else {
            $null
        }
        TrainScript = Join-Path $PSScriptRoot "train_commanded_walking.py"
        PlayScript = Join-Path $PSScriptRoot "play_commanded_walking.py"
    }
}

function Find-LatestWalkingCheckpoint {
    param([Parameter(Mandatory = $true)][string]$ExperimentRoot)

    if (-not (Test-Path -LiteralPath $ExperimentRoot -PathType Container)) {
        return $null
    }
    return Get-ChildItem -LiteralPath $ExperimentRoot -Directory |
        # Underscore-prefixed directories are workflow or rejected diagnostic
        # checkpoints and must never silently become the user's next model.
        Where-Object { $_.Name -notlike "_*" } |
        ForEach-Object {
            Get-ChildItem -LiteralPath $_.FullName -Filter "model_*.pt" -File |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 1
        } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Resolve-WalkingCheckpoint {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [string]$Checkpoint = ""
    )

    if ($Checkpoint) {
        return (Resolve-Path -LiteralPath $Checkpoint).Path
    }
    $latest = Find-LatestWalkingCheckpoint -ExperimentRoot $Context.ExperimentRoot
    if ($null -ne $latest) {
        return $latest.FullName
    }
    if ($Context.BundledCheckpoint -and (Test-Path -LiteralPath $Context.BundledCheckpoint -PathType Leaf)) {
        return $Context.BundledCheckpoint
    }
    return $null
}

function Invoke-WalkingTraining {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][int]$EnvironmentCount,
        [Parameter(Mandatory = $true)][int]$IterationCount,
        [Parameter(Mandatory = $true)][int]$Seed,
        [Parameter(Mandatory = $true)][string]$RunName,
        [Parameter(Mandatory = $true)][ValidateSet("kit", "none")][string]$Visualizer,
        [string]$Checkpoint = "",
        [switch]$Fresh
    )

    $source = if ($Fresh) { $null } else {
        Resolve-WalkingCheckpoint -Context $Context -Checkpoint $Checkpoint
    }
    $arguments = @(
        $Context.TrainScript,
        "--rl_library", "rsl_rl",
        "--task", $Context.Task,
        "--num_envs", "$EnvironmentCount",
        "--seed", "$Seed",
        "--max_iterations", "$IterationCount",
        "--run_name", $RunName,
        "--visualizer", $Visualizer
    )
    $curriculumOffsetSteps = 0
    if ($null -ne $source) {
        $checkpointStem = [System.IO.Path]::GetFileNameWithoutExtension($source)
        if ($checkpointStem -match '^model_(\d+)$') {
            # Each saved iteration represents one 64-step rollout per environment.
            $curriculumOffsetSteps = ([int64]$Matches[1] + 1) * 64
        }
        $bootstrapRun = Join-Path $Context.ExperimentRoot "_workflow-bootstrap"
        $bootstrapCheckpoint = Join-Path $bootstrapRun "model_0.pt"
        New-Item -ItemType Directory -Force -Path $bootstrapRun | Out-Null
        Copy-Item -LiteralPath $source -Destination $bootstrapCheckpoint -Force
        $arguments += @(
            "--resume",
            "--load_run", "^_workflow-bootstrap$",
            "--checkpoint", "^model_0[.]pt$"
        )
        Write-Host "Continuing from: $source"
    }
    else {
        Write-Host "Starting a fresh $($Context.Task) policy."
    }
    $arguments += "env.command_curriculum_offset_steps=$curriculumOffsetSteps"
    Write-Host "Command curriculum offset: $curriculumOffsetSteps policy steps"
    if ($Visualizer -eq "kit") {
        $arguments += @(
            "--max_visible_envs", "$EnvironmentCount",
            "agent.algorithm.num_mini_batches=1"
        )
    }
    Write-Host "Robots: $EnvironmentCount  PPO iterations: $IterationCount  Visualizer: $Visualizer"
    & $Context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab training exited with code $LASTEXITCODE"
    }
}
