$ErrorActionPreference = "Stop"

function Get-WalkingContext {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("forward", "directional", "smooth-payload", "external-rear-payload", "low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload", "higher-speed-straight-crawl-external-rear-payload", "padded-feet-forward-bias-external-rear-payload", "robust-straight-low-stance-external-rear-payload", "balanced-four-leg-straight-crawl-external-rear-payload", "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload", "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload", "schedule-matched-support-straight-crawl-external-rear-payload", "symmetry-gated-robust-straight-crawl-external-rear-payload")]
        [string]$CommandSet
    )

    $repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")).Path
    $isaacPython = "C:\isaacsim\python.bat"
    if (-not (Test-Path -LiteralPath $isaacPython -PathType Leaf)) {
        throw "Isaac Sim Python was not found at $isaacPython"
    }
    if ($CommandSet -eq "forward") {
        $task = "Drobot-Commanded-Walk-Forward-Direct"
        $experimentName = "drobot_commanded_walk_forward_v18_coordinated_trot_selected"
    }
    elseif ($CommandSet -eq "directional") {
        $task = "Drobot-Commanded-Walk-Directional-Direct"
        $experimentName = "drobot_commanded_walk_directional_v18_coordinated_trot_selected"
    }
    elseif ($CommandSet -eq "smooth-payload") {
        $task = "Drobot-Commanded-Walk-Smooth-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v19_smooth_rear_payload"
    }
    elseif ($CommandSet -eq "external-rear-payload") {
        $task = "Drobot-Commanded-Walk-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v20_external_rear_payload_straight"
    }
    elseif ($CommandSet -eq "low-speed-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Low-Speed-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v21_low_speed_external_rear_payload"
    }
    elseif ($CommandSet -eq "low-speed-crawl-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Low-Speed-Crawl-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v22_low_speed_crawl_external_rear_payload"
    }
    elseif ($CommandSet -eq "higher-speed-straight-crawl-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Higher-Speed-Straight-Crawl-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v23_higher_speed_straight_crawl_external_rear_payload"
    }
    elseif ($CommandSet -eq "padded-feet-forward-bias-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Padded-Feet-Forward-Bias-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v24_padded_feet_forward_bias_external_rear_payload"
    }
    elseif ($CommandSet -eq "robust-straight-low-stance-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Robust-Straight-Low-Stance-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v25_robust_straight_low_stance_external_rear_payload"
    }
    elseif ($CommandSet -eq "balanced-four-leg-straight-crawl-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Balanced-Four-Leg-Straight-Crawl-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v26_balanced_four_leg_straight_crawl_external_rear_payload"
    }
    elseif ($CommandSet -eq "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Adaptive-Asymmetric-Four-Leg-Straight-Crawl-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v27_adaptive_asymmetric_four_leg_straight_crawl_external_rear_payload"
    }
    elseif ($CommandSet -eq "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Forward-Biased-Cycle-Gated-Four-Leg-Straight-Crawl-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v28_forward_biased_cycle_gated_four_leg_straight_crawl_external_rear_payload"
    }
    elseif ($CommandSet -eq "schedule-matched-support-straight-crawl-external-rear-payload") {
        $task = "Drobot-Commanded-Walk-Schedule-Matched-Support-Straight-Crawl-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v29_schedule_matched_support_straight_crawl_external_rear_payload"
    }
    else {
        $task = "Drobot-Commanded-Walk-Symmetry-Gated-Robust-Straight-Crawl-External-Rear-Payload-Direct"
        $experimentName = "drobot_commanded_walk_v30_symmetry_gated_robust_straight_crawl_external_rear_payload"
    }
    return [PSCustomObject]@{
        CommandSet = $CommandSet
        RepoRoot = $repoRoot
        IsaacPython = $isaacPython
        Task = $task
        ExperimentName = $experimentName
        ExperimentRoot = Join-Path $repoRoot "logs\rsl_rl\$experimentName"
        BundledCheckpoint = if ($CommandSet -in @(
            "balanced-four-leg-straight-crawl-external-rear-payload",
            "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload",
            "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload",
            "schedule-matched-support-straight-crawl-external-rear-payload",
            "symmetry-gated-robust-straight-crawl-external-rear-payload"
        )) {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v25-neutral-bootstrap\model_3248.pt"
        }
        elseif ($CommandSet -eq "robust-straight-low-stance-external-rear-payload") {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v24-padded-feet-forward-bias\model_3248.pt"
        }
        elseif ($CommandSet -eq "padded-feet-forward-bias-external-rear-payload") {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v23-higher-speed-straight-residual-crawl\model_1500.pt"
        }
        elseif ($CommandSet -eq "higher-speed-straight-crawl-external-rear-payload") {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v22-low-speed-residual-crawl\model_500.pt"
        }
        elseif ($CommandSet -in @("low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload")) {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v20-external-rear-payload\model_900.pt"
        }
        elseif ($CommandSet -eq "external-rear-payload") {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v19-smooth-rear-payload\model_899.pt"
        }
        elseif ($CommandSet -in @("forward", "smooth-payload")) {
            Join-Path $repoRoot "simulation\isaac\models\parallel-walking-v18-coordinated\model_299.pt"
        }
        else {
            $null
        }
        ResetCurriculumOffset = ($CommandSet -in @("low-speed-external-rear-payload", "low-speed-crawl-external-rear-payload", "higher-speed-straight-crawl-external-rear-payload", "padded-feet-forward-bias-external-rear-payload", "robust-straight-low-stance-external-rear-payload", "balanced-four-leg-straight-crawl-external-rear-payload", "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload", "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload", "schedule-matched-support-straight-crawl-external-rear-payload", "symmetry-gated-robust-straight-crawl-external-rear-payload"))
        CurriculumIterationBase = if ($CommandSet -eq "symmetry-gated-robust-straight-crawl-external-rear-payload") { 4572 } elseif ($CommandSet -in @("robust-straight-low-stance-external-rear-payload", "balanced-four-leg-straight-crawl-external-rear-payload", "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload", "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload", "schedule-matched-support-straight-crawl-external-rear-payload")) { 3248 } else { $null }
        TrainScript = Join-Path $PSScriptRoot "train_commanded_walking.py"
        PlayScript = Join-Path $PSScriptRoot "play_commanded_walking.py"
    }
}

function Read-WalkingCurriculumOffset {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][string]$Checkpoint
    )

    $sidecar = "$Checkpoint.curriculum.json"
    if (Test-Path -LiteralPath $sidecar -PathType Leaf) {
        try {
            $record = Get-Content -LiteralPath $sidecar -Raw | ConvertFrom-Json
            if (
                $record.profile -eq $Context.CommandSet -and
                [int64]$record.curriculum_policy_steps -ge 0
            ) {
                return [int64]$record.curriculum_policy_steps
            }
        }
        catch {
            Write-Warning "Ignoring unreadable curriculum sidecar: $sidecar"
        }
    }

    $experimentPrefix = $Context.ExperimentRoot.TrimEnd('\') + '\'
    $sourceIsCurrentExperiment = $Checkpoint.StartsWith(
        $experimentPrefix,
        [System.StringComparison]::OrdinalIgnoreCase
    )
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Checkpoint)
    if (
        $sourceIsCurrentExperiment -and
        $null -ne $Context.CurriculumIterationBase -and
        $stem -match '^model_(\d+)$'
    ) {
        $profileIterations = [Math]::Max(
            0,
            [int64]$Matches[1] - [int64]$Context.CurriculumIterationBase
        )
        Write-Warning (
            "Curriculum sidecar missing; inferring $profileIterations $($Context.CommandSet) updates " +
            "from checkpoint iteration $($Matches[1])."
        )
        return $profileIterations * 64
    }
    return $null
}

function Write-WalkingCurriculumSidecars {
    param(
        [Parameter(Mandatory = $true)]$Context,
        [Parameter(Mandatory = $true)][string]$SourceCheckpoint,
        [Parameter(Mandatory = $true)][int64]$StartingOffsetSteps,
        [Parameter(Mandatory = $true)][datetime]$TrainingStartedUtc
    )

    if ($null -eq $Context.CurriculumIterationBase) {
        return
    }
    $sourceStem = [System.IO.Path]::GetFileNameWithoutExtension($SourceCheckpoint)
    if ($sourceStem -notmatch '^model_(\d+)$') {
        Write-Warning "Cannot persist curriculum age for nonstandard checkpoint $SourceCheckpoint"
        return
    }
    $sourceIteration = [int64]$Matches[1]
    Get-ChildItem -LiteralPath $Context.ExperimentRoot -Directory |
        Where-Object { $_.Name -notlike "_*" } |
        ForEach-Object {
            Get-ChildItem -LiteralPath $_.FullName -Filter "model_*.pt" -File
        } |
        Where-Object { $_.LastWriteTimeUtc -ge $TrainingStartedUtc.AddSeconds(-2) } |
        ForEach-Object {
            if ($_.BaseName -match '^model_(\d+)$') {
                $modelIteration = [int64]$Matches[1]
                $newUpdates = [Math]::Max(0, $modelIteration - $sourceIteration)
                $record = [ordered]@{
                    profile = $Context.CommandSet
                    curriculum_policy_steps = $StartingOffsetSteps + 64 * $newUpdates
                    source_checkpoint = [System.IO.Path]::GetFileName($SourceCheckpoint)
                    source_iteration = $sourceIteration
                    model_iteration = $modelIteration
                    recorded_at_utc = [DateTime]::UtcNow.ToString("o")
                }
                $record | ConvertTo-Json | Set-Content `
                    -LiteralPath "$($_.FullName).curriculum.json" `
                    -Encoding utf8
            }
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
        [ValidateSet("auto", "nominal", "robust")][string]$V25Phase = "auto",
        [switch]$Fresh
    )

    if (
        $Fresh -and
        $Context.CommandSet -in @(
            "robust-straight-low-stance-external-rear-payload",
            "balanced-four-leg-straight-crawl-external-rear-payload",
            "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload",
            "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload",
            "schedule-matched-support-straight-crawl-external-rear-payload",
            "symmetry-gated-robust-straight-crawl-external-rear-payload"
        )
    ) {
        throw "$($Context.CommandSet) is a checkpoint-continuation profile; do not use -Fresh for this profile."
    }
    $source = if ($Fresh) { $null } else {
        Resolve-WalkingCheckpoint -Context $Context -Checkpoint $Checkpoint
    }
    $curriculumOffsetSteps = 0
    if ($null -ne $source) {
        $persistedOffset = Read-WalkingCurriculumOffset `
            -Context $Context `
            -Checkpoint $source
        $checkpointStem = [System.IO.Path]::GetFileNameWithoutExtension($source)
        $experimentPrefix = $Context.ExperimentRoot.TrimEnd('\') + '\'
        $sourceIsCurrentExperiment = $source.StartsWith(
            $experimentPrefix,
            [System.StringComparison]::OrdinalIgnoreCase
        )
        $resetTransferredCurriculum = (
            $Context.ResetCurriculumOffset -and -not $sourceIsCurrentExperiment
        )
        if ($null -ne $persistedOffset) {
            $curriculumOffsetSteps = $persistedOffset
        }
        elseif (-not $resetTransferredCurriculum -and $checkpointStem -match '^model_(\d+)$') {
            # Each saved iteration represents one 64-step rollout per environment.
            $curriculumOffsetSteps = ([int64]$Matches[1] + 1) * 64
        }
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
    if ($null -ne $source) {
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
    $usesPhasedRandomization = $Context.CommandSet -in @(
        "robust-straight-low-stance-external-rear-payload",
        "balanced-four-leg-straight-crawl-external-rear-payload",
        "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload",
        "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload",
        "schedule-matched-support-straight-crawl-external-rear-payload",
        "symmetry-gated-robust-straight-crawl-external-rear-payload"
    )
    if ($usesPhasedRandomization) {
        # V29/V30 first have to learn exact contact topology and then complete their
        # 128,000-step speed curriculum.  Switching it to the broad robust
        # domain after the legacy 350-update adaptation stage would happen at
        # only 22,400 steps, while its command ceiling is still about 0.024 m/s.
        $nominalAdaptationSteps = if (
            $Context.CommandSet -in @(
                "schedule-matched-support-straight-crawl-external-rear-payload",
                "symmetry-gated-robust-straight-crawl-external-rear-payload"
            )
        ) {
            2000 * 64
        }
        else {
            350 * 64
        }
        $resolvedV25Phase = if ($V25Phase -eq "auto") {
            if ($curriculumOffsetSteps -lt $nominalAdaptationSteps) { "nominal" } else { "robust" }
        }
        else {
            $V25Phase
        }
        $nominalFraction = if ($resolvedV25Phase -eq "nominal") { 1.0 } else { 0.25 }
        $arguments += "env.physical_randomization_nominal_fraction=$nominalFraction"
        if ($resolvedV25Phase -eq "nominal") {
            # V24's common payload and supply domains predate the nominal-mask
            # mechanism, so pin them explicitly during gait/stance adaptation.
            $arguments += @(
                "env.rear_payload_combined_mass_scale_range=[1.0,1.0]",
                "env.rear_payload_combined_com_jitter_m=[0.0,0.0,0.0]",
                "env.actuator_effort_scale_range=[1.0,1.0]",
                "env.target_velocity_scale_range=[1.0,1.0]"
            )
        }
        elseif ($Context.CommandSet -in @(
            "balanced-four-leg-straight-crawl-external-rear-payload",
            "adaptive-asymmetric-four-leg-straight-crawl-external-rear-payload",
            "forward-biased-cycle-gated-four-leg-straight-crawl-external-rear-payload",
            "schedule-matched-support-straight-crawl-external-rear-payload",
            "symmetry-gated-robust-straight-crawl-external-rear-payload"
        )) {
            # V26-V30 deliberately include asymmetric physical randomization.
            # An exact left/right mirror is therefore not a valid robust sample.
            $arguments += @(
                "agent.algorithm.symmetry_cfg.use_data_augmentation=false",
                "agent.algorithm.symmetry_cfg.use_mirror_loss=false"
            )
        }
        Write-Host "Phased randomization: $resolvedV25Phase (nominal environment fraction $nominalFraction)"
    }
    elseif ($V25Phase -ne "auto") {
        throw "-V25Phase only applies to the V25-V30 phased-randomization profiles."
    }
    Write-Host "Command curriculum offset: $curriculumOffsetSteps policy steps"
    if ($Visualizer -eq "kit") {
        $arguments += @(
            "--max_visible_envs", "$EnvironmentCount",
            "agent.algorithm.num_mini_batches=1"
        )
    }
    Write-Host "Robots: $EnvironmentCount  PPO iterations: $IterationCount  Visualizer: $Visualizer"
    $trainingStartedUtc = [DateTime]::UtcNow
    & $Context.IsaacPython @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Isaac Lab training exited with code $LASTEXITCODE"
    }
    if (
        $null -ne $source -and
        $usesPhasedRandomization
    ) {
        Write-WalkingCurriculumSidecars `
            -Context $Context `
            -SourceCheckpoint $source `
            -StartingOffsetSteps $curriculumOffsetSteps `
            -TrainingStartedUtc $trainingStartedUtc
    }
}
