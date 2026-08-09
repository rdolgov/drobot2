param(
    [string]$IsaacPython = 'C:\isaacsim\python.bat',
    [string]$OutputDir = 'simulation/isaac/output/rl/ppo-stairs-v40-rear-right-swing-landing-1024-seed856',
    [int]$Seed = 856
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$robotCadRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    throw "Isaac Sim Python launcher not found: $IsaacPython"
}

Push-Location $robotCadRoot
try {
    & $IsaacPython simulation/isaac/rl/stairs/train_stairs_ppo.py `
        --config simulation/isaac/rl/stairs/quadruped_stairs_v40_rear_right_swing_landing.yaml `
        --output-dir $OutputDir `
        --total-timesteps 1024 `
        --seed $Seed `
        --device cpu `
        --fixed-active-steps 1 `
        --fixed-placement-level left-center-tread-load `
        --phase-train-leg rear_right `
        --phase-base-model simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
        --phase-base-swing-only `
        --phase-base-residual-model simulation/isaac/models/ppo-stairs-v38-rear-right-190mm-small/drobot_stairs_ppo_final.zip `
        --phase-base-residual-scale 1.0 `
        --phase-base-residual-mode support_only `
        --phase-base-residual-compact-action `
        --phase-residual-scale 0.5 `
        --phase-residual-swing-only `
        --phase-compact-residual-action `
        --initialize-from-stairs simulation/isaac/models/ppo-stairs-v35-rear-right-190mm-lift-small/drobot_stairs_ppo_final.zip `
        --precursor-leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
        --precursor-leg-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
        --phase-reset-attempts 4 `
        --phase-precursor-max-steps 3600

    if ($LASTEXITCODE -ne 0) {
        throw "V40 rear-right swing landing PPO failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
