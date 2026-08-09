param(
    [string]$IsaacPython = 'C:\isaacsim\python.bat',
    [string]$OutputDir = 'simulation/isaac/output/rl/ppo-stairs-v38-positive-margin-transfer-safe-1024-seed845',
    [int]$Seed = 845
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
        --config simulation/isaac/rl/stairs/quadruped_stairs_v38_positive_margin_rear_transfer.yaml `
        --output-dir $OutputDir `
        --total-timesteps 1024 `
        --seed $Seed `
        --device cpu `
        --fixed-active-steps 1 `
        --fixed-placement-level left-center-tread-load `
        --phase-train-leg rear_right `
        --phase-train-transfer `
        --phase-residual-support-only `
        --phase-compact-residual-action `
        --precursor-leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
        --precursor-leg-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
        --phase-reset-attempts 2 `
        --phase-precursor-max-steps 3600

    if ($LASTEXITCODE -ne 0) {
        throw "V38 positive-margin transfer PPO failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
