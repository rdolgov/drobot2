param(
    [string]$IsaacPython = 'C:\isaacsim\python.bat',
    [string]$OutputDir = 'simulation/isaac/output/rl/ppo-stairs-v36-transfer-support-hold-4096-seed840'
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
        --config simulation/isaac/rl/stairs/quadruped_stairs_v36_transfer_support_residual.yaml `
        --output-dir $OutputDir `
        --total-timesteps 4096 `
        --seed 840 `
        --device cpu `
        --fixed-active-steps 1 `
        --fixed-placement-level left-center-tread-load `
        --phase-train-leg rear_right `
        --phase-train-transfer `
        --phase-post-transfer-hold-only `
        --phase-residual-support-only `
        --phase-compact-residual-action `
        --precursor-leg-model front_right=simulation/isaac/models/ppo-stairs-v10-180mm-25cm-front-right-placement-small/drobot_stairs_ppo_final.zip `
        --precursor-leg-model front_left=simulation/isaac/models/ppo-stairs-v17-single-foot-190mm-small/drobot_stairs_ppo_final.zip `
        --phase-reset-attempts 8 `
        --phase-precursor-max-steps 3600

    if ($LASTEXITCODE -ne 0) {
        throw "V36 transfer-support PPO training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
