$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Push-Location $repoRoot
try {
    & "C:\isaacsim\python.bat" `
        simulation/isaac/rl/foot_lift/train_foot_lift_ppo.py `
        --config simulation/isaac/rl/foot_lift/quadruped_foot_lift_v3_rear_right_balance.yaml `
        --output-dir simulation/isaac/output/rl/ppo-foot-lift-v3-rear-right-190mm-small `
        --total-timesteps 512 `
        --seed 13190 `
        --device cpu `
        --smoke-test `
        --final-stage-only
    if ($LASTEXITCODE -ne 0) {
        throw "Rear-right 190 mm foot-lift training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
