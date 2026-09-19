[CmdletBinding()]
param(
    [ValidateRange(1, 100000)][int]$Iterations = 500,
    [ValidateRange(1, 4096)][int]$NumEnvs = 128,
    [int]$Seed = 1901,
    [ValidateSet('Active', 'Baseline', 'Locked')][string[]]$Variants = @('Active', 'Baseline', 'Locked')
)
$ErrorActionPreference = 'Stop'
$pilotRoot = (Resolve-Path (Join-Path $PSScriptRoot '../../../..')).Path
$pilotStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$pilotDir = Join-Path $pilotRoot "simulation/reviews/ankle-v2-training/$pilotStamp-seed$Seed"
New-Item -ItemType Directory -Path $pilotDir -Force | Out-Null
$snapshotDir = Join-Path $pilotDir 'source'
New-Item -ItemType Directory -Path $snapshotDir -Force | Out-Null
$sources = Get-ChildItem -LiteralPath $PSScriptRoot -File | Where-Object Extension -in '.py', '.json', '.ps1'
$sources | Copy-Item -Destination $snapshotDir
$hashes = @($sources | ForEach-Object { @{path=$_.FullName; sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash} })
$state = @{
    schema='drobot.ankle-pilot-batch.v1'; status='running'; started=(Get-Date).ToString('o')
    iterations=$Iterations; num_envs=$NumEnvs; seed=$Seed; variants=$Variants
    output_directory=$pilotDir; source_hashes=$hashes; completed=@()
    scope='Equal-budget first-seed training pilot; not held-out evaluation or hardware qualification'
}
$statePath = Join-Path $pilotDir 'batch.json'
$state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
Write-Output "ANKLE_PILOT_DIRECTORY=$pilotDir"
Push-Location $pilotRoot
try {
    foreach ($variant in $Variants) {
        $variantLower = $variant.ToLowerInvariant()
        $asset = Join-Path $pilotRoot "simulation/exports/isaac/ankle_v2/$variantLower/robot.usdc"
        if (-not (Test-Path -LiteralPath $asset)) { throw "Missing training asset $asset. Run import_assets.py first." }
        $state.current_variant = $variant
        $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
        $log = Join-Path $pilotDir "$variantLower.log"
        $metrics = (Join-Path $pilotDir "$variantLower.metrics.jsonl").Replace('\', '/')
        $trainingArgs = @(
            (Join-Path $PSScriptRoot 'train.py'), '--rl_library', 'rsl_rl',
            '--task', "Drobot-Ankle-V2-$variant", '--num_envs', "$NumEnvs",
            '--seed', "$Seed", '--max_iterations', "$Iterations",
            '--run_name', "ankle-$pilotStamp-seed$Seed", '--visualizer', 'none',
            "env.metrics_path=$metrics"
        )
        $commandRecord = @{executable='C:/isaacsim/python.bat'; arguments=$trainingArgs; asset=$asset; asset_sha256=(Get-FileHash -LiteralPath $asset -Algorithm SHA256).Hash}
        $commandRecord | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $pilotDir "$variantLower.command.json") -Encoding utf8
        Write-Output "ANKLE_TRAIN_START=$variant"
        & C:/isaacsim/python.bat @trainingArgs *> $log
        if ($LASTEXITCODE -ne 0) { throw "$variant training exited $LASTEXITCODE; see $log" }
        $state.completed += $variant
        $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
        Write-Output "ANKLE_TRAIN_RETURNED=$variant"
    }
    $state.status = 'training_processes_returned'
    $state.finished = (Get-Date).ToString('o')
}
catch {
    $state.status = 'failed'
    $state.error = $_.Exception.Message
    throw
}
finally {
    $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statePath -Encoding utf8
    Pop-Location
}
