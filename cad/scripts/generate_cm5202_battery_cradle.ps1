[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $python = "python"
}

$cadSkillRoot = $env:ROBOT_CAD_SKILL_ROOT
if (-not $cadSkillRoot) {
    $cacheRoot = Join-Path $env:USERPROFILE ".codex\plugins\cache\text-to-cad\cad"
    $cadSkillRoot = Get-ChildItem -LiteralPath $cacheRoot -Directory |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "skills\cad" } |
        Where-Object { Test-Path -LiteralPath (Join-Path $_ "scripts\gen") } |
        Select-Object -First 1
}

$genTool = Join-Path $cadSkillRoot "scripts\gen"
$exportTool = Join-Path $cadSkillRoot "scripts\export"
if (
    -not $cadSkillRoot -or
    -not (Test-Path -LiteralPath $genTool) -or
    -not (Test-Path -LiteralPath $exportTool)
) {
    throw "Could not find the installed CAD gen/export tools. Set ROBOT_CAD_SKILL_ROOT."
}

$partEntry = "drobot_cad/parts/cm5202_battery_cradle.step.py"
$partStep = "exports/step/cm5202_battery_cradle.step"
$partStl = (
    Join-Path $projectRoot "exports\stl\cm5202_battery_cradle.stl"
).Replace("\", "/")
$part3mf = (
    Join-Path $projectRoot "exports\3mf\cm5202_battery_cradle.3mf"
).Replace("\", "/")

$previousPythonPath = $env:PYTHONPATH
$pythonPathParts = @($projectRoot)
if ($previousPythonPath) {
    $pythonPathParts += $previousPythonPath
}
$env:PYTHONPATH = $pythonPathParts -join [IO.Path]::PathSeparator

Push-Location $projectRoot
try {
    & $python $genTool $partEntry --write $partStep
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 cradle STEP generation failed with exit code $LASTEXITCODE."
    }

    & $python $exportTool $partEntry --stl $partStl --3mf $part3mf
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 cradle mesh export failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
