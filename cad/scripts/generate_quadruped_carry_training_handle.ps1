[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "CAD Python environment not found at $python"
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
if (-not (Test-Path -LiteralPath $genTool) -or -not (Test-Path -LiteralPath $exportTool)) {
    throw "Could not find the installed CAD generation tools."
}

$handleEntry = "drobot_cad/parts/quadruped_carry_training_handle.step.py"
$previewEntry = "drobot_cad/assembly/quadruped_carry_training_handle_fit_preview.step.py"
$handleStep = "exports/step/quadruped_carry_training_handle.step"
$previewStep = "exports/step/quadruped_carry_training_handle_fit_preview.step"
$previewGlb = (Join-Path $projectRoot "exports\step\.quadruped_carry_training_handle_fit_preview.step.glb").Replace("\", "/")
$handleStl = (Join-Path $projectRoot "exports\stl\quadruped_carry_training_handle.stl").Replace("\", "/")
$handle3mf = (Join-Path $projectRoot "exports\3mf\quadruped_carry_training_handle.3mf").Replace("\", "/")

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @($projectRoot, $previousPythonPath) -join [IO.Path]::PathSeparator

Push-Location $projectRoot
try {
    & $python $genTool $handleEntry --write $handleStep
    if ($LASTEXITCODE -ne 0) { throw "Carry/training handle generation failed." }

    & $python $genTool $previewEntry --write $previewStep
    if ($LASTEXITCODE -ne 0) { throw "Carry/training handle preview generation failed." }

    & $python $exportTool $previewStep --glb $previewGlb
    if ($LASTEXITCODE -ne 0) { throw "Carry/training handle preview GLB generation failed." }

    & $python $exportTool $handleEntry --stl $handleStl --3mf $handle3mf
    if ($LASTEXITCODE -ne 0) { throw "Carry/training handle mesh export failed." }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}

