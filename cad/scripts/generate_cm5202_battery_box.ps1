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

$boxEntry = "drobot_cad/parts/cm5202_battery_box.step.py"
$lidEntry = "drobot_cad/parts/cm5202_battery_box_lid.step.py"
$previewEntry = "drobot_cad/assembly/cm5202_battery_box_fit_preview.step.py"
$boxStep = "exports/step/cm5202_battery_box.step"
$lidStep = "exports/step/cm5202_battery_box_lid.step"
$previewStep = "exports/step/cm5202_battery_box_fit_preview.step"
$previewGlb = (Join-Path $projectRoot "exports\step\.cm5202_battery_box_fit_preview.step.glb").Replace("\", "/")
$boxStl = (Join-Path $projectRoot "exports\stl\cm5202_battery_box.stl").Replace("\", "/")
$lidStl = (Join-Path $projectRoot "exports\stl\cm5202_battery_box_lid.stl").Replace("\", "/")
$box3mf = (Join-Path $projectRoot "exports\3mf\cm5202_battery_box.3mf").Replace("\", "/")
$lid3mf = (Join-Path $projectRoot "exports\3mf\cm5202_battery_box_lid.3mf").Replace("\", "/")

$previousPythonPath = $env:PYTHONPATH
$pythonPathParts = @($projectRoot)
if ($previousPythonPath) {
    $pythonPathParts += $previousPythonPath
}
$env:PYTHONPATH = $pythonPathParts -join [IO.Path]::PathSeparator

Push-Location $projectRoot
try {
    & $python $genTool $boxEntry --write $boxStep
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 battery-box STEP generation failed with exit code $LASTEXITCODE."
    }

    & $python $genTool $lidEntry --write $lidStep
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 lid STEP generation failed with exit code $LASTEXITCODE."
    }

    & $python $genTool $previewEntry --write $previewStep
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 box preview STEP generation failed with exit code $LASTEXITCODE."
    }

    # The project-owned CAD Viewer runtime uses an adjacent hidden GLB for
    # exported STEP previews.
    & $python $exportTool $previewStep --glb $previewGlb
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 box preview GLB generation failed with exit code $LASTEXITCODE."
    }

    & $python $exportTool $boxEntry --stl $boxStl --3mf $box3mf
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 battery-box mesh export failed with exit code $LASTEXITCODE."
    }

    & $python $exportTool $lidEntry --stl $lidStl --3mf $lid3mf
    if ($LASTEXITCODE -ne 0) {
        throw "CM5202 lid mesh export failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
