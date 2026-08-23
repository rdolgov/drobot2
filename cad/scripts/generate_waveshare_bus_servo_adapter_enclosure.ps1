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

$baseEntry = "drobot_cad/parts/waveshare_bus_servo_adapter_enclosure_base.step.py"
$lidEntry = "drobot_cad/parts/waveshare_bus_servo_adapter_enclosure_lid.step.py"
$previewEntry = "drobot_cad/assembly/waveshare_bus_servo_adapter_enclosure_fit_preview.step.py"
$baseStep = "exports/step/waveshare_bus_servo_adapter_enclosure_base.step"
$lidStep = "exports/step/waveshare_bus_servo_adapter_enclosure_lid.step"
$previewStep = "exports/step/waveshare_bus_servo_adapter_enclosure_fit_preview.step"
$previewGlb = (Join-Path $projectRoot "exports\step\.waveshare_bus_servo_adapter_enclosure_fit_preview.step.glb").Replace("\", "/")
$baseStl = (Join-Path $projectRoot "exports\stl\waveshare_bus_servo_adapter_enclosure_base.stl").Replace("\", "/")
$lidStl = (Join-Path $projectRoot "exports\stl\waveshare_bus_servo_adapter_enclosure_lid.stl").Replace("\", "/")
$base3mf = (Join-Path $projectRoot "exports\3mf\waveshare_bus_servo_adapter_enclosure_base.3mf").Replace("\", "/")
$lid3mf = (Join-Path $projectRoot "exports\3mf\waveshare_bus_servo_adapter_enclosure_lid.3mf").Replace("\", "/")

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @($projectRoot, $previousPythonPath) -join [IO.Path]::PathSeparator

Push-Location $projectRoot
try {
    & $python $genTool $baseEntry --write $baseStep
    if ($LASTEXITCODE -ne 0) { throw "Waveshare enclosure base generation failed." }

    & $python $genTool $lidEntry --write $lidStep
    if ($LASTEXITCODE -ne 0) { throw "Waveshare enclosure lid generation failed." }

    & $python $genTool $previewEntry --write $previewStep
    if ($LASTEXITCODE -ne 0) { throw "Waveshare enclosure preview generation failed." }

    & $python $exportTool $previewStep --glb $previewGlb
    if ($LASTEXITCODE -ne 0) { throw "Waveshare enclosure preview GLB generation failed." }

    & $python $exportTool $baseEntry --stl $baseStl --3mf $base3mf
    if ($LASTEXITCODE -ne 0) { throw "Waveshare enclosure base mesh export failed." }

    & $python $exportTool $lidEntry --stl $lidStl --3mf $lid3mf
    if ($LASTEXITCODE -ne 0) { throw "Waveshare enclosure lid mesh export failed." }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
