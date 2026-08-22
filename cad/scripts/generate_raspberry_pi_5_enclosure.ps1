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

$baseEntry = "drobot_cad/parts/raspberry_pi_5_enclosure_base.step.py"
$lidEntry = "drobot_cad/parts/raspberry_pi_5_enclosure_lid.step.py"
$coverEntry = "drobot_cad/parts/raspberry_pi_5_imu_cover.step.py"
$previewEntry = "drobot_cad/assembly/raspberry_pi_5_enclosure_fit_preview.step.py"
$baseStep = "exports/step/raspberry_pi_5_enclosure_base.step"
$lidStep = "exports/step/raspberry_pi_5_enclosure_lid.step"
$coverStep = "exports/step/raspberry_pi_5_imu_cover.step"
$previewStep = "exports/step/raspberry_pi_5_enclosure_fit_preview.step"
$previewGlb = (Join-Path $projectRoot "exports\step\.raspberry_pi_5_enclosure_fit_preview.step.glb").Replace("\", "/")
$baseStl = (Join-Path $projectRoot "exports\stl\raspberry_pi_5_enclosure_base.stl").Replace("\", "/")
$lidStl = (Join-Path $projectRoot "exports\stl\raspberry_pi_5_enclosure_lid.stl").Replace("\", "/")
$coverStl = (Join-Path $projectRoot "exports\stl\raspberry_pi_5_imu_cover.stl").Replace("\", "/")
$base3mf = (Join-Path $projectRoot "exports\3mf\raspberry_pi_5_enclosure_base.3mf").Replace("\", "/")
$lid3mf = (Join-Path $projectRoot "exports\3mf\raspberry_pi_5_enclosure_lid.3mf").Replace("\", "/")
$cover3mf = (Join-Path $projectRoot "exports\3mf\raspberry_pi_5_imu_cover.3mf").Replace("\", "/")

$previousPythonPath = $env:PYTHONPATH
$pythonPathParts = @($projectRoot)
if ($previousPythonPath) {
    $pythonPathParts += $previousPythonPath
}
$env:PYTHONPATH = $pythonPathParts -join [IO.Path]::PathSeparator

Push-Location $projectRoot
try {
    & $python $genTool $baseEntry --write $baseStep
    if ($LASTEXITCODE -ne 0) { throw "Pi enclosure base generation failed." }

    & $python $genTool $lidEntry --write $lidStep
    if ($LASTEXITCODE -ne 0) { throw "Pi enclosure lid generation failed." }

    & $python $genTool $coverEntry --write $coverStep
    if ($LASTEXITCODE -ne 0) { throw "Pi IMU cover generation failed." }

    & $python $genTool $previewEntry --write $previewStep
    if ($LASTEXITCODE -ne 0) { throw "Pi enclosure preview generation failed." }

    & $python $exportTool $previewStep --glb $previewGlb
    if ($LASTEXITCODE -ne 0) { throw "Pi enclosure preview GLB generation failed." }

    & $python $exportTool $baseEntry --stl $baseStl --3mf $base3mf
    if ($LASTEXITCODE -ne 0) { throw "Pi enclosure base mesh export failed." }

    & $python $exportTool $lidEntry --stl $lidStl --3mf $lid3mf
    if ($LASTEXITCODE -ne 0) { throw "Pi enclosure lid mesh export failed." }

    & $python $exportTool $coverEntry --stl $coverStl --3mf $cover3mf
    if ($LASTEXITCODE -ne 0) { throw "Pi IMU cover mesh export failed." }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
