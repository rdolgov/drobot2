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

# The current CAD CLI requires POSIX separators in every source/output pair,
# including on Windows.  These paths resolve from $projectRoot below.
$partEntry = "drobot_cad/parts/rectangular_fork_shoe.step.py"
$previewEntry = "drobot_cad/assembly/rectangular_fork_shoe_fit_preview.step.py"
$partStep = "exports/step/rectangular_fork_shoe.step"
$previewStep = "exports/step/rectangular_fork_shoe_fit_preview.step"
# Mesh output paths resolve beside the entry generator, unlike --write STEP
# paths, so use explicit absolute POSIX paths for the project export folders.
$partStl = (Join-Path $projectRoot "exports\stl\rectangular_fork_shoe.stl").Replace("\", "/")
$part3mf = (Join-Path $projectRoot "exports\3mf\rectangular_fork_shoe.3mf").Replace("\", "/")

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
        throw "Rectangular fork-shoe STEP generation failed with exit code $LASTEXITCODE."
    }

    & $python $genTool $previewEntry --write $previewStep
    if ($LASTEXITCODE -ne 0) {
        throw "Rectangular fork-shoe preview generation failed with exit code $LASTEXITCODE."
    }

    & $python $exportTool $partEntry --stl $partStl --3mf $part3mf
    if ($LASTEXITCODE -ne 0) {
        throw "Rectangular fork-shoe mesh export failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previousPythonPath
}
