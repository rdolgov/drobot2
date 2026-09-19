[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$strideCadRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$stridePython = Join-Path $strideCadRoot '.venv/Scripts/python.exe'
$strideSkill = Join-Path $env:USERPROFILE '.codex/plugins/cache/text-to-cad/cad/0.4.21/skills/cad'
$strideOldPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $strideCadRoot + [IO.Path]::PathSeparator + $strideOldPythonPath
Push-Location $strideCadRoot
try {
    foreach ($stridePart in @('ankle_v23_shank_link','ankle_v23_foot_link')) {
        & $stridePython "$strideSkill/scripts/gen" "drobot_cad/parts/$stridePart.step.py" --write "exports/step/$stridePart.step"
        if ($LASTEXITCODE -ne 0) { throw "STEP generation failed: $stridePart" }
        & $stridePython "$strideSkill/scripts/export" "drobot_cad/parts/$stridePart.step.py" --stl (Join-Path $strideCadRoot "exports/stl/$stridePart.stl")
        if ($LASTEXITCODE -ne 0) { throw "Mesh generation failed: $stridePart" }
    }
} finally {
    Pop-Location
    $env:PYTHONPATH = $strideOldPythonPath
}
