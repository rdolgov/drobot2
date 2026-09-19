[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$ankleRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$anklePython = Join-Path $ankleRoot '.venv/Scripts/python.exe'
$ankleSkill = Join-Path $env:USERPROFILE '.codex/plugins/cache/text-to-cad/cad/0.4.21/skills/cad'
$anklePreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $ankleRoot + [IO.Path]::PathSeparator + $anklePreviousPythonPath
Push-Location $ankleRoot
try {
    $entry = 'drobot_cad/parts/ankle_v23_existing_leg_foot.step.py'
    & $anklePython "$ankleSkill/scripts/gen" $entry --write exports/step/ankle_v23_existing_leg_foot.step
    if ($LASTEXITCODE -ne 0) { throw 'Existing-leg foot generation failed' }
    & $anklePython "$ankleSkill/scripts/export" $entry --stl (Join-Path $ankleRoot 'exports/stl/ankle_v23_existing_leg_foot.stl') --3mf (Join-Path $ankleRoot 'exports/3mf/ankle_v23_existing_leg_foot.3mf')
    if ($LASTEXITCODE -ne 0) { throw 'Existing-leg foot mesh export failed' }
    & $anklePython "$ankleSkill/scripts/gen" drobot_cad/assembly/ankle_v23_existing_leg_assembly.step.py --write exports/step/ankle_v23_existing_leg_assembly.step
    if ($LASTEXITCODE -ne 0) { throw 'Existing-leg assembly generation failed' }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $anklePreviousPythonPath
}
