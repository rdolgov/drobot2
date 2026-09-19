[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$ankleProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$anklePython = Join-Path $ankleProjectRoot '.venv/Scripts/python.exe'
$ankleSkill = Join-Path $env:USERPROFILE '.codex/plugins/cache/text-to-cad/cad/0.4.21/skills/cad'
$anklePreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $ankleProjectRoot + [IO.Path]::PathSeparator + $anklePreviousPythonPath
Push-Location $ankleProjectRoot
try {
    foreach ($part in @('ankle_v2_lower_leg','ankle_v2_sole','ankle_v2_cheek')) {
        $entry = "drobot_cad/parts/$part.step.py"
        & $anklePython "$ankleSkill/scripts/gen" $entry --write "exports/step/$part.step"
        if ($LASTEXITCODE -ne 0) { throw "$part STEP generation failed" }
        $stl = (Join-Path $ankleProjectRoot "exports/stl/$part.stl").Replace('\','/')
        $threeMf = (Join-Path $ankleProjectRoot "exports/3mf/$part.3mf").Replace('\','/')
        & $anklePython "$ankleSkill/scripts/export" $entry --stl $stl --3mf $threeMf
        if ($LASTEXITCODE -ne 0) { throw "$part mesh export failed" }
    }
    & $anklePython "$ankleSkill/scripts/gen" drobot_cad/assembly/ankle_v2_printable_assembly.step.py --write exports/step/ankle_v2_printable_assembly.step
    if ($LASTEXITCODE -ne 0) { throw 'Ankle assembly STEP generation failed' }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $anklePreviousPythonPath
}
