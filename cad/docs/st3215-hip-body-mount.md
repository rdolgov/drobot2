# ST3215 body-side hip mount

## Plate-to-fork reinforcement, 2026-09-05

The user reported separation between the robot mounting plate and the fork.
The previous design joined the plate to the curved tip of the half-oval fork
root with only 4 mm of axial overlap. A solid tapered collar now spreads this
connection across the plate and overlaps the solid root over a longer distance.

The collar has a rounded 50 x 64 mm footprint (8 mm corner radius) at the
plate's fork-facing surface, X=-26 mm. It extends 2 mm into the plate and
tapers over 24 mm to a rounded 20 x 48 mm section (6 mm corner radius) at
X=-2 mm. The final section lies inside the original half-oval root. The collar
is fused into the printed part; it is not a separate attachment.

The 76 x 76 x 6 mm mounting plate, flat body face at X=-32 mm, four 4.5 mm
through-holes on the 60 x 60 mm bolt pattern, original fork and output-axis
placement are retained. New material remains at X<0, behind the open fork.
At the plate face the closest collar corner is nominally about 6.32 mm from
each bolt axis, calculated from the rounded-rectangle profile. This provides
space for a 12 mm diameter axial access envelope; it is a source calculation,
not a performed collision check or a specification for any particular tool.

## Sources of truth

- Generator: `drobot_cad/parts/st3215_hip_body_mount.py`.
- Dimensions and intent: `specs/st3215-hip-body-mount.yaml`.
- Reused fork: `drobot_cad/parts/st3215_servo_output_fork.py`.
- Immutable fork reference: `vendor/references/so101/Upper_arm_SO101.step`.
- Coordinate convention: `specs/coordinate-system.md`.

No new purchased components or substitutes were introduced.

## Regeneration

From `cad/`, with the installed CAD skill:

```powershell
$cadSkill = 'C:\Users\roman\.codex\plugins\cache\text-to-cad\cad\0.4.21\skills\cad'
.\.venv\Scripts\python.exe "$cadSkill\scripts\gen" drobot_cad/parts/st3215_hip_body_mount.py --write exports/step/st3215_hip_body_mount.step
.\.venv\Scripts\python.exe "$cadSkill\scripts\export" drobot_cad/parts/st3215_hip_body_mount.py --stl "$PWD\exports\stl\st3215_hip_body_mount.stl"
```

The source is a shared module imported by existing assembly generators, so its
public filename and interface are retained. The standalone manufacturing
outputs are `exports/step/st3215_hip_body_mount.step` and
`exports/stl/st3215_hip_body_mount.stl`.

Refresh the dependent leg, handle, and full-robot STEP assemblies with the
repository's bundled legacy assembly helper on the Python import path:

```powershell
$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = @("$PWD", "$PWD\tools\cad-viewer-markup\runtime\packages\cadpy\src", $previousPythonPath) -join [IO.Path]::PathSeparator
try {
    .\.venv\Scripts\python.exe "$cadSkill\scripts\gen" drobot_cad/assembly/robot_leg.py --write exports/step/robot_leg.step
    .\.venv\Scripts\python.exe "$cadSkill\scripts\gen" drobot_cad/assembly/quadruped_carry_training_handle_fit_preview.step.py --write exports/step/quadruped_carry_training_handle_fit_preview.step
    .\.venv\Scripts\python.exe "$cadSkill\scripts\gen" drobot_cad/assembly/quadruped_robot.py --write exports/step/quadruped_robot_fusion360.step
    if ($LASTEXITCODE -ne 0) { throw 'Full-robot STEP generation failed.' }
    Copy-Item -LiteralPath exports/step/quadruped_robot_fusion360.step -Destination exports/step/quadruped_robot.step -Force
} finally {
    $env:PYTHONPATH = $previousPythonPath
}
```

These commands update static assembly deliverables; they do not exercise robot
motion or update mass/inertia values in the URDF.

## Verification and limitations

Automated tests, standalone geometry validation, collision checks, snapshots,
and browser review were not run, following the user's standing instruction.
No Viewer server was started. The part STEP and STL exports completed, and
the STL exporter emitted no triangulation warning for this part. The initial
leg-assembly export could not import `cadpy`; using the bundled helper path
shown above resolved that dependency without changing assembly source.
Generation uses the existing generator's connected-solid guard, which is not
a structural analysis or a print-strength test. The design increases the
connection footprint and load-transfer length; actual break resistance still
depends on filament, layer direction, and slicing settings. CAD solidity does
not prescribe 100% slicer infill.
