# TPU distal-fork shoe

> Superseded on 2026-08-11 by the
> [rigid low-profile fork shoe](rigid-fork-shoe.md) for indoor hardwood use.

## Design

`tpu_fork_shoe` is a one-piece, replaceable TPU foot for the free fork at the
end of each lower leg. Its local origin is the existing distal revolute axis,
so the fit preview places it from the stable `frame_distal_fork_axis` datum
rather than transient STEP faces.

The attachment hub clears the closest inward fork bosses by 0.4 mm per side.
Four 3.4 mm TPU bores reuse the fork's existing 3.0 mm holes on a 9.899494 mm
square. Install two diagonal M3 x 75 mm threaded rods with washers and nylon
lock nuts to lock rotation; the other diagonal remains available. Do not
tighten until the TPU visibly crushes. Existing printed fork holes may need a
3.0 mm hand reamer.

The contact body is a fully convex hollow near-round ellipsoid rather than a
sphere with a flat circular pad. Its principal radii are 27 x 24 x 24 mm,
giving a 54 x 48 mm rocker profile along the leg axis with no planar contact
face. The 1.125:1 axial/radial ratio keeps a small amount of directional
rocker behavior while appearing substantially closer to a circle.

The ellipsoid extends 5 mm rearward into the screw attachment hub and ends at
X=60.5 mm. The earlier exposed 8 mm cylindrical load core remains removed. The
overlapping ellipsoid and hub form the load path directly, producing a broad
monolithic junction intended to reduce wobble without changing the fork
interface. A 4 mm principal-axis shell and four pairs of side vents retain the
compliant contact behavior.

TPU 95A, 0.20 mm layers, four to six walls, a broad brim, and supports tuned
for flexible filament are reasonable prototype settings. Because the contact
surface is deliberately fully rounded, choose the actual slicer orientation
after inspecting overhangs; there is no longer a flat contact pad to place on
the print bed.

The generated CAD material volume and mass estimate are recorded in the
validation section below. A slicer's sparse infill in the hub/core can reduce
the fully dense estimate, but distal mass should be checked against the
walking controller and ST3215 torque budget before fitting four shoes.

## Hardware provenance

The step.parts catalog was searched on 2026-08-08. It contains
`iso_metric_threaded_rod_m3_l075_simple`, a generic M3 x 75 mm threaded rod.
The fit preview intentionally uses smooth 3.0 mm clearance envelopes because
thread detail is irrelevant to the mating check; no vendor file is added.

## Generate

From `cad/`:

```powershell
$cadSkill = "$env:USERPROFILE\.codex\plugins\cache\text-to-cad\cad\0.3.13\skills\cad"
$cadpy = Join-Path $cadSkill "scripts\packages\cadpy\src"
$env:PYTHONPATH = ((Resolve-Path ".").Path, $cadpy) -join [IO.Path]::PathSeparator

.\.venv\Scripts\python.exe "$cadSkill\scripts\step" `
  drobot_cad/parts/tpu_fork_shoe.py=exports/step/tpu_fork_shoe.step `
  --stl ../stl/tpu_fork_shoe.stl `
  --3mf ../3mf/tpu_fork_shoe.3mf `
  --force

.\.venv\Scripts\python.exe "$cadSkill\scripts\step" `
  drobot_cad/assembly/tpu_fork_shoe_fit_preview.py=exports/step/tpu_fork_shoe_fit_preview.step `
  --force
```

## Validate

Run the focused geometry and fit tests, Ruff, baseline STEP inspection,
spec-driven measurements, snapshots, and CAD Viewer handoff:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_tpu_fork_shoe.py
.\.venv\Scripts\python.exe -m ruff check `
  drobot_cad/parts/tpu_fork_shoe.py `
  drobot_cad/assembly/tpu_fork_shoe_fit_preview.py `
  tests/test_tpu_fork_shoe.py
```

## Outputs

- `exports/step/tpu_fork_shoe.step`: primary printable CAD
- `exports/stl/tpu_fork_shoe.stl`: slicer mesh
- `exports/3mf/tpu_fork_shoe.3mf`: slicer-ready mesh container
- `exports/step/tpu_fork_shoe_fit_preview.step`: lower-leg installation review

The STEP, STL, 3MF, and fit-preview STEP were regenerated from the near-round
direct-fusion source on 2026-08-10. Generation completed successfully. They
have not undergone the separate automated geometry, fit, or regression checks
listed below.

## Known limitations

The design has not been load-tested, fatigue-tested, or printed on the target
machine. TPU grade, actual fork shrinkage, rod retention, abrasion, and the
robot controller's changed contact geometry all require a physical prototype.
This first pass is intentionally separate from the robot URDF and full
quadruped assembly until hardware contact behavior is measured.

## Previous validation record

The following record applies only to the superseded 2026-08-08 generated
artifacts, not the pending direct-fusion revision:

- focused shoe tests: 6 passed; 17 shoe/upper-arm/leg regression tests passed;
  Ruff passed for the part, preview, and tests
- primary STEP re-import: one valid solid, 36 faces, 91 edges, and a nominal
  84 x 48 x 48 mm envelope
- fit preview: four labeled occurrences, 517 faces, 1,457 edges
- principal-axis shell thickness: 4.0 mm
- contact surface: outer surface-of-revolution extends from X=12.5 to 72.5 mm;
  inner surface extends from X=16.5 to 68.5 mm, with no planar nose face
- measured rod pitch: 9.899494 mm in both X and Y
- measured main hub width: 32.6 mm; source clearances are 0.4 mm per side
- installed shoe/leg intersection: 0.0 mm3 in the exact source-level check
- shoe occurrence frame translation: `[65.084989, 12.0, 0.0]` mm
- material volume: 44.60 cm3; fully dense 1.20 g/cm3 estimate: 53.5 g per shoe
- opposed isometric, top, and front snapshots reviewed for both artifacts

For the 2026-08-10 direct-fusion revision, the explicit STEP/STL/3MF generation
and visual handoff snapshots were run. No automated geometry inspection,
targeted measurements, collision checks, tests, linting, print validation, or
hardware validation were run.
