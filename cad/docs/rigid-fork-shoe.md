# Rigid low-profile distal-fork shoe

## Design

This design supersedes the compliant oval-rocker concept for normal indoor
walking. It uses a rigid 48 mm circular sole only 6 mm thick, positioned close
to the existing fork nose so it adds approximately 6.6 mm of structural reach.
A recommended 1.5 mm traction disk sits in a 0.8 mm recess, bringing the total
projection beyond the original fork nose to roughly 7.3 mm.

The attachment is a close-fitting saddle rather than a narrow neck. A central
solid hub fills the gap between the fork cheeks with 0.4 mm side clearance. A
5.5 mm-deep annular rim cups the rounded fork nose with an assumed 0.5 mm
radial clearance. Together they wrap around the fork tip and create a broad,
rigid load path into the sole. Two diagonal M3 x 75 mm threaded rods, washers,
and nylon lock nuts reuse the existing holes; all four bores remain available.

The structural part should be printed in PETG, ABS/ASA, nylon, or a suitably
tough fiber-filled material—not flexible TPU. For hardwood traction, bond a
44 mm diameter, approximately 1.5 mm thick rubber or TPU sheet disk into the
front recess. This follows a real-shoe architecture: rigid support with a thin
high-friction outsole. Hard printed plastic should not be the floor-contact
material because it is more likely to slide, click, and scratch the finish.

## Source and outputs

- Source: `drobot_cad/parts/rigid_fork_shoe.py`
- Fit preview: `drobot_cad/assembly/rigid_fork_shoe_fit_preview.py`
- Primary target: `exports/step/rigid_fork_shoe.step`
- Printable target: `exports/stl/rigid_fork_shoe.stl`
- Mesh container: `exports/3mf/rigid_fork_shoe.3mf`
- Assembly target: `exports/step/rigid_fork_shoe_fit_preview.step`

## Generation

From `cad/`, use the installed CAD skill's explicit STEP workflow:

```powershell
$cadSkill = "$env:USERPROFILE\.codex\plugins\cache\text-to-cad\cad\0.3.13\skills\cad"
$cadpy = Join-Path $cadSkill "scripts\packages\cadpy\src"
$env:PYTHONPATH = ((Resolve-Path ".").Path, $cadpy) -join [IO.Path]::PathSeparator

.\.venv\Scripts\python.exe "$cadSkill\scripts\step" `
  drobot_cad/parts/rigid_fork_shoe.py=exports/step/rigid_fork_shoe.step `
  --stl ../stl/rigid_fork_shoe.stl `
  --3mf ../3mf/rigid_fork_shoe.3mf `
  --force

.\.venv\Scripts\python.exe "$cadSkill\scripts\step" `
  drobot_cad/assembly/rigid_fork_shoe_fit_preview.py=exports/step/rigid_fork_shoe_fit_preview.step `
  --force
```

## Validation status

The STEP, STL, 3MF, and installed fit-preview STEP were generated on
2026-08-11. Opposed isometric, top, and front snapshots were visually reviewed.
The views show the recessed flat sole, central hub, and annular rear saddle
nesting around the fork nose without the earlier narrow-neck architecture.

No automated geometry inspection, dimensional or collision measurement,
tests, linting, slicing, or print validation were run. The nominal 0.5 mm
saddle clearance is therefore unverified and the part should be treated as a
review prototype rather than a confirmed fit.

The existing M3 x 75 mm rod provenance remains the step.parts record
`iso_metric_threaded_rod_m3_l075_simple`; no new catalog component was added.
