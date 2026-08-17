# Rectangular PLA distal-fork shoe

## Purpose

This is a rigid, long-footprint replacement for the compliant TPU rocker. The
TPU design begins only `1.55 mm` beyond the forward fork-hole center and wraps
the attachment in a hollow curved body. That leaves little practical space for
washers, lock nuts, or a driver and provides a compliant point-like contact.

The rectangular design instead uses a `100 x 60 x 6 mm` PLA sole. Its entire
floor face is flat, and its long axis follows the robot's fore/aft direction.
Two-millimeter corner radii only soften the printable edges, leaving an
unmistakably rectangular footprint. Two upper-side longitudinal ribs connect
the plate into the central spine while preserving the fastener-access envelope.

The sole now begins at `X=24 mm`, raising the attachment axis `6 mm` farther
above the sole than the first rectangular version. A narrow `11.5 mm`-radius
central spine bridges that distance. The inherited outer circular fork cup was
removed: an installed boolean check found that it occupied `448.076642 mm^3`
of the real fork. The internal hub and four rods locate the shoe while the
bottom of the fork remains open and unobstructed.

## Attachment access

The sole's upper face begins at local `X=24 mm`; the closest forward fork-hole
center is at `X=4.949747 mm`. This provides `19.050253 mm` from the hole center
to the main sole. The sole ribs begin at `X=20 mm`, providing `15.050253 mm`
from the hole center.

The fit-preview source models a `9 mm`-diameter nut-driver envelope around all
four M3 axes, approaching the nuts from both ends of the specified `75 mm`
rods. The nearest reinforcement remains `10.550253 mm` beyond that envelope.
This is the design response to the earlier
TPU shoe being too close to the attachment holes.

These values are source dimensions and were confirmed in the generated fit
preview. They have not yet been confirmed by a physical installation.

## Material and printing intent

The structural part is intended for PLA or PLA+. Use all four existing M3
locations with washers and nylon lock nuts so the longer plate's bending load
is distributed across the fork. Tighten only enough to retain the part; PLA
can creep or crack under excessive point load.

The large contact face is intentionally unrecessed. Bond an approximately
`1.0 mm` rubber or TPU sheet to it, inset about `3 mm` from the perimeter. The
thin layer supplies traction without recreating the spring and rocker behavior
of the monolithic TPU shoe.

The natural print orientation places the large contact face on the build
plate, with the ribs, spine, and attachment hub growing upward. A first slicing
review should select wall count, infill, and any local support after the source
has been generated. No slicer settings are claimed as validated yet.

## Source and outputs

- Editable part: `drobot_cad/parts/rectangular_fork_shoe.py`
- Build entry: `drobot_cad/parts/rectangular_fork_shoe.step.py`
- Editable preview: `drobot_cad/assembly/rectangular_fork_shoe_fit_preview.py`
- Preview entry: `drobot_cad/assembly/rectangular_fork_shoe_fit_preview.step.py`
- Specification: `specs/rectangular-fork-shoe.yaml`
- STEP: `exports/step/rectangular_fork_shoe.step`
- STL: `exports/stl/rectangular_fork_shoe.stl`
- 3MF: `exports/3mf/rectangular_fork_shoe.3mf`
- Fit preview: `exports/step/rectangular_fork_shoe_fit_preview.step`

## Generation

From `cad/`:

```powershell
.\scripts\generate_rectangular_fork_shoe.ps1
```

That focused script uses the installed CAD skill's current `gen` and `export`
entry points. It generates only this shoe and its fit preview.

The focused script was run successfully on 2026-08-13. Its CLI paths use `/`
separators because the current generator requires POSIX-style source/output
pairs on Windows. Mesh outputs use absolute paths because `scripts/export`
otherwise resolves relative destinations beside the `.step.py` entry rather
than from the repository's `cad/` root.

## Controller impact

The bare PLA contact face is `30 mm` beyond the fork axis. A `1.0 mm` bonded
tread makes the nominal contact extension approximately `31 mm`, compared with
the current rigid circular-shoe controller assumption of `20.7 mm`. Update the
gait geometry only after the printed shoe and compressed tread thickness are
measured.

## Validation status

The source was revised and regenerated on 2026-08-13 with the `100 x 60 mm`
sole and the attachment axis raised by `6 mm`. The part inspection reports one
valid, closed, positive-volume solid. The 14-piece installed preview also
reports valid closed solids. Focused boolean checks measured `0 mm^3` overlap
for shoe-to-fork, shoe-to-driver, and fork-to-driver pairs.

Local assembly and part snapshots are stored in
`reviews/rectangular-fork-shoe/`. No automated test suite, lint check, slicing,
motion simulation, hardware test, or physical fit check was run. A first PLA
print should still be treated as a fit prototype.
