# CM5202 battery cradle

## Purpose

This printable open cradle secures the user's CM5202 3S LiPo inside the
quadruped body's existing battery rail.  It uses four existing M3 floor-grid
holes and two 20 mm hook-and-loop straps; it does not modify the body base.

## Source-of-truth inputs

- User-measured battery hard-case envelope on 2026-08-21: `135 x 45 x 33 mm`.
- The battery has additional wire-exit bulges on its left side.  Their exact
  size and position were not measured.
- Robot coordinates follow `specs/coordinate-system.md`: `+X` is robot front,
  `+Y` is robot left, and `+Z` is up.
- The existing body provides a nominal `170 x 70 mm` battery-rail opening and
  a 10 mm-pitch M3 floor grid.

The step.parts catalog was searched for `CM5202` and
`Hilldow 5200mAh 3S battery`; both searches returned zero matches.  Product
listings describe nominal envelopes near `139 x 47 x 39.5-40 mm`, but the
user's physical measurements control this design.

## Design

The editable geometry is
`drobot_cad/parts/cm5202_battery_cradle.py`; the build entry is
`drobot_cad/parts/cm5202_battery_cradle.step.py`.

- Battery pocket: `138 x 48 mm`, providing 1.5 mm nominal clearance on every
  measured side.
- Base: `143 x 53 x 3 mm`.
- Overall envelope including mounting tabs: `143 x 68 x 15 mm`.
- Robot-right side wall: 12 mm above the base.
- Robot-left wire side: only a 4 mm-high locating lip above the base, leaving
  the rest of that side open for unmeasured wire bulges.
- End stops: 8 mm above the base.
- Mounting holes: four 3.4 mm M3 clearances at `(X, Y) = (+/-60, +/-30) mm`.
- Retention: two pairs of `22 x 4.5 mm` slots for nominal 20 mm straps, centered
  at `X = +/-35 mm`.

The mounting tabs end at `Y = +/-34 mm`, leaving 1 mm nominal clearance to the
body's battery-rail opening on each side.  The M3 heads sit outside the battery
footprint.  The battery should still receive a thin nonconductive foam liner.
Because the four-hole pattern is symmetric, the cradle can also be rotated in
the body so the low lip faces whichever physical side carries the wires.

## Installation

1. Print the cradle flat with its base on the build plate.  PETG or ASA is
   preferred over low-temperature PLA for an enclosed robot body.
2. Thread both 20 mm straps through the slots before mounting the cradle.
3. Align the four cradle holes with body-floor grid points
   `(X, Y) = (+/-60, +/-30) mm`.
4. Fasten from inside with four M3 screws and retain them below the body with
   washers and locking nuts.  Confirm that no sharp thread or hardware edge can
   contact the battery.
5. Add approximately 1 mm closed-cell foam, install the battery, and tighten
   both straps only enough to prevent movement.
6. Route the left-side wires clear of hip joints, linkages, lid edges, and
   abrasion points.

## Reproduction

From `cad/`, generate the STEP, STL, and 3MF deliverables with:

```powershell
.\scripts\generate_cm5202_battery_cradle.ps1
```

Generated outputs:

- `exports/step/cm5202_battery_cradle.step`
- `exports/stl/cm5202_battery_cradle.stl`
- `exports/3mf/cm5202_battery_cradle.3mf`

## Validation status and limitations

Generated and reviewed on 2026-08-21:

- STEP generation completed from the `.step.py` entry.
- STL and 3MF secondary exports completed from the same parametric source.
- Baseline inspection reported one leaf occurrence with 52 faces, 148 edges,
  and `143 x 68 x 15 mm` bounds.
- Geometry validation passed: one closed, positive-volume solid with no
  reported invalid topology or self-intersection.
- Targeted measurements confirmed a `138 x 48 mm` battery pocket, `120 x 60
  mm` mounting-hole spacing, and `22 x 4.5 mm` strap slots.
- Opposed isometric, top, and front snapshots were reviewed.  The mounting
  tabs and holes were symmetric, all four strap slots were open, the pocket
  was unobstructed, and the low wire-side lip appeared as designed.
- The repository-owned local CAD Viewer was launched in its orthographic
  markup workspace for both the generator and STL.

No slicer check or physical print/fit check was performed.

The unmeasured wire bulges are assumed to sit above the 4 mm left locating lip.
Before printing the final part, check that assumption against the physical
battery.  Also verify strap fit, body-grid alignment, screw length, lid and
electronics clearance, and full leg motion with the robot unpowered.

This cradle is not a fire-resistant LiPo enclosure and does not make in-robot
charging safe.  Do not use a swollen or damaged pack, do not charge it in the
robot, and do not allow fasteners or printed edges to press into the case.
