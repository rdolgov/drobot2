# CM5202 screw-lid battery box

## Purpose

This design replaces the open strap cradle with a larger two-piece enclosure:
a body-grid-mounted main box and a separate screw-on lid.  The battery inserts
from above.  The short left end uses two large cable openings separated by a
narrow center support rib.

## Inputs and orientation

- User-measured battery envelope: `135 x 45 x 33 mm`.
- Additional unmeasured wire bulges sit at the battery's left end.
- Top-view assumption: the wire location is the short `-X` end.
- The quadruped body provides a nominal `170 x 70 mm` battery-rail opening and
  a 10 mm-pitch M3 floor grid.

If the physical wire position differs, adjust `WIRE_PORT_CENTER_Y_MM` or rotate
the battery/box in the parametric source.

## Main box

- Internal cavity: `139 x 49 x 37 mm`.
- Nominal battery clearance: 2 mm at each end, 2 mm at each side, and 4 mm
  above the measured hard case.
- Floor: 3 mm.
- Walls: 2.5 mm.
- Box envelope before external tabs/bosses: `144 x 54 x 40 mm`.
- Overall width including body-mount tabs and lid bosses: 68 mm.
- Body mounting: four 3.4 mm M3 holes at `(X, Y) = (+/-60, +/-30) mm`.
- Lid retention: four 2.8 mm blind self-tapping M3 pilots in full-height
  printed towers at `(X, Y) = (+/-35, +/-29.5) mm`.
- Wire opening: two 21 mm-wide openings extending 32 mm down from the top of
  the short `-X` end wall.
- End-wall support: a 6 mm full-height center rib, approximately 3 mm corner
  returns, and an 8 mm lower wall/floor section.
- Edge treatment: 1.5 mm main-envelope fillets, 1 mm cavity and wire-opening
  radii, and 0.8 mm mounting-tab radii. Screw bores, pilot holes, and boss
  mating circles remain dimensionally sharp.

## Lid

- Solid 3 mm plate matching the box footprint.
- Four 3.4 mm M3 clearance holes aligned to the printed screw towers.
- Two 21 x 18 mm edge reliefs aligned to the short-end openings, leaving a
  matching 6 mm center tongue.
- Edge treatment: 0.8 mm plate/ear fillets and 1 mm split-relief radii.
- No strap slots or other openings.

Use four short M3 screws suitable for printed plastic.  Do not overtighten.
Heat-set inserts are not modeled in this revision; the 2.8 mm pilots are for a
first physical self-tapping trial.

## Editable sources and outputs

- Shared geometry: `drobot_cad/parts/cm5202_battery_box.py`
- Main-box entry: `drobot_cad/parts/cm5202_battery_box.step.py`
- Lid entry: `drobot_cad/parts/cm5202_battery_box_lid.step.py`
- Exploded fit preview:
  `drobot_cad/assembly/cm5202_battery_box_fit_preview.step.py`
- Generation script: `scripts/generate_cm5202_battery_box.ps1`

Generated outputs:

- `exports/step/cm5202_battery_box.step`
- `exports/step/cm5202_battery_box_lid.step`
- `exports/step/cm5202_battery_box_fit_preview.step`
- `exports/stl/cm5202_battery_box.stl`
- `exports/stl/cm5202_battery_box_lid.stl`
- `exports/3mf/cm5202_battery_box.3mf`
- `exports/3mf/cm5202_battery_box_lid.3mf`

## Generation status

The split-open-end wire revision was regenerated on 2026-08-21.
STEP, STL, 3MF, and Viewer GLB export completed successfully.  Automated
geometry inspection, new review snapshots, slicing, and physical battery/robot
fit were not run for this revision.

## Safety and limitations

The wire-bulge size and precise location remain unmeasured.  The split
openings are intentionally an assumption for review and may need adjustment.
The design is not a fire-resistant LiPo enclosure and provides no swelling,
thermal, impact, or flame certification.  Do not charge the LiPo in the robot,
and do not use a swollen or damaged pack.
