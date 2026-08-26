# Two-color AprilTag body marker

This printable marker provides external body-pose ground truth for synchronized
robot-walk video and internal RL recordings. It uses **tag36h11 ID 0** from the
[official AprilRobotics image family](https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag36h11).
The marker code and bit positions are derived from AprilRobotics' BSD-2-Clause
[`tag36h11.c`](https://github.com/AprilRobotics/apriltag/blob/master/tag36h11.c).

## Geometry

| Quantity | Value |
| --- | ---: |
| Plate envelope | 104 x 80 x 2.24 mm |
| White structural plate | 2.00 mm |
| Raised black layer | 0.24 mm |
| Marker image | 10 x 10 modules, 80 x 80 mm |
| Module size | 8.00 mm |
| Detector tag size | **64.00 mm / 0.064 m** |
| Mounting | Four 4 x 16 mm slots; optional front-wall bracket pair |
| Orientation mark | 8 mm cut-through arrow in the white side margin |

The `0.064 m` detector size is the distance between the outside edges of the
black border, not the full white plate or the 80 mm image including its quiet
border. Do not scale the model in the slicer unless the configured detector
size is scaled by the same factor.

For the recommended face-up installation, mount the plate rigidly to the main
chassis with the cut-through arrow pointing toward robot front. The arrow is a
human-readable mounting reference only; AprilTag detection uses the encoded
black-and-white pattern. Do not attach the marker to a flexible wire bundle,
moving leg, or battery lid that can move relative to the chassis.

## Front-facing bracket mount

`apriltag_front_mount_brackets` is a pair of 8 mm-thick printed standoff rails
for mounting the marker vertically on the robot's forward-facing body wall.
The rails use four existing holes in the body's 10 mm-pitch M3 wall grid; no
new robot-body holes are required. The tag then fastens through its existing
side slots, so no screw, bracket, or recess overlaps the 80 x 80 mm marker
image.

Mount the tag centered at **50 mm above the body bottom**. The body-wall screw
centers are then `(robot Y, robot Z) = (+/-50, 20)` and `(+/-50, 80)` mm. The
assembled marker spans Z = 10 through 90 mm and its visible face sits 10.24 mm
forward of the wall, including the white and black printed layers.

Recommended hardware:

- Four M3 x 16 mm screws, washers, and nuts to fasten the rails to the front
  body wall. The screw heads sit in the round counterbores on the rail fronts.
- Four standard M3 hex nuts in the hexagonal front pockets.
- Four M3 x 8 mm screws through the marker slots and into those captive nuts.

Assembly order:

1. Print both rails flat with their uninterrupted robot-facing sides on the
   build plate and the counterbores/nut pockets facing upward. PLA, PETG, or
   ASA is suitable; use at least three walls and 25% infill.
2. Bolt the two rails to the robot front wall at the four grid locations above.
3. Press one M3 nut into each hexagonal pocket.
4. Place the marker against the rail fronts and install the four M3 x 8 mm
   screws through its vertical slots. Leave them loose until the tag is level,
   then tighten gently.
5. When looking at the robot from the front, keep the orientation arrow pointing
   **up**. This gives the fixed transform `marker +Z = robot +X`,
   `marker +Y = robot +Z`, and `marker +X = robot +Y`.

The brackets assume the current 220 x 170 x 100 mm body with its documented
front-wall mounting grid. If the physical body is an older revision without
those holes, use the bracket as a drilling template only after checking wall
clearance and internal wiring. The plate partially covers the front ventilation
field, so keep the 8 mm standoff gap unobstructed and monitor electronics
temperature during the first powered trial.

## Multi-material printing

Preferred workflow:

1. Open `apriltag_body_marker.3mf` directly in Bambu Studio. It is a complete
   project containing one object with the aligned `white_plate` and
   `black_tag36h11_id_0` parts.
2. If Bambu Studio says the included stock P2S profile differs from the
   connected printer, select **Switch now** to use the connected printer's
   profile. This is an expected printer-profile prompt, not a damaged-file
   warning.
3. Confirm filament 1 is matte white and filament 2 is matte black. The part
   assignments are already stored in the project.
4. Confirm the object remains 104 x 80 x 2.24 mm and centered on the plate.
5. Print the marker face upward at 0.20 or 0.24 mm layer height without support.
6. Disable automatic scaling and avoid glossy, translucent, silk, or reflective
   filament.

The packaged project uses only stock Bambu printer, process, and PLA presets;
it does not contain custom machine G-code. It was verified by opening it in
Bambu Studio 2.8.2 with two active white/black filaments and both parts in the
correct shared position.

If the slicer does not preserve 3MF part labels, import the aligned white and
black STL files together **as parts of one object**. Keep their original
coordinates and do not auto-arrange them separately.

The front-wall brackets are a separate single-color print. Import
`apriltag_front_mount_brackets.stl` or
`apriltag_front_mount_brackets.3mf`; both rails are already aligned as a pair.
The checked-in 3MF is packaged as a Bambu-native, one-filament project rather
than the generic assembly 3MF produced by the CAD exporter. Regenerate it with:

```powershell
uv run python cad/scripts/generate_apriltag_front_mount_brackets_3mf.py
```

To regenerate the Bambu-compatible file after changing either aligned STL:

```powershell
uv run python cad/scripts/generate_apriltag_body_marker_3mf.py
```

The generator flattens the marker's disconnected black regions into one mesh,
keeps the white and black meshes as two direct parts of one object, assigns
them to filament 1 and 2, and packages them using the validated stock project
template in `cad/templates/apriltag_body_marker_bambu_source.3mf`.

## Recording setup

- Keep the complete marker visible with margin around it throughout the walk.
- Prefer a fixed camera, short exposure, bright diffuse lighting, and at least
  1080p/60 fps.
- Place the camera high enough to see the marker but not so obliquely that the
  plate becomes a thin line.
- Include a stationary tag or measured floor references if world-frame metric
  accuracy matters.
- Record the internal RL trial at the same time; the tag supplies external pose,
  while the internal file supplies IMU, joint, policy, and power data.

AprilTag tracking improves pose and speed measurement but does not replace the
internal recording or show foot forces directly.
