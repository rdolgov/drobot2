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
| Mounting | Four 4 x 16 mm zip-tie slots |

The `0.064 m` detector size is the distance between the outside edges of the
black border, not the full white plate or the 80 mm image including its quiet
border. Do not scale the model in the slicer unless the configured detector
size is scaled by the same factor.

The small round hole near one end marks robot front. Mount the plate rigidly to
the main chassis with that hole toward the robot's forward direction. Do not
attach it to a flexible wire bundle, moving leg, or battery lid that can move
relative to the chassis.

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
