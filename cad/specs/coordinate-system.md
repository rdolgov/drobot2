# Coordinate-system contract

All linear dimensions are millimeters and all angles are degrees unless a file
explicitly states otherwise.

## Imported SO-101 upper arm

The upper-arm generator intentionally preserves the immutable reference STEP
coordinates:

- The original perforated mounting panel is at the negative-X end.
- The original reference bounds are
  `[-65.084989, 0.0, -35.6]` to `[78.516747, 24.5, 31.7]`.
- The imported-body center is near `[6.715879, 12.25, -1.95]`.
- Subtractive edits are evaluated in these original coordinates.
- Optional final rotations are applied about `ROTATION_PIVOT_MM` in X, Y, Z
  order, followed by translation.

Do not recenter or reorient the imported reference as cleanup. Existing
placement parameters, source markups, and fit previews rely on this frame.

## ST3215 motor bay

- The flat attachment datum is the YZ plane at X=0.
- The bay extends from X=0 toward negative X.
- The servo inserts from the open negative-X end.
- Servo local X follows global X.
- Servo local Z follows global negative Y.
- Servo local Y follows global positive Z.
- The installed catalog servo is rotated +90 degrees about X.

In upper-arm coordinates, the bay attachment datum is at
`[-58.2, 12.0, -1.95]` after the configured 30 mm negative-X extension.

## Named interfaces

Named frames live in `drobot_cad/interfaces.py` and the YAML specifications.
Never create persistent fit intent from a transient face or edge index. CAD
Viewer selectors may change after regeneration even when the solid set is
geometrically equivalent.
