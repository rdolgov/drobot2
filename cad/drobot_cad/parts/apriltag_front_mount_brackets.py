"""Front-wall standoff rails for the two-color AprilTag body marker.

The two rails use the quadruped body's 10 mm-pitch front-wall M3 grid and the
four existing 4 x 16 mm slots in the AprilTag plate.  They keep all hardware
outside the 80 x 80 mm marker image.

Coordinate convention:
- XY registers with the marker plane.
- Z = 0 is the flat robot-wall and print-bed datum.
- +Z points out through the visible marker face.
- The marker rear face seats on the bracket front at Z = 8 mm.

For the intended front-wall installation, marker axes map to the robot as:
- marker +Z = robot +X (forward)
- marker +Y = robot +Z (up)
- marker +X = robot +Y (left)
"""

from __future__ import annotations

from math import sqrt

from build123d import (
    Align,
    Box,
    BuildSketch,
    Color,
    Compound,
    Cylinder,
    Pos,
    RegularPolygon,
    Shape,
    extrude,
)

from drobot_cad.parts.apriltag_body_marker import (
    ZIP_SLOT_X_MM,
    ZIP_SLOT_Y_MM,
)

# Each solid rail is a printable standoff.  Its flat robot-facing side goes on
# the print bed; all counterbores and nut pockets then open upward.
BRACKET_WIDTH_MM = 20.0
BRACKET_HEIGHT_MM = 72.0
BRACKET_STANDOFF_MM = 8.0
BRACKET_CENTER_X_MM = 48.0

M3_CLEARANCE_DIAMETER_MM = 3.4
BODY_MOUNT_X_MM = 50.0
BODY_MOUNT_Y_MM = 30.0
BODY_SCREW_HEAD_COUNTERBORE_DIAMETER_MM = 6.2
BODY_SCREW_HEAD_COUNTERBORE_DEPTH_MM = 3.2

# A regular hexagonal prism uses its circumradius.  This gives a 5.8 mm
# across-flats pocket for a nominal 5.5 mm M3 nut.
TAG_NUT_POCKET_ACROSS_FLATS_MM = 5.8
TAG_NUT_POCKET_RADIUS_MM = TAG_NUT_POCKET_ACROSS_FLATS_MM / sqrt(3.0)
TAG_NUT_POCKET_DEPTH_MM = 3.0
BOOLEAN_OVERTRAVEL_MM = 0.5


def _through_hole(x_mm: float, y_mm: float) -> Shape:
    return Pos(x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM) * Cylinder(
        M3_CLEARANCE_DIAMETER_MM / 2.0,
        BRACKET_STANDOFF_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )


def _front_recess(
    x_mm: float,
    y_mm: float,
    *,
    radius_mm: float,
    depth_mm: float,
    sides: int | None = None,
) -> Shape:
    if sides is None:
        recess = Cylinder(
            radius_mm,
            depth_mm + BOOLEAN_OVERTRAVEL_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    else:
        with BuildSketch() as recess_profile:
            RegularPolygon(radius_mm, sides)
        recess = extrude(
            recess_profile.sketch,
            amount=depth_mm + BOOLEAN_OVERTRAVEL_MM,
        )
    return Pos(x_mm, y_mm, BRACKET_STANDOFF_MM - depth_mm) * recess


def make_front_mount_bracket(side: int) -> Shape:
    """Return one left/right rail in its assembled marker coordinate frame."""
    if side not in (-1, 1):
        raise ValueError("side must be -1 or 1")

    rail_center_x = side * BRACKET_CENTER_X_MM
    rail = Pos(
        rail_center_x,
        0.0,
        0.0,
    ) * Box(
        BRACKET_WIDTH_MM,
        BRACKET_HEIGHT_MM,
        BRACKET_STANDOFF_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )

    body_x = side * BODY_MOUNT_X_MM
    tag_x = side * ZIP_SLOT_X_MM
    cutters: list[Shape] = []

    for y_mm in (-BODY_MOUNT_Y_MM, BODY_MOUNT_Y_MM):
        cutters.append(_through_hole(body_x, y_mm))
        cutters.append(
            _front_recess(
                body_x,
                y_mm,
                radius_mm=BODY_SCREW_HEAD_COUNTERBORE_DIAMETER_MM / 2.0,
                depth_mm=BODY_SCREW_HEAD_COUNTERBORE_DEPTH_MM,
            )
        )

    for y_mm in (-ZIP_SLOT_Y_MM, ZIP_SLOT_Y_MM):
        cutters.append(_through_hole(tag_x, y_mm))
        cutters.append(
            _front_recess(
                tag_x,
                y_mm,
                radius_mm=TAG_NUT_POCKET_RADIUS_MM,
                depth_mm=TAG_NUT_POCKET_DEPTH_MM,
                sides=6,
            )
        )

    bracket = rail - cutters
    side_name = "left" if side > 0 else "right"
    bracket.label = f"apriltag_front_mount_bracket_{side_name}"
    bracket.color = Color(0.82, 0.82, 0.82)
    return bracket


def make_apriltag_front_mount_brackets() -> Compound:
    """Return the aligned left and right front-wall bracket rails."""
    brackets = Compound(
        children=[make_front_mount_bracket(-1), make_front_mount_bracket(1)]
    )
    brackets.label = "apriltag_front_mount_brackets"
    return brackets
