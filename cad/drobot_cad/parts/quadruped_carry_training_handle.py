"""Rigid carry and resistance-band handle for the quadruped body.

The one-piece U handle bolts to the upper two rows of the three-column M3
mounting grid on both long side walls. Its plates remain above the side wire
ports, and two crossbar slots accept looped resistance bands while leaving a
clear central hand grip.

Coordinate convention:
    - origin and axes match the quadruped body base
    - +X: robot front
    - +Y: robot left
    - +Z: up
"""

from __future__ import annotations

from build123d import Align, Axis, Box, Cylinder, Location, Shape, fillet

from drobot_cad.parts import quadruped_body

# Side plates seat directly against the body's exterior Y faces. The selected
# rows are the two highest complete rows in the central three-column field.
# The neighboring 76 mm hip plates terminate at X=-22 and X=+22 mm. A
# 36 mm handle plate spans only X=-18..+18, preserving 4 mm per-side clearance.
MOUNT_PLATE_LENGTH_X_MM = 36.0
MOUNT_PLATE_THICKNESS_Y_MM = 6.0
MOUNT_PLATE_HEIGHT_Z_MM = 28.0
MOUNT_PLATE_CENTER_Z_MM = 80.0
MOUNT_HOLE_X_MM = (-10.0, 0.0, 10.0)
MOUNT_HOLE_Z_MM = (70.0, 80.0)
M3_CLEARANCE_DIAMETER_MM = 3.4

BODY_OUTER_SIDE_Y_MM = quadruped_body.BODY_WIDTH_Y_MM / 2.0
MOUNT_PLATE_CENTER_Y_MM = (
    BODY_OUTER_SIDE_Y_MM + MOUNT_PLATE_THICKNESS_Y_MM / 2.0
)

# The inner crossbar surface is 40 mm above the 100 mm installed body lid.
POST_LENGTH_X_MM = 22.0
POST_WIDTH_Y_MM = 12.0
POST_BOTTOM_Z_MM = 80.0
CROSSBAR_CENTER_Z_MM = 150.0
POST_HEIGHT_Z_MM = CROSSBAR_CENTER_Z_MM - POST_BOTTOM_Z_MM
POST_CENTER_Y_MM = BODY_OUTER_SIDE_Y_MM + POST_WIDTH_Y_MM / 2.0

CROSSBAR_LENGTH_Y_MM = 194.0
CROSSBAR_DEPTH_X_MM = 22.0
CROSSBAR_HEIGHT_Z_MM = 20.0
ELBOW_RADIUS_MM = 11.0

# Two band slots preserve a 100 mm clear central hand-grip region.
BAND_SLOT_CENTER_Y_MM = (-62.0, 62.0)
BAND_SLOT_WIDTH_Y_MM = 24.0
BAND_SLOT_HEIGHT_Z_MM = 9.0
BAND_SLOT_CUTTER_DEPTH_X_MM = CROSSBAR_DEPTH_X_MM + 2.0

BOOLEAN_OVERTRAVEL_MM = 1.0
MOUNT_PLATE_FILLET_MM = 2.0
POST_FILLET_MM = 2.5
CROSSBAR_FILLET_MM = 4.0
BAND_SLOT_FILLET_MM = 3.0

RECOMMENDED_FASTENER = (
    "12 x M3 bolts with wide washers and nyloc nuts; select length after "
    "checking the printed plate and 3.2 mm body wall"
)


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def _rounded_box(
    length_x_mm: float,
    width_y_mm: float,
    height_z_mm: float,
    radius_mm: float,
) -> Shape:
    blank = Box(
        length_x_mm,
        width_y_mm,
        height_z_mm,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    )
    return fillet(blank.edges(), radius_mm)


def _make_mount_plates() -> tuple[Shape, ...]:
    return tuple(
        _rounded_box(
            MOUNT_PLATE_LENGTH_X_MM,
            MOUNT_PLATE_THICKNESS_Y_MM,
            MOUNT_PLATE_HEIGHT_Z_MM,
            MOUNT_PLATE_FILLET_MM,
        ).moved(
            Location(
                (
                    0.0,
                    side * MOUNT_PLATE_CENTER_Y_MM,
                    MOUNT_PLATE_CENTER_Z_MM,
                )
            )
        )
        for side in (-1.0, 1.0)
    )


def _make_vertical_posts() -> tuple[Shape, ...]:
    post_center_z = POST_BOTTOM_Z_MM + POST_HEIGHT_Z_MM / 2.0
    return tuple(
        _rounded_box(
            POST_LENGTH_X_MM,
            POST_WIDTH_Y_MM,
            POST_HEIGHT_Z_MM,
            POST_FILLET_MM,
        ).moved(
            Location((0.0, side * POST_CENTER_Y_MM, post_center_z))
        )
        for side in (-1.0, 1.0)
    )


def _make_crossbar() -> Shape:
    return _rounded_box(
        CROSSBAR_DEPTH_X_MM,
        CROSSBAR_LENGTH_Y_MM,
        CROSSBAR_HEIGHT_Z_MM,
        CROSSBAR_FILLET_MM,
    ).moved(Location((0.0, 0.0, CROSSBAR_CENTER_Z_MM)))


def _make_elbows() -> tuple[Shape, ...]:
    """Return round gussets that thicken both post-to-crossbar junctions."""
    return tuple(
        Cylinder(
            ELBOW_RADIUS_MM,
            CROSSBAR_DEPTH_X_MM,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        .rotate(Axis.Y, 90.0)
        .moved(
            Location(
                (
                    0.0,
                    side * POST_CENTER_Y_MM,
                    CROSSBAR_CENTER_Z_MM - CROSSBAR_HEIGHT_Z_MM / 2.0,
                )
            )
        )
        for side in (-1.0, 1.0)
    )


def _make_mount_hole_tools() -> tuple[Shape, ...]:
    cutter_length = MOUNT_PLATE_THICKNESS_Y_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_length,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        .rotate(Axis.X, 90.0)
        .moved(
            Location(
                (
                    x_mm,
                    side * MOUNT_PLATE_CENTER_Y_MM,
                    z_mm,
                )
            )
        )
        for side in (-1.0, 1.0)
        for x_mm in MOUNT_HOLE_X_MM
        for z_mm in MOUNT_HOLE_Z_MM
    )


def _make_band_slot_tools() -> tuple[Shape, ...]:
    return tuple(
        _rounded_box(
            BAND_SLOT_CUTTER_DEPTH_X_MM,
            BAND_SLOT_WIDTH_Y_MM,
            BAND_SLOT_HEIGHT_Z_MM,
            BAND_SLOT_FILLET_MM,
        ).moved(Location((0.0, y_mm, CROSSBAR_CENTER_Z_MM)))
        for y_mm in BAND_SLOT_CENTER_Y_MM
    )


def make_handle() -> Shape:
    """Return the installed-orientation, one-piece printable handle."""
    with_plates = _make_crossbar().fuse(*_make_mount_plates())
    with_posts = with_plates.fuse(*_make_vertical_posts())
    reinforced = with_posts.fuse(*_make_elbows())
    drilled = reinforced - _make_mount_hole_tools()
    finished = drilled - _make_band_slot_tools()
    return _one_valid_solid(finished, "quadruped_carry_training_handle")


def gen_step() -> Shape:
    return make_handle()
