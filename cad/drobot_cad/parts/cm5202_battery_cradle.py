"""Printable CM5202 LiPo cradle for the quadruped body floor grid.

Coordinate convention:
    - origin: center of the cradle footprint at its bottom surface
    - +X: robot front / battery length
    - +Y: robot left / battery wire-bulge side
    - +Z: up

The cradle is intentionally open.  Two hook-and-loop straps provide vertical
retention, while the low +Y lip leaves clearance for the battery's unmeasured
wire-exit bulges.
"""

from __future__ import annotations

from build123d import Align, Box, Cylinder, Location, Shape


# User-measured hard-case envelope.  Wire bulges are excluded from these
# dimensions and are accommodated by the low, open +Y retaining lip.
BATTERY_LENGTH_X_MM = 135.0
BATTERY_WIDTH_Y_MM = 45.0
BATTERY_HEIGHT_Z_MM = 33.0
FIT_CLEARANCE_PER_SIDE_MM = 1.5

BASE_THICKNESS_Z_MM = 3.0
WALL_THICKNESS_MM = 2.5
RIGHT_WALL_RISE_Z_MM = 12.0
LEFT_WIRE_LIP_RISE_Z_MM = 4.0
END_WALL_RISE_Z_MM = 8.0
FUSION_OVERLAP_MM = 0.5
BOOLEAN_OVERTRAVEL_MM = 1.0

POCKET_LENGTH_X_MM = BATTERY_LENGTH_X_MM + 2.0 * FIT_CLEARANCE_PER_SIDE_MM
POCKET_WIDTH_Y_MM = BATTERY_WIDTH_Y_MM + 2.0 * FIT_CLEARANCE_PER_SIDE_MM
BASE_LENGTH_X_MM = POCKET_LENGTH_X_MM + 2.0 * WALL_THICKNESS_MM
BASE_WIDTH_Y_MM = POCKET_WIDTH_Y_MM + 2.0 * WALL_THICKNESS_MM

# These four locations are members of the quadruped body's 10 mm M3 floor
# grid and remain inside its 170 x 70 mm battery-rail opening.
MOUNT_HOLE_CENTERS_XY_MM = (
    (-60.0, -30.0),
    (-60.0, 30.0),
    (60.0, -30.0),
    (60.0, 30.0),
)
MOUNT_TAB_LENGTH_X_MM = 14.0
MOUNT_TAB_WIDTH_Y_MM = 9.0
MOUNT_TAB_CENTER_Y_MM = 29.5
M3_CLEARANCE_DIAMETER_MM = 3.4

# Each pair of slots accepts one nominal 20 mm hook-and-loop strap.  Straps
# wrap across Y and over the battery; thread them before bolting down the tray.
STRAP_WIDTH_X_MM = 22.0
STRAP_SLOT_WIDTH_Y_MM = 4.5
STRAP_STATION_X_MM = (-35.0, 35.0)
STRAP_SLOT_CENTER_Y_MM = (-18.0, 18.0)


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def make_base() -> Shape:
    """Return the flat battery-support plate."""
    return Box(
        BASE_LENGTH_X_MM,
        BASE_WIDTH_Y_MM,
        BASE_THICKNESS_Z_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )


def make_mount_tabs() -> tuple[Shape, ...]:
    """Return four tabs that reach the selected body-floor grid holes."""
    return tuple(
        Box(
            MOUNT_TAB_LENGTH_X_MM,
            MOUNT_TAB_WIDTH_Y_MM,
            BASE_THICKNESS_Z_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    side * MOUNT_TAB_CENTER_Y_MM,
                    0.0,
                )
            )
        )
        for x_mm in (-60.0, 60.0)
        for side in (-1.0, 1.0)
    )


def _wall_box(
    length_x_mm: float,
    width_y_mm: float,
    rise_z_mm: float,
    center_x_mm: float,
    center_y_mm: float,
) -> Shape:
    """Return a wall overlapping the base for a dependable fused joint."""
    return Box(
        length_x_mm,
        width_y_mm,
        rise_z_mm + FUSION_OVERLAP_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).moved(
        Location(
            (
                center_x_mm,
                center_y_mm,
                BASE_THICKNESS_Z_MM - FUSION_OVERLAP_MM,
            )
        )
    )


def make_retaining_walls() -> tuple[Shape, ...]:
    """Return asymmetric side restraints and two longitudinal end stops."""
    side_center_y = POCKET_WIDTH_Y_MM / 2.0 + WALL_THICKNESS_MM / 2.0
    end_center_x = POCKET_LENGTH_X_MM / 2.0 + WALL_THICKNESS_MM / 2.0
    right_wall = _wall_box(
        POCKET_LENGTH_X_MM,
        WALL_THICKNESS_MM,
        RIGHT_WALL_RISE_Z_MM,
        0.0,
        -side_center_y,
    )
    left_wire_lip = _wall_box(
        POCKET_LENGTH_X_MM,
        WALL_THICKNESS_MM,
        LEFT_WIRE_LIP_RISE_Z_MM,
        0.0,
        side_center_y,
    )
    end_walls = tuple(
        _wall_box(
            WALL_THICKNESS_MM,
            BASE_WIDTH_Y_MM,
            END_WALL_RISE_Z_MM,
            side * end_center_x,
            0.0,
        )
        for side in (-1.0, 1.0)
    )
    return (right_wall, left_wire_lip) + end_walls


def make_mount_hole_tools() -> tuple[Shape, ...]:
    """Return four through-cutters for M3 body-floor fasteners."""
    cutter_height = BASE_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM)))
        for x_mm, y_mm in MOUNT_HOLE_CENTERS_XY_MM
    )


def make_strap_slot_tools() -> tuple[Shape, ...]:
    """Return two pairs of through-slots for 20 mm battery straps."""
    cutter_height = BASE_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Box(
            STRAP_WIDTH_X_MM,
            STRAP_SLOT_WIDTH_Y_MM,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    -BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for x_mm in STRAP_STATION_X_MM
        for y_mm in STRAP_SLOT_CENTER_Y_MM
    )


def make_cm5202_battery_cradle() -> Shape:
    """Return the complete open, screw-mounted CM5202 cradle."""
    blank = make_base().fuse(*make_mount_tabs(), *make_retaining_walls())
    finished = blank - (make_mount_hole_tools() + make_strap_slot_tools())
    return _one_valid_solid(finished, "cm5202_battery_cradle")
