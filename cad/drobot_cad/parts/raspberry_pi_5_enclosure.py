"""Screw-mounted Raspberry Pi 5 enclosure for the quadruped body floor.

The printable system has three pieces: a ventilated base, a screw-on lid, and
an open-sided BNO085 protection roof.  The base attaches independently to four
members of the robot body's 10 mm-pitch M3 floor grid.

Coordinate convention:
    - origin: center of the enclosure floor footprint at its bottom surface
    - +X: robot front / Pi length
    - +Y: robot left / Pi width
    - +Z: up
"""

from __future__ import annotations

from build123d import Align, Box, Cylinder, Location, Shape, fillet

from drobot_cad.parts import (
    adafruit_bno085,
    quadruped_imu_cover,
    raspberry_pi_5,
)

FIT_CAVITY_LENGTH_X_MM = 96.0
FIT_CAVITY_WIDTH_Y_MM = 66.0
FIT_CAVITY_HEIGHT_Z_MM = 31.0
FLOOR_THICKNESS_Z_MM = 3.0
WALL_THICKNESS_MM = 2.5
BASE_OUTER_LENGTH_X_MM = FIT_CAVITY_LENGTH_X_MM + 2.0 * WALL_THICKNESS_MM
BASE_OUTER_WIDTH_Y_MM = FIT_CAVITY_WIDTH_Y_MM + 2.0 * WALL_THICKNESS_MM
BASE_TOTAL_HEIGHT_Z_MM = FLOOR_THICKNESS_Z_MM + FIT_CAVITY_HEIGHT_Z_MM

PI_STANDOFF_HEIGHT_MM = 6.0
PI_STANDOFF_OUTER_DIAMETER_MM = 7.0
PI_M2_5_PILOT_DIAMETER_MM = 2.2
PI_M2_5_PILOT_DEPTH_MM = 5.0
PI_PCB_BOTTOM_Z_MM = FLOOR_THICKNESS_Z_MM + PI_STANDOFF_HEIGHT_MM

# Each coordinate is a valid point on the body's 10 mm-pitch M3 floor grid.
BODY_MOUNT_HOLE_CENTERS_XY_MM = (
    (-40.0, -50.0),
    (-40.0, 50.0),
    (40.0, -50.0),
    (40.0, 50.0),
)
BODY_MOUNT_TAB_LENGTH_X_MM = 18.0
BODY_MOUNT_TAB_WIDTH_Y_MM = 30.0
BODY_MOUNT_TAB_CENTER_Y_MM = 43.0
BODY_M3_CLEARANCE_DIAMETER_MM = 3.4

LID_THICKNESS_Z_MM = 3.0
LID_M3_CLEARANCE_DIAMETER_MM = 3.4
LID_SCREW_PILOT_DIAMETER_MM = 2.8
LID_SCREW_PILOT_DEPTH_MM = 10.0
LID_BOSS_RADIUS_MM = 5.0
LID_SCREW_CENTERS_XY_MM = (
    (-38.0, -38.5),
    (-38.0, 38.5),
    (38.0, -38.5),
    (38.0, 38.5),
)

# Broad open windows provide connector access and airflow without assuming
# one cable routing scheme. Corner posts and lower sills retain stiffness.
SHORT_END_WINDOW_WIDTH_Y_MM = 56.0
SHORT_END_WINDOW_BOTTOM_Z_MM = 10.0
LONG_SIDE_WINDOW_LENGTH_X_MM = 84.0
LONG_SIDE_WINDOW_BOTTOM_Z_MM = 14.0

IMU_STANDOFF_HEIGHT_MM = 4.0
IMU_STANDOFF_OUTER_DIAMETER_MM = 6.0
IMU_M2_CLEARANCE_DIAMETER_MM = 2.4
IMU_BOARD_BOTTOM_ON_LID_Z_MM = LID_THICKNESS_Z_MM + IMU_STANDOFF_HEIGHT_MM

FUSION_OVERLAP_MM = 0.5
BOOLEAN_OVERTRAVEL_MM = 1.0
OUTER_FILLET_RADIUS_MM = 2.0
LID_EDGE_FILLET_RADIUS_MM = 1.2
INNER_FILLET_RADIUS_MM = 1.5
SMALL_FILLET_RADIUS_MM = 0.8
WINDOW_FILLET_RADIUS_MM = 2.0


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
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    return fillet(blank.edges(), radius_mm)


def _make_body_mount_tabs() -> tuple[Shape, ...]:
    return tuple(
        _rounded_box(
            BODY_MOUNT_TAB_LENGTH_X_MM,
            BODY_MOUNT_TAB_WIDTH_Y_MM,
            FLOOR_THICKNESS_Z_MM,
            SMALL_FILLET_RADIUS_MM,
        ).moved(Location((x_mm, side * BODY_MOUNT_TAB_CENTER_Y_MM, 0.0)))
        for x_mm in (-40.0, 40.0)
        for side in (-1.0, 1.0)
    )


def _make_lid_bosses() -> tuple[Shape, ...]:
    return tuple(
        Cylinder(
            LID_BOSS_RADIUS_MM,
            BASE_TOTAL_HEIGHT_Z_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, 0.0)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def _make_pi_standoffs() -> tuple[Shape, ...]:
    height = PI_STANDOFF_HEIGHT_MM + FUSION_OVERLAP_MM
    return tuple(
        Cylinder(
            PI_STANDOFF_OUTER_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, FLOOR_THICKNESS_Z_MM - FUSION_OVERLAP_MM)))
        for x_mm, y_mm in raspberry_pi_5.MOUNT_HOLE_CENTERS_XY_MM
    )


def _make_body_hole_tools() -> tuple[Shape, ...]:
    height = FLOOR_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            BODY_M3_CLEARANCE_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM)))
        for x_mm, y_mm in BODY_MOUNT_HOLE_CENTERS_XY_MM
    )


def _make_pi_pilot_tools() -> tuple[Shape, ...]:
    height = PI_M2_5_PILOT_DEPTH_MM + BOOLEAN_OVERTRAVEL_MM
    minimum_z = PI_PCB_BOTTOM_Z_MM - PI_M2_5_PILOT_DEPTH_MM
    return tuple(
        Cylinder(
            PI_M2_5_PILOT_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, minimum_z)))
        for x_mm, y_mm in raspberry_pi_5.MOUNT_HOLE_CENTERS_XY_MM
    )


def _make_lid_pilot_tools() -> tuple[Shape, ...]:
    height = LID_SCREW_PILOT_DEPTH_MM + BOOLEAN_OVERTRAVEL_MM
    minimum_z = BASE_TOTAL_HEIGHT_Z_MM - LID_SCREW_PILOT_DEPTH_MM
    return tuple(
        Cylinder(
            LID_SCREW_PILOT_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, minimum_z)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def _make_service_window_tools() -> tuple[Shape, ...]:
    short_depth = WALL_THICKNESS_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    short_height = (
        BASE_TOTAL_HEIGHT_Z_MM - SHORT_END_WINDOW_BOTTOM_Z_MM
        + BOOLEAN_OVERTRAVEL_MM
    )
    short_x = BASE_OUTER_LENGTH_X_MM / 2.0 - WALL_THICKNESS_MM / 2.0
    short_tools = tuple(
        _rounded_box(
            short_depth,
            SHORT_END_WINDOW_WIDTH_Y_MM,
            short_height,
            WINDOW_FILLET_RADIUS_MM,
        ).moved(Location((side * short_x, 0.0, SHORT_END_WINDOW_BOTTOM_Z_MM)))
        for side in (-1.0, 1.0)
    )

    long_depth = WALL_THICKNESS_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    long_height = (
        BASE_TOTAL_HEIGHT_Z_MM - LONG_SIDE_WINDOW_BOTTOM_Z_MM
        + BOOLEAN_OVERTRAVEL_MM
    )
    long_y = BASE_OUTER_WIDTH_Y_MM / 2.0 - WALL_THICKNESS_MM / 2.0
    long_tools = tuple(
        _rounded_box(
            LONG_SIDE_WINDOW_LENGTH_X_MM,
            long_depth,
            long_height,
            WINDOW_FILLET_RADIUS_MM,
        ).moved(Location((0.0, side * long_y, LONG_SIDE_WINDOW_BOTTOM_Z_MM)))
        for side in (-1.0, 1.0)
    )
    return short_tools + long_tools


def make_base() -> Shape:
    """Return the printable Pi box base with independent body mounts."""
    outer = _rounded_box(
        BASE_OUTER_LENGTH_X_MM,
        BASE_OUTER_WIDTH_Y_MM,
        BASE_TOTAL_HEIGHT_Z_MM,
        OUTER_FILLET_RADIUS_MM,
    )
    cavity = _rounded_box(
        FIT_CAVITY_LENGTH_X_MM,
        FIT_CAVITY_WIDTH_Y_MM,
        FIT_CAVITY_HEIGHT_Z_MM + BOOLEAN_OVERTRAVEL_MM,
        INNER_FILLET_RADIUS_MM,
    ).moved(Location((0.0, 0.0, FLOOR_THICKNESS_Z_MM)))
    shell = outer - cavity
    reinforced = shell.fuse(
        *_make_body_mount_tabs(),
        *_make_lid_bosses(),
        *_make_pi_standoffs(),
    )
    finished = reinforced - (
        _make_body_hole_tools()
        + _make_pi_pilot_tools()
        + _make_lid_pilot_tools()
        + _make_service_window_tools()
    )
    return _one_valid_solid(finished, "raspberry_pi_5_enclosure_base")


def _make_lid_ears() -> tuple[Shape, ...]:
    return tuple(
        _rounded_box(
            2.0 * LID_BOSS_RADIUS_MM + 2.0,
            2.0 * LID_BOSS_RADIUS_MM,
            LID_THICKNESS_Z_MM,
            SMALL_FILLET_RADIUS_MM,
        ).moved(Location((x_mm, y_mm, 0.0)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def _make_imu_standoffs() -> tuple[Shape, ...]:
    height = IMU_STANDOFF_HEIGHT_MM + FUSION_OVERLAP_MM
    return tuple(
        Cylinder(
            IMU_STANDOFF_OUTER_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, LID_THICKNESS_Z_MM - FUSION_OVERLAP_MM)))
        for x_mm, y_mm in adafruit_bno085.MOUNT_HOLE_CENTERS_SENSOR_XY_MM
    )


def _make_lid_hole_tools() -> tuple[Shape, ...]:
    height = LID_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            LID_M3_CLEARANCE_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def _make_imu_hole_tools() -> tuple[Shape, ...]:
    height = IMU_BOARD_BOTTOM_ON_LID_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            IMU_M2_CLEARANCE_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM)))
        for x_mm, y_mm in adafruit_bno085.MOUNT_HOLE_CENTERS_SENSOR_XY_MM
    )


def _make_vent_slot_tools() -> tuple[Shape, ...]:
    return tuple(
        _rounded_box(20.0, 4.0, LID_THICKNESS_Z_MM + 2.0, 1.5).moved(
            Location((x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM))
        )
        for x_mm in (-32.0, 32.0)
        for y_mm in (-20.0, 0.0, 20.0)
    )


def make_lid() -> Shape:
    """Return the ventilated screw-on lid with integral IMU standoffs."""
    plate = _rounded_box(
        BASE_OUTER_LENGTH_X_MM,
        BASE_OUTER_WIDTH_Y_MM,
        LID_THICKNESS_Z_MM,
        LID_EDGE_FILLET_RADIUS_MM,
    )
    blank = plate.fuse(*_make_lid_ears(), *_make_imu_standoffs())
    finished = blank - (
        _make_lid_hole_tools()
        + _make_imu_hole_tools()
        + _make_vent_slot_tools()
    )
    return _one_valid_solid(finished, "raspberry_pi_5_enclosure_lid")


def make_imu_cover() -> Shape:
    """Return the exact-pattern, open-sided BNO085 protection roof."""
    cover = quadruped_imu_cover.make_imu_cover()
    cover.label = "raspberry_pi_5_imu_cover"
    return cover
