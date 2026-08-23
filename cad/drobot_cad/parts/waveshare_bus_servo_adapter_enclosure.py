"""Protective box and screw-on lid for the Waveshare Bus Servo Adapter (A).

The enclosure uses the exact official Waveshare board model and its 37 x 28 mm
mounting pattern. Broad side openings preserve plug access, while a dedicated
lid opening leaves the external-power terminal screws serviceable.

Coordinate convention:
    - origin: center of the enclosure footprint at its bottom surface
    - +X/+Y: aligned to the adapter PCB axes
    - +Z: up, toward the removable lid
"""

from __future__ import annotations

from build123d import Align, Box, Cylinder, Location, Shape, fillet

from drobot_cad.parts import waveshare_bus_servo_adapter_a

FIT_CAVITY_LENGTH_X_MM = 48.0
FIT_CAVITY_WIDTH_Y_MM = 40.0
FIT_CAVITY_HEIGHT_Z_MM = 21.0
FLOOR_THICKNESS_Z_MM = 3.0
WALL_THICKNESS_MM = 2.5
BASE_OUTER_LENGTH_X_MM = FIT_CAVITY_LENGTH_X_MM + 2.0 * WALL_THICKNESS_MM
BASE_OUTER_WIDTH_Y_MM = FIT_CAVITY_WIDTH_Y_MM + 2.0 * WALL_THICKNESS_MM
BASE_TOTAL_HEIGHT_Z_MM = FLOOR_THICKNESS_Z_MM + FIT_CAVITY_HEIGHT_Z_MM

BOARD_STANDOFF_HEIGHT_MM = 5.0
BOARD_STANDOFF_OUTER_DIAMETER_MM = 6.0
BOARD_M2_PILOT_DIAMETER_MM = 2.0
BOARD_M2_PILOT_DEPTH_MM = 4.5
BOARD_DATUM_Z_MM = FLOOR_THICKNESS_Z_MM + BOARD_STANDOFF_HEIGHT_MM

# Four points on the robot body's universal 10 mm-pitch M3 floor grid.
BODY_MOUNT_HOLE_CENTERS_XY_MM = (
    (-20.0, -30.0),
    (-20.0, 30.0),
    (20.0, -30.0),
    (20.0, 30.0),
)
BODY_MOUNT_TAB_LENGTH_X_MM = 14.0
BODY_MOUNT_TAB_WIDTH_Y_MM = 17.0
BODY_MOUNT_TAB_CENTER_Y_MM = 26.0
BODY_M3_CLEARANCE_DIAMETER_MM = 3.4

LID_THICKNESS_Z_MM = 3.0
LID_SCREW_CENTERS_XY_MM = (
    (-22.0, -23.5),
    (-22.0, 23.5),
    (22.0, -23.5),
    (22.0, 23.5),
)
LID_BOSS_RADIUS_MM = 4.5
LID_M3_CLEARANCE_DIAMETER_MM = 3.4
LID_M3_PILOT_DIAMETER_MM = 2.8
LID_M3_PILOT_DEPTH_MM = 9.0

# Exact-board connector access. The -Y window serves USB-C and the 9-12.6 V
# terminal; the +Y window serves the two three-pin servo-bus connectors.
USB_POWER_WINDOW_LENGTH_X_MM = 42.0
USB_POWER_WINDOW_BOTTOM_Z_MM = 3.8
SERVO_WINDOW_LENGTH_X_MM = 32.0
SERVO_WINDOW_CENTER_X_MM = -4.5
SERVO_WINDOW_BOTTOM_Z_MM = 4.0
UART_WINDOW_WIDTH_Y_MM = 18.0
UART_WINDOW_CENTER_Y_MM = 2.0
UART_WINDOW_BOTTOM_Z_MM = 5.0

# Allows a screwdriver to reach the terminal clamp screws without removing
# the complete protective lid.
POWER_TERMINAL_ACCESS_LENGTH_X_MM = 15.0
POWER_TERMINAL_ACCESS_WIDTH_Y_MM = 16.0
POWER_TERMINAL_ACCESS_CENTER_XY_MM = (-11.0, -10.0)

BOOLEAN_OVERTRAVEL_MM = 1.0
FUSION_OVERLAP_MM = 0.5
OUTER_FILLET_RADIUS_MM = 1.5
LID_FILLET_RADIUS_MM = 1.2
INNER_FILLET_RADIUS_MM = 1.0
SMALL_FILLET_RADIUS_MM = 0.7
WINDOW_FILLET_RADIUS_MM = 1.5
FIT_PREVIEW_LID_GAP_Z_MM = 8.0


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
        for x_mm in (-20.0, 20.0)
        for side in (-1.0, 1.0)
    )


def _make_board_standoffs() -> tuple[Shape, ...]:
    height = BOARD_STANDOFF_HEIGHT_MM + FUSION_OVERLAP_MM
    return tuple(
        Cylinder(
            BOARD_STANDOFF_OUTER_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, FLOOR_THICKNESS_Z_MM - FUSION_OVERLAP_MM)))
        for x_mm, y_mm in waveshare_bus_servo_adapter_a.MOUNT_HOLE_CENTERS_XY_MM
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


def _make_board_pilot_tools() -> tuple[Shape, ...]:
    height = BOARD_M2_PILOT_DEPTH_MM + BOOLEAN_OVERTRAVEL_MM
    minimum_z = BOARD_DATUM_Z_MM - BOARD_M2_PILOT_DEPTH_MM
    return tuple(
        Cylinder(
            BOARD_M2_PILOT_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, minimum_z)))
        for x_mm, y_mm in waveshare_bus_servo_adapter_a.MOUNT_HOLE_CENTERS_XY_MM
    )


def _make_lid_pilot_tools() -> tuple[Shape, ...]:
    height = LID_M3_PILOT_DEPTH_MM + BOOLEAN_OVERTRAVEL_MM
    minimum_z = BASE_TOTAL_HEIGHT_Z_MM - LID_M3_PILOT_DEPTH_MM
    return tuple(
        Cylinder(
            LID_M3_PILOT_DIAMETER_MM / 2.0,
            height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, minimum_z)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def _make_connector_window_tools() -> tuple[Shape, ...]:
    wall_depth = WALL_THICKNESS_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM

    usb_power_height = (
        BASE_TOTAL_HEIGHT_Z_MM - USB_POWER_WINDOW_BOTTOM_Z_MM
        + BOOLEAN_OVERTRAVEL_MM
    )
    usb_power_y = -BASE_OUTER_WIDTH_Y_MM / 2.0 + WALL_THICKNESS_MM / 2.0
    usb_power = _rounded_box(
        USB_POWER_WINDOW_LENGTH_X_MM,
        wall_depth,
        usb_power_height,
        WINDOW_FILLET_RADIUS_MM,
    ).moved(Location((0.0, usb_power_y, USB_POWER_WINDOW_BOTTOM_Z_MM)))

    servo_height = (
        BASE_TOTAL_HEIGHT_Z_MM - SERVO_WINDOW_BOTTOM_Z_MM
        + BOOLEAN_OVERTRAVEL_MM
    )
    servo_y = BASE_OUTER_WIDTH_Y_MM / 2.0 - WALL_THICKNESS_MM / 2.0
    servo = _rounded_box(
        SERVO_WINDOW_LENGTH_X_MM,
        wall_depth,
        servo_height,
        WINDOW_FILLET_RADIUS_MM,
    ).moved(
        Location((SERVO_WINDOW_CENTER_X_MM, servo_y, SERVO_WINDOW_BOTTOM_Z_MM))
    )

    uart_height = (
        BASE_TOTAL_HEIGHT_Z_MM - UART_WINDOW_BOTTOM_Z_MM
        + BOOLEAN_OVERTRAVEL_MM
    )
    uart_x = BASE_OUTER_LENGTH_X_MM / 2.0 - WALL_THICKNESS_MM / 2.0
    uart = _rounded_box(
        wall_depth,
        UART_WINDOW_WIDTH_Y_MM,
        uart_height,
        WINDOW_FILLET_RADIUS_MM,
    ).moved(Location((uart_x, UART_WINDOW_CENTER_Y_MM, UART_WINDOW_BOTTOM_Z_MM)))
    return (usb_power, servo, uart)


def make_base() -> Shape:
    """Return the printable controller box with an open top."""
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
    # Fuse each connected feature family separately. A single mixed fuse of
    # the exterior tabs, interior standoffs, and wall bosses can preserve
    # detached operands in OCC even though every feature overlaps the shell.
    with_body_tabs = shell.fuse(*_make_body_mount_tabs())
    with_board_standoffs = with_body_tabs.fuse(*_make_board_standoffs())
    reinforced = with_board_standoffs.fuse(*_make_lid_bosses())
    finished = reinforced - (
        _make_body_hole_tools()
        + _make_board_pilot_tools()
        + _make_lid_pilot_tools()
        + _make_connector_window_tools()
    )
    return _one_valid_solid(finished, "waveshare_bus_servo_adapter_enclosure_base")


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


def _make_lid_access_tools() -> tuple[Shape, ...]:
    height = LID_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    power_access = _rounded_box(
        POWER_TERMINAL_ACCESS_LENGTH_X_MM,
        POWER_TERMINAL_ACCESS_WIDTH_Y_MM,
        height,
        2.0,
    ).moved(
        Location(
            (
                *POWER_TERMINAL_ACCESS_CENTER_XY_MM,
                -BOOLEAN_OVERTRAVEL_MM,
            )
        )
    )
    vent_slots = tuple(
        _rounded_box(15.0, 3.5, height, 1.2).moved(
            Location((12.0, y_mm, -BOOLEAN_OVERTRAVEL_MM))
        )
        for y_mm in (-10.0, 0.0, 10.0)
    )
    return (power_access,) + vent_slots


def make_lid() -> Shape:
    """Return the screw-on protective lid with power-terminal access."""
    plate = _rounded_box(
        BASE_OUTER_LENGTH_X_MM,
        BASE_OUTER_WIDTH_Y_MM,
        LID_THICKNESS_Z_MM,
        LID_FILLET_RADIUS_MM,
    )
    blank = plate.fuse(*_make_lid_ears())
    finished = blank - (_make_lid_hole_tools() + _make_lid_access_tools())
    return _one_valid_solid(finished, "waveshare_bus_servo_adapter_enclosure_lid")
