"""Two-piece CM5202 LiPo battery box for the quadruped body floor grid.

Coordinate convention:
    - origin: center of the box footprint at its bottom surface
    - +X: robot front / battery length
    - +Y: robot left
    - +Z: up

The short -X end wall has two large cable openings separated by a narrow
full-height center support rib.  The lid carries matching split edge reliefs
so it cannot pinch the battery leads.
"""

from __future__ import annotations

from build123d import (
    Align,
    Box,
    Color,
    Compound,
    Cylinder,
    Location,
    Shape,
    fillet,
)


BATTERY_LENGTH_X_MM = 135.0
BATTERY_WIDTH_Y_MM = 45.0
BATTERY_HEIGHT_Z_MM = 33.0
FIT_CLEARANCE_X_PER_END_MM = 2.0
FIT_CLEARANCE_Y_PER_SIDE_MM = 2.0
FIT_CLEARANCE_ABOVE_MM = 4.0

FLOOR_THICKNESS_Z_MM = 3.0
WALL_THICKNESS_MM = 2.5
POCKET_LENGTH_X_MM = BATTERY_LENGTH_X_MM + 2.0 * FIT_CLEARANCE_X_PER_END_MM
POCKET_WIDTH_Y_MM = BATTERY_WIDTH_Y_MM + 2.0 * FIT_CLEARANCE_Y_PER_SIDE_MM
POCKET_HEIGHT_Z_MM = BATTERY_HEIGHT_Z_MM + FIT_CLEARANCE_ABOVE_MM
BOX_OUTER_LENGTH_X_MM = POCKET_LENGTH_X_MM + 2.0 * WALL_THICKNESS_MM
BOX_OUTER_WIDTH_Y_MM = POCKET_WIDTH_Y_MM + 2.0 * WALL_THICKNESS_MM
BOX_TOTAL_HEIGHT_Z_MM = FLOOR_THICKNESS_Z_MM + POCKET_HEIGHT_Z_MM

LID_THICKNESS_Z_MM = 3.0
LID_M3_CLEARANCE_DIAMETER_MM = 3.4
LID_SCREW_PILOT_DIAMETER_MM = 2.8
LID_SCREW_PILOT_DEPTH_MM = 10.0
LID_BOSS_RADIUS_MM = 4.5
LID_SCREW_X_MM = (-35.0, 35.0)
LID_SCREW_Y_MM = (-29.5, 29.5)
LID_SCREW_CENTERS_XY_MM = tuple(
    (x_mm, y_mm)
    for x_mm in LID_SCREW_X_MM
    for y_mm in LID_SCREW_Y_MM
)

# Body mounting remains independent of the removable lid.  These four holes
# are members of the quadruped body's 10 mm-pitch M3 floor grid.
BODY_MOUNT_HOLE_CENTERS_XY_MM = (
    (-60.0, -30.0),
    (-60.0, 30.0),
    (60.0, -30.0),
    (60.0, 30.0),
)
BODY_MOUNT_TAB_LENGTH_X_MM = 14.0
BODY_MOUNT_TAB_WIDTH_Y_MM = 9.0
BODY_MOUNT_TAB_CENTER_Y_MM = 29.5
BODY_M3_CLEARANCE_DIAMETER_MM = 3.4

# Two large wire exits in the short -X end wall.  A narrow full-height center
# rib remains for support, along with small corner returns and a lower sill.
WIRE_PORT_TOTAL_SPAN_Y_MM = 48.0
WIRE_PORT_CENTER_RIB_WIDTH_Y_MM = 6.0
WIRE_PORT_SIDE_OPENING_WIDTH_Y_MM = (
    WIRE_PORT_TOTAL_SPAN_Y_MM - WIRE_PORT_CENTER_RIB_WIDTH_Y_MM
) / 2.0
WIRE_PORT_SIDE_CENTER_OFFSET_Y_MM = (
    WIRE_PORT_CENTER_RIB_WIDTH_Y_MM / 2.0
    + WIRE_PORT_SIDE_OPENING_WIDTH_Y_MM / 2.0
)
WIRE_PORT_HEIGHT_Z_MM = 32.0
LID_WIRE_RELIEF_DEPTH_X_MM = 18.0

FUSION_OVERLAP_MM = 0.5
BOOLEAN_OVERTRAVEL_MM = 1.0
FIT_PREVIEW_LID_GAP_Z_MM = 10.0
FIT_PREVIEW_FOAM_ALLOWANCE_Z_MM = 1.0
OUTER_BOX_FILLET_RADIUS_MM = 1.5
CAVITY_FILLET_RADIUS_MM = 1.0
SMALL_FEATURE_FILLET_RADIUS_MM = 0.8
WIRE_OPENING_FILLET_RADIUS_MM = 1.0


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
    """Return a centered, bottom-aligned box with every primitive edge rounded."""
    blank = Box(
        length_x_mm,
        width_y_mm,
        height_z_mm,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    )
    return fillet(blank.edges(), radius_mm)


def make_body_mount_tabs() -> tuple[Shape, ...]:
    """Return four floor-height tabs for the robot body mounting screws."""
    return tuple(
        _rounded_box(
            BODY_MOUNT_TAB_LENGTH_X_MM,
            BODY_MOUNT_TAB_WIDTH_Y_MM,
            FLOOR_THICKNESS_Z_MM,
            SMALL_FEATURE_FILLET_RADIUS_MM,
        ).moved(
            Location(
                (
                    x_mm,
                    side * BODY_MOUNT_TAB_CENTER_Y_MM,
                    0.0,
                )
            )
        )
        for x_mm in (-60.0, 60.0)
        for side in (-1.0, 1.0)
    )


def make_lid_screw_bosses() -> tuple[Shape, ...]:
    """Return four wall-connected towers for self-tapping M3 lid screws."""
    return tuple(
        Cylinder(
            LID_BOSS_RADIUS_MM,
            BOX_TOTAL_HEIGHT_Z_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, 0.0)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def make_body_mount_hole_tools() -> tuple[Shape, ...]:
    """Return four through-cutters for the robot body mounting screws."""
    cutter_height = FLOOR_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            BODY_M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM)))
        for x_mm, y_mm in BODY_MOUNT_HOLE_CENTERS_XY_MM
    )


def make_lid_screw_pilot_tools() -> tuple[Shape, ...]:
    """Return blind pilot cutters entering the screw towers from above."""
    cutter_height = LID_SCREW_PILOT_DEPTH_MM + BOOLEAN_OVERTRAVEL_MM
    cutter_min_z = BOX_TOTAL_HEIGHT_Z_MM - LID_SCREW_PILOT_DEPTH_MM
    return tuple(
        Cylinder(
            LID_SCREW_PILOT_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, cutter_min_z)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def make_wire_port_tools() -> tuple[Shape, ...]:
    """Return split short-end openings with a narrow center support rib."""
    cutter_depth_x = WALL_THICKNESS_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    cutter_height = WIRE_PORT_HEIGHT_Z_MM + BOOLEAN_OVERTRAVEL_MM
    cutter_center_x = -BOX_OUTER_LENGTH_X_MM / 2.0 + WALL_THICKNESS_MM / 2.0
    cutter_min_z = BOX_TOTAL_HEIGHT_Z_MM - WIRE_PORT_HEIGHT_Z_MM
    return tuple(
        _rounded_box(
            cutter_depth_x,
            WIRE_PORT_SIDE_OPENING_WIDTH_Y_MM,
            cutter_height,
            WIRE_OPENING_FILLET_RADIUS_MM,
        ).moved(
            Location(
                (
                    cutter_center_x,
                    side * WIRE_PORT_SIDE_CENTER_OFFSET_Y_MM,
                    cutter_min_z,
                )
            )
        )
        for side in (-1.0, 1.0)
    )


def make_cm5202_battery_box() -> Shape:
    """Return the printable main battery box with an open top."""
    outer = _rounded_box(
        BOX_OUTER_LENGTH_X_MM,
        BOX_OUTER_WIDTH_Y_MM,
        BOX_TOTAL_HEIGHT_Z_MM,
        OUTER_BOX_FILLET_RADIUS_MM,
    )
    cavity = _rounded_box(
        POCKET_LENGTH_X_MM,
        POCKET_WIDTH_Y_MM,
        POCKET_HEIGHT_Z_MM + BOOLEAN_OVERTRAVEL_MM,
        CAVITY_FILLET_RADIUS_MM,
    ).moved(Location((0.0, 0.0, FLOOR_THICKNESS_Z_MM)))
    shell = outer - cavity
    reinforced = shell.fuse(
        *make_body_mount_tabs(),
        *make_lid_screw_bosses(),
    )
    finished = reinforced - (
        make_body_mount_hole_tools()
        + make_lid_screw_pilot_tools()
        + make_wire_port_tools()
    )
    return _one_valid_solid(finished, "cm5202_battery_box")


def make_lid_mount_tabs() -> tuple[Shape, ...]:
    """Return four lid ears aligned to the main-box screw towers."""
    return tuple(
        _rounded_box(
            2.0 * LID_BOSS_RADIUS_MM + 2.0,
            2.0 * LID_BOSS_RADIUS_MM,
            LID_THICKNESS_Z_MM,
            SMALL_FEATURE_FILLET_RADIUS_MM,
        ).moved(Location((x_mm, y_mm, 0.0)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def make_lid_clearance_hole_tools() -> tuple[Shape, ...]:
    """Return four M3 through-cutters for the removable lid."""
    cutter_height = LID_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            LID_M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, -BOOLEAN_OVERTRAVEL_MM)))
        for x_mm, y_mm in LID_SCREW_CENTERS_XY_MM
    )


def make_lid_wire_relief_tools() -> tuple[Shape, ...]:
    """Return split lid reliefs matching the short-end wire openings."""
    cutter_center_x = (
        -BOX_OUTER_LENGTH_X_MM / 2.0
        + LID_WIRE_RELIEF_DEPTH_X_MM / 2.0
        - BOOLEAN_OVERTRAVEL_MM / 2.0
    )
    return tuple(
        _rounded_box(
            LID_WIRE_RELIEF_DEPTH_X_MM + BOOLEAN_OVERTRAVEL_MM,
            WIRE_PORT_SIDE_OPENING_WIDTH_Y_MM,
            LID_THICKNESS_Z_MM + 2.0 * BOOLEAN_OVERTRAVEL_MM,
            WIRE_OPENING_FILLET_RADIUS_MM,
        ).moved(
            Location(
                (
                    cutter_center_x,
                    side * WIRE_PORT_SIDE_CENTER_OFFSET_Y_MM,
                    -BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for side in (-1.0, 1.0)
    )


def make_cm5202_battery_box_lid() -> Shape:
    """Return the printable screw-on lid with one wire-edge relief."""
    plate = _rounded_box(
        BOX_OUTER_LENGTH_X_MM,
        BOX_OUTER_WIDTH_Y_MM,
        LID_THICKNESS_Z_MM,
        SMALL_FEATURE_FILLET_RADIUS_MM,
    )
    blank = plate.fuse(*make_lid_mount_tabs())
    finished = blank - (
        make_lid_clearance_hole_tools() + make_lid_wire_relief_tools()
    )
    return _one_valid_solid(finished, "cm5202_battery_box_lid")


def make_battery_fit_proxy() -> Shape:
    """Return the measured battery envelope at its installed height."""
    battery = Box(
        BATTERY_LENGTH_X_MM,
        BATTERY_WIDTH_Y_MM,
        BATTERY_HEIGHT_Z_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).moved(
        Location(
            (
                0.0,
                0.0,
                FLOOR_THICKNESS_Z_MM + FIT_PREVIEW_FOAM_ALLOWANCE_Z_MM,
            )
        )
    )
    battery.label = "cm5202_measured_battery_envelope"
    battery.color = Color(0.55, 0.16, 0.20)
    return battery


def make_cm5202_battery_box_fit_preview() -> Compound:
    """Return an exploded lid, inserted battery, and main box review assembly."""
    box = make_cm5202_battery_box()
    box.label = "printable_battery_box"
    box.color = Color(0.26, 0.42, 0.62)
    battery = make_battery_fit_proxy()
    lid = make_cm5202_battery_box_lid().moved(
        Location(
            (
                0.0,
                0.0,
                BOX_TOTAL_HEIGHT_Z_MM + FIT_PREVIEW_LID_GAP_Z_MM,
            )
        )
    )
    lid.label = "printable_lid_exploded_10mm"
    lid.color = Color(0.70, 0.76, 0.84)
    preview = Compound(children=[box, battery, lid])
    preview.label = "cm5202_battery_box_fit_preview"
    return preview
