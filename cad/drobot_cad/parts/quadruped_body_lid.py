"""Removable locating lid for the quadruped body tub."""

from __future__ import annotations

from math import hypot
from pathlib import Path

from build123d import Align, Box, Cylinder, Location, Shape

from drobot_cad.parts import quadruped_body

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LID_LOCATOR_CLEARANCE_PER_SIDE_MM = 0.30
LID_LOCATOR_WALL_MM = 2.0
LID_LOCATOR_DEPTH_MM = 3.0
LID_LOCATOR_RELIEF_MM = 0.50
LID_FUSION_OVERLAP_MM = 0.4
LID_M3_CLEARANCE_DIAMETER_MM = 3.4
LID_VENT_SLOT_LENGTH_X_MM = 30.0
LID_VENT_SLOT_WIDTH_Y_MM = 4.0
LID_VENT_X_MM = (-60.0, 0.0, 60.0)
LID_VENT_Y_MM = (-42.0, 0.0, 42.0)
LID_CABLE_PORT_LENGTH_X_MM = 26.0
LID_CABLE_PORT_WIDTH_Y_MM = 14.0
LID_CABLE_PORT_X_MM = (-quadruped_body.HIP_MOUNT_CENTER_X_MM, quadruped_body.HIP_MOUNT_CENTER_X_MM)
LID_CABLE_PORT_Y_MM = (-76.0, 76.0)
LID_UTILITY_M2_X_MM = (-30.0, 30.0)
LID_UTILITY_Y_MM = (-24.0, 24.0)
LID_REAR_UTILITY_M3_CENTERS_XY_MM = (
    (-90.0, -24.0),
    (-90.0, 24.0),
)

# The official LeKiwi base camera mount has three M3 clearance holes on a
# 20 mm pitch.  This front row accepts that published mount without modifying
# it.  The dedicated cable port replaces the otherwise-overlapping +X/center
# ventilation slot.
LEKIWI_CAMERA_MOUNT_CENTER_X_MM = 90.0
LEKIWI_CAMERA_MOUNT_HOLE_PITCH_Y_MM = 20.0
LEKIWI_CAMERA_MOUNT_HOLE_CENTERS_XY_MM = tuple(
    (LEKIWI_CAMERA_MOUNT_CENTER_X_MM, y_mm)
    for y_mm in (-20.0, 0.0, 20.0)
)
LEKIWI_CAMERA_CABLE_PORT_LENGTH_X_MM = 20.0
LEKIWI_CAMERA_CABLE_PORT_WIDTH_Y_MM = 12.0
LEKIWI_CAMERA_CABLE_PORT_CENTER_X_MM = 65.0
LEKIWI_CAMERA_CABLE_PORT_CENTER_Y_MM = 0.0

# Universal M3 mounting field for future compute, fuse, power-distribution,
# sensor, and cable-management hardware.  The 10 mm pitch is dense enough to
# adapt common board standoffs with small brackets while leaving 6.6 mm of
# material between neighboring 3.4 mm holes.  Keep-outs preserve at least a
# 2 mm web around every existing opening.  The three camera holes happen to
# land on the grid and are counted as usable grid locations without cutting
# them twice.
LID_MOUNTING_GRID_PITCH_MM = 10.0
LID_MOUNTING_GRID_X_MM = tuple(float(value) for value in range(-90, 91, 10))
LID_MOUNTING_GRID_Y_MM = tuple(float(value) for value in range(-60, 61, 10))
LID_MOUNTING_GRID_M3_CLEARANCE_DIAMETER_MM = 3.4
LID_MOUNTING_GRID_MIN_WEB_MM = 2.0


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def make_lid_blank() -> Shape:
    """Return the rounded top plate before its locating lip and openings."""
    return quadruped_body._rounded_prism(
        quadruped_body.BODY_LENGTH_X_MM,
        quadruped_body.BODY_WIDTH_Y_MM,
        quadruped_body.BODY_LID_THICKNESS_Z_MM,
        minimum_z_mm=0.0,
        corner_radius_mm=quadruped_body.BODY_CORNER_RADIUS_MM,
    )


def make_locator_lip() -> Shape:
    """Return the downward rounded ring that locates inside the body cavity."""
    clearance = 2.0 * LID_LOCATOR_CLEARANCE_PER_SIDE_MM
    outer_length = (
        quadruped_body.BODY_LENGTH_X_MM
        - 2.0 * quadruped_body.BODY_WALL_MM
        - clearance
    )
    outer_width = (
        quadruped_body.BODY_WIDTH_Y_MM
        - 2.0 * quadruped_body.BODY_WALL_MM
        - clearance
    )
    outer_radius = (
        quadruped_body.BODY_CORNER_RADIUS_MM
        - quadruped_body.BODY_WALL_MM
        - LID_LOCATOR_CLEARANCE_PER_SIDE_MM
    )
    lip_height = LID_LOCATOR_DEPTH_MM + LID_FUSION_OVERLAP_MM
    lip_min_z = -LID_LOCATOR_DEPTH_MM
    outer = quadruped_body._rounded_prism(
        outer_length,
        outer_width,
        lip_height,
        minimum_z_mm=lip_min_z,
        corner_radius_mm=outer_radius,
    )
    inner = quadruped_body._rounded_prism(
        outer_length - 2.0 * LID_LOCATOR_WALL_MM,
        outer_width - 2.0 * LID_LOCATOR_WALL_MM,
        lip_height + 2.0 * LID_FUSION_OVERLAP_MM,
        minimum_z_mm=lip_min_z - LID_FUSION_OVERLAP_MM,
        corner_radius_mm=outer_radius - LID_LOCATOR_WALL_MM,
    )
    return _one_valid_solid(outer - inner, "quadruped_body_lid_locator")


def make_lid_hole_tools() -> tuple[Shape, ...]:
    """Return four through cutters aligned to the tub lid bosses."""
    cutter_height = (
        quadruped_body.BODY_LID_THICKNESS_Z_MM
        + LID_LOCATOR_DEPTH_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    cutter_min_z = -LID_LOCATOR_DEPTH_MM - quadruped_body.BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            LID_M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, cutter_min_z)))
        for x_mm, y_mm in quadruped_body.LID_BOSS_CENTERS_XY_MM
    )


def make_locator_relief_tools() -> tuple[Shape, ...]:
    """Return underside cutters that clear the tub pads and lid bosses."""
    cutter_min_z = (
        -LID_LOCATOR_DEPTH_MM - quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    cutter_height = (
        LID_LOCATOR_DEPTH_MM + quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    side_relief_width = (
        quadruped_body.HIP_BACKING_TOTAL_THICKNESS_Y_MM
        + 2.0 * LID_LOCATOR_WALL_MM
        + 2.0 * LID_LOCATOR_RELIEF_MM
    )
    side_relief_center_y = (
        quadruped_body.BODY_WIDTH_Y_MM / 2.0
        - side_relief_width / 2.0
    )
    pad_reliefs = tuple(
        Box(
            quadruped_body.HIP_BACKING_PAD_LENGTH_X_MM
            + 2.0 * LID_LOCATOR_RELIEF_MM,
            side_relief_width,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    side * side_relief_center_y,
                    cutter_min_z,
                )
            )
        )
        for x_mm in quadruped_body.HIP_MOUNT_CENTERS_X_MM
        for side in (-1.0, 1.0)
    )
    boss_reliefs = tuple(
        Cylinder(
            quadruped_body.LID_BOSS_RADIUS_MM + LID_LOCATOR_RELIEF_MM,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, cutter_min_z)))
        for x_mm, y_mm in quadruped_body.LID_BOSS_CENTERS_XY_MM
    )
    return pad_reliefs + boss_reliefs


def make_lid_vent_tools() -> tuple[Shape, ...]:
    """Return vents clear of the dedicated front-camera cable opening."""
    cutter_height = (
        quadruped_body.BODY_LID_THICKNESS_Z_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Box(
            LID_VENT_SLOT_LENGTH_X_MM,
            LID_VENT_SLOT_WIDTH_Y_MM,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    -quadruped_body.BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for x_mm in LID_VENT_X_MM
        for y_mm in LID_VENT_Y_MM
        if not (x_mm == 60.0 and y_mm == 0.0)
    )


def _rounded_slot_tool_xy(
    length_x_mm: float,
    width_y_mm: float,
    height_z_mm: float,
    *,
    center_x_mm: float,
    center_y_mm: float,
    minimum_z_mm: float,
) -> Shape:
    """Return a horizontal stadium-shaped through-cutter."""
    radius = width_y_mm / 2.0
    straight_length = length_x_mm - 2.0 * radius
    center_box = Box(
        straight_length,
        width_y_mm,
        height_z_mm,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).moved(Location((center_x_mm, center_y_mm, minimum_z_mm)))
    cap_offset_x = straight_length / 2.0
    caps = tuple(
        Cylinder(
            radius,
            height_z_mm,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    center_x_mm + side * cap_offset_x,
                    center_y_mm,
                    minimum_z_mm,
                )
            )
        )
        for side in (-1.0, 1.0)
    )
    return center_box.fuse(*caps)


def make_lid_cable_port_tools() -> tuple[Shape, ...]:
    """Return four rounded cable ports, one beside each hip location."""
    cutter_height = (
        quadruped_body.BODY_LID_THICKNESS_Z_MM
        + LID_LOCATOR_DEPTH_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        _rounded_slot_tool_xy(
            LID_CABLE_PORT_LENGTH_X_MM,
            LID_CABLE_PORT_WIDTH_Y_MM,
            cutter_height,
            center_x_mm=x_mm,
            center_y_mm=y_mm,
            minimum_z_mm=(
                -LID_LOCATOR_DEPTH_MM
                - quadruped_body.BOOLEAN_OVERTRAVEL_MM
            ),
        )
        for x_mm in LID_CABLE_PORT_X_MM
        for y_mm in LID_CABLE_PORT_Y_MM
    )


def make_lekiwi_camera_cable_port_tool() -> Shape:
    """Return the top-entry cable slot behind the LeKiwi camera mount."""
    cutter_height = (
        quadruped_body.BODY_LID_THICKNESS_Z_MM
        + LID_LOCATOR_DEPTH_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return _rounded_slot_tool_xy(
        LEKIWI_CAMERA_CABLE_PORT_LENGTH_X_MM,
        LEKIWI_CAMERA_CABLE_PORT_WIDTH_Y_MM,
        cutter_height,
        center_x_mm=LEKIWI_CAMERA_CABLE_PORT_CENTER_X_MM,
        center_y_mm=LEKIWI_CAMERA_CABLE_PORT_CENTER_Y_MM,
        minimum_z_mm=(
            -LID_LOCATOR_DEPTH_MM - quadruped_body.BOOLEAN_OVERTRAVEL_MM
        ),
    )


def _center_clears_rectangular_opening(
    center_xy_mm: tuple[float, float],
    opening_center_xy_mm: tuple[float, float],
    opening_length_x_mm: float,
    opening_width_y_mm: float,
) -> bool:
    """Return whether a grid hole leaves the required web around an opening."""
    grid_radius = LID_MOUNTING_GRID_M3_CLEARANCE_DIAMETER_MM / 2.0
    keepout = grid_radius + LID_MOUNTING_GRID_MIN_WEB_MM
    return (
        abs(center_xy_mm[0] - opening_center_xy_mm[0])
        > opening_length_x_mm / 2.0 + keepout
        or abs(center_xy_mm[1] - opening_center_xy_mm[1])
        > opening_width_y_mm / 2.0 + keepout
    )


def _center_clears_round_hole(
    center_xy_mm: tuple[float, float],
    hole_center_xy_mm: tuple[float, float],
    hole_diameter_mm: float,
) -> bool:
    """Return whether a grid hole leaves the required web around another hole."""
    required_distance = (
        LID_MOUNTING_GRID_M3_CLEARANCE_DIAMETER_MM / 2.0
        + hole_diameter_mm / 2.0
        + LID_MOUNTING_GRID_MIN_WEB_MM
    )
    return hypot(
        center_xy_mm[0] - hole_center_xy_mm[0],
        center_xy_mm[1] - hole_center_xy_mm[1],
    ) >= required_distance


def lid_mounting_grid_centers_xy_mm() -> tuple[tuple[float, float], ...]:
    """Return usable 10 mm-pitch M3 grid centers after feature keep-outs."""
    vent_openings = tuple(
        (x_mm, y_mm)
        for x_mm in LID_VENT_X_MM
        for y_mm in LID_VENT_Y_MM
        if not (x_mm == 60.0 and y_mm == 0.0)
    )
    rectangular_openings = (
        tuple(
            (
                center,
                LID_VENT_SLOT_LENGTH_X_MM,
                LID_VENT_SLOT_WIDTH_Y_MM,
            )
            for center in vent_openings
        )
        + tuple(
            (
                (x_mm, y_mm),
                LID_CABLE_PORT_LENGTH_X_MM,
                LID_CABLE_PORT_WIDTH_Y_MM,
            )
            for x_mm in LID_CABLE_PORT_X_MM
            for y_mm in LID_CABLE_PORT_Y_MM
        )
        + (
            (
                (
                    LEKIWI_CAMERA_CABLE_PORT_CENTER_X_MM,
                    LEKIWI_CAMERA_CABLE_PORT_CENTER_Y_MM,
                ),
                LEKIWI_CAMERA_CABLE_PORT_LENGTH_X_MM,
                LEKIWI_CAMERA_CABLE_PORT_WIDTH_Y_MM,
            ),
        )
    )
    round_holes = (
        tuple(
            ((x_mm, y_mm), quadruped_body.UTILITY_M2_CLEARANCE_DIAMETER_MM)
            for x_mm in LID_UTILITY_M2_X_MM
            for y_mm in LID_UTILITY_Y_MM
        )
        + tuple(
            (
                center,
                quadruped_body.UTILITY_M3_CLEARANCE_DIAMETER_MM,
            )
            for center in LID_REAR_UTILITY_M3_CENTERS_XY_MM
        )
        + tuple(
            (center, LID_M3_CLEARANCE_DIAMETER_MM)
            for center in quadruped_body.LID_BOSS_CENTERS_XY_MM
        )
    )

    return tuple(
        (x_mm, y_mm)
        for x_mm in LID_MOUNTING_GRID_X_MM
        for y_mm in LID_MOUNTING_GRID_Y_MM
        if all(
            _center_clears_rectangular_opening(
                (x_mm, y_mm),
                opening_center,
                opening_length,
                opening_width,
            )
            for opening_center, opening_length, opening_width in (
                rectangular_openings
            )
        )
        and all(
            _center_clears_round_hole(
                (x_mm, y_mm),
                hole_center,
                hole_diameter,
            )
            for hole_center, hole_diameter in round_holes
        )
    )


def make_lid_mounting_grid_hole_tools() -> tuple[Shape, ...]:
    """Return new M3 cutters for the universal top mounting field."""
    cutter_height = (
        quadruped_body.BODY_LID_THICKNESS_Z_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    cutter_min_z = -quadruped_body.BOOLEAN_OVERTRAVEL_MM
    return tuple(
        Cylinder(
            LID_MOUNTING_GRID_M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, cutter_min_z)))
        for x_mm, y_mm in lid_mounting_grid_centers_xy_mm()
        if (x_mm, y_mm) not in LEKIWI_CAMERA_MOUNT_HOLE_CENTERS_XY_MM
    )


def make_lid_utility_mount_hole_tools() -> tuple[Shape, ...]:
    """Return utility holes plus the LeKiwi-compatible camera mount row."""
    cutter_height = (
        quadruped_body.BODY_LID_THICKNESS_Z_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    cutter_min_z = -quadruped_body.BOOLEAN_OVERTRAVEL_MM
    m2_tools = tuple(
        Cylinder(
            quadruped_body.UTILITY_M2_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, cutter_min_z)))
        for x_mm in LID_UTILITY_M2_X_MM
        for y_mm in LID_UTILITY_Y_MM
    )
    m3_centers = (
        LID_REAR_UTILITY_M3_CENTERS_XY_MM
        + LEKIWI_CAMERA_MOUNT_HOLE_CENTERS_XY_MM
    )
    m3_tools = tuple(
        Cylinder(
            quadruped_body.UTILITY_M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((x_mm, y_mm, cutter_min_z)))
        for x_mm, y_mm in m3_centers
    )
    return m2_tools + m3_tools


def make_lid() -> Shape:
    """Return the complete removable lid."""
    lid_with_lip = make_lid_blank().fuse(make_locator_lip())
    finished = lid_with_lip - (
        make_locator_relief_tools()
        + make_lid_hole_tools()
        + make_lid_vent_tools()
        + make_lid_cable_port_tools()
        + (make_lekiwi_camera_cable_port_tool(),)
        + make_lid_mounting_grid_hole_tools()
        + make_lid_utility_mount_hole_tools()
    )
    return _one_valid_solid(finished, "quadruped_body_lid")


def gen_step() -> Shape:
    """Return the STEP-ready removable lid."""
    return make_lid()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "quadruped_body_lid.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
