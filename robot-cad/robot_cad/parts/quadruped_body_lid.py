"""Removable locating lid for the quadruped body tub."""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Box, Cylinder, Location, Shape

from robot_cad.parts import quadruped_body

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
LID_CABLE_PORT_LENGTH_X_MM = 18.0
LID_CABLE_PORT_WIDTH_Y_MM = 12.0
LID_CABLE_PORT_X_MM = (-quadruped_body.HIP_MOUNT_CENTER_X_MM, quadruped_body.HIP_MOUNT_CENTER_X_MM)
LID_CABLE_PORT_Y_MM = (-76.0, 76.0)


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
    """Return a nine-slot electronics ventilation pattern."""
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
    )


def make_lid_cable_port_tools() -> tuple[Shape, ...]:
    """Return four provisional cable ports beside the hip locations."""
    cutter_height = (
        quadruped_body.BODY_LID_THICKNESS_Z_MM
        + LID_LOCATOR_DEPTH_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Box(
            LID_CABLE_PORT_LENGTH_X_MM,
            LID_CABLE_PORT_WIDTH_Y_MM,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    -LID_LOCATOR_DEPTH_MM
                    - quadruped_body.BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for x_mm in LID_CABLE_PORT_X_MM
        for y_mm in LID_CABLE_PORT_Y_MM
    )


def make_lid() -> Shape:
    """Return the complete removable lid."""
    lid_with_lip = make_lid_blank().fuse(make_locator_lip())
    finished = lid_with_lip - (
        make_locator_relief_tools()
        + make_lid_hole_tools()
        + make_lid_vent_tools()
        + make_lid_cable_port_tools()
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
