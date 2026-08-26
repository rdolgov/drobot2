"""Printable ST3215 hip made from the oval-ended fork and motor bay.

The component preserves the user-approved orientation:
    - global +Z is up and global +X is right
    - the fork local +X direction points down along global -Z
    - the bay first rolls +90 degrees about its local X centerline between the
      screw-access holes, placing the diamond face upward
    - viewed from that diamond face, the complete rolled bay then turns
      +90 degrees left about the vertical axis through the face center
    - the bay finally turns 90 degrees clockwise about its vertical centerline,
      making its open local -X end face global left
    - the bay is centered over the top of the fork's new half-oval end

The bay's lower broad wall seats into the oval tip by one wall thickness.  A
pair of tapered external side webs then carries the load farther down into the
fork shoulders.  The webs begin behind the open end of the motor bay, stay
outside the servo cavity, and retain clearance around the side screw-access
holes.
"""

from __future__ import annotations

from pathlib import Path

from build123d import (
    Align,
    Axis,
    Box,
    BuildSketch,
    Cylinder,
    Location,
    Plane,
    Polygon,
    Shape,
    extrude,
    import_step,
)

from drobot_cad.parts.st3215_motor_bay import (
    SOCKET_CLEARANCE_Y_PER_SIDE_MM,
    SOCKET_CLEARANCE_Z_TOTAL_MM,
    SOCKET_LENGTH_X_MM,
    SOCKET_WALL_MM,
    SERVO_BOTTOM_MOUNT_HOLE_X_MM,
    SERVO_MOUNT_COUNTERBORE_DIAMETER_MM,
    SERVO_MOUNT_HOLE_Y_SPACING_MM,
    SERVO_TOP_MOUNT_HOLE_X_MM,
    ST3215_CATALOG_HEIGHT_Y_MM,
    ST3215_CATALOG_WIDTH_Z_MM,
    ST3215_SERVO_STEP,
    VENT_DIAMOND_COLUMNS_X_MM,
    VENT_DIAMOND_HEIGHT_Z_MM,
    VENT_DIAMOND_ROWS_Z_MM,
    VENT_DIAMOND_WIDTH_X_MM,
    st3215_installed_location,
)
from drobot_cad.parts.st3215_motor_bay import gen_step as gen_motor_bay
from drobot_cad.parts.st3215_servo_output_fork import (
    ROOT_EXTENSION_CENTER_X_MM,
    ROOT_EXTENSION_CENTER_Z_MM,
    ROOT_EXTENSION_RADIUS_X_MM,
    ROOT_EXTENSION_WIDTH_Y_MM,
)
from drobot_cad.parts.st3215_servo_output_fork import gen_step as gen_servo_output_fork

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORK_FACE_DOWN_ROTATION_XYZ_DEG = (0.0, 90.0, 0.0)
MOTOR_BAY_FIRST_ROLL_ROTATION_XYZ_DEG = (90.0, 0.0, 0.0)
MOTOR_BAY_TOP_VIEW_LEFT_TURN_DEG = 90.0
MOTOR_BAY_FINAL_TOP_VIEW_CLOCKWISE_TURN_DEG = -90.0

# The fork's negative local-X half-oval maps upward after the face-down
# rotation.  Its nose is the new attachment datum at global Z=30 mm.
FORK_OVAL_CENTER_WORLD_X_MM = ROOT_EXTENSION_CENTER_Z_MM
FORK_OVAL_TOP_WORLD_Z_MM = (
    ROOT_EXTENSION_RADIUS_X_MM - ROOT_EXTENSION_CENTER_X_MM
)

# The bay extends toward local/global -X.  Offset its X=0 datum by half its
# length so the complete bay is centered over the oval tip.
MOTOR_BAY_CENTER_WORLD_X_MM = FORK_OVAL_CENTER_WORLD_X_MM
MOTOR_BAY_DATUM_WORLD_X_MM = (
    MOTOR_BAY_CENTER_WORLD_X_MM + SOCKET_LENGTH_X_MM / 2.0
)

# A +90-degree roll maps the bay's local Y width into world Z.  The approved
# centerline height seats the lower broad wall on the oval's top tangent plane.
MOTOR_BAY_OUTER_Y_MM = (
    ST3215_CATALOG_WIDTH_Z_MM
    + 2.0 * SOCKET_CLEARANCE_Y_PER_SIDE_MM
    + 2.0 * SOCKET_WALL_MM
)
MOTOR_BAY_APPROVED_CENTERLINE_WORLD_Z_MM = (
    FORK_OVAL_TOP_WORLD_Z_MM + MOTOR_BAY_OUTER_Y_MM / 2.0
)

# Seat through the bay's lower wall thickness.  This gives the curved oval tip
# a broad union while preserving zero collision with the installed ST3215.
HIP_JOIN_OVERLAP_MM = SOCKET_WALL_MM
MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM = (
    MOTOR_BAY_APPROVED_CENTERLINE_WORLD_Z_MM - HIP_JOIN_OVERLAP_MM
)

# The original join relied almost entirely on the bay's 3 mm-deep seating
# overlap at the oval crown.  These external side webs spread that load into
# the fork shoulders without changing the fit-critical motor bay itself.
MOTOR_BAY_OPEN_WORLD_X_MM = (
    MOTOR_BAY_CENTER_WORLD_X_MM - SOCKET_LENGTH_X_MM / 2.0
)
MOTOR_BAY_CLOSED_WORLD_X_MM = (
    MOTOR_BAY_CENTER_WORLD_X_MM + SOCKET_LENGTH_X_MM / 2.0
)
MOTOR_BAY_OUTER_WORLD_Y_MM = (
    ST3215_CATALOG_HEIGHT_Y_MM
    + SOCKET_CLEARANCE_Z_TOTAL_MM
    + 2.0 * SOCKET_WALL_MM
)
MOTOR_BAY_INNER_WORLD_Y_MM = (
    ST3215_CATALOG_HEIGHT_Y_MM + SOCKET_CLEARANCE_Z_TOTAL_MM
)
MOTOR_BAY_INNER_WORLD_Z_MM = (
    ST3215_CATALOG_WIDTH_Z_MM + 2.0 * SOCKET_CLEARANCE_Y_PER_SIDE_MM
)
MOTOR_BAY_CAVITY_BOTTOM_WORLD_Z_MM = (
    MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM
    - MOTOR_BAY_INNER_WORLD_Z_MM / 2.0
)

HIP_SIDE_WEB_OPENING_SETBACK_MM = 3.5
HIP_SIDE_WEB_CLOSED_END_OVERLAP_MM = 0.5
HIP_SIDE_WEB_FORK_INNER_Y_MM = ROOT_EXTENSION_WIDTH_Y_MM / 2.0 - 1.5
HIP_SIDE_WEB_FORK_OUTER_Y_MM = ROOT_EXTENSION_WIDTH_Y_MM / 2.0 + 0.5
HIP_SIDE_WEB_BAY_INNER_Y_MM = MOTOR_BAY_INNER_WORLD_Y_MM / 2.0 + 0.5
HIP_SIDE_WEB_BAY_OUTER_Y_MM = MOTOR_BAY_OUTER_WORLD_Y_MM / 2.0 + 0.5
HIP_SIDE_WEB_FORK_BOTTOM_Z_MM = 18.0
HIP_SIDE_WEB_FORK_TOP_Z_MM = 27.0
HIP_SIDE_WEB_BAY_TOE_Z_MM = FORK_OVAL_TOP_WORLD_Z_MM - 2.0
HIP_SIDE_WEB_CAVITY_KNEE_Z_MM = MOTOR_BAY_CAVITY_BOTTOM_WORLD_Z_MM - 0.5
HIP_SIDE_WEB_TOP_Z_MM = MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM + 2.0
HIP_SIDE_WEB_SCREW_CLEARANCE_DIAMETER_MM = (
    SERVO_MOUNT_COUNTERBORE_DIAMETER_MM + 1.0
)
HIP_SIDE_WEB_SCREW_CLEARANCE_OVERTRAVEL_MM = 1.0

# The standalone motor bay is symmetric and ventilates both broad walls.  In
# this hip orientation its local -Y wall becomes the load-bearing lower floor,
# so the hip backfills that vent field while retaining the upper diamonds.
HIP_LOWER_WALL_FILL_MARGIN_MM = 0.5


def _motor_bay_first_roll_location(centerline_world_z_mm: float) -> Location:
    """Apply the first 90-degree roll about the screw-hole centerline."""
    return Location(
        (
            MOTOR_BAY_DATUM_WORLD_X_MM,
            0.0,
            centerline_world_z_mm,
        ),
        MOTOR_BAY_FIRST_ROLL_ROTATION_XYZ_DEG,
    )


def _motor_bay_top_view_left_turn(centerline_world_z_mm: float) -> Location:
    """Turn left around the center of the upward-facing diamond surface."""
    diamond_face_center_z = centerline_world_z_mm + MOTOR_BAY_OUTER_Y_MM / 2.0
    pivot_to_world = Location((0.0, 0.0, diamond_face_center_z))
    left_turn = Location(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, MOTOR_BAY_TOP_VIEW_LEFT_TURN_DEG),
    )
    world_to_pivot = Location((0.0, 0.0, -diamond_face_center_z))
    return pivot_to_world * left_turn * world_to_pivot


def _motor_bay_two_stage_location(centerline_world_z_mm: float) -> Location:
    """Compose the screw-axis roll followed by the top-view left turn."""
    first_roll = _motor_bay_first_roll_location(centerline_world_z_mm)
    second_turn = _motor_bay_top_view_left_turn(centerline_world_z_mm)
    return second_turn * first_roll


def _motor_bay_final_clockwise_turn(centerline_world_z_mm: float) -> Location:
    """Turn clockwise about the bay's vertical centerline in top view."""
    pivot_to_world = Location((0.0, 0.0, centerline_world_z_mm))
    clockwise_turn = Location(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, MOTOR_BAY_FINAL_TOP_VIEW_CLOCKWISE_TURN_DEG),
    )
    world_to_pivot = Location((0.0, 0.0, -centerline_world_z_mm))
    return pivot_to_world * clockwise_turn * world_to_pivot


def _motor_bay_three_stage_location(centerline_world_z_mm: float) -> Location:
    """Compose both prior rotations and the final clockwise turn."""
    first_two_stages = _motor_bay_two_stage_location(centerline_world_z_mm)
    final_turn = _motor_bay_final_clockwise_turn(centerline_world_z_mm)
    return final_turn * first_two_stages


def fork_face_down_location() -> Location:
    """Place the fork with its longitudinal +X direction pointing down."""
    return Location((0.0, 0.0, 0.0), FORK_FACE_DOWN_ROTATION_XYZ_DEG)


def motor_bay_approved_location() -> Location:
    """Return the flush placement after all three requested rotations."""
    return _motor_bay_three_stage_location(
        MOTOR_BAY_APPROVED_CENTERLINE_WORLD_Z_MM
    )


def motor_bay_fused_location() -> Location:
    """Return the printable three-stage placement with seating overlap."""
    return _motor_bay_three_stage_location(
        MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM
    )


def placed_fork() -> Shape:
    """Return the new oval-ended fork in the approved face-down pose."""
    return gen_servo_output_fork().moved(fork_face_down_location())


def placed_motor_bay(*, printable: bool = True) -> Shape:
    """Return the unchanged bay in either the approved or printable pose."""
    location = (
        motor_bay_fused_location()
        if printable
        else motor_bay_approved_location()
    )
    return gen_motor_bay().moved(location)


def placed_installed_servo(*, printable: bool = True) -> Shape:
    """Return the exact catalog ST3215 seated in the placed motor bay."""
    bay_location = (
        motor_bay_fused_location()
        if printable
        else motor_bay_approved_location()
    )
    servo_pose = bay_location * st3215_installed_location()
    return import_step(ST3215_SERVO_STEP).moved(servo_pose)


def motor_insertion_keep_clear() -> Shape:
    """Return the full motor-entry envelope protected from reinforcement."""
    keep_clear_min_x = MOTOR_BAY_OPEN_WORLD_X_MM - 20.0
    keep_clear_max_x = (
        MOTOR_BAY_OPEN_WORLD_X_MM + HIP_SIDE_WEB_OPENING_SETBACK_MM
    )
    keep_clear = Box(
        keep_clear_max_x - keep_clear_min_x,
        MOTOR_BAY_INNER_WORLD_Y_MM,
        MOTOR_BAY_INNER_WORLD_Z_MM,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(
        Location(
            (
                (keep_clear_min_x + keep_clear_max_x) / 2.0,
                0.0,
                MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM,
            )
        )
    )
    keep_clear.label = "st3215_motor_insertion_keep_clear"
    return keep_clear


def _side_web_profile_points(y_sign: float) -> list[tuple[float, float]]:
    """Return a CCW YZ profile for one tapered reinforcement web."""
    if y_sign not in (-1.0, 1.0):
        raise ValueError("y_sign must be -1 or +1")

    positive_points = [
        (HIP_SIDE_WEB_FORK_INNER_Y_MM, HIP_SIDE_WEB_FORK_BOTTOM_Z_MM),
        (HIP_SIDE_WEB_FORK_OUTER_Y_MM, HIP_SIDE_WEB_FORK_BOTTOM_Z_MM),
        (HIP_SIDE_WEB_BAY_OUTER_Y_MM, HIP_SIDE_WEB_BAY_TOE_Z_MM),
        (HIP_SIDE_WEB_BAY_OUTER_Y_MM, HIP_SIDE_WEB_TOP_Z_MM),
        (HIP_SIDE_WEB_BAY_INNER_Y_MM, HIP_SIDE_WEB_TOP_Z_MM),
        (HIP_SIDE_WEB_BAY_INNER_Y_MM, HIP_SIDE_WEB_CAVITY_KNEE_Z_MM),
        (HIP_SIDE_WEB_FORK_INNER_Y_MM, HIP_SIDE_WEB_FORK_TOP_Z_MM),
    ]
    if y_sign > 0.0:
        return positive_points
    # Mirroring reverses winding, so reverse the point order at the same time.
    return [(-y, z) for y, z in reversed(positive_points)]


def _side_web_screw_clearance(y_sign: float) -> Shape:
    """Return clearance for the existing lower side screw-access openings."""
    local_mount_x = (
        SERVO_BOTTOM_MOUNT_HOLE_X_MM
        if y_sign > 0.0
        else SERVO_TOP_MOUNT_HOLE_X_MM
    )
    world_mount_x = MOTOR_BAY_DATUM_WORLD_X_MM + local_mount_x
    web_depth_y = (
        2.0 * HIP_SIDE_WEB_BAY_OUTER_Y_MM
        + 2.0 * HIP_SIDE_WEB_SCREW_CLEARANCE_OVERTRAVEL_MM
    )
    hole_z = (
        MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM
        - SERVO_MOUNT_HOLE_Y_SPACING_MM / 2.0
    )
    cutter = Cylinder(
        HIP_SIDE_WEB_SCREW_CLEARANCE_DIAMETER_MM / 2.0,
        web_depth_y,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).rotate(Axis.X, 90.0)
    return cutter.moved(Location((world_mount_x, 0.0, hole_z)))


def make_side_reinforcement_webs() -> Shape:
    """Return paired external webs joining the bay to both fork shoulders."""
    web_min_x = (
        MOTOR_BAY_OPEN_WORLD_X_MM + HIP_SIDE_WEB_OPENING_SETBACK_MM
    )
    web_max_x = (
        MOTOR_BAY_CLOSED_WORLD_X_MM + HIP_SIDE_WEB_CLOSED_END_OVERLAP_MM
    )
    web_plane = Plane(
        origin=(web_min_x, 0.0, 0.0),
        x_dir=(0.0, 1.0, 0.0),
        z_dir=(1.0, 0.0, 0.0),
    )

    webs = []
    for y_sign in (-1.0, 1.0):
        with BuildSketch(web_plane) as profile:
            Polygon(*_side_web_profile_points(y_sign))
        web = extrude(profile.sketch, amount=web_max_x - web_min_x)
        web = web - _side_web_screw_clearance(y_sign)
        webs.append(_one_valid_solid(web, f"st3215_hip_side_web_{y_sign:+.0f}"))

    reinforced = webs[0].fuse(webs[1])
    reinforced.label = "st3215_hip_side_reinforcement_webs"
    return reinforced


def make_solid_lower_motor_bay_wall() -> Shape:
    """Backfill every diamond opening in the hip's lower motor-bay wall."""
    margin = float(HIP_LOWER_WALL_FILL_MARGIN_MM)
    columns_x = tuple(float(value) for value in VENT_DIAMOND_COLUMNS_X_MM)
    rows_z = tuple(float(value) for value in VENT_DIAMOND_ROWS_Z_MM)
    diamond_width_x = float(VENT_DIAMOND_WIDTH_X_MM)
    diamond_height_z = float(VENT_DIAMOND_HEIGHT_Z_MM)
    if min(margin, diamond_width_x, diamond_height_z) <= 0.0:
        raise ValueError("Lower-wall fill dimensions must be positive")

    fill_min_x = min(columns_x) - diamond_width_x / 2.0 - margin
    fill_max_x = max(columns_x) + diamond_width_x / 2.0 + margin
    fill_min_z = min(rows_z) - diamond_height_z / 2.0 - margin
    fill_max_z = max(rows_z) + diamond_height_z / 2.0 + margin
    inner_local_y = MOTOR_BAY_OUTER_Y_MM - 2.0 * SOCKET_WALL_MM
    lower_wall_center_local_y = -(
        MOTOR_BAY_OUTER_Y_MM + inner_local_y
    ) / 4.0

    fill = Box(
        fill_max_x - fill_min_x,
        SOCKET_WALL_MM,
        fill_max_z - fill_min_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(
        Location(
            (
                (fill_min_x + fill_max_x) / 2.0,
                lower_wall_center_local_y,
                (fill_min_z + fill_max_z) / 2.0,
            )
        )
    )
    placed_fill = fill.moved(motor_bay_fused_location())
    placed_fill.label = "st3215_hip_solid_lower_motor_bay_wall"
    return placed_fill


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def gen_step() -> Shape:
    """Return the STEP-ready, fused ST3215 hip component."""
    fused = placed_fork().fuse(
        placed_motor_bay(printable=True),
        make_side_reinforcement_webs(),
        make_solid_lower_motor_bay_wall(),
    )
    return _one_valid_solid(fused, "st3215_hip")


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "st3215_hip.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
