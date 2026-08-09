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

The bay's lower broad wall seats into the oval tip by one wall thickness,
creating a strong union without entering the servo cavity.
"""

from __future__ import annotations

from pathlib import Path

from build123d import Location, Shape, import_step

from drobot_cad.parts.st3215_motor_bay import (
    SOCKET_CLEARANCE_Y_PER_SIDE_MM,
    SOCKET_LENGTH_X_MM,
    SOCKET_WALL_MM,
    ST3215_CATALOG_WIDTH_Z_MM,
    ST3215_SERVO_STEP,
    st3215_installed_location,
)
from drobot_cad.parts.st3215_motor_bay import gen_step as gen_motor_bay
from drobot_cad.parts.st3215_servo_output_fork import (
    ROOT_EXTENSION_CENTER_X_MM,
    ROOT_EXTENSION_CENTER_Z_MM,
    ROOT_EXTENSION_RADIUS_X_MM,
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
    fused = placed_fork().fuse(placed_motor_bay(printable=True))
    return _one_valid_solid(fused, "st3215_hip")


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "st3215_hip.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
