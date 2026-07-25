"""Printable ST3215 hip made from the fork core and motor bay.

The component preserves the user-approved orientation:
    - global +Z is up and global +X is right
    - the fork local +X direction points down along global -Z
    - the motor bay extends left along global -X
    - the bay is rolled +90 degrees about its local X centerline between the
      screw-access holes
    - the bay and fork root left edges align

The hip deliberately retains the previously approved positive-X fork core.
The reusable fork's optional negative-X fusion envelope is excluded here
because its 30 mm extension maps upward in this pose and intersects the
installed servo.  The approved flush pose is retained as a public placement
datum.  The printable part lowers the bay by a small, named seating overlap so
the two source components form one robust B-rep solid.
"""

from __future__ import annotations

from pathlib import Path

from build123d import Location, Shape, import_step

from robot_cad.parts.st3215_motor_bay import (
    SOCKET_CLEARANCE_Y_PER_SIDE_MM,
    SOCKET_LENGTH_X_MM,
    SOCKET_WALL_MM,
    ST3215_CATALOG_WIDTH_Z_MM,
    ST3215_SERVO_STEP,
    st3215_installed_location,
)
from robot_cad.parts.st3215_motor_bay import gen_step as gen_motor_bay
from robot_cad.parts.st3215_servo_output_fork import (
    gen_core_step as gen_servo_output_fork,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORK_FACE_DOWN_ROTATION_XYZ_DEG = (0.0, 90.0, 0.0)
MOTOR_BAY_LEFT_ROLL_ROTATION_XYZ_DEG = (90.0, 0.0, 0.0)

# The rotated fork root spans X=-31.7..31.7 mm.  The bay is 16 mm long
# toward local -X, so placing its X=0 datum at -15.7 mm aligns both left edges.
FORK_ROOT_LEFT_EDGE_WORLD_X_MM = -31.7
MOTOR_BAY_DATUM_WORLD_X_MM = (
    FORK_ROOT_LEFT_EDGE_WORLD_X_MM + SOCKET_LENGTH_X_MM
)

# A +90-degree roll maps the bay's local Y width into world Z.  The approved
# centerline height seats the lower broad wall exactly on the fork root plane.
MOTOR_BAY_OUTER_Y_MM = (
    ST3215_CATALOG_WIDTH_Z_MM
    + 2.0 * SOCKET_CLEARANCE_Y_PER_SIDE_MM
    + 2.0 * SOCKET_WALL_MM
)
MOTOR_BAY_APPROVED_CENTERLINE_WORLD_Z_MM = MOTOR_BAY_OUTER_Y_MM / 2.0

# Hidden seating depth used only to make the approved touching parts one solid.
HIP_JOIN_OVERLAP_MM = 0.2
MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM = (
    MOTOR_BAY_APPROVED_CENTERLINE_WORLD_Z_MM - HIP_JOIN_OVERLAP_MM
)


def fork_face_down_location() -> Location:
    """Place the fork with its longitudinal +X direction pointing down."""
    return Location((0.0, 0.0, 0.0), FORK_FACE_DOWN_ROTATION_XYZ_DEG)


def motor_bay_approved_location() -> Location:
    """Return the approved flush motor-bay placement."""
    return Location(
        (
            MOTOR_BAY_DATUM_WORLD_X_MM,
            0.0,
            MOTOR_BAY_APPROVED_CENTERLINE_WORLD_Z_MM,
        ),
        MOTOR_BAY_LEFT_ROLL_ROTATION_XYZ_DEG,
    )


def motor_bay_fused_location() -> Location:
    """Return the printable placement with the hidden seating overlap."""
    return Location(
        (
            MOTOR_BAY_DATUM_WORLD_X_MM,
            0.0,
            MOTOR_BAY_FUSED_CENTERLINE_WORLD_Z_MM,
        ),
        MOTOR_BAY_LEFT_ROLL_ROTATION_XYZ_DEG,
    )


def placed_fork() -> Shape:
    """Return the collision-safe legacy fork core in the approved pose."""
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
