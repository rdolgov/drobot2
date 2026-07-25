"""Position-check assembly for the perpendicular ST3215 hip link.

This file intentionally stops at an assembly-level orientation review.  The
fork and motor bay remain separate labeled components until the pose is
confirmed, so a printable transition cannot accidentally lock in the wrong
handedness.

Hip review coordinates:
    - +Z is up, +X is right, and +Y is the shared transverse centerline.
    - The fork local +X direction points down along global -Z.
    - The motor bay extends left: its local -X direction follows global -X.
    - The motor bay is rolled +90 degrees about its local X centerline between
      the screw-access holes.
    - The rolled bay sits on top of the fork with both left edges aligned.
"""

from __future__ import annotations

from pathlib import Path

from build123d import Location, Shape
from cadpy.assembly import AssemblyHelper

from robot_cad.parts import st3215_hip

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FORK_FACE_DOWN_ROTATION_XYZ_DEG = st3215_hip.FORK_FACE_DOWN_ROTATION_XYZ_DEG
FORK_ROOT_LEFT_EDGE_WORLD_X_MM = st3215_hip.FORK_ROOT_LEFT_EDGE_WORLD_X_MM
MOTOR_BAY_DATUM_WORLD_X_MM = st3215_hip.MOTOR_BAY_DATUM_WORLD_X_MM
MOTOR_BAY_CENTERLINE_WORLD_Z_MM = (
    st3215_hip.MOTOR_BAY_APPROVED_CENTERLINE_WORLD_Z_MM
)
MOTOR_BAY_LEFT_ROLL_ROTATION_XYZ_DEG = (
    st3215_hip.MOTOR_BAY_LEFT_ROLL_ROTATION_XYZ_DEG
)


def fork_face_down_location() -> Location:
    """Return the approved face-down fork location."""
    return st3215_hip.fork_face_down_location()


def motor_bay_facing_left_location() -> Location:
    """Place the left-aligned, rolled motor bay on top of the fork."""
    return st3215_hip.motor_bay_approved_location()


def placed_fork() -> Shape:
    """Return the fork in the review pose."""
    return st3215_hip.placed_fork()


def placed_motor_bay() -> Shape:
    """Return the bay in the approved flush review pose."""
    return st3215_hip.placed_motor_bay(printable=False)


def placed_installed_servo() -> Shape:
    """Return the exact ST3215 seated in the approved bay pose."""
    return st3215_hip.placed_installed_servo(printable=False)


def gen_step():
    """Return a labeled, non-printable hip orientation-review assembly."""
    assembly = AssemblyHelper("hip_orientation_preview")
    assembly.add(
        placed_fork(),
        "st3215_servo_output_fork_facing_down",
    )
    assembly.add(
        placed_motor_bay(),
        "st3215_rear_motor_bay_facing_left_rolled_90",
    )
    return assembly.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "hip_orientation_preview.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
