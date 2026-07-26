"""Complete body-side hip and two-link ST3215 leg assembly.

The non-printable assembly follows the physical chain from the robot body:

    body hip mount -> ST3215 -> hip -> ST3215 -> upper arm
    -> ST3215 -> upper arm

Every moving child is located by mating its installed servo output axis to the
preceding fork axis.  The three preview angles only make the full chain easier
to inspect; they do not alter any printable component.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from build123d import Color, Location, import_step

from robot_cad.assembly import robot_arm
from robot_cad.parts import (
    st3215_hip,
    st3215_hip_body_mount,
    st3215_motor_bay,
    st3215_servo_output_fork,
    upper_arm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BODY_HIP_PREVIEW_ANGLE_DEG = 15.0
HIP_FLEXION_PREVIEW_ANGLE_DEG = 20.0
KNEE_PREVIEW_ANGLE_DEG = robot_arm.ELBOW_PREVIEW_ANGLE_DEG

# Exact shaft frame in the immutable step.parts ST3215 model.  Local +Z is
# aligned with the servo's dual-output shaft direction.
ST3215_CATALOG_OUTPUT_AXIS_LOCATION = Location(
    (-25.5, -7.475, 0.0),
    (-90.0, 0.0, 0.0),
)


@dataclass(frozen=True)
class RobotLegAssemblySpec:
    name: str = "complete_st3215_robot_leg"
    component_order: tuple[str, ...] = (
        "body_side_hip_mount",
        "hip_abduction_st3215_servo",
        "st3215_hip",
        "hip_flexion_st3215_servo",
        "proximal_upper_arm",
        "knee_st3215_servo",
        "distal_upper_arm",
    )
    body_hip_preview_angle_deg: float = BODY_HIP_PREVIEW_ANGLE_DEG
    hip_flexion_preview_angle_deg: float = HIP_FLEXION_PREVIEW_ANGLE_DEG
    knee_preview_angle_deg: float = KNEE_PREVIEW_ANGLE_DEG


FINAL_ASSEMBLY_SPEC = RobotLegAssemblySpec()


def body_mount_fork_axis_location() -> Location:
    """Return the body-side mount fork axis in body-mount coordinates."""
    return st3215_servo_output_fork.output_axis_location_local()


def body_mount_preview_location() -> Location:
    """Orient the root so the approved hip pose hangs down at zero degrees."""
    return (
        hip_installed_servo_axis_location()
        * body_mount_fork_axis_location().inverse()
    )


def hip_installed_servo_axis_location() -> Location:
    """Return the installed hip servo shaft frame in hip coordinates."""
    return (
        st3215_hip.motor_bay_fused_location()
        * st3215_motor_bay.st3215_installed_location()
        * ST3215_CATALOG_OUTPUT_AXIS_LOCATION
    )


def hip_distal_fork_axis_location() -> Location:
    """Return the downward-facing hip fork axis in hip coordinates."""
    return (
        st3215_hip.fork_face_down_location()
        * st3215_servo_output_fork.output_axis_location_local()
    )


def upper_arm_installed_servo_axis_location() -> Location:
    """Return an upper arm's installed servo shaft frame."""
    return upper_arm.st3215_preview_location() * ST3215_CATALOG_OUTPUT_AXIS_LOCATION


def upper_arm_distal_fork_axis_location() -> Location:
    """Return an upper arm's distal fork axis frame."""
    return upper_arm.distal_fork_axis_location()


def mated_child_location(
    parent_location: Location,
    parent_axis_location: Location,
    child_axis_location: Location,
    angle_deg: float,
) -> Location:
    """Mate child and parent shaft frames, then rotate around their common axis."""
    joint_rotation = Location((0.0, 0.0, 0.0), (0.0, 0.0, float(angle_deg)))
    return (
        parent_location
        * parent_axis_location
        * joint_rotation
        * child_axis_location.inverse()
    )


def component_locations(
    body_hip_angle_deg: float = BODY_HIP_PREVIEW_ANGLE_DEG,
    hip_flexion_angle_deg: float = HIP_FLEXION_PREVIEW_ANGLE_DEG,
    knee_angle_deg: float = KNEE_PREVIEW_ANGLE_DEG,
) -> dict[str, Location]:
    """Return all seven component poses in physical body-to-foot order."""
    body_mount_pose = body_mount_preview_location()
    hip_pose = mated_child_location(
        body_mount_pose,
        body_mount_fork_axis_location(),
        hip_installed_servo_axis_location(),
        body_hip_angle_deg,
    )
    proximal_arm_pose = mated_child_location(
        hip_pose,
        hip_distal_fork_axis_location(),
        upper_arm_installed_servo_axis_location(),
        hip_flexion_angle_deg,
    )
    distal_arm_pose = mated_child_location(
        proximal_arm_pose,
        upper_arm_distal_fork_axis_location(),
        upper_arm_installed_servo_axis_location(),
        knee_angle_deg,
    )

    hip_servo_pose = (
        hip_pose
        * st3215_hip.motor_bay_fused_location()
        * st3215_motor_bay.st3215_installed_location()
    )
    proximal_servo_pose = proximal_arm_pose * upper_arm.st3215_preview_location()
    distal_servo_pose = distal_arm_pose * upper_arm.st3215_preview_location()

    return {
        "body_side_hip_mount": body_mount_pose,
        "hip_abduction_st3215_servo": hip_servo_pose,
        "st3215_hip": hip_pose,
        "hip_flexion_st3215_servo": proximal_servo_pose,
        "proximal_upper_arm": proximal_arm_pose,
        "knee_st3215_servo": distal_servo_pose,
        "distal_upper_arm": distal_arm_pose,
    }


def gen_step():
    """Return the labeled seven-occurrence, three-servo leg assembly."""
    from cadpy.assembly import AssemblyHelper

    locations = component_locations()
    body_mount = st3215_hip_body_mount.gen_step()
    hip = st3215_hip.gen_step()
    proximal_arm = upper_arm.gen_step()
    distal_arm = upper_arm.gen_step()
    servo = import_step(st3215_motor_bay.ST3215_SERVO_STEP)

    components = {
        "body_side_hip_mount": body_mount,
        "hip_abduction_st3215_servo": servo,
        "st3215_hip": hip,
        "hip_flexion_st3215_servo": servo,
        "proximal_upper_arm": proximal_arm,
        "knee_st3215_servo": servo,
        "distal_upper_arm": distal_arm,
    }
    colors = {
        "body_side_hip_mount": Color(0.72, 0.74, 0.78),
        "hip_abduction_st3215_servo": Color(0.22, 0.25, 0.30),
        "st3215_hip": Color(0.92, 0.48, 0.12),
        "hip_flexion_st3215_servo": Color(0.22, 0.25, 0.30),
        "proximal_upper_arm": Color(0.20, 0.55, 0.82),
        "knee_st3215_servo": Color(0.22, 0.25, 0.30),
        "distal_upper_arm": Color(0.25, 0.70, 0.48),
    }

    asm = AssemblyHelper(FINAL_ASSEMBLY_SPEC.name)
    for label in FINAL_ASSEMBLY_SPEC.component_order:
        asm.add(
            components[label].moved(locations[label]),
            label,
            color=colors[label],
        )
    return asm.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "robot_leg.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
