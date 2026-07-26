"""Four complete ST3215 legs mounted to the printable quadruped body."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from build123d import Color, Location, Shape, import_step

from robot_cad.assembly import robot_leg
from robot_cad.parts import (
    quadruped_body,
    quadruped_body_lid,
    quadruped_electronics_tray,
    st3215_hip,
    st3215_hip_body_mount,
    st3215_motor_bay,
    upper_arm,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LegMountSpec:
    name: str
    center_x_mm: float
    side_sign: float
    body_hip_angle_deg: float
    hip_flexion_angle_deg: float
    knee_angle_deg: float


@dataclass(frozen=True)
class QuadrupedAssemblySpec:
    name: str = "four_leg_st3215_quadruped"
    component_order: tuple[str, ...] = (
        "body_base",
        "electronics_tray",
        "body_lid",
        "front_left_leg",
        "rear_left_leg",
        "front_right_leg",
        "rear_right_leg",
    )


# Mirrored flexion/knee signs keep the front and rear legs separated in the
# static review stance.  These are preview poses, not validated joint limits.
LEG_MOUNT_SPECS = (
    LegMountSpec(
        "front_left",
        quadruped_body.HIP_MOUNT_CENTER_X_MM,
        1.0,
        15.0,
        -20.0,
        -35.0,
    ),
    LegMountSpec(
        "rear_left",
        -quadruped_body.HIP_MOUNT_CENTER_X_MM,
        1.0,
        15.0,
        20.0,
        35.0,
    ),
    LegMountSpec(
        "front_right",
        quadruped_body.HIP_MOUNT_CENTER_X_MM,
        -1.0,
        15.0,
        20.0,
        35.0,
    ),
    LegMountSpec(
        "rear_right",
        -quadruped_body.HIP_MOUNT_CENTER_X_MM,
        -1.0,
        15.0,
        -20.0,
        -35.0,
    ),
)
FINAL_ASSEMBLY_SPEC = QuadrupedAssemblySpec()

LEG_COMPONENT_COLORS = {
    "body_side_hip_mount": Color(0.72, 0.74, 0.78),
    "hip_abduction_st3215_servo": Color(0.22, 0.25, 0.30),
    "st3215_hip": Color(0.92, 0.48, 0.12),
    "hip_flexion_st3215_servo": Color(0.22, 0.25, 0.30),
    "proximal_upper_arm": Color(0.20, 0.55, 0.82),
    "knee_st3215_servo": Color(0.22, 0.25, 0.30),
    "distal_upper_arm": Color(0.25, 0.70, 0.48),
}


def body_mount_location(spec: LegMountSpec) -> Location:
    """Return a rolled hip-mount pose with its body face flush to a side wall."""
    body_side_face_local_x = (
        st3215_hip_body_mount.MOUNTING_PLATE_BODY_SIDE_FACE_X_MM
    )
    origin_y = spec.side_sign * (
        quadruped_body.BODY_WIDTH_Y_MM / 2.0
        - body_side_face_local_x
    )
    translation = Location(
        (
            spec.center_x_mm,
            origin_y,
            quadruped_body.HIP_MOUNT_CENTER_Z_MM,
        )
    )
    side_yaw = Location((0.0, 0.0, 0.0), (0.0, 0.0, 90.0 * spec.side_sign))
    approved_hip_roll = Location((0.0, 0.0, 0.0), (90.0, 0.0, 0.0))
    return translation * side_yaw * approved_hip_roll


def leg_component_locations(spec: LegMountSpec) -> dict[str, Location]:
    """Return one leg's seven world poses relative to the body."""
    local_locations = robot_leg.component_locations(
        spec.body_hip_angle_deg,
        spec.hip_flexion_angle_deg,
        spec.knee_angle_deg,
    )
    local_root = local_locations["body_side_hip_mount"]
    world_from_leg_preview = body_mount_location(spec) * local_root.inverse()
    return {
        label: world_from_leg_preview * local_location
        for label, local_location in local_locations.items()
    }


def _leg_prototypes() -> dict[str, Shape]:
    """Create one geometry prototype for every distinct leg role."""
    servo = import_step(st3215_motor_bay.ST3215_SERVO_STEP)
    return {
        "body_side_hip_mount": st3215_hip_body_mount.gen_step(),
        "hip_abduction_st3215_servo": servo,
        "st3215_hip": st3215_hip.gen_step(),
        "hip_flexion_st3215_servo": servo,
        "proximal_upper_arm": upper_arm.gen_step(),
        "knee_st3215_servo": servo,
        "distal_upper_arm": upper_arm.gen_step(),
    }


def _placed_leg_children(
    spec: LegMountSpec,
    prototypes: dict[str, Shape],
) -> list[Shape]:
    """Return labeled children for one nested leg module."""
    from cadpy.assembly import label_shape

    locations = leg_component_locations(spec)
    children = []
    for label in robot_leg.FINAL_ASSEMBLY_SPEC.component_order:
        child = prototypes[label].moved(locations[label])
        label_shape(child, label, color=LEG_COMPONENT_COLORS[label])
        children.append(child)
    return children


def gen_step():
    """Return the labeled body, internals, and four complete leg modules."""
    from cadpy.assembly import AssemblyHelper

    asm = AssemblyHelper(FINAL_ASSEMBLY_SPEC.name)
    asm.add(
        quadruped_body.gen_step(),
        "body_base",
        color=Color(0.16, 0.23, 0.32),
    )
    asm.add(
        quadruped_electronics_tray.gen_step().moved(
            Location((0.0, 0.0, quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM))
        ),
        "electronics_tray",
        color=Color(0.95, 0.70, 0.18),
    )
    asm.add(
        quadruped_body_lid.gen_step().moved(
            Location((0.0, 0.0, quadruped_body.BODY_BASE_HEIGHT_Z_MM))
        ),
        "body_lid",
        color=Color(0.38, 0.48, 0.60),
    )

    prototypes = _leg_prototypes()
    for leg_spec in LEG_MOUNT_SPECS:
        asm.add_module(
            f"{leg_spec.name}_leg",
            _placed_leg_children(leg_spec, prototypes),
        )
    return asm.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "quadruped_robot.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
