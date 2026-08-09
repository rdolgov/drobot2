"""Two complete upper arms connected by an exact ST3215 servo at the elbow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from build123d import Location, import_step

from drobot_cad.interfaces import UPPER_ARM_INTERFACES, InterfaceFrame
from drobot_cad.parts.st3215_motor_bay import ST3215_SERVO_STEP
from drobot_cad.parts.upper_arm import gen_step as gen_upper_arm
from drobot_cad.parts.upper_arm import st3215_preview_location

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ELBOW_PREVIEW_ANGLE_DEG = 35.0
ELBOW_MINIMUM_DEG = -90.0
ELBOW_MAXIMUM_DEG = 90.0


@dataclass(frozen=True)
class RevoluteConnection:
    name: str
    parent_component: str
    child_component: str
    parent_frame: InterfaceFrame
    child_frame: InterfaceFrame
    minimum_deg: float
    maximum_deg: float


@dataclass(frozen=True)
class RobotArmAssemblySpec:
    root_component: str = "upper_arm_link_1"
    child_component: str = "upper_arm_link_2"
    base_servo_component: str = "st3215_base_servo"
    elbow_servo_component: str = "st3215_elbow_servo"
    preview_angle_deg: float = ELBOW_PREVIEW_ANGLE_DEG


ELBOW_CONNECTION = RevoluteConnection(
    name="st3215_elbow_joint",
    parent_component="upper_arm_link_1",
    child_component="upper_arm_link_2",
    parent_frame=UPPER_ARM_INTERFACES["frame_distal_fork_axis"],
    child_frame=UPPER_ARM_INTERFACES["frame_st3215_output_axis"],
    minimum_deg=ELBOW_MINIMUM_DEG,
    maximum_deg=ELBOW_MAXIMUM_DEG,
)
FINAL_ASSEMBLY_SPEC = RobotArmAssemblySpec()


def child_arm_location(angle_deg: float = ELBOW_PREVIEW_ANGLE_DEG) -> Location:
    """Align link 2's installed servo output axis to link 1's distal fork."""
    angle = float(angle_deg)
    if not ELBOW_MINIMUM_DEG <= angle <= ELBOW_MAXIMUM_DEG:
        raise ValueError(
            f"Elbow angle must be within [{ELBOW_MINIMUM_DEG}, "
            f"{ELBOW_MAXIMUM_DEG}] degrees"
        )

    parent_axis = Location(ELBOW_CONNECTION.parent_frame.xyz_mm, (0.0, 0.0, angle))
    child_axis = Location(ELBOW_CONNECTION.child_frame.xyz_mm)
    return parent_axis * child_axis.inverse()


def elbow_servo_location(angle_deg: float = ELBOW_PREVIEW_ANGLE_DEG) -> Location:
    """Return the catalog servo pose for the elbow-connected second link."""
    return child_arm_location(angle_deg) * st3215_preview_location()


def gen_step():
    """Return the labeled, non-printable two-link final assembly."""
    from cadpy.assembly import AssemblyHelper

    root_arm = gen_upper_arm()
    child_pose = child_arm_location()
    catalog_servo = import_step(ST3215_SERVO_STEP)

    asm = AssemblyHelper("dual_upper_arm_st3215_final_assembly")
    asm.add(root_arm, FINAL_ASSEMBLY_SPEC.root_component)
    asm.add(
        catalog_servo.moved(st3215_preview_location()),
        FINAL_ASSEMBLY_SPEC.base_servo_component,
    )
    asm.add(
        root_arm.moved(child_pose),
        FINAL_ASSEMBLY_SPEC.child_component,
    )
    asm.add(
        catalog_servo.moved(elbow_servo_location()),
        FINAL_ASSEMBLY_SPEC.elbow_servo_component,
    )
    return asm.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "robot_arm.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
