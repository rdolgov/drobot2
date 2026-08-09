"""Generate the wall-mounted one-leg URDF used for Isaac range testing.

The fixture reproduces the physical bench arrangement: the printable
ST3215 hip body mount is bolted to a vertical wall and the three-servo leg
hangs from it.  Moving-link geometry, frames, masses, and inertias are shared
with the quadruped URDF so this experiment isolates fixture and controller
effects instead of introducing a second leg model.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from drobot_cad.urdf import quadruped_robot as quadruped

ROBOT_NAME = "st3215_one_leg_wall_testbed"

# Wall/world convention: +X is robot-forward along the wall, +Y points away
# from the wall, and +Z points up.  The hip-abduction axis is at the origin.
ROOT_JOINT_RPY_RAD = (math.pi / 2.0, 0.0, math.pi / 2.0)
PITCH_AXIS = (0.0, 0.0, -1.0)

# Snapshot of the physically exercised local testbed configuration on
# 2026-07-28.  These are not automatically promoted to the complete robot:
# cable routing, body/leg interference, and support loads differ there.
PHYSICALLY_EXERCISED_LIMITS_DEG = {
    "hip_abduction": (-45.0, 45.0),
    "hip_flexion": (-90.0, 90.0),
    "knee": (-120.0, 120.0),
}
HARDWARE_ENCODER_DIRECTIONS = {
    "hip_abduction": 1,
    "hip_flexion": -1,
    "knee": -1,
}
HARDWARE_CENTER_TICK = 2048
HARDWARE_TORQUE_LIMIT_FRACTION = 0.30
NOMINAL_HARDWARE_EFFORT_CAP_NM = (
    HARDWARE_TORQUE_LIMIT_FRACTION * quadruped.SERVO_STALL_TORQUE_NM
)

# The quadruped places the body-side mount 53.084989 mm behind the root axis.
# Its local plate back face is another 32 mm behind that point, so this puts
# the wall surface exactly flush to the plate back face.
BODY_MOUNT_ORIGIN_M = (0.0, -0.053084989, 0.0)
WALL_SURFACE_Y_M = -0.085084989
WALL_THICKNESS_M = 0.020
WALL_SIZE_M = (0.50, WALL_THICKNESS_M, 0.60)
WALL_CENTER_M = (
    0.0,
    WALL_SURFACE_Y_M - WALL_THICKNESS_M / 2.0,
    -0.12,
)

# The fixed wall is only a simulation fixture.  Its inertial values are a
# uniform 5 kg box approximation and do not affect the fixed-base articulation.
FIXTURE_MASS_KG = 5.0
FIXTURE_INERTIA_KG_M2 = (
    FIXTURE_MASS_KG / 12.0 * (WALL_SIZE_M[1] ** 2 + WALL_SIZE_M[2] ** 2),
    0.0,
    0.0,
    FIXTURE_MASS_KG / 12.0 * (WALL_SIZE_M[0] ** 2 + WALL_SIZE_M[2] ** 2),
    0.0,
    FIXTURE_MASS_KG / 12.0 * (WALL_SIZE_M[0] ** 2 + WALL_SIZE_M[1] ** 2),
)


def _add_box_visual(link, name: str, size, xyz, material: str):
    visual = ET.SubElement(link, "visual", {"name": name})
    quadruped._origin(visual, xyz)
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "box", {"size": quadruped._numbers(size)})
    ET.SubElement(visual, "material", {"name": material})


def _add_fixture(robot):
    wall = ET.SubElement(robot, "link", {"name": "wall_link"})
    quadruped._add_inertial(
        wall,
        FIXTURE_MASS_KG,
        WALL_CENTER_M,
        FIXTURE_INERTIA_KG_M2,
    )
    _add_box_visual(
        wall,
        "fixed_vertical_wall",
        WALL_SIZE_M,
        WALL_CENTER_M,
        "fixture_gray",
    )
    quadruped._add_mesh_visual(
        wall,
        "exact_printable_hip_body_mount",
        "../stl/st3215_hip_body_mount.stl",
        "mount_light",
        xyz=BODY_MOUNT_ORIGIN_M,
        rpy=ROOT_JOINT_RPY_RAD,
    )
    quadruped._add_box_collision(
        wall,
        "vertical_wall_collision",
        WALL_SIZE_M,
        WALL_CENTER_M,
    )


def _add_leg(robot):
    hip = ET.SubElement(robot, "link", {"name": "hip_link"})
    quadruped._add_inertial(
        hip,
        quadruped.HIP_LINK_MASS_KG,
        quadruped.HIP_LINK_COM_M,
        quadruped.HIP_LINK_INERTIA_KG_M2,
    )
    quadruped._add_mesh_visual(
        hip,
        "printable_hip",
        "../stl/st3215_hip.stl",
        "hip_orange",
        xyz=quadruped.HIP_VISUAL_XYZ_M,
        rpy=quadruped.HIP_VISUAL_RPY_RAD,
    )
    quadruped._add_mesh_visual(
        hip,
        "exact_st3215_servo",
        quadruped.SERVO_VISUAL_MESH,
        "servo_black",
        xyz=quadruped.SERVO_MESH_XYZ_M,
        rpy=quadruped.SERVO_MESH_RPY_RAD,
    )
    quadruped._add_box_collision(
        hip,
        "hip_printable_proxy",
        quadruped.HIP_PRINTABLE_COLLISION_SIZE_M,
        quadruped.HIP_PRINTABLE_COLLISION_XYZ_M,
    )
    quadruped._add_box_collision(
        hip,
        "hip_servo_case_collision",
        quadruped.SERVO_BOX_SIZE_M,
        quadruped.SERVO_COLLISION_XYZ_M,
        quadruped.SERVO_COLLISION_RPY_RAD,
    )

    for role, link_name, material in (
        ("proximal", "proximal_link", "proximal_blue"),
        ("distal", "distal_link", "distal_green"),
    ):
        arm = ET.SubElement(robot, "link", {"name": link_name})
        quadruped._add_inertial(
            arm,
            quadruped.ARM_LINK_MASS_KG,
            quadruped.ARM_LINK_COM_M,
            quadruped.ARM_LINK_INERTIA_KG_M2,
        )
        quadruped._add_mesh_visual(
            arm,
            f"printable_{role}_upper_arm",
            "../stl/upper_arm.stl",
            material,
            xyz=quadruped.ARM_VISUAL_XYZ_M,
        )
        quadruped._add_mesh_visual(
            arm,
            "exact_st3215_servo",
            quadruped.SERVO_VISUAL_MESH,
            "servo_black",
            xyz=quadruped.SERVO_MESH_XYZ_M,
            rpy=quadruped.SERVO_MESH_RPY_RAD,
        )
        quadruped._add_box_collision(
            arm,
            f"{role}_arm_printable_proxy",
            quadruped.ARM_PRINTABLE_COLLISION_SIZE_M,
            quadruped.ARM_PRINTABLE_COLLISION_XYZ_M,
        )
        quadruped._add_box_collision(
            arm,
            "st3215_servo_case_collision",
            quadruped.SERVO_BOX_SIZE_M,
            quadruped.SERVO_COLLISION_XYZ_M,
            quadruped.SERVO_COLLISION_RPY_RAD,
        )

    quadruped._add_revolute_joint(
        robot,
        "hip_abduction",
        "wall_link",
        "hip_link",
        (0.0, 0.0, 0.0),
        ROOT_JOINT_RPY_RAD,
        (0.0, 0.0, 1.0),
        math.radians(PHYSICALLY_EXERCISED_LIMITS_DEG["hip_abduction"][0]),
        math.radians(PHYSICALLY_EXERCISED_LIMITS_DEG["hip_abduction"][1]),
    )
    quadruped._add_revolute_joint(
        robot,
        "hip_flexion",
        "hip_link",
        "proximal_link",
        quadruped.HIP_LINK_TO_FLEXION_XYZ_M,
        quadruped.HIP_LINK_TO_FLEXION_RPY_RAD,
        PITCH_AXIS,
        math.radians(PHYSICALLY_EXERCISED_LIMITS_DEG["hip_flexion"][0]),
        math.radians(PHYSICALLY_EXERCISED_LIMITS_DEG["hip_flexion"][1]),
    )
    quadruped._add_revolute_joint(
        robot,
        "knee",
        "proximal_link",
        "distal_link",
        (quadruped.ARM_LINK_LENGTH_M, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        PITCH_AXIS,
        math.radians(PHYSICALLY_EXERCISED_LIMITS_DEG["knee"][0]),
        math.radians(PHYSICALLY_EXERCISED_LIMITS_DEG["knee"][1]),
    )


def gen_urdf():
    """Return the complete wall-mounted one-leg URDF XML root."""
    robot = ET.Element("robot", {"name": ROBOT_NAME})
    robot.append(
        ET.Comment(
            "Fixed wall fixture with physically exercised 2026-07-28 "
            "one-leg limits; moving-link geometry is shared with the "
            "quadruped URDF."
        )
    )
    quadruped._add_material(robot, "fixture_gray", (0.36, 0.39, 0.43, 1.0))
    quadruped._add_material(robot, "mount_light", (0.65, 0.68, 0.74, 1.0))
    quadruped._add_material(robot, "servo_black", (0.025, 0.03, 0.04, 1.0))
    quadruped._add_material(robot, "hip_orange", (0.95, 0.38, 0.08, 1.0))
    quadruped._add_material(robot, "proximal_blue", (0.20, 0.62, 0.92, 1.0))
    quadruped._add_material(robot, "distal_green", (0.18, 0.72, 0.48, 1.0))
    _add_fixture(robot)
    _add_leg(robot)
    return robot


if __name__ == "__main__":
    ET.indent(tree := ET.ElementTree(gen_urdf()), space="  ")
    tree.write(
        "one_leg_wall_testbed.urdf",
        encoding="unicode",
        xml_declaration=True,
    )
