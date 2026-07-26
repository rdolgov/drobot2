"""Generate the simulation URDF for the four-leg ST3215 quadruped.

The printable CAD is authored in millimeters.  This URDF is authored in SI
units and uses joint-axis frames measured from the source CAD.  The adjacent
``specs/quadruped-urdf-ledger.md`` records the physical assumptions.

The current printable design ends in open motor forks.  Each distal link
therefore carries a conspicuous, simulation-only spherical contact proxy at
the distal fork axis.  It is not a designed or printable foot.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

ROBOT_NAME = "st3215_quadruped"
MESH_SCALE_MM_TO_M = (0.001, 0.001, 0.001)

# Exact Feetech ST-3215-C018 / Waveshare ST3215 values at 12 V.
SERVO_MASS_KG = 0.055
SERVO_RATED_TORQUE_NM = 0.980665
SERVO_STALL_TORQUE_NM = 2.941995
SERVO_NO_LOAD_VELOCITY_RAD_S = 4.712389
SERVO_BOX_SIZE_M = (0.045223408, 0.0378, 0.024723408)
SERVO_VISUAL_MESH = "../stl/st3215_servo_visual.stl"

# These safe initial ranges are assumptions pending physical cable-routing and
# CAD collision sweeps.  Only the knee's +/-90 degree range was pre-specified.
HIP_ABDUCTION_LIMIT_RAD = math.radians(25.0)
HIP_FLEXION_LIMIT_RAD = math.radians(60.0)
KNEE_LIMIT_RAD = math.radians(90.0)
JOINT_DAMPING_NM_S_RAD = 0.08
JOINT_FRICTION_NM = 0.02

# Exact zero-pose joint geometry derived from robot_cad.assembly.robot_leg and
# robot_cad.assembly.quadruped_robot.  Link frames sit on servo output axes.
HIP_LINK_TO_FLEXION_XYZ_M = (0.0286117, -0.095696689, -0.00195)
HIP_LINK_TO_FLEXION_RPY_RAD = (-math.pi / 2.0, 0.0, -math.pi / 2.0)
ARM_LINK_LENGTH_M = 0.159896689

# The actual distal part is an open servo fork.  These spheres make a first
# contact experiment possible without pretending that a printable foot exists.
VIRTUAL_FORK_TIP_RADIUS_M = 0.0125

# Solid PLA at 1240 kg/m^3, exact 55 g servos, a provisional 0.45 kg battery,
# and 0.15 kg electronics.  Fasteners, wiring, and future feet are omitted.
BASE_MASS_KG = 2.049119
HIP_LINK_MASS_KG = 0.169697
ARM_LINK_MASS_KG = 0.215137
TOTAL_ROBOT_MASS_KG = BASE_MASS_KG + 4.0 * (
    HIP_LINK_MASS_KG + 2.0 * ARM_LINK_MASS_KG
)

# Inertias are about each listed COM and expressed in its link frame.
BASE_COM_M = (0.0, 0.0, 0.046485537)
BASE_INERTIA_KG_M2 = (
    0.0132456004,
    0.0000110328,
    0.0,
    0.0097442040,
    0.0,
    0.0196972212,
)
HIP_LINK_COM_M = (0.023744808, -0.031777881, -0.001911095)
HIP_LINK_INERTIA_KG_M2 = (
    0.0001832517,
    0.0000248550,
    0.0000002392,
    0.0000718098,
    -0.0000002529,
    0.0002314034,
)
ARM_LINK_COM_M = (0.073011680, -0.000021551, -0.000924466)
ARM_LINK_INERTIA_KG_M2 = (
    0.0000747088,
    0.0000005653,
    -0.0000069897,
    0.0005598933,
    0.0000000356,
    0.0005075793,
)

HIP_VISUAL_XYZ_M = (0.0286117, -0.0426117, -0.00195)
HIP_VISUAL_RPY_RAD = (-math.pi / 2.0, 0.0, 0.0)
ARM_VISUAL_XYZ_M = (0.0948117, -0.012, 0.0)

# The exact vendor mesh transform is the inverse of its catalog output-axis
# frame.  Its transformed envelope center is different from the mesh origin,
# so the primitive collision box has its own audited center.
SERVO_MESH_XYZ_M = (0.0255, 0.0, 0.007475)
SERVO_MESH_RPY_RAD = (math.pi / 2.0, 0.0, 0.0)
SERVO_COLLISION_XYZ_M = (0.0125, 0.0, -0.001825)
SERVO_COLLISION_RPY_RAD = SERVO_MESH_RPY_RAD


@dataclass(frozen=True)
class LegSpec:
    """One mirrored leg's root pose and positive flexion-axis convention."""

    name: str
    center_x_m: float
    side_sign: float

    @property
    def root_xyz_m(self) -> tuple[float, float, float]:
        return (
            self.center_x_m,
            self.side_sign * 0.170084989,
            0.050,
        )

    @property
    def root_rpy_rad(self) -> tuple[float, float, float]:
        return (
            math.pi / 2.0,
            0.0,
            self.side_sign * math.pi / 2.0,
        )

    @property
    def pitch_axis(self) -> tuple[float, float, float]:
        # Positive pitch has the same robot-forward meaning on both sides.
        return (0.0, 0.0, -self.side_sign)


LEGS = (
    LegSpec("front_left", 0.060, 1.0),
    LegSpec("rear_left", -0.060, 1.0),
    LegSpec("front_right", 0.060, -1.0),
    LegSpec("rear_right", -0.060, -1.0),
)


def _numbers(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _origin(parent, xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0)):
    return ET.SubElement(
        parent,
        "origin",
        {"xyz": _numbers(xyz), "rpy": _numbers(rpy)},
    )


def _add_material(robot, name: str, rgba):
    material = ET.SubElement(robot, "material", {"name": name})
    ET.SubElement(material, "color", {"rgba": _numbers(rgba)})


def _add_inertial(link, mass_kg: float, com_xyz, inertia):
    inertial = ET.SubElement(link, "inertial")
    _origin(inertial, com_xyz)
    ET.SubElement(inertial, "mass", {"value": f"{mass_kg:.10g}"})
    ixx, ixy, ixz, iyy, iyz, izz = inertia
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": f"{ixx:.10g}",
            "ixy": f"{ixy:.10g}",
            "ixz": f"{ixz:.10g}",
            "iyy": f"{iyy:.10g}",
            "iyz": f"{iyz:.10g}",
            "izz": f"{izz:.10g}",
        },
    )


def _add_mesh_visual(
    link,
    name: str,
    filename: str,
    material: str,
    xyz=(0.0, 0.0, 0.0),
    rpy=(0.0, 0.0, 0.0),
):
    visual = ET.SubElement(link, "visual", {"name": name})
    _origin(visual, xyz, rpy)
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(
        geometry,
        "mesh",
        {"filename": filename, "scale": _numbers(MESH_SCALE_MM_TO_M)},
    )
    ET.SubElement(visual, "material", {"name": material})


def _add_sphere_visual(link, name: str, radius: float, xyz, material: str):
    visual = ET.SubElement(link, "visual", {"name": name})
    _origin(visual, xyz)
    geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(geometry, "sphere", {"radius": f"{radius:.10g}"})
    ET.SubElement(visual, "material", {"name": material})


def _add_box_collision(
    link,
    name: str,
    size,
    xyz,
    rpy=(0.0, 0.0, 0.0),
):
    collision = ET.SubElement(link, "collision", {"name": name})
    _origin(collision, xyz, rpy)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "box", {"size": _numbers(size)})


def _add_sphere_collision(link, name: str, radius: float, xyz):
    collision = ET.SubElement(link, "collision", {"name": name})
    _origin(collision, xyz)
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(geometry, "sphere", {"radius": f"{radius:.10g}"})


def _add_revolute_joint(
    robot,
    name: str,
    parent_link: str,
    child_link: str,
    xyz,
    rpy,
    axis,
    lower: float,
    upper: float,
):
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "revolute"})
    ET.SubElement(joint, "parent", {"link": parent_link})
    ET.SubElement(joint, "child", {"link": child_link})
    _origin(joint, xyz, rpy)
    ET.SubElement(joint, "axis", {"xyz": _numbers(axis)})
    ET.SubElement(
        joint,
        "limit",
        {
            "lower": f"{lower:.10g}",
            "upper": f"{upper:.10g}",
            "effort": f"{SERVO_STALL_TORQUE_NM:.10g}",
            "velocity": f"{SERVO_NO_LOAD_VELOCITY_RAD_S:.10g}",
        },
    )
    ET.SubElement(
        joint,
        "dynamics",
        {
            "damping": f"{JOINT_DAMPING_NM_S_RAD:.10g}",
            "friction": f"{JOINT_FRICTION_NM:.10g}",
        },
    )


def _add_base(robot):
    base = ET.SubElement(robot, "link", {"name": "base_link"})
    _add_inertial(base, BASE_MASS_KG, BASE_COM_M, BASE_INERTIA_KG_M2)
    _add_mesh_visual(
        base,
        "printable_body_tub",
        "../stl/quadruped_body_base.stl",
        "body_dark",
    )
    _add_mesh_visual(
        base,
        "printable_body_lid",
        "../stl/quadruped_body_lid.stl",
        "lid_blue",
        xyz=(0.0, 0.0, 0.096),
    )
    _add_mesh_visual(
        base,
        "printable_electronics_tray",
        "../stl/quadruped_electronics_tray.stl",
        "tray_gold",
        xyz=(0.0, 0.0, 0.056),
    )

    # Fixed body-side mounts are part of base_link, while the four root servo
    # cases live with their moving hip links.
    for leg in LEGS:
        _add_mesh_visual(
            base,
            f"{leg.name}_body_side_mount",
            "../stl/st3215_hip_body_mount.stl",
            "mount_light",
            xyz=(
                leg.center_x_m,
                leg.side_sign * 0.117,
                0.050,
            ),
            rpy=(math.pi / 2.0, 0.0, leg.side_sign * math.pi / 2.0),
        )

    # A deliberately simple enclosure proxy is more stable in PhysX than a
    # thin-walled concave mesh.  Internal payload mass is represented inertially.
    _add_box_collision(
        base,
        "body_enclosure_collision",
        (0.220, 0.170, 0.100),
        (0.0, 0.0, 0.050),
    )


def _add_leg(robot, leg: LegSpec):
    hip_name = f"{leg.name}_hip_link"
    proximal_name = f"{leg.name}_proximal_link"
    distal_name = f"{leg.name}_distal_link"
    leg_material = f"{leg.name}_plastic"

    hip = ET.SubElement(robot, "link", {"name": hip_name})
    _add_inertial(hip, HIP_LINK_MASS_KG, HIP_LINK_COM_M, HIP_LINK_INERTIA_KG_M2)
    _add_mesh_visual(
        hip,
        "printable_hip",
        "../stl/st3215_hip.stl",
        leg_material,
        xyz=HIP_VISUAL_XYZ_M,
        rpy=HIP_VISUAL_RPY_RAD,
    )
    _add_mesh_visual(
        hip,
        "exact_st3215_servo",
        SERVO_VISUAL_MESH,
        "servo_black",
        xyz=SERVO_MESH_XYZ_M,
        rpy=SERVO_MESH_RPY_RAD,
    )
    _add_box_collision(
        hip,
        "hip_printable_proxy",
        (0.0673, 0.123308, 0.04405),
        (0.0266617, -0.0460425, -0.00195),
    )
    _add_box_collision(
        hip,
        "hip_servo_case_collision",
        SERVO_BOX_SIZE_M,
        SERVO_COLLISION_XYZ_M,
        SERVO_COLLISION_RPY_RAD,
    )

    for role, link_name in (
        ("proximal", proximal_name),
        ("distal", distal_name),
    ):
        arm = ET.SubElement(robot, "link", {"name": link_name})
        _add_inertial(
            arm,
            ARM_LINK_MASS_KG,
            ARM_LINK_COM_M,
            ARM_LINK_INERTIA_KG_M2,
        )
        _add_mesh_visual(
            arm,
            f"printable_{role}_upper_arm",
            "../stl/upper_arm.stl",
            leg_material,
            xyz=ARM_VISUAL_XYZ_M,
        )
        _add_mesh_visual(
            arm,
            "exact_st3215_servo",
            SERVO_VISUAL_MESH,
            "servo_black",
            xyz=SERVO_MESH_XYZ_M,
            rpy=SERVO_MESH_RPY_RAD,
        )
        _add_box_collision(
            arm,
            f"{role}_arm_printable_proxy",
            (0.151285, 0.031229, 0.0673),
            (0.096254, 0.0, -0.00195),
        )
        _add_box_collision(
            arm,
            "st3215_servo_case_collision",
            SERVO_BOX_SIZE_M,
            SERVO_COLLISION_XYZ_M,
            SERVO_COLLISION_RPY_RAD,
        )

    _add_sphere_visual(
        distal_name_element := next(
            link
            for link in robot.findall("link")
            if link.attrib["name"] == distal_name
        ),
        "simulation_only_fork_tip_contact_proxy",
        VIRTUAL_FORK_TIP_RADIUS_M,
        (ARM_LINK_LENGTH_M, 0.0, 0.0),
        "virtual_contact_magenta",
    )
    _add_sphere_collision(
        distal_name_element,
        "simulation_only_fork_tip_contact_proxy",
        VIRTUAL_FORK_TIP_RADIUS_M,
        (ARM_LINK_LENGTH_M, 0.0, 0.0),
    )

    _add_revolute_joint(
        robot,
        f"{leg.name}_hip_abduction",
        "base_link",
        hip_name,
        leg.root_xyz_m,
        leg.root_rpy_rad,
        (0.0, 0.0, 1.0),
        -HIP_ABDUCTION_LIMIT_RAD,
        HIP_ABDUCTION_LIMIT_RAD,
    )
    _add_revolute_joint(
        robot,
        f"{leg.name}_hip_flexion",
        hip_name,
        proximal_name,
        HIP_LINK_TO_FLEXION_XYZ_M,
        HIP_LINK_TO_FLEXION_RPY_RAD,
        leg.pitch_axis,
        -HIP_FLEXION_LIMIT_RAD,
        HIP_FLEXION_LIMIT_RAD,
    )
    _add_revolute_joint(
        robot,
        f"{leg.name}_knee",
        proximal_name,
        distal_name,
        (ARM_LINK_LENGTH_M, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        leg.pitch_axis,
        -KNEE_LIMIT_RAD,
        KNEE_LIMIT_RAD,
    )


def gen_urdf():
    """Return the complete URDF XML tree root."""
    robot = ET.Element("robot", {"name": ROBOT_NAME})
    robot.append(
        ET.Comment(
            "Mass model: solid PLA, 0.45 kg battery, 0.15 kg electronics; "
            "magenta fork-tip contacts are simulation-only."
        )
    )
    _add_material(robot, "body_dark", (0.08, 0.13, 0.20, 1.0))
    _add_material(robot, "lid_blue", (0.22, 0.36, 0.50, 1.0))
    _add_material(robot, "tray_gold", (0.95, 0.65, 0.12, 1.0))
    _add_material(robot, "mount_light", (0.65, 0.68, 0.74, 1.0))
    _add_material(robot, "servo_black", (0.025, 0.03, 0.04, 1.0))
    _add_material(robot, "virtual_contact_magenta", (0.95, 0.05, 0.48, 1.0))
    _add_material(robot, "front_left_plastic", (0.95, 0.38, 0.08, 1.0))
    _add_material(robot, "rear_left_plastic", (0.20, 0.62, 0.92, 1.0))
    _add_material(robot, "front_right_plastic", (0.18, 0.72, 0.48, 1.0))
    _add_material(robot, "rear_right_plastic", (0.50, 0.28, 0.82, 1.0))

    _add_base(robot)
    for leg in LEGS:
        _add_leg(robot, leg)
    return robot


if __name__ == "__main__":
    ET.indent(tree := ET.ElementTree(gen_urdf()), space="  ")
    tree.write("quadruped_robot.urdf", encoding="unicode", xml_declaration=True)
