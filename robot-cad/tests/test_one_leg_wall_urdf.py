from __future__ import annotations

import math
from pathlib import Path

import pytest

from robot_cad.urdf import one_leg_wall_testbed as wall_testbed
from robot_cad.urdf import quadruped_robot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
URDF_OUTPUT = PROJECT_ROOT / "exports" / "urdf" / "one_leg_wall_testbed.urdf"


def _numbers(value: str) -> tuple[float, ...]:
    return tuple(float(number) for number in value.split())


def test_wall_testbed_has_one_fixed_fixture_and_three_joint_leg():
    robot = wall_testbed.gen_urdf()

    assert robot.attrib["name"] == wall_testbed.ROBOT_NAME
    assert [link.attrib["name"] for link in robot.findall("link")] == [
        "wall_link",
        "hip_link",
        "proximal_link",
        "distal_link",
    ]
    assert [joint.attrib["name"] for joint in robot.findall("joint")] == [
        "hip_abduction",
        "hip_flexion",
        "knee",
    ]
    assert all(
        joint.attrib["type"] == "revolute"
        for joint in robot.findall("joint")
    )
    assert all(link.find("inertial") is not None for link in robot.findall("link"))


def test_wall_surface_is_flush_to_exact_body_mount_back_face():
    robot = wall_testbed.gen_urdf()
    wall = robot.find("link[@name='wall_link']")
    assert wall is not None
    mount = wall.find("visual[@name='exact_printable_hip_body_mount']")
    collision = wall.find("collision[@name='vertical_wall_collision']")
    assert mount is not None
    assert collision is not None

    mount_origin = _numbers(mount.find("origin").attrib["xyz"])
    # The CAD mount's local body-side face is x=-32 mm.  Under the approved
    # root rotation local +X maps to wall-frame +Y.
    transformed_back_face_y = mount_origin[1] - 0.032
    assert transformed_back_face_y == pytest.approx(
        wall_testbed.WALL_SURFACE_Y_M
    )
    wall_center_y = _numbers(collision.find("origin").attrib["xyz"])[1]
    wall_size_y = _numbers(collision.find("geometry/box").attrib["size"])[1]
    assert wall_center_y + wall_size_y / 2.0 == pytest.approx(
        wall_testbed.WALL_SURFACE_Y_M
    )


def test_joint_frames_match_front_left_quadruped_but_use_tested_limits():
    robot = wall_testbed.gen_urdf()
    reference_leg = quadruped_robot.LEGS[0]

    hip = robot.find("joint[@name='hip_abduction']")
    flexion = robot.find("joint[@name='hip_flexion']")
    knee = robot.find("joint[@name='knee']")
    assert hip is not None
    assert flexion is not None
    assert knee is not None

    assert _numbers(hip.find("origin").attrib["rpy"]) == pytest.approx(
        reference_leg.root_rpy_rad
    )
    assert _numbers(hip.find("axis").attrib["xyz"]) == pytest.approx((0, 0, 1))
    assert _numbers(flexion.find("origin").attrib["xyz"]) == pytest.approx(
        quadruped_robot.HIP_LINK_TO_FLEXION_XYZ_M
    )
    assert _numbers(flexion.find("origin").attrib["rpy"]) == pytest.approx(
        quadruped_robot.HIP_LINK_TO_FLEXION_RPY_RAD
    )
    assert _numbers(flexion.find("axis").attrib["xyz"]) == pytest.approx(
        reference_leg.pitch_axis
    )
    assert _numbers(knee.find("axis").attrib["xyz"]) == pytest.approx(
        reference_leg.pitch_axis
    )

    for name, expected_deg in wall_testbed.PHYSICALLY_EXERCISED_LIMITS_DEG.items():
        limit = robot.find(f"joint[@name='{name}']/limit")
        assert limit is not None
        assert math.degrees(float(limit.attrib["lower"])) == pytest.approx(
            expected_deg[0]
        )
        assert math.degrees(float(limit.attrib["upper"])) == pytest.approx(
            expected_deg[1]
        )


def test_moving_link_geometry_and_inertia_are_shared_with_quadruped():
    robot = wall_testbed.gen_urdf()
    reference = quadruped_robot.gen_urdf()
    pairs = (
        ("hip_link", "front_left_hip_link"),
        ("proximal_link", "front_left_proximal_link"),
        ("distal_link", "front_left_distal_link"),
    )

    for actual_name, reference_name in pairs:
        actual = robot.find(f"link[@name='{actual_name}']")
        expected = reference.find(f"link[@name='{reference_name}']")
        assert actual is not None
        assert expected is not None
        assert actual.find("inertial/mass").attrib == expected.find(
            "inertial/mass"
        ).attrib
        assert actual.find("inertial/inertia").attrib == expected.find(
            "inertial/inertia"
        ).attrib
        # The one-leg fixture intentionally omits the quadruped's virtual foot
        # sphere, which has no matching physical part.
        expected_visuals = [
            element
            for element in expected.findall("visual")
            if element.attrib["name"]
            != "simulation_only_fork_tip_contact_proxy"
        ]
        expected_collisions = [
            element
            for element in expected.findall("collision")
            if element.attrib["name"]
            != "simulation_only_fork_tip_contact_proxy"
        ]
        assert len(actual.findall("visual")) == len(expected_visuals)
        assert len(actual.findall("collision")) == len(expected_collisions)


def test_generated_wall_urdf_mesh_references_resolve():
    if not URDF_OUTPUT.is_file():
        pytest.skip("Generated wall testbed URDF has not been emitted yet")
    import xml.etree.ElementTree as ET

    root = ET.parse(URDF_OUTPUT).getroot()
    for mesh in root.findall(".//mesh"):
        mesh_path = (URDF_OUTPUT.parent / mesh.attrib["filename"]).resolve()
        assert mesh_path.is_file(), mesh_path
        assert mesh.attrib["scale"] == "0.001 0.001 0.001"
