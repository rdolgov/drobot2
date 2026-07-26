"""Contract tests for the simulation URDF source."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from build123d import import_step, import_stl

from robot_cad.urdf import quadruped_robot

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _numbers(value: str) -> tuple[float, ...]:
    return tuple(float(number) for number in value.split())


def test_generated_urdf_is_synchronized_with_generator():
    generated_path = PROJECT_ROOT / "exports" / "urdf" / "quadruped_robot.urdf"
    source_path = Path(quadruped_robot.__file__).resolve()
    expected_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    generated_xml = generated_path.read_text(encoding="utf-8")

    assert f"cadpy:sourceHash={expected_hash}" in generated_xml
    expected_xml = ET.tostring(
        quadruped_robot.gen_urdf(),
        encoding="unicode",
    )
    assert ET.canonicalize(generated_xml, strip_text=True) == ET.canonicalize(
        expected_xml,
        strip_text=True,
    )


def test_urdf_has_expected_tree_and_physics_elements():
    robot = quadruped_robot.gen_urdf()
    links = robot.findall("link")
    joints = robot.findall("joint")

    assert robot.attrib["name"] == "st3215_quadruped"
    assert len(links) == 15
    assert len(joints) == 14
    assert {link.attrib["name"] for link in links} >= {
        "base_link",
        "camera_link",
        "camera_optical_frame",
    }

    for link in links:
        if link.attrib["name"] == "camera_optical_frame":
            assert link.find("inertial") is None
            assert not link.findall("visual")
            assert not link.findall("collision")
            continue
        assert link.find("inertial") is not None
        assert link.findall("visual")
        assert link.findall("collision")

    for joint in joints:
        assert joint.find("parent") is not None
        assert joint.find("child") is not None
        assert joint.find("origin") is not None
        if joint.attrib["type"] == "fixed":
            assert joint.find("axis") is None
            assert joint.find("limit") is None
            assert joint.find("dynamics") is None
        else:
            assert joint.attrib["type"] == "revolute"
            assert joint.find("axis") is not None
            assert joint.find("limit") is not None
            assert joint.find("dynamics") is not None

    assert len(robot.findall("joint[@type='revolute']")) == 12
    assert len(robot.findall("joint[@type='fixed']")) == 2


def test_mesh_references_resolve_from_generated_urdf_directory():
    robot = quadruped_robot.gen_urdf()
    urdf_directory = PROJECT_ROOT / "exports" / "urdf"
    meshes = robot.findall(".//mesh")

    assert len(meshes) == 33
    assert {mesh.attrib["filename"] for mesh in meshes} == {
        "../stl/quadruped_body_base.stl",
        "../stl/quadruped_body_lid.stl",
        "../stl/quadruped_electronics_tray.stl",
        "../stl/st3215_hip_body_mount.stl",
        "../stl/st3215_hip.stl",
        "../stl/upper_arm.stl",
        "../stl/st3215_servo_visual.stl",
        "../../vendor/references/lekiwi/base_camera_mount.stl",
        "../../vendor/references/lekiwi/arducam_5mp_camera_model.stl",
    }
    for mesh in meshes:
        mesh_path = (urdf_directory / mesh.attrib["filename"]).resolve()
        assert mesh_path.is_file(), mesh_path
        assert mesh.attrib["scale"] == "0.001 0.001 0.001"


def test_all_moving_links_use_exact_servo_mesh_visuals():
    robot = quadruped_robot.gen_urdf()
    moving_links = [
        link
        for link in robot.findall("link")
        if any(
            link.attrib["name"].endswith(suffix)
            for suffix in ("_hip_link", "_proximal_link", "_distal_link")
        )
    ]

    assert len(moving_links) == 12
    for link in moving_links:
        servo_visuals = link.findall("visual[@name='exact_st3215_servo']")
        assert len(servo_visuals) == 1, link.attrib["name"]
        mesh = servo_visuals[0].find("geometry/mesh")
        assert mesh is not None
        assert mesh.attrib["filename"] == "../stl/st3215_servo_visual.stl"
        assert not link.findall("visual/geometry/box"), link.attrib["name"]


def test_servo_mesh_and_collision_use_separately_audited_origins():
    robot = quadruped_robot.gen_urdf()

    assert quadruped_robot.SERVO_MESH_XYZ_M != (
        quadruped_robot.SERVO_COLLISION_XYZ_M
    )
    for link in robot.findall("link"):
        if not any(
            link.attrib["name"].endswith(suffix)
            for suffix in ("_hip_link", "_proximal_link", "_distal_link")
        ):
            continue

        visual_origin = link.find(
            "visual[@name='exact_st3215_servo']/origin"
        )
        collision = link.find("collision[@name='hip_servo_case_collision']")
        if collision is None:
            collision = link.find(
                "collision[@name='st3215_servo_case_collision']"
            )
        assert visual_origin is not None, link.attrib["name"]
        assert collision is not None, link.attrib["name"]
        collision_origin = collision.find("origin")
        assert collision_origin is not None

        assert _numbers(visual_origin.attrib["xyz"]) == pytest.approx(
            quadruped_robot.SERVO_MESH_XYZ_M
        )
        assert _numbers(visual_origin.attrib["rpy"]) == pytest.approx(
            quadruped_robot.SERVO_MESH_RPY_RAD
        )
        assert _numbers(collision_origin.attrib["xyz"]) == pytest.approx(
            quadruped_robot.SERVO_COLLISION_XYZ_M
        )
        assert _numbers(collision_origin.attrib["rpy"]) == pytest.approx(
            quadruped_robot.SERVO_COLLISION_RPY_RAD
        )


def test_printable_collision_boxes_include_visible_surface_guard():
    robot = quadruped_robot.gen_urdf()

    assert quadruped_robot.PRINTABLE_COLLISION_GUARD_PER_SIDE_M == pytest.approx(
        0.002
    )
    for leg in quadruped_robot.LEGS:
        hip = robot.find(f"link[@name='{leg.name}_hip_link']")
        assert hip is not None
        hip_proxy = hip.find(
            "collision[@name='hip_printable_proxy']/geometry/box"
        )
        assert hip_proxy is not None
        assert _numbers(hip_proxy.attrib["size"]) == pytest.approx(
            quadruped_robot.HIP_PRINTABLE_COLLISION_SIZE_M
        )

        for role in ("proximal", "distal"):
            arm = robot.find(f"link[@name='{leg.name}_{role}_link']")
            assert arm is not None
            arm_proxy = arm.find(
                f"collision[@name='{role}_arm_printable_proxy']/geometry/box"
            )
            assert arm_proxy is not None
            assert _numbers(arm_proxy.attrib["size"]) == pytest.approx(
                quadruped_robot.ARM_PRINTABLE_COLLISION_SIZE_M
            )

    for guarded, exact in (
        (
            quadruped_robot.HIP_PRINTABLE_COLLISION_SIZE_M,
            quadruped_robot.HIP_PRINTABLE_BOUNDS_SIZE_M,
        ),
        (
            quadruped_robot.ARM_PRINTABLE_COLLISION_SIZE_M,
            quadruped_robot.ARM_PRINTABLE_BOUNDS_SIZE_M,
        ),
    ):
        assert tuple(
            guarded_value - exact_value
            for guarded_value, exact_value in zip(
                guarded,
                exact,
                strict=True,
            )
        ) == pytest.approx((0.004, 0.004, 0.004))


def test_upper_arm_stl_matches_current_step_bounds():
    step = import_step(PROJECT_ROOT / "exports" / "step" / "upper_arm.step")
    stl = import_stl(PROJECT_ROOT / "exports" / "stl" / "upper_arm.stl")
    step_bounds = step.bounding_box()
    stl_bounds = stl.bounding_box()

    assert tuple(step_bounds.size) == pytest.approx(
        (151.284989, 31.228608, 67.3),
        abs=1e-5,
    )
    assert tuple(stl_bounds.size) == pytest.approx(
        tuple(step_bounds.size),
        abs=0.05,
    )
    assert tuple(stl_bounds.min) == pytest.approx(
        tuple(step_bounds.min),
        abs=0.05,
    )
    assert tuple(stl_bounds.max) == pytest.approx(
        tuple(step_bounds.max),
        abs=0.05,
    )


def test_motor_bay_stl_matches_current_step_bounds():
    step = import_step(
        PROJECT_ROOT / "exports" / "step" / "st3215_motor_bay.step"
    )
    stl = import_stl(
        PROJECT_ROOT / "exports" / "stl" / "st3215_motor_bay.stl"
    )
    step_bounds = step.bounding_box()
    stl_bounds = stl.bounding_box()

    assert tuple(step_bounds.size) == pytest.approx(
        (16.0, 31.2234, 44.05),
        abs=1e-5,
    )
    assert tuple(stl_bounds.min) == pytest.approx(
        tuple(step_bounds.min),
        abs=0.05,
    )
    assert tuple(stl_bounds.max) == pytest.approx(
        tuple(step_bounds.max),
        abs=0.05,
    )


def test_joint_limits_use_exact_st3215_hard_limits():
    robot = quadruped_robot.gen_urdf()
    for limit in robot.findall(".//joint/limit"):
        assert math.isclose(
            float(limit.attrib["effort"]),
            quadruped_robot.SERVO_STALL_TORQUE_NM,
        )
        assert math.isclose(
            float(limit.attrib["velocity"]),
            quadruped_robot.SERVO_NO_LOAD_VELOCITY_RAD_S,
        )


def test_mass_model_matches_ledger_total():
    robot = quadruped_robot.gen_urdf()
    total_mass = sum(
        float(mass.attrib["value"])
        for mass in robot.findall(".//link/inertial/mass")
    )
    assert math.isclose(
        total_mass,
        quadruped_robot.TOTAL_ROBOT_MASS_KG,
        abs_tol=1e-9,
    )
    assert math.isclose(total_mass, 4.523638989, abs_tol=1e-6)


def test_camera_payload_and_optical_frame_match_approved_cad_pose():
    robot = quadruped_robot.gen_urdf()
    camera = robot.find("link[@name='camera_link']")
    optical = robot.find("link[@name='camera_optical_frame']")
    mount_joint = robot.find("joint[@name='base_to_camera_mount']")
    optical_joint = robot.find("joint[@name='camera_to_optical']")

    assert camera is not None
    assert optical is not None
    assert mount_joint is not None
    assert optical_joint is not None
    assert mount_joint.attrib["type"] == "fixed"
    assert optical_joint.attrib["type"] == "fixed"
    assert _numbers(mount_joint.find("origin").attrib["xyz"]) == pytest.approx(
        quadruped_robot.BASE_TO_CAMERA_LINK_XYZ_M
    )
    assert _numbers(optical_joint.find("origin").attrib["xyz"]) == pytest.approx(
        quadruped_robot.CAMERA_LINK_TO_OPTICAL_XYZ_M
    )
    assert _numbers(optical_joint.find("origin").attrib["rpy"]) == pytest.approx(
        quadruped_robot.CAMERA_LINK_TO_OPTICAL_RPY_RAD
    )
    assert quadruped_robot.CAMERA_OPTICAL_XYZ_FROM_BASE_M == pytest.approx(
        (0.1145, 0.0, 0.123)
    )
    assert float(camera.find("inertial/mass").attrib["value"]) == pytest.approx(
        quadruped_robot.CAMERA_ASSEMBLY_MASS_KG
    )


def test_camera_intrinsics_match_lightweight_lekiwi_profile():
    assert quadruped_robot.CAMERA_RESOLUTION_HW == (480, 640)
    assert quadruped_robot.CAMERA_TICK_RATE_HZ == pytest.approx(30.0)
    assert quadruped_robot.CAMERA_HORIZONTAL_FOV_DEG == pytest.approx(95.0)
    calculated_hfov = math.degrees(
        2.0
        * math.atan(
            quadruped_robot.CAMERA_HORIZONTAL_APERTURE_MM
            / (2.0 * quadruped_robot.CAMERA_FOCAL_LENGTH_MM)
        )
    )
    assert calculated_hfov == pytest.approx(
        quadruped_robot.CAMERA_HORIZONTAL_FOV_DEG
    )


def test_each_distal_link_discloses_virtual_contact_proxy():
    robot = quadruped_robot.gen_urdf()
    for leg in quadruped_robot.LEGS:
        distal = next(
            link
            for link in robot.findall("link")
            if link.attrib["name"] == f"{leg.name}_distal_link"
        )
        proxy = distal.find(
            "collision[@name='simulation_only_fork_tip_contact_proxy']"
        )
        assert proxy is not None
