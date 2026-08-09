from functools import lru_cache

import pytest
from build123d import Vertex

from drobot_cad.assembly import quadruped_robot
from drobot_cad.parts import quadruped_body, st3215_hip_body_mount


def _mapped_point(location, point) -> tuple[float, float, float]:
    return tuple(Vertex(*point).moved(location).center())


@lru_cache(maxsize=1)
def generated_quadruped():
    return quadruped_robot.gen_step()


def test_four_leg_modules_are_declared_in_body_order() -> None:
    assert quadruped_robot.FINAL_ASSEMBLY_SPEC.component_order == (
        "body_base",
        "body_battery",
        "electronics_tray",
        "body_servo_bus_adapter",
        "body_imu",
        "body_imu_cover",
        "body_lid",
        "lekiwi_camera_assembly",
        "front_left_leg",
        "rear_left_leg",
        "front_right_leg",
        "rear_right_leg",
    )
    assert tuple(spec.name for spec in quadruped_robot.LEG_MOUNT_SPECS) == (
        "front_left",
        "rear_left",
        "front_right",
        "rear_right",
    )


@pytest.mark.parametrize("spec", quadruped_robot.LEG_MOUNT_SPECS)
def test_each_hip_plate_body_face_is_flush_to_its_side_wall(spec) -> None:
    mount_pose = quadruped_robot.body_mount_location(spec)
    mapped_center = _mapped_point(
        mount_pose,
        (
            st3215_hip_body_mount.MOUNTING_PLATE_BODY_SIDE_FACE_X_MM,
            0.0,
            0.0,
        ),
    )

    assert mapped_center == pytest.approx(
        (
            spec.center_x_mm,
            spec.side_sign * quadruped_body.BODY_WIDTH_Y_MM / 2.0,
            quadruped_body.HIP_MOUNT_CENTER_Z_MM,
        )
    )


@pytest.mark.parametrize("spec", quadruped_robot.LEG_MOUNT_SPECS)
def test_each_hip_plate_bolt_pattern_maps_to_body_holes(spec) -> None:
    mount_pose = quadruped_robot.body_mount_location(spec)
    mapped_centers = {
        tuple(
            round(value, 6)
            for value in _mapped_point(
                mount_pose,
                (
                    st3215_hip_body_mount.MOUNTING_PLATE_BODY_SIDE_FACE_X_MM,
                    local_y_mm,
                    local_z_mm,
                ),
            )
        )
        for local_y_mm, local_z_mm in (
            st3215_hip_body_mount.BODY_BOLT_HOLE_CENTERS_YZ_MM
        )
    }
    expected_centers = {
        (
            spec.center_x_mm + offset_x_mm,
            spec.side_sign * quadruped_body.BODY_WIDTH_Y_MM / 2.0,
            quadruped_body.HIP_MOUNT_CENTER_Z_MM + offset_z_mm,
        )
        for offset_x_mm in (-30.0, 30.0)
        for offset_z_mm in (-30.0, 30.0)
    }

    assert mapped_centers == expected_centers


@pytest.mark.parametrize("spec", quadruped_robot.LEG_MOUNT_SPECS)
def test_leg_root_pose_equals_the_declared_body_mount_pose(spec) -> None:
    locations = quadruped_robot.leg_component_locations(spec)
    actual = locations["body_side_hip_mount"]
    expected = quadruped_robot.body_mount_location(spec)

    assert tuple(actual.position) == pytest.approx(tuple(expected.position))
    assert tuple(actual.orientation) == pytest.approx(tuple(expected.orientation))


def test_generated_quadruped_has_body_and_four_nested_leg_modules() -> None:
    assembly = generated_quadruped()

    assert assembly.label == quadruped_robot.FINAL_ASSEMBLY_SPEC.name
    assert tuple(child.label for child in assembly.children) == (
        quadruped_robot.FINAL_ASSEMBLY_SPEC.component_order
    )
    camera_module = next(
        child
        for child in assembly.children
        if child.label == "lekiwi_camera_assembly"
    )
    assert tuple(child.label for child in camera_module.children) == (
        "lekiwi_base_camera_mount_reference",
        "arducam_5mp_reference",
    )
    leg_modules = [
        child for child in assembly.children if child.label.endswith("_leg")
    ]
    assert len(leg_modules) == 4
    for leg_module in leg_modules:
        assert tuple(child.label for child in leg_module.children) == (
            quadruped_robot.robot_leg.FINAL_ASSEMBLY_SPEC.component_order
        )


def test_complete_stance_has_expected_proportional_envelope() -> None:
    bounds = generated_quadruped().bounding_box()

    assert bounds.size.X == pytest.approx(519.235, abs=0.2)
    assert bounds.size.Y == pytest.approx(635.397, abs=0.2)
    assert bounds.size.Z == pytest.approx(432.609, abs=0.2)
