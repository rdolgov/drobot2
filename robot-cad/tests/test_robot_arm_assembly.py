import pytest
from build123d import Vertex

from robot_cad.assembly import robot_arm


def test_child_servo_axis_maps_to_parent_fork_axis() -> None:
    connection = robot_arm.ELBOW_CONNECTION
    child_pose = robot_arm.child_arm_location()
    mapped_axis = Vertex(*connection.child_frame.xyz_mm).moved(child_pose).center()

    assert tuple(mapped_axis) == pytest.approx(connection.parent_frame.xyz_mm)
    assert tuple(child_pose.orientation) == pytest.approx(
        (0.0, 0.0, robot_arm.ELBOW_PREVIEW_ANGLE_DEG)
    )


def test_elbow_pose_rejects_angles_outside_declared_limits() -> None:
    with pytest.raises(ValueError):
        robot_arm.child_arm_location(robot_arm.ELBOW_MAXIMUM_DEG + 0.1)


def test_final_assembly_declares_two_arms_and_two_servos() -> None:
    spec = robot_arm.FINAL_ASSEMBLY_SPEC

    assert spec.root_component == "upper_arm_link_1"
    assert spec.child_component == "upper_arm_link_2"
    assert spec.root_fork_component == "st3215_servo_output_fork_1"
    assert spec.child_fork_component == "st3215_servo_output_fork_2"
    assert spec.base_servo_component == "st3215_base_servo"
    assert spec.elbow_servo_component == "st3215_elbow_servo"
