from functools import lru_cache

import pytest
from build123d import Location, Vertex

from robot_cad.assembly import robot_arm, robot_leg


def _mapped_origin(location: Location) -> tuple[float, float, float]:
    return tuple(Vertex(0.0, 0.0, 0.0).moved(location).center())


def _mapped_z_direction(location: Location) -> tuple[float, float, float]:
    origin = Vertex(0.0, 0.0, 0.0).moved(location).center()
    endpoint = Vertex(0.0, 0.0, 1.0).moved(location).center()
    return tuple(endpoint - origin)


@lru_cache(maxsize=1)
def _generated_leg():
    return robot_leg.gen_step()


def test_component_order_matches_requested_body_to_foot_chain() -> None:
    assert robot_leg.FINAL_ASSEMBLY_SPEC.component_order == (
        "body_side_hip_mount",
        "hip_abduction_st3215_servo",
        "st3215_hip",
        "hip_flexion_st3215_servo",
        "proximal_upper_arm",
        "knee_st3215_servo",
        "distal_upper_arm",
    )
    assert tuple(robot_leg.component_locations()) == (
        robot_leg.FINAL_ASSEMBLY_SPEC.component_order
    )


@pytest.mark.parametrize(
    ("parent_pose_name", "parent_frame", "child_pose_name", "child_frame"),
    [
        (
            "body_side_hip_mount",
            robot_leg.body_mount_fork_axis_location(),
            "st3215_hip",
            robot_leg.hip_installed_servo_axis_location(),
        ),
        (
            "st3215_hip",
            robot_leg.hip_distal_fork_axis_location(),
            "proximal_upper_arm",
            robot_leg.upper_arm_installed_servo_axis_location(),
        ),
        (
            "proximal_upper_arm",
            robot_leg.upper_arm_distal_fork_axis_location(),
            "distal_upper_arm",
            robot_leg.upper_arm_installed_servo_axis_location(),
        ),
    ],
)
def test_all_servo_shaft_frames_are_coaxial_with_parent_forks(
    parent_pose_name: str,
    parent_frame: Location,
    child_pose_name: str,
    child_frame: Location,
) -> None:
    poses = robot_leg.component_locations()
    parent_world_frame = poses[parent_pose_name] * parent_frame
    child_world_frame = poses[child_pose_name] * child_frame

    assert _mapped_origin(child_world_frame) == pytest.approx(
        _mapped_origin(parent_world_frame),
        abs=1.0e-8,
    )
    assert _mapped_z_direction(child_world_frame) == pytest.approx(
        _mapped_z_direction(parent_world_frame),
        abs=1.0e-8,
    )


def test_distal_arm_uses_the_existing_upper_arm_joint_transform() -> None:
    poses = robot_leg.component_locations()
    expected = (
        poses["proximal_upper_arm"]
        * robot_arm.child_arm_location(robot_leg.KNEE_PREVIEW_ANGLE_DEG)
    )
    actual = poses["distal_upper_arm"]

    assert tuple(actual.position) == pytest.approx(tuple(expected.position))
    assert tuple(actual.orientation) == pytest.approx(tuple(expected.orientation))


def test_generated_leg_has_seven_labeled_top_level_occurrences() -> None:
    assembly = _generated_leg()

    assert assembly.label == robot_leg.FINAL_ASSEMBLY_SPEC.name
    assert tuple(child.label for child in assembly.children) == (
        robot_leg.FINAL_ASSEMBLY_SPEC.component_order
    )
