import pytest

from robot_cad.interfaces import (
    ST3215_MOTOR_BAY_INTERFACES,
    ST3215_SERVO_OUTPUT_FORK_INTERFACES,
    UPPER_ARM_INTERFACES,
)
from robot_cad.parts import st3215_servo_output_fork
from robot_cad.parts.st3215_motor_bay import st3215_installed_location
from robot_cad.parts.upper_arm import (
    distal_fork_axis_location,
    st3215_output_axis_location,
    st3215_preview_location,
)
from robot_cad.validation import validate_interface_frame


@pytest.mark.parametrize(
    "frame",
    [
        *ST3215_MOTOR_BAY_INTERFACES.values(),
        *ST3215_SERVO_OUTPUT_FORK_INTERFACES.values(),
        *UPPER_ARM_INTERFACES.values(),
    ],
)
def test_interface_axes_are_normalized(frame) -> None:
    validate_interface_frame(frame)


def test_motor_bay_servo_frame_matches_source_location() -> None:
    frame = ST3215_MOTOR_BAY_INTERFACES["frame_servo_install"]

    assert tuple(st3215_installed_location().position) == pytest.approx(frame.xyz_mm)


def test_upper_arm_servo_frame_matches_source_location() -> None:
    frame = UPPER_ARM_INTERFACES["frame_st3215_servo_install"]

    assert tuple(st3215_preview_location().position) == pytest.approx(frame.xyz_mm)


def test_upper_arm_joint_frames_match_source_locations() -> None:
    output_frame = UPPER_ARM_INTERFACES["frame_st3215_output_axis"]
    fork_frame = UPPER_ARM_INTERFACES["frame_distal_fork_axis"]

    assert tuple(st3215_output_axis_location().position) == pytest.approx(
        output_frame.xyz_mm
    )
    assert tuple(distal_fork_axis_location().position) == pytest.approx(
        fork_frame.xyz_mm
    )


def test_servo_output_fork_frames_match_source_locations() -> None:
    attachment = ST3215_SERVO_OUTPUT_FORK_INTERFACES["frame_attachment_datum"]
    output = ST3215_SERVO_OUTPUT_FORK_INTERFACES["frame_st3215_output_axis"]

    assert tuple(st3215_servo_output_fork.cut_plane_normal_global()) == pytest.approx(
        UPPER_ARM_INTERFACES["frame_servo_output_fork_attachment"].axis
    )
    assert attachment.xyz_mm == (0.0, 0.0, 0.0)
    assert tuple(
        st3215_servo_output_fork.output_axis_location_local().position
    ) == pytest.approx(output.xyz_mm)
    assert st3215_servo_output_fork.output_axis_direction_local() == pytest.approx(
        output.axis
    )
