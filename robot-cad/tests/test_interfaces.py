import pytest

from robot_cad.interfaces import ST3215_MOTOR_BAY_INTERFACES, UPPER_ARM_INTERFACES
from robot_cad.parts.st3215_motor_bay import st3215_installed_location
from robot_cad.parts.upper_arm import st3215_preview_location
from robot_cad.validation import validate_interface_frame


@pytest.mark.parametrize(
    "frame",
    [*ST3215_MOTOR_BAY_INTERFACES.values(), *UPPER_ARM_INTERFACES.values()],
)
def test_interface_axes_are_normalized(frame) -> None:
    validate_interface_frame(frame)


def test_motor_bay_servo_frame_matches_source_location() -> None:
    frame = ST3215_MOTOR_BAY_INTERFACES["frame_servo_install"]

    assert tuple(st3215_installed_location().position) == pytest.approx(frame.xyz_mm)


def test_upper_arm_servo_frame_matches_source_location() -> None:
    frame = UPPER_ARM_INTERFACES["frame_st3215_servo_install"]

    assert tuple(st3215_preview_location().position) == pytest.approx(frame.xyz_mm)
