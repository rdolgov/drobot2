from __future__ import annotations

from pathlib import Path

import pytest

from drobot_leg_testbed.controller import LegController
from drobot_leg_testbed.model import calibration_from_centers, load_config

ROOT = Path(__file__).resolve().parents[1]


class FakeBus:
    def __init__(self) -> None:
        self.positions = {1: 2048, 2: 2048, 3: 2048}
        self.calls: list[tuple] = []

    def read_position(self, servo_id: int) -> int:
        self.calls.append(("read", servo_id))
        return self.positions[servo_id]

    def write_position(self, servo_id: int, raw: int, _bus_config) -> None:
        self.calls.append(("write", servo_id, raw))
        self.positions[servo_id] = raw

    def enable_torque(self, servo_id: int) -> None:
        self.calls.append(("enable", servo_id))

    def disable_torque(self, servo_id: int) -> None:
        self.calls.append(("disable", servo_id))


def _controller() -> tuple[LegController, FakeBus]:
    config = load_config(ROOT / "leg.example.toml")
    calibration = calibration_from_centers(
        config,
        {motor.name: 2048 for motor in config.motors},
    )
    bus = FakeBus()
    return LegController(config, calibration, bus), bus


def test_arm_writes_present_position_before_enabling_torque() -> None:
    controller, bus = _controller()
    motor = controller.config.motor(1)

    state = controller.arm(motor)

    assert state.degrees == 0.0
    assert bus.calls == [
        ("read", 1),
        ("write", 1, 2048),
        ("enable", 1),
    ]


def test_command_requires_arm_and_rejects_large_step() -> None:
    controller, _bus = _controller()
    motor = controller.config.motor(1)

    with pytest.raises(RuntimeError, match="disarmed"):
        controller.command(motor, 1.0)

    controller.arm(motor)
    with pytest.raises(ValueError, match="maximum"):
        controller.command(motor, 5.1)


def test_nudge_moves_only_selected_motor_and_disarm_is_explicit() -> None:
    controller, bus = _controller()
    motor = controller.config.motor(2)
    controller.arm(motor)

    state = controller.nudge(motor, -2.0)
    controller.disarm(motor)

    assert state.motor.name == "hip_flexion"
    assert state.degrees == -2.0
    assert ("write", 2, state.raw_position) in bus.calls
    assert bus.calls[-1] == ("disable", 2)
