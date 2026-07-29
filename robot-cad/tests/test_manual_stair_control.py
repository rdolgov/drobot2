"""Pure tests for the interactive real-stair leg controller."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_DIR = PROJECT_ROOT / "simulation" / "isaac"
EXPERIMENT_DIR = (
    ISAAC_DIR / "experiments" / "stair_feasibility"
)
for module_dir in (str(ISAAC_DIR), str(EXPERIMENT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _contract import validate_config
from _manual_control import (
    CONTROL_HELP,
    LEG_SELECTION_KEYS,
    MOTOR_NUMBER_TO_JOINT,
    ManualLegController,
    controller_from_experiment,
    motor_controller_from_experiment,
)

CONFIG_PATH = EXPERIMENT_DIR / "real_stair_feasibility.yaml"


def _controller() -> ManualLegController:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return controller_from_experiment(validate_config(config))


def _motor_controller():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return motor_controller_from_experiment(validate_config(config))


def test_documented_leg_selection_is_complete_and_stable():
    assert LEG_SELECTION_KEYS == {
        "KEY_1": "front_left",
        "KEY_2": "front_right",
        "KEY_3": "rear_left",
        "KEY_4": "rear_right",
    }
    assert "W / S" in CONTROL_HELP
    assert "E / D" in CONTROL_HELP
    assert "Q / A" in CONTROL_HELP


def test_only_selected_leg_moves_and_reset_restores_nominal_pose():
    controller = _controller()
    original = controller.snapshot()
    assert controller.select_from_key("KEY_2")
    assert controller.advance(
        {"W", "E"},
        dt_s=0.1,
        foot_speed_m_s=0.04,
        abduction_speed_rad_s=math.radians(15.0),
    )
    moved = controller.snapshot()
    assert moved["targets"]["front_right"]["forward_m"] > (
        original["targets"]["front_right"]["forward_m"]
    )
    assert moved["targets"]["front_right"]["down_m"] < (
        original["targets"]["front_right"]["down_m"]
    )
    for leg in ("front_left", "rear_left", "rear_right"):
        assert moved["targets"][leg] == original["targets"][leg]
    controller.reset()
    assert controller.snapshot()["targets"] == original["targets"]


def test_target_motion_stays_inside_full_joint_margins():
    controller = _controller()
    for _ in range(1000):
        controller.advance(
            {"W", "E", "Q"},
            dt_s=1.0 / 60.0,
            foot_speed_m_s=0.10,
            abduction_speed_rad_s=math.radians(40.0),
        )
    joints = controller.joint_targets_by_name()
    assert abs(joints["front_left_hip_abduction"]) < math.radians(25.0)
    assert abs(joints["front_left_hip_flexion"]) < math.radians(60.0)
    assert abs(joints["front_left_knee"]) < math.radians(90.0)
    assert controller.last_rejection is not None


@pytest.mark.parametrize(
    ("key", "field", "direction"),
    [
        ("W", "forward_m", 1.0),
        ("S", "forward_m", -1.0),
        ("E", "down_m", -1.0),
        ("D", "down_m", 1.0),
        ("Q", "hip_abduction_rad", 1.0),
        ("A", "hip_abduction_rad", -1.0),
    ],
)
def test_each_motion_key_changes_the_documented_axis(key, field, direction):
    controller = _controller()
    before = getattr(controller.targets["front_left"], field)
    assert controller.advance(
        {key},
        dt_s=0.05,
        foot_speed_m_s=0.02,
        abduction_speed_rad_s=math.radians(10.0),
    )
    after = getattr(controller.targets["front_left"], field)
    assert (after - before) * direction > 0.0


def test_motor_numbers_are_stable_leg_by_leg():
    assert MOTOR_NUMBER_TO_JOINT == {
        1: "front_left_hip_abduction",
        2: "front_left_hip_flexion",
        3: "front_left_knee",
        4: "front_right_hip_abduction",
        5: "front_right_hip_flexion",
        6: "front_right_knee",
        7: "rear_left_hip_abduction",
        8: "rear_left_hip_flexion",
        9: "rear_left_knee",
        10: "rear_right_hip_abduction",
        11: "rear_right_hip_flexion",
        12: "rear_right_knee",
    }


def test_two_digit_motor_selection_requires_enter_and_reports_number():
    controller = _motor_controller()
    assert controller.selected_motor_number == 1
    assert controller.select_from_key("KEY_1")
    assert controller.select_from_key("KEY_0")
    assert controller.selection_buffer == "10"
    assert controller.selected_motor_number == 1
    assert controller.select_from_key("ENTER")
    assert controller.selected_motor_number == 10
    assert controller.selected_joint_name == "rear_right_hip_abduction"
    assert controller.snapshot()["selected_motor_number"] == 10


def test_direct_motor_target_persists_and_only_selected_motor_changes():
    controller = _motor_controller()
    controller.select_from_key("KEY_1")
    controller.select_from_key("KEY_2")
    controller.select_from_key("ENTER")
    before = controller.joint_targets_by_name()
    assert controller.advance(
        {"UP"},
        dt_s=0.1,
        foot_speed_m_s=0.01,
        abduction_speed_rad_s=math.radians(10.0),
        motor_speed_rad_s=math.radians(20.0),
    )
    after = controller.joint_targets_by_name()
    changed = [
        name for name in after if not math.isclose(after[name], before[name])
    ]
    assert changed == ["rear_right_knee"]
    held_value = after["rear_right_knee"]
    assert not controller.advance(
        set(),
        dt_s=0.1,
        foot_speed_m_s=0.01,
        abduction_speed_rad_s=math.radians(10.0),
        motor_speed_rad_s=math.radians(20.0),
    )
    assert controller.targets_rad["rear_right_knee"] == held_value


def test_direct_motor_target_stops_at_joint_margin_and_can_zero():
    controller = _motor_controller()
    for _ in range(1000):
        controller.advance(
            {"UP"},
            dt_s=1.0 / 60.0,
            foot_speed_m_s=0.01,
            abduction_speed_rad_s=math.radians(10.0),
            motor_speed_rad_s=math.radians(90.0),
        )
    assert abs(controller.targets_rad[controller.selected_joint_name]) < (
        math.radians(25.0)
    )
    assert controller.last_rejection is not None
    controller.zero_selected()
    assert controller.targets_rad[controller.selected_joint_name] == 0.0
