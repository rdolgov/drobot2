"""Pure tests for the simplified 190 mm single-foot-lift RL task."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_DIR = PROJECT_ROOT / "simulation" / "isaac"
RL_DIR = ISAAC_DIR / "rl"
FOOT_LIFT_DIR = RL_DIR / "foot_lift"
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(FOOT_LIFT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _foot_lift_contract import (  # noqa: E402
    FOOT_LIFT_OBSERVATION_FIELDS,
    FOOT_LIFT_OBSERVATION_SIZE,
    desired_foot_lift_m,
    foot_lift_failure_reasons,
    foot_lift_reward_terms,
    foot_lift_success_reached,
    lift_curriculum_level,
    pack_foot_lift_observation,
    support_triangle_signed_margin_m,
)
from _quadruped_runtime import leg_ik  # noqa: E402


@pytest.fixture
def config() -> dict:
    with (FOOT_LIFT_DIR / "quadruped_foot_lift_v1.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def test_lift_reference_commands_200_mm_for_a_190_mm_gate(config: dict) -> None:
    lift = config["task"]["foot_lift"]
    common = {
        "target_lift_m": lift["reference_lift_m"],
        "ramp_start_seconds": lift["ramp_start_seconds"],
        "ramp_duration_seconds": lift["ramp_duration_seconds"],
    }

    assert desired_foot_lift_m(0.50, **common) == pytest.approx(0.0)
    assert desired_foot_lift_m(1.75, **common) == pytest.approx(0.10)
    assert desired_foot_lift_m(3.00, **common) == pytest.approx(0.20)
    assert desired_foot_lift_m(5.00, **common) == pytest.approx(0.20)


def test_190mm_reference_is_inside_measured_joint_limits(config: dict) -> None:
    task = config["task"]
    down_m = task["nominal_stance"]["down_m"] - task["foot_lift"]["reference_lift_m"]
    hip, knee = leg_ik(
        "front_left",
        down_m,
        task["nominal_stance"]["fore_aft_m"] + task["foot_lift"]["target_forward_offset_m"],
    )

    assert down_m == pytest.approx(0.09)
    assert math.radians(-90.0) < hip < math.radians(90.0)
    assert math.radians(-120.0) < knee < math.radians(120.0)


def test_full_weight_shift_and_lift_reference_remain_reachable(config: dict) -> None:
    task = config["task"]
    stance_down = task["nominal_stance"]["down_m"]
    lateral_shift = task["weight_shift"]["lateral_m"]
    forward_shift = task["weight_shift"]["forward_m"]
    nominal_forward = task["nominal_stance"]["fore_aft_m"]
    support_down = math.hypot(stance_down, lateral_shift)
    swing_down = math.hypot(
        stance_down - task["foot_lift"]["reference_lift_m"],
        lateral_shift,
    )
    support_forward = nominal_forward + forward_shift
    swing_forward = support_forward + task["foot_lift"]["target_forward_offset_m"]

    support_hip, support_knee = leg_ik(
        "front_right",
        support_down,
        support_forward,
    )
    swing_hip, swing_knee = leg_ik(
        "front_left",
        swing_down,
        swing_forward,
    )
    for angle in (support_hip, swing_hip):
        assert math.radians(-90.0) < angle < math.radians(90.0)
    for angle in (support_knee, swing_knee):
        assert math.radians(-120.0) < angle < math.radians(120.0)


def test_skill_observation_extends_hardware_walk_state_to_56_values() -> None:
    observation = pack_foot_lift_observation(
        walking_observation=np.zeros(48),
        target_lift_m=0.19,
        desired_lift_m=0.095,
        measured_lift_m=0.08,
        maximum_lift_m=0.09,
        base_height_error_m=-0.01,
        base_displacement_xy_m=(0.01, -0.02),
        maximum_support_foot_lift_m=0.005,
    )

    assert FOOT_LIFT_OBSERVATION_SIZE == 56
    assert len(FOOT_LIFT_OBSERVATION_FIELDS) == 56
    assert len(set(FOOT_LIFT_OBSERVATION_FIELDS)) == 56
    assert observation.shape == (56,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))


def test_success_requires_full_lift_upright_body_and_three_support_feet() -> None:
    common = {
        "desired_lift_m": 0.19,
        "target_lift_m": 0.19,
        "minimum_success_lift_m": 0.19,
        "projected_gravity_xyz": (0.0, 0.0, -1.0),
        "base_height_error_m": 0.01,
        "base_displacement_xy_m": (0.02, 0.01),
        "maximum_support_foot_lift_m": 0.01,
        "minimum_upright_cosine": math.cos(math.radians(12.0)),
        "maximum_base_height_error_m": 0.05,
        "maximum_base_displacement_m": 0.06,
        "maximum_support_foot_lift_allowed_m": 0.025,
    }

    assert foot_lift_success_reached(**common, measured_lift_m=0.19)
    assert not foot_lift_success_reached(**common, measured_lift_m=0.189)
    assert not foot_lift_success_reached(
        **{**common, "maximum_support_foot_lift_m": 0.03},
        measured_lift_m=0.19,
    )


def test_failure_reasons_distinguish_fall_drift_and_support_loss() -> None:
    reasons = foot_lift_failure_reasons(
        base_height_m=0.20,
        projected_gravity_xyz=(0.0, 0.0, -0.5),
        base_displacement_xy_m=(0.20, 0.0),
        maximum_support_foot_lift_m=0.07,
        minimum_base_height_m=0.25,
        minimum_upright_cosine=0.90,
        maximum_base_displacement_m=0.15,
        maximum_support_foot_lift_allowed_m=0.06,
    )

    assert reasons == (
        "base_too_low",
        "body_tipped",
        "base_drifted",
        "support_foot_lost",
    )


def test_stable_full_lift_outscores_a_failed_low_lift(config: dict) -> None:
    reward = config["task"]["reward"]
    common = {
        "desired_lift_m": 0.19,
        "lift_progress_m": 0.001,
        "base_displacement_xy_m": (0.0, 0.0),
        "maximum_support_foot_lift_m": 0.0,
        "body_angular_velocity_xyz": (0.0, 0.0, 0.0),
        "projected_gravity_xyz": (0.0, 0.0, -1.0),
        "joint_velocities_normalized": np.zeros(12),
        "action": np.zeros(12),
        "previous_action": np.zeros(12),
        "reward_config": reward,
    }
    success = foot_lift_reward_terms(
        **common,
        measured_lift_m=0.19,
        tracking_target_reached=True,
        base_height_error_m=0.0,
        failed=False,
        succeeded=True,
    )
    failed = foot_lift_reward_terms(
        **common,
        measured_lift_m=0.04,
        tracking_target_reached=False,
        base_height_error_m=-0.12,
        failed=True,
        succeeded=False,
    )

    assert success["total"] > failed["total"]
    assert success["success"] == pytest.approx(500.0)
    assert failed["failure"] == pytest.approx(-250.0)


def test_config_keeps_real_torque_limit_and_removes_vision(config: dict) -> None:
    task = config["task"]

    assert task["id"] == "Drobot-Quadruped-Foot-Lift-v1-190mm-Supported"
    assert task["foot_lift"]["target_lift_m"] == pytest.approx(0.19)
    assert task["foot_lift"]["reference_lift_m"] == pytest.approx(0.20)
    assert task["foot_lift"]["target_forward_offset_m"] == pytest.approx(0.11)
    assert task["weight_shift"] == {
        "start_seconds": 0.25,
        "duration_seconds": 1.25,
        "forward_m": 0.0,
        "lateral_m": 0.0,
    }
    assert task["robot_hardware_profile"]["effort_cap_nm"] == pytest.approx(0.8825985)
    assert "terrain_perception" not in task
    assert task["target_velocity_body_m_s"] == [0.0, 0.0, 0.0]
    assert task["base_support"]["mode"] == "pose_hold"
    assert config["ppo"]["zero_action_mean_init"] is True


def test_unsupported_balance_curriculum_reaches_190mm_final_stage() -> None:
    with (FOOT_LIFT_DIR / "quadruped_foot_lift_v2_balance.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        balance_config = yaml.safe_load(stream)

    task = balance_config["task"]
    levels = task["lift_curriculum"]["levels"]
    assert task["base_support"]["mode"] == "none"
    assert task["foot_lift"]["target_lift_m"] == pytest.approx(0.19)
    assert task["weight_shift"]["forward_m"] != 0.0
    assert task["weight_shift"]["lateral_m"] != 0.0
    assert lift_curriculum_level(levels, 0.0)["id"] == "unload-20mm"
    assert lift_curriculum_level(levels, 0.51)["id"] == "lift-90mm"
    assert lift_curriculum_level(levels, 1.0)["target_lift_m"] == pytest.approx(0.19)

    reward = task["reward"]
    common = {
        "desired_lift_m": 0.19,
        "lift_progress_m": 0.0,
        "tracking_target_reached": False,
        "base_height_error_m": 0.0,
        "base_displacement_xy_m": (0.0, 0.0),
        "maximum_support_foot_lift_m": 0.0,
        "body_angular_velocity_xyz": (0.0, 0.0, 0.0),
        "projected_gravity_xyz": (0.0, 0.0, -1.0),
        "joint_velocities_normalized": np.zeros(12),
        "action": np.zeros(12),
        "previous_action": np.zeros(12),
        "failed": False,
        "succeeded": False,
        "reward_config": reward,
    }
    low = foot_lift_reward_terms(**common, measured_lift_m=0.0)
    higher = foot_lift_reward_terms(**common, measured_lift_m=0.05)
    assert higher["total"] > low["total"]
    assert higher["lift_height"] > low["lift_height"]
    assert higher["lift_error"] > low["lift_error"]


def test_support_triangle_margin_is_positive_inside_and_negative_outside() -> None:
    support = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))

    assert support_triangle_signed_margin_m((0.2, 0.2), support) == pytest.approx(0.2)
    assert support_triangle_signed_margin_m((-0.1, 0.2), support) == pytest.approx(-0.1)
