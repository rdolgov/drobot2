"""Tests for the Isaac walking-policy observation and reward contract."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_DIR = PROJECT_ROOT / "simulation" / "isaac"
RL_DIR = ISAAC_DIR / "rl"
for module_dir in (str(ISAAC_DIR), str(RL_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _rl_contract import (  # noqa: E402
    POLICY_OBSERVATION_FIELDS,
    POLICY_OBSERVATION_SIZE,
    pack_policy_observation,
    termination_reasons,
    walking_reward_terms,
)


@pytest.fixture
def task_config() -> dict:
    with (RL_DIR / "quadruped_walk_v1.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def _reward(
    reward_config: dict,
    *,
    forward_m_s: float,
    gravity_z: float = -1.0,
    terminated: bool = False,
) -> dict[str, float]:
    return walking_reward_terms(
        command_velocity_xyz=(0.15, 0.0, 0.0),
        body_linear_velocity_xyz=(forward_m_s, 0.0, 0.0),
        body_angular_velocity_xyz=(0.0, 0.0, 0.0),
        projected_gravity_xyz=(0.0, 0.0, gravity_z),
        base_height_m=0.373,
        joint_velocities_normalized=np.zeros(12),
        action=np.zeros(12),
        previous_action=np.zeros(12),
        terminated=terminated,
        reward_config=reward_config,
    )


def test_policy_observation_is_hardware_reproducible_48_value_contract() -> None:
    observation = pack_policy_observation(
        command_velocity_xyz=(0.15, 0.0, 0.0),
        imu_observation=(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0),
        joint_positions=np.linspace(-0.4, 0.4, 12),
        nominal_joint_positions=np.zeros(12),
        joint_velocities=np.zeros(12),
        joint_max_velocities=np.full(12, 4.712389),
        previous_action=np.zeros(12),
    )

    assert POLICY_OBSERVATION_SIZE == 48
    assert observation.shape == (48,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert len(POLICY_OBSERVATION_FIELDS) == len(set(POLICY_OBSERVATION_FIELDS))
    assert not any("base_linear_velocity" in name for name in POLICY_OBSERVATION_FIELDS)


def test_forward_velocity_tracking_is_better_at_the_command(task_config: dict) -> None:
    reward_config = task_config["task"]["reward"]

    tracking = _reward(reward_config, forward_m_s=0.15)
    standing = _reward(reward_config, forward_m_s=0.0)

    assert tracking["forward_velocity_tracking"] > standing[
        "forward_velocity_tracking"
    ]
    assert tracking["total"] > standing["total"]


def test_upright_state_beats_tipped_and_termination_is_penalized(
    task_config: dict,
) -> None:
    reward_config = task_config["task"]["reward"]

    upright = _reward(reward_config, forward_m_s=0.15)
    tipped = _reward(
        reward_config,
        forward_m_s=0.15,
        gravity_z=-0.4,
        terminated=True,
    )

    assert upright["upright"] > tipped["upright"]
    assert tipped["termination"] == pytest.approx(-5.0)
    assert upright["total"] > tipped["total"]


def test_fall_termination_reasons_are_explicit() -> None:
    assert termination_reasons(
        base_height_m=0.373,
        projected_gravity_xyz=(0.0, 0.0, -1.0),
        minimum_base_height_m=0.22,
        minimum_upright_cosine=0.78,
    ) == ()
    assert termination_reasons(
        base_height_m=0.19,
        projected_gravity_xyz=(0.0, 0.0, -0.5),
        minimum_base_height_m=0.22,
        minimum_upright_cosine=0.78,
    ) == ("base_too_low", "body_tipped")


def test_rl_config_points_to_the_validated_sensor_world(task_config: dict) -> None:
    task = task_config["task"]
    ppo = task_config["ppo"]

    assert task_config["schema_version"] == 1
    assert (PROJECT_ROOT / task["world"]).is_file()
    assert task["imu_prim"].endswith("/body_imu")
    assert task["camera_prim"].endswith("/lekiwi_camera")
    assert task["physics_hz"] == 120
    assert task["control_hz"] == 60
    assert set(task["action_scale_rad"]) == {
        "hip_abduction",
        "hip_flexion",
        "knee",
    }
    assert ppo["rollout_steps"] % ppo["batch_size"] == 0
