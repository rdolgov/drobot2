"""Pure tests for the hardware-reproducible stair depth observation."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ISAAC_DIR = PROJECT_ROOT / "simulation" / "isaac"
RL_DIR = ISAAC_DIR / "rl"
STAIRS_DIR = RL_DIR / "stairs"
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(STAIRS_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _stair_rl_contract import (  # noqa: E402
    pack_stair_policy_observation,
    stair_observation_fields,
)
from _vl53l5cx_contract import (  # noqa: E402
    apply_vl53l5cx_noise,
    compress_vl53l5cx_depth_grid,
    validate_vl53l5cx_config,
    vl53l5cx_observation_fields,
    vl53l5cx_ray_directions,
)


@pytest.fixture
def v7_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v7_vl53l5cx.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def test_v7_keeps_real_test_profile_and_exact_25cm_treads(
    v7_config: dict,
) -> None:
    task = v7_config["task"]
    sensor = task["terrain_perception"]
    validate_vl53l5cx_config(sensor, control_hz=task["control_hz"])
    assert task["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert task["staircase"]["rise_m"] == pytest.approx(0.18)
    assert task["robot_hardware_profile"]["id"] == (
        "one-leg-real-test-2026-07-28"
    )
    assert task["robot_hardware_profile"]["effort_cap_nm"] == pytest.approx(
        0.8825985
    )
    assert sensor["rows"] == sensor["columns"] == 8
    assert sensor["update_rate_hz"] == 15
    assert sensor["latency_frames"] == 1
    assert sensor["pitch_down_deg"] == pytest.approx(40.0)


def test_vl53l5cx_rays_have_reviewed_orientation(v7_config: dict) -> None:
    sensor = v7_config["task"]["terrain_perception"]
    rays = vl53l5cx_ray_directions(sensor)
    assert rays.shape == (8, 8, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=2), 1.0, atol=1e-6)
    assert np.all(rays[:, :, 0] > 0.0)
    assert np.all(rays[:, 0, 1] > 0.0)
    assert np.all(rays[:, -1, 1] < 0.0)
    assert np.all(rays[0, :, 2] > rays[-1, :, 2])
    top_down_deg = np.rad2deg(np.arcsin(-rays[0, 3, 2]))
    bottom_down_deg = np.rad2deg(np.arcsin(-rays[-1, 3, 2]))
    assert top_down_deg == pytest.approx(20.3125, abs=1e-4)
    assert bottom_down_deg == pytest.approx(59.6875, abs=1e-4)


def test_depth_compression_is_lane_by_row_and_handles_missing_zones(
    v7_config: dict,
) -> None:
    sensor = v7_config["task"]["terrain_perception"]
    grid = np.full((8, 8), np.nan, dtype=np.float32)
    grid[0, 0:3] = (0.30, 0.60, 0.90)
    grid[0, 3:5] = (1.20, 1.40)
    grid[0, 5:8] = (2.00, 2.50, 3.00)
    observation = compress_vl53l5cx_depth_grid(grid, sensor)
    assert observation.shape == (24,)
    assert observation[0] == pytest.approx(0.60 / 1.50)
    assert observation[8] == pytest.approx(1.30 / 1.50)
    assert observation[16] == pytest.approx(1.0)
    assert np.all(observation[[1, 9, 17]] == 1.0)


def test_noise_is_seeded_bounded_and_dropout_is_explicit(v7_config: dict) -> None:
    sensor = deepcopy(v7_config["task"]["terrain_perception"])
    sensor["dropout_probability"] = 0.0
    grid = np.full((8, 8), 1.0, dtype=np.float32)
    grid[0, 0] = 0.10
    first = apply_vl53l5cx_noise(grid, sensor, np.random.default_rng(12))
    second = apply_vl53l5cx_noise(grid, sensor, np.random.default_rng(12))
    np.testing.assert_array_equal(first, second)
    assert abs(float(first[0, 0]) - 0.10) <= 0.015 + 1e-7
    assert np.max(np.abs(first[1:, :] - 1.0)) <= 0.05 + 1e-7

    sensor["dropout_probability"] = 1.0
    dropped = apply_vl53l5cx_noise(grid, sensor, np.random.default_rng(12))
    assert np.all(np.isnan(dropped))


def test_v7_replaces_privileged_terrain_profile_with_24_depth_values(
    v7_config: dict,
) -> None:
    task = v7_config["task"]
    sensor_fields = vl53l5cx_observation_fields(task["terrain_perception"])
    fields = stair_observation_fields(
        task["staircase"]["terrain_sample_offsets_m"],
        include_navigation_observation=True,
        include_foot_progress_observation=True,
        terrain_observation_fields=sensor_fields,
    )
    observation = pack_stair_policy_observation(
        walking_observation=np.zeros(48, dtype=np.float32),
        base_world_x_m=0.20,
        goal_world_x_m=0.70,
        staircase=task["staircase"],
        include_navigation_observation=True,
        include_foot_progress_observation=True,
        foot_progress_normalized=(0.0, 0.0, 0.0, 0.0),
        next_foot_target_one_hot=(1.0, 0.0, 0.0, 0.0),
        terrain_observation_values=np.ones(24, dtype=np.float32),
    )
    assert len(sensor_fields) == 24
    assert len(fields) == observation.shape[0] == 84
    assert not any(field.startswith("terrain_height_delta") for field in fields)


def test_8x8_mode_rejects_the_4x4_60hz_headline(v7_config: dict) -> None:
    sensor = deepcopy(v7_config["task"]["terrain_perception"])
    sensor["update_rate_hz"] = 60
    with pytest.raises(ValueError, match="8 x 8 mode is limited to 15 Hz"):
        validate_vl53l5cx_config(sensor, control_hz=120)
