"""Pure tests for the separate stair-climbing RL experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISAAC_DIR = PROJECT_ROOT / "simulation" / "isaac"
RL_DIR = ISAAC_DIR / "rl"
STAIRS_DIR = RL_DIR / "stairs"
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(STAIRS_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _policy_transfer import transfer_policy_state  # noqa: E402
from _run_support import (  # noqa: E402
    build_model_manifest,
    expected_ppo_algorithm_contract,
    model_manifest_path,
    validate_model_manifest,
    write_model_manifest,
)
from _stair_geometry import stair_layer_boxes  # noqa: E402
from _stair_rl_contract import (  # noqa: E402
    curriculum_active_steps,
    goal_x_for_active_steps,
    pack_stair_policy_observation,
    progress_gate_failures,
    stair_failure_reasons,
    stair_height_at_x,
    stair_observation_fields,
    stair_reward_terms,
)


@pytest.fixture
def config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v1.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v2_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v2.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


def test_stair_layers_match_the_reviewed_four_step_profile(config: dict) -> None:
    staircase = config["task"]["staircase"]
    boxes = stair_layer_boxes(staircase)

    assert len(boxes) == 4
    assert boxes[0]["center_xyz_m"] == pytest.approx((1.26, 0.0, 0.02))
    assert boxes[0]["size_xyz_m"] == pytest.approx((1.42, 1.0, 0.04))
    assert boxes[1]["center_xyz_m"] == pytest.approx((1.375, 0.0, 0.06))
    assert boxes[1]["size_xyz_m"] == pytest.approx((1.19, 1.0, 0.04))
    assert boxes[2]["center_xyz_m"] == pytest.approx((1.49, 0.0, 0.10))
    assert boxes[2]["size_xyz_m"] == pytest.approx((0.96, 1.0, 0.04))
    assert boxes[3]["center_xyz_m"] == pytest.approx((1.605, 0.0, 0.14))
    assert boxes[3]["size_xyz_m"] == pytest.approx((0.73, 1.0, 0.04))
    assert boxes[-1]["exposed_top_z_m"] == pytest.approx(0.16)
    assert boxes[-1]["exposed_tread_end_x_m"] == pytest.approx(1.97)


def test_height_query_matches_treads_and_landing(config: dict) -> None:
    staircase = config["task"]["staircase"]

    assert stair_height_at_x(0.54, staircase) == 0.0
    assert stair_height_at_x(0.55, staircase) == pytest.approx(0.04)
    assert stair_height_at_x(0.78, staircase) == pytest.approx(0.08)
    assert stair_height_at_x(1.01, staircase) == pytest.approx(0.12)
    assert stair_height_at_x(1.24, staircase) == pytest.approx(0.16)
    assert stair_height_at_x(1.90, staircase) == pytest.approx(0.16)
    assert stair_height_at_x(1.97, staircase) == 0.0


def test_stair_policy_observation_is_a_separate_57_value_contract(
    config: dict,
) -> None:
    staircase = config["task"]["staircase"]
    walking_observation = np.zeros(48, dtype=np.float32)
    observation = pack_stair_policy_observation(
        walking_observation=walking_observation,
        base_world_x_m=0.50,
        goal_world_x_m=1.57,
        staircase=staircase,
    )
    fields = stair_observation_fields(
        staircase["terrain_sample_offsets_m"]
    )

    assert observation.shape == (57,)
    assert observation.dtype == np.float32
    assert np.all(np.isfinite(observation))
    assert len(fields) == 57
    assert len(fields) == len(set(fields))
    assert np.any(observation[48:-1] > 0.0)
    assert observation[-1] > 0.0


def test_curriculum_reaches_the_top_platform_goal(config: dict) -> None:
    task = config["task"]
    staircase = task["staircase"]
    levels = task["curriculum"]["levels"]

    assert curriculum_active_steps(
        0.0,
        levels,
        maximum_steps=4,
    ) == 1
    assert curriculum_active_steps(
        0.30,
        levels,
        maximum_steps=4,
    ) == 3
    assert curriculum_active_steps(
        1.0,
        levels,
        maximum_steps=4,
    ) == 4
    assert goal_x_for_active_steps(staircase, 1) == pytest.approx(0.68)
    assert goal_x_for_active_steps(staircase, 4) == pytest.approx(1.57)


def test_forward_ascent_and_success_outscore_no_progress(config: dict) -> None:
    reward_config = config["task"]["reward"]
    common = {
        "command_velocity_xyz": (0.11, 0.0, 0.0),
        "body_angular_velocity_xyz": (0.0, 0.0, 0.0),
        "projected_gravity_xyz": (0.0, 0.0, -1.0),
        "base_clearance_m": 0.373,
        "lateral_position_m": 0.0,
        "heading_error_rad": 0.0,
        "joint_velocities_normalized": np.zeros(12),
        "action": np.zeros(12),
        "previous_action": np.zeros(12),
        "failed": False,
        "reward_config": reward_config,
    }
    progress = stair_reward_terms(
        **common,
        body_linear_velocity_xyz=(0.11, 0.0, 0.0),
        forward_progress_m=0.002,
        base_height_gain_m=0.04,
        terrain_height_gain_m=0.04,
        succeeded=True,
    )
    stuck = stair_reward_terms(
        **common,
        body_linear_velocity_xyz=(0.0, 0.0, 0.0),
        forward_progress_m=0.0,
        base_height_gain_m=0.0,
        terrain_height_gain_m=0.0,
        succeeded=False,
    )

    assert progress["forward_progress"] > stuck["forward_progress"]
    assert progress["terrain_height_gain"] > stuck["terrain_height_gain"]
    assert progress["success"] == pytest.approx(400.0)
    assert progress["total"] > stuck["total"]
    discounted_wait_value = stuck["total"] / (1.0 - config["ppo"]["gamma"])
    assert progress["success"] > discounted_wait_value


def test_stair_failures_use_local_clearance_and_corridor() -> None:
    assert stair_failure_reasons(
        base_clearance_m=0.373,
        lateral_position_m=0.0,
        world_x_m=0.7,
        projected_gravity_xyz=(0.0, 0.0, -1.0),
        minimum_base_clearance_m=0.20,
        minimum_upright_cosine=0.70,
        maximum_lateral_deviation_m=0.48,
        minimum_world_x_m=-1.20,
    ) == ()
    assert stair_failure_reasons(
        base_clearance_m=0.18,
        lateral_position_m=0.50,
        world_x_m=-1.30,
        projected_gravity_xyz=(0.0, 0.0, -0.5),
        minimum_base_clearance_m=0.20,
        minimum_upright_cosine=0.70,
        maximum_lateral_deviation_m=0.48,
        minimum_world_x_m=-1.20,
    ) == (
        "base_clearance_too_low",
        "body_tipped",
        "left_stair_corridor",
        "moved_too_far_backward",
    )


def test_flat_policy_transfer_expands_only_the_two_input_layers() -> None:
    torch = pytest.importorskip("torch")
    source = {
        "mlp_extractor.policy_net.0.weight": torch.full((4, 48), 2.0),
        "mlp_extractor.value_net.0.weight": torch.full((4, 48), 3.0),
        "action_net.bias": torch.full((2,), 4.0),
    }
    target = {
        "mlp_extractor.policy_net.0.weight": torch.ones((4, 57)),
        "mlp_extractor.value_net.0.weight": torch.ones((4, 57)),
        "action_net.bias": torch.zeros((2,)),
    }

    transferred, report = transfer_policy_state(
        source,
        target,
        source_observation_size=48,
    )

    assert report["expanded_input_count"] == 2
    assert report["copied_exact_count"] == 1
    assert torch.all(
        transferred["mlp_extractor.policy_net.0.weight"][:, :48] == 2.0
    )
    assert torch.all(
        transferred["mlp_extractor.policy_net.0.weight"][:, 48:] == 0.0
    )
    assert torch.all(transferred["action_net.bias"] == 4.0)


def test_stairs_config_is_separate_and_consistent(config: dict) -> None:
    task = config["task"]
    ppo = config["ppo"]

    assert config["schema_version"] == 1
    assert task["id"] == "Drobot-Quadruped-Stairs-v1"
    assert task["world"].endswith("quadruped_robot_stairs_world.usda")
    assert task["world_dependencies"] == [
        "exports/isaac/quadruped_robot_manual_world.usda",
        "exports/isaac/quadruped_robot_floating.usdc",
    ]
    assert task["physics_hz"] == 120
    assert task["control_hz"] == 60
    assert task["staircase"]["step_count"] == 4
    assert ppo["rollout_steps"] % ppo["batch_size"] == 0
    assert ppo["policy_hidden_layers"] == [256, 256]


def test_v2_starts_close_and_exposes_navigation_state(v2_config: dict) -> None:
    task = v2_config["task"]
    staircase = task["staircase"]
    start_x = task["reset_start_x_range_m"]
    observation = pack_stair_policy_observation(
        walking_observation=np.zeros(48, dtype=np.float32),
        base_world_x_m=0.20,
        base_world_y_m=0.10,
        heading_error_rad=0.25,
        goal_world_x_m=0.68,
        staircase=staircase,
        include_navigation_observation=True,
    )
    fields = stair_observation_fields(
        staircase["terrain_sample_offsets_m"],
        include_navigation_observation=True,
    )

    assert task["id"] == "Drobot-Quadruped-Stairs-v2"
    assert task["include_navigation_observation"] is True
    assert float(staircase["start_x_m"]) - max(start_x) <= 0.37
    assert observation.shape == (60,)
    assert len(fields) == 60
    assert fields[-3:] == (
        "lateral_offset_normalized",
        "heading_error_sin",
        "heading_error_cos",
    )
    assert observation[-3] == pytest.approx(0.20)
    assert observation[-2] == pytest.approx(np.sin(0.25))
    assert observation[-1] == pytest.approx(np.cos(0.25))


def test_v2_rewards_physical_height_and_has_an_early_abort_gate(
    v2_config: dict,
) -> None:
    task = v2_config["task"]
    reward = task["reward"]
    watchdog = task["progress_watchdog"]
    height_step = stair_reward_terms(
        command_velocity_xyz=(0.11, 0.0, 0.0),
        body_linear_velocity_xyz=(0.11, 0.0, 0.0),
        body_angular_velocity_xyz=(0.0, 0.0, 0.0),
        projected_gravity_xyz=(0.0, 0.0, -1.0),
        base_clearance_m=0.373,
        lateral_position_m=0.0,
        forward_progress_m=0.0,
        base_height_gain_m=0.04,
        terrain_height_gain_m=0.0,
        heading_error_rad=0.0,
        joint_velocities_normalized=np.zeros(12),
        action=np.zeros(12),
        previous_action=np.zeros(12),
        failed=False,
        succeeded=False,
        reward_config=reward,
    )

    assert reward["base_height_gain"] > reward["terrain_height_gain"]
    assert height_step["base_height_gain"] == pytest.approx(6.0)
    assert reward["centerline"] <= -8.0
    assert reward["heading_error"] < 0.0
    assert watchdog["enabled"] is True
    assert watchdog["initial_gate_steps"] == 100000
    assert watchdog["minimum_first_step_climb_episodes"] >= 3
    assert watchdog["minimum_base_elevation_gain_m"] >= 0.02
    assert task["curriculum"]["mode"] == "mastery"

    failures = progress_gate_failures(
        completed_episodes=100,
        first_step_climb_episodes=0,
        minimum_completed_episodes=40,
        minimum_first_step_climb_episodes=3,
        minimum_first_step_climb_rate=0.02,
    )
    assert failures == (
        "first_step_climb_episodes=0<3",
        "first_step_climb_rate=0.0000<0.0200",
    )
    assert progress_gate_failures(
        completed_episodes=100,
        first_step_climb_episodes=3,
        minimum_completed_episodes=40,
        minimum_first_step_climb_episodes=3,
        minimum_first_step_climb_rate=0.02,
    ) == ()


def test_full_and_smoke_ppo_contracts_cannot_be_confused(config: dict) -> None:
    ppo = config["ppo"]
    full = expected_ppo_algorithm_contract(
        ppo,
        training_mode="full",
        rollout_steps=2048,
        batch_size=256,
        epochs=10,
        observation_size=57,
        action_size=12,
    )
    smoke = expected_ppo_algorithm_contract(
        ppo,
        training_mode="smoke",
        rollout_steps=128,
        batch_size=64,
        epochs=2,
        observation_size=57,
        action_size=12,
    )

    assert full["training_mode"] == "full"
    assert smoke["training_mode"] == "smoke"
    assert full != smoke


def test_v2_ppo_contract_caps_policy_update_size(v2_config: dict) -> None:
    contract = expected_ppo_algorithm_contract(
        v2_config["ppo"],
        training_mode="full",
        rollout_steps=2048,
        batch_size=256,
        epochs=10,
        observation_size=60,
        action_size=12,
    )

    assert contract["learning_rate"] == 0.00005
    assert contract["target_kl"] == 0.03


def test_manifest_binds_all_composed_world_dependencies(
    config: dict,
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model.zip"
    config_path = tmp_path / "task.yaml"
    world_path = tmp_path / "stairs.usda"
    base_world_path = tmp_path / "base.usda"
    robot_path = tmp_path / "robot.usdc"
    for path, content in (
        (model_path, b"model"),
        (config_path, b"config"),
        (world_path, b"stairs"),
        (base_world_path, b"base"),
        (robot_path, b"robot"),
    ):
        path.write_bytes(content)
    environment_contract = {
        "task_id": "Drobot-Quadruped-Stairs-v1",
        "dof_names": ["joint"],
        "observation_fields": ["field"],
        "observation_size": 57,
        "action_size": 12,
        "physics_steps_per_control": 2,
        "staircase": config["task"]["staircase"],
    }
    algorithm_contract = expected_ppo_algorithm_contract(
        config["ppo"],
        training_mode="full",
        rollout_steps=2048,
        batch_size=256,
        epochs=10,
        observation_size=57,
        action_size=12,
    )
    dependencies = (base_world_path, robot_path)
    manifest = build_model_manifest(
        model_path=model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=dependencies,
        environment_contract=environment_contract,
        algorithm_contract=algorithm_contract,
        training_seed=142,
        transferred_from=None,
    )
    write_model_manifest(model_manifest_path(model_path), manifest)

    verification = validate_model_manifest(
        model_path=model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=dependencies,
        environment_contract=environment_contract,
        allow_unverified=False,
        expected_algorithm_contract=algorithm_contract,
    )
    assert verification["status"] == "PASS"

    smoke_contract = dict(algorithm_contract)
    smoke_contract.update(
        {
            "training_mode": "smoke",
            "rollout_steps": 128,
            "batch_size": 64,
            "epochs": 2,
        }
    )
    with pytest.raises(RuntimeError, match="PPO algorithm contract differs"):
        validate_model_manifest(
            model_path=model_path,
            config_path=config_path,
            world_path=world_path,
            world_dependencies=dependencies,
            environment_contract=environment_contract,
            allow_unverified=False,
            expected_algorithm_contract=smoke_contract,
        )

    robot_path.write_bytes(b"changed robot")
    with pytest.raises(RuntimeError, match="Model contract mismatch"):
        validate_model_manifest(
            model_path=model_path,
            config_path=config_path,
            world_path=world_path,
            world_dependencies=dependencies,
            environment_contract=environment_contract,
            allow_unverified=False,
            expected_algorithm_contract=algorithm_contract,
        )
