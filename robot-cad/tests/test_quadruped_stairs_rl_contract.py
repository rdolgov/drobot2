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

from _policy_transfer import (  # noqa: E402
    physical_action_output_ratios,
    transfer_policy_state,
)
from _run_support import (  # noqa: E402
    build_model_manifest,
    expected_ppo_algorithm_contract,
    model_manifest_path,
    validate_model_manifest,
    write_model_manifest,
)
from _stair_geometry import stair_layer_boxes  # noqa: E402
from _stair_rl_contract import (  # noqa: E402
    config_for_height_stage,
    curriculum_active_steps,
    foot_tread_progress,
    goal_x_for_active_steps,
    next_foot_target_index,
    pack_stair_policy_observation,
    progress_gate_failures,
    stair_failure_reasons,
    stair_goal_reached,
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


@pytest.fixture
def v3_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v3.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v4_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v4.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v5_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v5.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v6_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v6_180mm.yaml").open(
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


def test_strict_stair_goal_requires_simultaneous_foot_placement() -> None:
    common = {
        "base_world_x_m": 1.80,
        "base_elevation_gain_m": 0.68,
        "goal_world_x_m": 1.77,
        "minimum_base_elevation_gain_m": 0.648,
        "active_steps": 4,
        "required_feet_on_goal_tread": 4,
    }

    assert stair_goal_reached(
        **common,
        current_foot_steps=(4, 4, 4, 4),
    )
    assert not stair_goal_reached(
        **common,
        current_foot_steps=(4, 4, 4, 3),
    )
    assert not stair_goal_reached(
        **{**common, "base_elevation_gain_m": 0.40},
        current_foot_steps=(4, 4, 4, 4),
    )
    with pytest.raises(ValueError, match="required_feet_on_goal_tread"):
        stair_goal_reached(
            **{**common, "required_feet_on_goal_tread": 5},
            current_foot_steps=(4, 4, 4, 4),
        )


def test_foot_tread_progress_requires_forward_and_vertical_motion(
    v6_config: dict,
) -> None:
    staircase = v6_config["task"]["staircase"]
    common = {
        "highest_foot_steps": (0,),
        "staircase": staircase,
        "active_steps": 1,
        "approach_distance_m": 0.22,
    }
    no_lift = foot_tread_progress(
        **common,
        foot_tip_positions_m=((0.60, 0.0, 0.0),),
    )
    lift_too_early = foot_tread_progress(
        **common,
        foot_tip_positions_m=((0.20, 0.0, 0.18),),
    )
    coordinated = foot_tread_progress(
        **common,
        foot_tip_positions_m=((0.60, 0.0, 0.18),),
    )

    assert no_lift[0] == pytest.approx(0.0)
    assert lift_too_early[0] == pytest.approx(0.0)
    assert coordinated[0] > 0.8


def test_next_foot_target_advances_one_tread_in_configured_sequence() -> None:
    sequence = (0, 1, 2, 3)

    assert next_foot_target_index(
        (0, 0, 0, 0),
        active_steps=4,
        sequence_indices=sequence,
    ) == 0
    assert next_foot_target_index(
        (1, 0, 0, 0),
        active_steps=4,
        sequence_indices=sequence,
    ) == 1
    assert next_foot_target_index(
        (1, 1, 1, 1),
        active_steps=4,
        sequence_indices=sequence,
    ) == 0
    assert next_foot_target_index(
        (4, 4, 4, 4),
        active_steps=4,
        sequence_indices=sequence,
    ) is None


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


def test_flat_policy_transfer_can_preserve_physical_action_mean() -> None:
    torch = pytest.importorskip("torch")
    dof_names = (
        "front_left_hip_abduction",
        "front_left_hip_flexion",
        "front_left_knee",
    )
    ratios = physical_action_output_ratios(
        dof_names,
        {
            "hip_abduction": 0.12,
            "hip_flexion": 0.30,
            "knee": 0.40,
        },
        {
            "hip_abduction": 0.30,
            "hip_flexion": 0.75,
            "knee": 1.00,
        },
    )
    source = {
        "action_net.weight": torch.ones((3, 2)),
        "action_net.bias": torch.tensor([1.0, 2.0, 3.0]),
    }
    target = {
        "action_net.weight": torch.zeros((3, 2)),
        "action_net.bias": torch.zeros(3),
    }

    transferred, report = transfer_policy_state(
        source,
        target,
        source_observation_size=48,
        action_output_ratios=ratios,
    )

    assert ratios == pytest.approx((0.4, 0.4, 0.4))
    assert torch.allclose(
        transferred["action_net.weight"],
        torch.full((3, 2), 0.4),
    )
    assert torch.allclose(
        transferred["action_net.bias"],
        torch.tensor([0.4, 0.8, 1.2]),
    )
    assert report["physical_action_mean_preserved"] is True
    assert report["rescaled_action_outputs"] == [
        "action_net.weight",
        "action_net.bias",
    ]


def test_balance_transfer_keeps_only_the_shared_48_value_prefix() -> None:
    torch = pytest.importorskip("torch")
    source = {
        "mlp_extractor.policy_net.0.weight": torch.full((3, 56), 2.0),
        "mlp_extractor.value_net.0.weight": torch.full((3, 56), 3.0),
        "action_net.bias": torch.ones(2),
    }
    target = {
        "mlp_extractor.policy_net.0.weight": torch.full((3, 68), 9.0),
        "mlp_extractor.value_net.0.weight": torch.full((3, 68), 9.0),
        "action_net.bias": torch.zeros(2),
    }

    transferred, report = transfer_policy_state(
        source,
        target,
        source_observation_size=56,
        shared_observation_prefix_size=48,
    )

    assert torch.all(
        transferred["mlp_extractor.policy_net.0.weight"][:, :48] == 2.0
    )
    assert torch.all(
        transferred["mlp_extractor.policy_net.0.weight"][:, 48:] == 0.0
    )
    assert report["shared_observation_prefix_size"] == 48
    assert report["dropped_source_input_columns"] == 8


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


def test_v3_applies_the_physically_exercised_one_leg_profile(
    v3_config: dict,
    v2_config: dict,
) -> None:
    task = v3_config["task"]
    profile = task["robot_hardware_profile"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v3"
    assert profile["center_tick"] == 2048
    assert profile["encoder_directions"] == {
        "hip_abduction": 1,
        "hip_flexion": -1,
        "knee": -1,
    }
    assert profile["joint_limits_deg"] == {
        "hip_abduction": [-45.0, 45.0],
        "hip_flexion": [-90.0, 90.0],
        "knee": [-120.0, 120.0],
    }
    assert profile["effort_cap_nm"] == pytest.approx(
        profile["torque_limit_fraction"] * profile["stall_torque_nm"]
    )
    assert profile["speed_register"] == 350
    assert profile["speed_register_applied_as_rad_s"] is False
    assert task["action_scale_rad"]["hip_flexion"] > 0.36
    assert task["action_scale_rad"]["knee"] > 0.48
    assert (
        v3_config["ppo"]["initial_log_std"]
        < v2_config["ppo"]["initial_log_std"]
    )


def test_v4_transfer_preserves_flat_joint_target_amplitudes(
    v4_config: dict,
) -> None:
    task = v4_config["task"]
    transfer = task["flat_policy_transfer"]
    ratios = physical_action_output_ratios(
        (
            "front_left_hip_abduction",
            "front_left_hip_flexion",
            "front_left_knee",
        ),
        transfer["source_action_scale_rad"],
        task["action_scale_rad"],
    )

    assert task["id"] == "Drobot-Quadruped-Stairs-v4"
    assert transfer["preserve_physical_action_mean"] is True
    assert task["control_hz"] == task["physics_hz"] == 120
    assert task["target_velocity_body_m_s"] == (
        transfer["source_target_velocity_body_m_s"]
    )
    assert ratios == pytest.approx((1.0, 1.0, 1.0))
    assert v4_config["ppo"]["initial_log_std"] < -2.0
    assert task["success_minimum_base_elevation_fraction"] > 0.5
    assert task["reward"]["foot_lift_progress"] > 0.0
    assert task["reward"]["foot_step_placement"] > 0.0


def test_v4_foot_progress_shaping_is_bounded_to_new_progress(
    v4_config: dict,
) -> None:
    reward = v4_config["task"]["reward"]
    terms = stair_reward_terms(
        command_velocity_xyz=(0.15, 0.0, 0.0),
        body_linear_velocity_xyz=(0.15, 0.0, 0.0),
        body_angular_velocity_xyz=(0.0, 0.0, 0.0),
        projected_gravity_xyz=(0.0, 0.0, -1.0),
        base_clearance_m=0.373,
        lateral_position_m=0.0,
        forward_progress_m=0.0,
        base_height_gain_m=0.0,
        terrain_height_gain_m=0.0,
        heading_error_rad=0.0,
        joint_velocities_normalized=np.zeros(12),
        action=np.zeros(12),
        previous_action=np.zeros(12),
        failed=False,
        succeeded=False,
        reward_config=reward,
        foot_lift_progress_m=0.01,
        foot_step_placement_progress=1,
    )

    assert terms["foot_lift_progress"] == pytest.approx(3.0)
    assert terms["foot_step_placement"] == pytest.approx(100.0)
    standing = stair_reward_terms(
        command_velocity_xyz=(0.15, 0.0, 0.0),
        body_linear_velocity_xyz=(0.0, 0.0, 0.0),
        body_angular_velocity_xyz=(0.0, 0.0, 0.0),
        projected_gravity_xyz=(0.0, 0.0, -1.0),
        base_clearance_m=0.373,
        lateral_position_m=0.0,
        forward_progress_m=0.0,
        base_height_gain_m=0.0,
        terrain_height_gain_m=0.0,
        heading_error_rad=0.0,
        joint_velocities_normalized=np.zeros(12),
        action=np.zeros(12),
        previous_action=np.zeros(12),
        failed=False,
        succeeded=False,
        reward_config=reward,
    )

    assert standing["forward_velocity_tracking"] == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert standing["total"] < 0.10
    assert v4_config["task"]["stall_termination"]["after_seconds"] <= 5.0


def test_v5_learns_residuals_around_the_flat_gait(v5_config: dict) -> None:
    task = v5_config["task"]
    residual = task["residual_policy"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v5"
    assert residual["enabled"] is True
    assert residual["base_task_id"] == "Drobot-Quadruped-Walk-v1"
    assert residual["base_model"].endswith("drobot_walk_ppo_final.zip")
    assert residual["base_action_scale_rad"] == {
        "hip_abduction": 0.12,
        "hip_flexion": 0.30,
        "knee": 0.40,
    }
    assert task["action_scale_rad"]["hip_flexion"] > 0.30
    assert task["action_scale_rad"]["knee"] > 0.40
    assert v5_config["ppo"]["zero_action_mean_init"] is True

    stage_10 = config_for_height_stage(v5_config, "10mm")
    stage_40 = config_for_height_stage(v5_config, "40mm")
    assert stage_10["task"]["staircase"]["rise_m"] == pytest.approx(0.01)
    assert stage_10["task"]["world"].endswith("v5_10mm_world.usda")
    assert stage_40["task"]["staircase"]["rise_m"] == pytest.approx(0.04)
    assert stage_40["task"]["world"].endswith("stairs_v2_world.usda")


def test_v6_uses_exact_180mm_geometry_and_strict_success(
    v6_config: dict,
) -> None:
    task = v6_config["task"]
    staircase = task["staircase"]
    stage = config_for_height_stage(v6_config, "180mm")
    boxes = stair_layer_boxes(stage["task"]["staircase"])

    assert task["id"] == "Drobot-Quadruped-Stairs-v6-180mm"
    assert stage["task"]["id"] == "Drobot-Quadruped-Stairs-v6-180mm"
    assert staircase["rise_m"] == pytest.approx(0.18)
    assert staircase["tread_depth_m"] == pytest.approx(0.25)
    assert [stage["id"] for stage in v6_config["stair_height_stages"]] == [
        "10mm",
        "20mm",
        "30mm",
        "40mm",
        "60mm",
        "80mm",
        "100mm",
        "120mm",
        "140mm",
        "150mm",
        "160mm",
        "180mm",
    ]
    for height_stage in v6_config["stair_height_stages"]:
        staged_task = config_for_height_stage(v6_config, height_stage["id"])[
            "task"
        ]
        assert staged_task["staircase"]["tread_depth_m"] == pytest.approx(
            0.25
        )
    assert boxes[-1]["exposed_top_z_m"] == pytest.approx(0.72)
    assert task["success_required_feet_on_goal_tread"] == 4
    assert task["success_minimum_base_elevation_fraction"] >= 0.90
    assert task["robot_hardware_profile"]["effort_cap_nm"] == pytest.approx(
        0.8825985
    )
    assert task["action_scale_rad"]["hip_flexion"] >= 0.60
    assert task["action_scale_rad"]["knee"] >= 0.90
    fields = stair_observation_fields(
        staircase["terrain_sample_offsets_m"],
        include_navigation_observation=True,
        include_foot_progress_observation=True,
    )
    observation = pack_stair_policy_observation(
        walking_observation=np.zeros(48, dtype=np.float32),
        base_world_x_m=0.20,
        base_world_y_m=0.0,
        heading_error_rad=0.0,
        goal_world_x_m=0.70,
        staircase=staircase,
        include_navigation_observation=True,
        include_foot_progress_observation=True,
        foot_progress_normalized=(0.0, 0.0, 0.0, 0.0),
        next_foot_target_one_hot=(1.0, 0.0, 0.0, 0.0),
    )
    assert len(fields) == observation.shape[0] == 68
    assert task["foot_placement_sequence"] == [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ]
    assert task["reward"]["upright_deviation"] < 0.0
    assert abs(task["reward"]["failure"]) > (
        2
        * (
            task["reward"]["foot_tread_progress"]
            + task["reward"]["foot_step_placement"]
        )
    )
    assert task["reward"]["foot_tread_support"] > 0.0
    assert task["stall_termination"][
        "minimum_any_foot_tread_progress"
    ] > 0.0


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
    override = validate_model_manifest(
        model_path=model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=dependencies,
        environment_contract=environment_contract,
        allow_unverified=True,
        expected_algorithm_contract=algorithm_contract,
    )
    assert override["status"] == "SKIPPED"
    assert override["reason"] == "manifest_mismatch_and_override_enabled"
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
