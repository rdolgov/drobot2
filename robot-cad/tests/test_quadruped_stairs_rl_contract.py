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
    observation_prefix_compatibility,
    physical_action_output_ratios,
    predict_with_observation_prefix,
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
    PLACEMENT_REFERENCE_OBSERVATION_FIELDS,
    SUPPORT_REGULATION_OBSERVATION_FIELDS,
    balance_target_error_xy,
    bounded_support_incenter_target_xy,
    compose_bounded_residual_action,
    config_for_height_stage,
    curriculum_active_steps,
    foot_tread_progress,
    goal_x_for_active_steps,
    inter_leg_pre_unload_gate_failures,
    inter_leg_transfer_state,
    joint_effort_telemetry_sample,
    next_foot_target_index,
    overlay_masked_action,
    pack_placement_reference_observation,
    pack_stair_policy_observation,
    pack_support_regulation_observation,
    placement_advance_clearance_gate_state,
    placement_completion_settle_gate_failures,
    placement_contact_reached,
    placement_curriculum_level,
    placement_lift_hold_reached,
    placement_phase_ready,
    placement_policy_action_mask,
    placement_reference_state,
    placement_success_mode,
    placement_transfer_ready,
    post_landing_reposition_snapshot,
    progress_gate_failures,
    split_post_clearance_advance_fractions,
    stabilized_support_reference_base_delta,
    staged_support_rear_pitch_scale,
    staged_swing_outward_offset_m,
    staged_swing_reference_base_delta,
    stair_failure_reasons,
    stair_goal_reached,
    stair_height_at_x,
    stair_observation_fields,
    stair_reward_terms,
    support_load_share_vertical_corrections,
    support_margin_constrained_target_xy,
    support_pitch_vertical_corrections,
    support_triangle_incenter_xy,
    touchdown_load_lift_correction_m,
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


@pytest.fixture
def v8_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v8_single_tread_placement.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v9_config() -> dict:
    with (STAIRS_DIR / "quadruped_stairs_v9_front_pair_placement.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v10_config() -> dict:
    with (
        STAIRS_DIR
        / "quadruped_stairs_v10_front_right_single_tread_placement.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v11_config() -> dict:
    with (
        STAIRS_DIR
        / "quadruped_stairs_v11_front_right_after_left_training.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v12_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v12_front_right_lift_hold.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v13_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v13_front_right_stabilized_lift.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v14_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v14_front_pair_right_then_left.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v15_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v15_front_left_stabilized_lift.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v16_config() -> dict:
    with (
        STAIRS_DIR
        / "quadruped_stairs_v16_front_pair_proprioceptive_support.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v24_config() -> dict:
    with (
        STAIRS_DIR
        / "quadruped_stairs_v24_front_pair_conservative_support.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v29_config() -> dict:
    with (
        STAIRS_DIR
        / "quadruped_stairs_v29_preunload_com_gate.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v31_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v31_clearance_tracking.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v32_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v32_compact_support_regulation.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v33_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v33_compact_support_pitch.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v34_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v34_imu_pitch_feedback.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v35_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v35_full_first_tread.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v36_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v36_transfer_support_residual.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v37_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v37_joint_clearance_support.yaml"
    ).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def v38_config() -> dict:
    with (
        STAIRS_DIR / "quadruped_stairs_v38_positive_margin_rear_transfer.yaml"
    ).open("r", encoding="utf-8") as stream:
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
    assert stair_failure_reasons(
        base_clearance_m=0.373,
        lateral_position_m=0.0,
        world_x_m=0.7,
        projected_gravity_xyz=(0.0, 0.0, -1.0),
        minimum_base_clearance_m=0.20,
        minimum_upright_cosine=0.70,
        maximum_lateral_deviation_m=0.48,
        minimum_world_x_m=-1.20,
        support_slip_m=0.0251,
        maximum_support_slip_m=0.025,
    ) == ("support_slip_exceeded",)


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


def test_legacy_policy_uses_an_unchanged_observation_prefix() -> None:
    report = observation_prefix_compatibility(
        source_observation_fields=("a", "b", "c"),
        target_observation_fields=("a", "b", "c", "load", "com"),
        source_observation_size=3,
        target_observation_size=5,
    )
    assert report == {
        "mode": "target_prefix_adapter",
        "source_observation_size": 3,
        "target_observation_size": 5,
        "appended_target_observation_count": 2,
    }

    class FakePolicy:
        class ObservationSpace:
            shape = (3,)

        observation_space = ObservationSpace()

        def __init__(self) -> None:
            self.seen: np.ndarray | None = None

        def predict(
            self,
            observation: np.ndarray,
            *,
            deterministic: bool,
        ) -> tuple[np.ndarray, None]:
            assert deterministic is True
            self.seen = np.asarray(observation).copy()
            return np.ones(2, dtype=np.float32), None

    policy = FakePolicy()
    action, _ = predict_with_observation_prefix(
        policy,
        np.asarray([1.0, 2.0, 3.0, 99.0, 100.0], dtype=np.float32),
        deterministic=True,
    )
    assert policy.seen == pytest.approx((1.0, 2.0, 3.0))
    assert action == pytest.approx((1.0, 1.0))
    with pytest.raises(ValueError, match="not a target prefix"):
        observation_prefix_compatibility(
            source_observation_fields=("a", "wrong"),
            target_observation_fields=("a", "b", "c"),
            source_observation_size=2,
            target_observation_size=3,
        )


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


def test_v8_placement_curriculum_uses_blind_force_verified_250mm_tread(
    v8_config: dict,
) -> None:
    task = v8_config["task"]
    levels = task["placement_curriculum"]["levels"]

    assert task["id"] == (
        "Drobot-Quadruped-Stairs-v8-180mm-Single-Tread-Placement"
    )
    assert task["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert task["staircase"]["rise_m"] == pytest.approx(0.18)
    assert task["placement_reference"]["enabled"] is True
    assert task["placement_reference"]["swing_leg"] == "front_left"
    assert task["residual_policy"]["enabled"] is False
    assert task["target_velocity_body_m_s"] == [0.0, 0.0, 0.0]
    assert task["robot_hardware_profile"]["effort_cap_nm"] == pytest.approx(
        0.8825985
    )
    assert placement_curriculum_level(0.0, levels)["id"] == "near-edge-touch"
    assert placement_curriculum_level(0.5, levels)["id"] == "quarter-tread-load"
    assert placement_curriculum_level(1.0, levels)["id"] == "center-tread-load"
    assert levels[-1]["apex_lift_m"] >= 0.19
    assert task["placement_reference"]["contact_on_threshold_n"] > 0.0
    assert task["placement_reference"]["measurable_slip_threshold_m"] == pytest.approx(
        0.025
    )


def test_placement_reference_has_explicit_shift_lift_lower_and_hold_phases(
    v8_config: dict,
) -> None:
    task = v8_config["task"]
    timing = task["placement_reference"]["timing"]
    level = task["placement_curriculum"]["levels"][-1]

    shifted = placement_reference_state(2.25, timing=timing, level=level)
    lifted = placement_reference_state(5.5, timing=timing, level=level)
    advanced = placement_reference_state(9.5, timing=timing, level=level)
    lowering = placement_reference_state(10.25, timing=timing, level=level)
    held = placement_reference_state(12.0, timing=timing, level=level)

    assert shifted["phase"] == "weight_shift"
    assert shifted["shift_fraction"] == pytest.approx(1.0)
    assert lifted["phase"] == "advance"
    assert lifted["desired_lift_m"] == pytest.approx(level["apex_lift_m"])
    assert advanced["phase"] == "lower"
    assert advanced["desired_forward_offset_m"] == pytest.approx(
        level["swing_forward_offset_m"]
    )
    assert 0.0 < lowering["lower_fraction"] < 1.0
    assert held["phase"] == "hold"
    assert held["desired_lift_m"] == pytest.approx(level["landing_lift_m"])
    assert held["desired_forward_offset_m"] == pytest.approx(
        level["landing_forward_offset_m"]
    )
    assert held["contact_expected"] is True


def test_advance_clearance_gate_holds_then_times_out_safely() -> None:
    before_advance = placement_advance_clearance_gate_state(
        candidate_phase="lift",
        measured_clearance_m=0.160,
        minimum_clearance_m=0.190,
        held_steps=0,
        maximum_hold_steps=120,
    )
    assert before_advance == {
        "advance_due": False,
        "released": False,
        "hold_reference": False,
        "held_steps": 0,
        "timed_out": False,
        "clearance_error_m": pytest.approx(0.030),
    }

    held = placement_advance_clearance_gate_state(
        candidate_phase="advance",
        measured_clearance_m=0.170,
        minimum_clearance_m=0.190,
        held_steps=118,
        maximum_hold_steps=120,
    )
    assert held["hold_reference"] is True
    assert held["held_steps"] == 119
    assert held["timed_out"] is False

    timed_out = placement_advance_clearance_gate_state(
        candidate_phase="advance",
        measured_clearance_m=0.170,
        minimum_clearance_m=0.190,
        held_steps=119,
        maximum_hold_steps=120,
    )
    assert timed_out["hold_reference"] is True
    assert timed_out["timed_out"] is True

    released = placement_advance_clearance_gate_state(
        candidate_phase="advance",
        measured_clearance_m=0.191,
        minimum_clearance_m=0.190,
        held_steps=37,
        maximum_hold_steps=120,
    )
    assert released["released"] is True
    assert released["hold_reference"] is False
    assert released["held_steps"] == 37


def test_placement_observation_and_contact_gate_require_loaded_support(
    v8_config: dict,
) -> None:
    task = v8_config["task"]
    staircase = task["staircase"]
    base_fields = stair_observation_fields(
        staircase["terrain_sample_offsets_m"],
        include_navigation_observation=True,
        include_foot_progress_observation=True,
    )
    base_observation = np.zeros(len(base_fields), dtype=np.float32)
    observation = pack_placement_reference_observation(
        stair_observation=base_observation,
        phase_one_hot=(0.0, 0.0, 0.0, 0.0, 1.0),
        desired_swing_height_m=0.18,
        measured_swing_height_m=0.18,
        swing_x_error_m=0.0,
        swing_z_error_m=0.0,
        tread_normal_load_n=12.0,
        support_contact_fraction=1.0,
        support_margin_m=0.01,
        maximum_support_slip_m=0.004,
        staircase=staircase,
        contact_load_normalization_n=50.0,
    )
    assert observation.shape == (
        len(base_fields) + len(PLACEMENT_REFERENCE_OBSERVATION_FIELDS),
    )
    target_x = staircase["start_x_m"] + 0.24 * staircase["tread_depth_m"]
    common = {
        "swing_tip_position_m": (target_x, 0.18, 0.18),
        "swing_tread_normal_load_n": 12.0,
        "projected_gravity_xyz": (0.0, 0.0, -1.0),
        "staircase": staircase,
        "target_tread_fraction": 0.24,
        "target_x_tolerance_m": 0.065,
        "target_z_tolerance_m": 0.018,
        "contact_on_threshold_n": 1.0,
        "minimum_upright_cosine": 0.97,
    }
    assert placement_contact_reached(
        **common,
        support_ground_normal_loads_n=(10.0, 11.0, 12.0),
    )
    assert not placement_contact_reached(
        **common,
        support_ground_normal_loads_n=(10.0, 0.0, 12.0),
    )


def test_support_regulation_observation_exposes_load_com_and_saturation(
    v8_config: dict,
) -> None:
    staircase = v8_config["task"]["staircase"]
    base_fields = stair_observation_fields(
        staircase["terrain_sample_offsets_m"],
        include_placement_reference_observation=True,
        include_support_regulation_observation=True,
    )
    assert base_fields[-len(SUPPORT_REGULATION_OBSERVATION_FIELDS) :] == (
        SUPPORT_REGULATION_OBSERVATION_FIELDS
    )
    requested_effort = np.zeros(12, dtype=np.float32)
    requested_effort[[0, 4, 8]] = (0.2, 0.96, 1.2)
    requested_effort[[2, 6, 10]] = (0.1, 0.2, 0.3)
    requested_effort[[1, 5, 9]] = (0.95, 0.95, 0.1)
    requested_effort[[3, 7, 11]] = (0.0, 0.0, 0.0)
    packed = pack_support_regulation_observation(
        stair_observation=np.zeros(5, dtype=np.float32),
        total_foot_normal_loads_n=(10.0, 20.0, 30.0, 40.0),
        com_target_error_xy_m=(0.02, -0.03),
        requested_pd_effort_nm=requested_effort,
        effort_cap_nm=1.0,
        contact_load_normalization_n=50.0,
    )
    extras = packed[5:]
    assert extras[:4] == pytest.approx((0.2, 0.4, 0.6, 0.8))
    assert extras[4:6] == pytest.approx((0.2, -0.3))
    assert extras[6:10] == pytest.approx((1.2, 0.3, 0.95, 0.0))
    assert extras[10:] == pytest.approx((2 / 3, 0.0, 2 / 3, 0.0))
    with pytest.raises(ValueError, match="requires placement reference"):
        stair_observation_fields(
            staircase["terrain_sample_offsets_m"],
            include_support_regulation_observation=True,
        )


def test_support_regulation_reward_penalizes_com_and_drive_saturation(
    v8_config: dict,
) -> None:
    reward = {
        **v8_config["task"]["reward"],
        "balance_target_error": -4.0,
        "minimum_support_load": 2.0,
        "pd_effort_saturation": -0.5,
    }
    requested = np.zeros(12, dtype=np.float32)
    requested[:6] = 1.5
    terms = stair_reward_terms(
        command_velocity_xyz=(0.0, 0.0, 0.0),
        body_linear_velocity_xyz=(0.0, 0.0, 0.0),
        body_angular_velocity_xyz=(0.0, 0.0, 0.0),
        projected_gravity_xyz=(0.0, 0.0, -1.0),
        base_clearance_m=0.36,
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
        balance_target_error_xy_m=(0.05, -0.05),
        support_normal_loads_n=(10.0, 20.0, 30.0),
        requested_pd_effort_nm=requested,
        effort_cap_nm=1.0,
        contact_load_normalization_n=50.0,
    )
    assert terms["balance_target_error"] == pytest.approx(-2.0)
    assert terms["minimum_support_load"] == pytest.approx(0.4)
    assert terms["pd_effort_saturation"] < 0.0


def test_clearance_tracking_rewards_height_and_penalizes_gate_deficit(
    v29_config: dict,
) -> None:
    reward = {
        **v29_config["task"]["reward"],
        "swing_height_tracking_sigma_m": 0.02,
        "swing_height_tracking": 18.0,
        "clearance_gate_deficit": -6000.0,
    }
    common = {
        "command_velocity_xyz": (0.0, 0.0, 0.0),
        "body_linear_velocity_xyz": (0.0, 0.0, 0.0),
        "body_angular_velocity_xyz": (0.0, 0.0, 0.0),
        "projected_gravity_xyz": (0.0, 0.0, -1.0),
        "base_clearance_m": 0.36,
        "lateral_position_m": 0.0,
        "forward_progress_m": 0.0,
        "base_height_gain_m": 0.0,
        "terrain_height_gain_m": 0.0,
        "heading_error_rad": 0.0,
        "joint_velocities_normalized": np.zeros(12),
        "action": np.zeros(12),
        "previous_action": np.zeros(12),
        "failed": False,
        "succeeded": False,
        "reward_config": reward,
    }
    tracked = stair_reward_terms(
        **common,
        swing_height_error_m=0.005,
        clearance_gate_deficit_m=0.0,
    )
    low = stair_reward_terms(
        **common,
        swing_height_error_m=0.045,
        clearance_gate_deficit_m=0.025,
    )
    assert tracked["swing_height_tracking"] > low["swing_height_tracking"]
    assert tracked["clearance_gate_deficit"] == pytest.approx(0.0)
    assert low["clearance_gate_deficit"] == pytest.approx(-150.0)
    assert tracked["total"] > low["total"]
    with pytest.raises(ValueError, match="nonnegative"):
        stair_reward_terms(
            **common,
            swing_height_error_m=0.0,
            clearance_gate_deficit_m=-0.001,
        )


def test_support_load_sharing_extends_the_underloaded_leg() -> None:
    triangle = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    correction = support_load_share_vertical_corrections(
        support_points_xy_m=triangle,
        target_position_xy_m=(1.0 / 3.0, 1.0 / 3.0),
        measured_normal_loads_n=(0.0, 15.0, 15.0),
        proportional_gain_m=0.030,
        maximum_correction_m=0.012,
    )
    assert correction == pytest.approx((0.010, -0.005, -0.005))
    assert float(np.sum(correction)) == pytest.approx(0.0)
    assert support_load_share_vertical_corrections(
        support_points_xy_m=triangle,
        target_position_xy_m=(1.0 / 3.0, 1.0 / 3.0),
        measured_normal_loads_n=(10.0, 10.0, 10.0),
        proportional_gain_m=0.030,
        maximum_correction_m=0.012,
    ) == pytest.approx((0.0, 0.0, 0.0))


def test_v24_reward_and_termination_make_drift_worse_than_success(
    v24_config: dict,
) -> None:
    task = v24_config["task"]
    reward = task["reward"]
    assert task["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert task["include_support_regulation_observation"] is True
    assert task["termination"]["maximum_lateral_deviation_m"] == pytest.approx(
        0.20
    )
    assert task["termination"]["minimum_upright_cosine"] >= 0.94
    assert reward["failure"] < -2.0 * reward["success"]
    assert reward["support_margin"] < 100.0
    assert v24_config["ppo"]["learning_rate"] <= 1e-5
    assert v24_config["ppo"]["initial_log_std"] <= -4.0


def test_v29_requires_a_strict_stable_pre_unload_gate(
    v29_config: dict,
) -> None:
    task = v29_config["task"]
    transfer = task["placement_reference"]["inter_leg_transfer"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v29-Pre-Unload-COM-Gate"
    assert task["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert task["staircase"]["rise_m"] == pytest.approx(0.18)
    assert task["robot_hardware_profile"]["effort_cap_nm"] == pytest.approx(
        0.8825985
    )
    assert transfer["pre_unload_gate_hold_seconds"] == pytest.approx(0.50)
    assert transfer["minimum_next_swing_preload_n"] == pytest.approx(5.0)
    assert transfer["minimum_support_margin_m"] >= 0.025
    assert transfer["base_target_tolerance_m"] <= 0.020
    assert transfer["maximum_base_speed_m_s"] <= 0.020
    assert transfer["maximum_body_rate_rad_s"] <= 0.200
    assert v29_config["ppo"]["learning_rate"] == pytest.approx(1e-4)
    assert v29_config["ppo"]["initial_log_std"] == pytest.approx(-2.0)
    clearance_gate = task["placement_reference"]["advance_clearance_gate"]
    assert clearance_gate["enabled"] is True
    assert clearance_gate["legs"] == ["front_left"]
    assert clearance_gate["minimum_clearance_m"] == pytest.approx(0.190)
    assert clearance_gate["maximum_hold_seconds"] == pytest.approx(2.0)
    assert transfer["maximum_seconds"] > (
        transfer["duration_seconds"]
        + transfer["unload_duration_seconds"]
        + transfer["pre_unload_gate_hold_seconds"]
    )


def test_v31_targets_measured_clearance_without_changing_stair_contract(
    v29_config: dict,
    v31_config: dict,
) -> None:
    baseline = v29_config["task"]
    task = v31_config["task"]
    reward = task["reward"]
    lift_level = next(
        level
        for level in task["placement_curriculum"]["levels"]
        if level["id"] == "left-supported-190mm-lift"
    )

    assert task["id"] == "Drobot-Quadruped-Stairs-v31-Clearance-Tracking"
    assert task["staircase"] == baseline["staircase"]
    assert task["robot_hardware_profile"] == baseline["robot_hardware_profile"]
    assert task["placement_reference"]["inter_leg_transfer"] == (
        baseline["placement_reference"]["inter_leg_transfer"]
    )
    assert task["placement_reference"]["advance_clearance_gate"] == (
        baseline["placement_reference"]["advance_clearance_gate"]
    )
    assert lift_level["minimum_lift_m"] == pytest.approx(0.190)
    assert lift_level["apex_lift_m"] == pytest.approx(0.220)
    assert reward["swing_target_tracking"] == pytest.approx(0.0)
    assert reward["swing_height_tracking"] > 0.0
    assert reward["clearance_gate_deficit"] < 0.0


def test_v32_prioritizes_compact_support_regulation_without_changing_hardware(
    v31_config: dict,
    v32_config: dict,
) -> None:
    baseline = v31_config["task"]
    task = v32_config["task"]
    reward = task["reward"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v32-Compact-Support-Regulation"
    assert task["staircase"] == baseline["staircase"]
    assert task["robot_hardware_profile"] == baseline["robot_hardware_profile"]
    assert task["placement_reference"] == baseline["placement_reference"]
    assert reward["upright_deviation"] < baseline["reward"]["upright_deviation"]
    assert reward["roll_pitch_rate"] < baseline["reward"]["roll_pitch_rate"]
    assert reward["support_margin"] > baseline["reward"]["support_margin"]
    assert reward["balance_target_error"] < baseline["reward"]["balance_target_error"]


def test_v33_reduces_exploration_for_support_pitch_training(
    v32_config: dict,
    v33_config: dict,
) -> None:
    baseline = v32_config["task"]
    task = v33_config["task"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v33-Compact-Support-Pitch"
    assert task["staircase"] == baseline["staircase"]
    assert task["robot_hardware_profile"] == baseline["robot_hardware_profile"]
    assert task["placement_reference"] == baseline["placement_reference"]
    assert task["reward"] == baseline["reward"]
    assert v33_config["ppo"]["learning_rate"] < v32_config["ppo"]["learning_rate"]
    assert v33_config["ppo"]["target_kl"] < v32_config["ppo"]["target_kl"]
    assert v33_config["ppo"]["initial_log_std"] < v32_config["ppo"]["initial_log_std"]


def test_v34_enables_bounded_imu_pitch_feedback(
    v33_config: dict,
    v34_config: dict,
) -> None:
    baseline = v33_config["task"]
    task = v34_config["task"]
    feedback = task["placement_reference"]["inter_leg_transfer"][
        "com_regulation"
    ]["pitch_attitude_feedback"]
    center_level = next(
        level
        for level in task["placement_curriculum"]["levels"]
        if level["id"] == "left-center-tread-load"
    )

    assert task["id"] == "Drobot-Quadruped-Stairs-v34-IMU-Pitch-Feedback"
    assert task["staircase"] == baseline["staircase"]
    assert task["robot_hardware_profile"] == baseline["robot_hardware_profile"]
    assert feedback == {
        "enabled": True,
        "proportional_gain_m": pytest.approx(0.080),
        "maximum_correction_m": pytest.approx(0.025),
    }
    assert center_level["apex_lift_m"] == pytest.approx(0.235)
    assert center_level["landing_lift_m"] == pytest.approx(0.125)


def test_v35_preserves_rear_unload_pose_for_190mm_clearance(
    v34_config: dict,
    v35_config: dict,
) -> None:
    baseline = v34_config["task"]
    task = v35_config["task"]
    placement = task["placement_reference"]
    transfer = placement["inter_leg_transfer"]
    rear_right = transfer["override_by_next_swing_leg"]["rear_right"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v35-Full-First-Tread"
    assert task["staircase"] == baseline["staircase"]
    assert task["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert task["staircase"]["rise_m"] == pytest.approx(0.18)
    assert placement["sequence_legs"] == [
        "front_right",
        "front_left",
        "rear_right",
        "rear_left",
    ]
    assert placement["advance_clearance_gate"]["minimum_clearance_m"] == (
        pytest.approx(0.190)
    )
    assert rear_right["swing_unload_lift_m"] == pytest.approx(0.120)
    assert rear_right["minimum_support_margin_m"] == pytest.approx(-0.010)
    assert transfer["post_transfer_swing_reference_mode_by_leg"] == {
        "rear_right": "phase_baseline",
        "rear_left": "phase_baseline",
    }
    assert placement["level_override_by_leg"]["rear_right"][
        "lift_forward_offset_m"
    ] == pytest.approx(0.050)
    assert placement["level_override_by_leg"]["rear_left"][
        "lift_forward_offset_m"
    ] == pytest.approx(0.050)


def test_v36_exposes_only_bounded_support_residual_during_transfer(
    v35_config: dict,
    v36_config: dict,
) -> None:
    baseline = v35_config["task"]
    task = v36_config["task"]
    transfer = task["placement_reference"]["inter_leg_transfer"]
    support_scale = task["placement_residual_action_scale_rad"]["support"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v36-Transfer-Support-Residual"
    assert task["staircase"] == baseline["staircase"]
    assert task["robot_hardware_profile"] == baseline["robot_hardware_profile"]
    assert transfer["residual_action_scale"] == pytest.approx(0.15)
    assert transfer["policy_post_hold_seconds"] == pytest.approx(2.0)
    assert transfer["phase_snapshot_restore_zero_velocities"] is False
    assert transfer["phase_snapshot_restore_settle_control_steps"] == 0
    assert transfer["com_regulation"]["pitch_attitude_feedback"][
        "front_only_by_swing_leg"
    ] == ["rear_right", "rear_left"]
    assert 0.15 * support_scale["hip_abduction"] == pytest.approx(0.0375)
    assert 0.15 * support_scale["hip_flexion"] == pytest.approx(0.033)
    assert 0.15 * support_scale["knee"] == pytest.approx(0.045)
    assert transfer["override_by_next_swing_leg"]["rear_right"][
        "swing_unload_lift_m"
    ] == pytest.approx(0.120)
    assert v36_config["ppo"]["learning_rate"] == pytest.approx(1e-5)
    assert v36_config["ppo"]["initial_log_std"] == pytest.approx(-4.0)
    assert v36_config["ppo"]["initial_action_bias"] == pytest.approx(
        [
            -0.011061003,
            0.035005786,
            0.036190730,
            0.120000000,
            -0.009826610,
            0.003399614,
            -0.055805374,
            0.017830297,
            -0.001008772,
        ]
    )


def test_v38_requires_positive_margin_for_rear_right_transfer(
    v37_config: dict,
    v38_config: dict,
) -> None:
    baseline = v37_config["task"]
    task = v38_config["task"]
    transfer = task["placement_reference"]["inter_leg_transfer"]
    rear_right = transfer["override_by_next_swing_leg"]["rear_right"]
    regulation = transfer["com_regulation"]

    assert task["id"] == (
        "Drobot-Quadruped-Stairs-v38-Positive-Margin-Rear-Transfer"
    )
    assert task["staircase"] == baseline["staircase"]
    assert task["staircase"]["tread_depth_m"] == pytest.approx(0.250)
    assert task["staircase"]["rise_m"] == pytest.approx(0.180)
    assert task["robot_hardware_profile"] == baseline["robot_hardware_profile"]
    assert task["robot_hardware_profile"]["effort_cap_nm"] == pytest.approx(
        0.8825985
    )
    assert task["placement_reference"]["sequence_legs"] == [
        "front_right",
        "front_left",
        "rear_right",
    ]
    assert task["placement_reference"]["advance_clearance_gate"]["legs"] == [
        "front_left",
        "rear_right",
    ]
    assert rear_right["minimum_support_margin_m"] == pytest.approx(0.015)
    assert transfer["minimum_next_swing_preload_n"] == pytest.approx(5.0)
    assert rear_right["minimum_next_swing_preload_n"] == pytest.approx(3.0)
    assert rear_right["swing_unload_lift_m"] == pytest.approx(0.140)
    assert rear_right["minimum_upright_cosine"] == pytest.approx(0.9781476)
    assert transfer["residual_action_scale"] == pytest.approx(0.02)
    assert regulation["target_offset_by_swing_leg"]["rear_right"] == {
        "forward": pytest.approx(0.020),
        "lateral": pytest.approx(0.040),
    }
    assert regulation["maximum_correction_m_by_swing_leg"]["rear_right"][
        "forward"
    ] == pytest.approx(0.100)
    assert regulation["maximum_correction_m_by_swing_leg"]["rear_left"][
        "forward"
    ] == pytest.approx(0.080)


def test_v9_front_pair_uses_dynamic_swing_support_action_contract(
    v9_config: dict,
) -> None:
    task = v9_config["task"]
    residual = task["placement_residual_action_scale_rad"]

    assert task["id"] == "Drobot-Quadruped-Stairs-v9-180mm-Front-Pair-Placement"
    assert task["placement_reference"]["sequence_legs"] == [
        "front_left",
        "front_right",
    ]
    assert task["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert task["episode_seconds"] >= 2.0 * 10.75
    assert residual["support"]["hip_flexion"] > residual["swing"][
        "hip_flexion"
    ]
    assert task["action_scale_rad"] == residual["swing"]
    transfer = task["placement_reference"]["inter_leg_transfer"]
    assert transfer["enabled"] is True
    assert transfer["duration_seconds"] > 0.0
    assert transfer["maximum_seconds"] > transfer["duration_seconds"]
    assert transfer["minimum_support_margin_m"] > 0.0
    assert transfer["maximum_base_speed_m_s"] > 0.0
    assert transfer["maximum_body_rate_rad_s"] > 0.0
    assert 0.0 <= transfer["support_world_anchor_follow_gain"] <= 1.0
    assert transfer["post_transfer_weight_shift"]["forward_m"] == pytest.approx(
        0.0
    )
    assert transfer["post_transfer_weight_shift"]["lateral_m"] >= 0.0
    assert transfer["residual_action_scale"] == pytest.approx(0.0)
    assert transfer["com_regulation"]["enabled"] is True
    assert transfer["com_regulation"]["balance_point"] == "composite_com"
    assert transfer["swing_unload_lift_m"] == pytest.approx(0.080)
    traction = task["foot_contact_material"]
    assert traction["enabled"] is False
    assert traction["static_friction"] > 0.90
    assert traction["dynamic_friction"] > 0.75
    assert traction["dynamic_friction"] <= traction["static_friction"]


def test_v10_isolates_the_mirrored_front_right_placement(
    v8_config: dict,
    v10_config: dict,
) -> None:
    left = v8_config["task"]
    right = v10_config["task"]

    assert right["id"] == (
        "Drobot-Quadruped-Stairs-v10-180mm-Front-Right-Placement"
    )
    assert right["placement_reference"]["swing_leg"] == "front_right"
    assert right["foot_placement_sequence"][0] == "front_right"
    assert right["staircase"] == left["staircase"]
    assert right["placement_curriculum"] == left["placement_curriculum"]
    assert right["robot_hardware_profile"] == left["robot_hardware_profile"]


def test_v11_trains_only_the_mixed_height_handoff_before_strict_v9_eval(
    v9_config: dict,
    v11_config: dict,
) -> None:
    strict = v9_config["task"]
    training = v11_config["task"]

    assert training["placement_reference"]["sequence_legs"] == [
        "front_left",
        "front_right",
    ]
    assert training["staircase"] == strict["staircase"]
    assert training["robot_hardware_profile"] == strict["robot_hardware_profile"]
    assert training["termination"]["maximum_lateral_deviation_m"] > strict[
        "termination"
    ]["maximum_lateral_deviation_m"]
    assert training["placement_reference"][
        "measurable_slip_threshold_m"
    ] == pytest.approx(
        strict["placement_reference"]["measurable_slip_threshold_m"]
    )
    transfer = training["placement_reference"]["inter_leg_transfer"]
    assert transfer["unload_duration_seconds"] > 0.0
    assert transfer["swing_unload_lift_m"] > 0.0
    assert transfer["maximum_swing_unloaded_load_n"] == pytest.approx(1.0)
    assert len(training["reset_joint_offsets_rad"]) == 12
    assert training["reset_start_x_range_m"][0] == pytest.approx(
        training["reset_start_x_range_m"][1]
    )
    assert training["reward"]["centerline"] < strict["reward"]["centerline"]
    assert training["reward"]["support_slip"] < strict["reward"][
        "support_slip"
    ]
    assert v11_config["ppo"]["learning_rate"] > 0.0
    assert v11_config["ppo"]["target_kl"] < v9_config["ppo"]["target_kl"]
    assert v11_config["ppo"]["initial_log_std"] < v9_config["ppo"][
        "initial_log_std"
    ]


def test_v12_is_a_support_only_190mm_lift_hold_on_a_250mm_tread(
    v11_config: dict,
    v12_config: dict,
) -> None:
    baseline = v11_config["task"]
    lift = v12_config["task"]
    placement = lift["placement_reference"]

    assert lift["staircase"] == baseline["staircase"]
    assert lift["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert placement["sequence_legs"] == ["front_left", "front_right"]
    assert placement["success_mode"] == "tread_contact"
    assert placement["success_mode_by_leg"] == {
        "front_right": "swing_lift_hold"
    }
    assert placement["minimum_lift_m"] == pytest.approx(0.190)
    assert placement["minimum_lift_support_margin_m"] > 0.0
    assert lift["placement_curriculum"]["levels"][0][
        "lift_hold_seconds"
    ] == pytest.approx(0.50)


def test_v13_isolates_front_right_lift_under_the_strict_gate(
    v12_config: dict,
    v13_config: dict,
) -> None:
    baseline = v12_config["task"]
    stabilized = v13_config["task"]
    transfer = stabilized["placement_reference"]["inter_leg_transfer"]
    gains = transfer["support_base_error_feedback_gain"]

    assert stabilized["staircase"] == baseline["staircase"]
    assert stabilized["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert stabilized["placement_reference"]["minimum_lift_m"] == pytest.approx(
        0.190
    )
    lift_levels = stabilized["placement_curriculum"]["levels"]
    assert stabilized["placement_curriculum"]["mode"] == "mastery"
    assert stabilized["placement_curriculum"][
        "mastery_successes_per_level"
    ] == 2
    assert [level["minimum_lift_m"] for level in lift_levels] == pytest.approx(
        [0.015, 0.020, 0.035, 0.060, 0.100, 0.140, 0.190]
    )
    assert [level["minimum_support_margin_m"] for level in lift_levels] == (
        pytest.approx([0.003, 0.005, 0.008, 0.010, 0.012, 0.012, 0.015])
    )
    assert lift_levels[-1]["apex_lift_m"] > stabilized[
        "placement_reference"
    ]["minimum_lift_m"]
    assert lift_levels[-1]["start_fraction"] == pytest.approx(0.90)
    assert stabilized["placement_reference"]["success_mode_by_leg"] == {
        "front_right": "swing_lift_hold"
    }
    assert stabilized["placement_reference"]["sequence_legs"] == [
        "front_right"
    ]
    assert stabilized["foot_placement_sequence"][0] == "front_right"
    assert stabilized["placement_reference"]["weight_shift"][
        "scale_by_leg"
    ]["front_right"] == pytest.approx(1.0)
    assert stabilized["termination"][
        "maximum_lateral_deviation_m"
    ] == pytest.approx(0.20)
    assert transfer["enabled"] is False
    assert transfer["support_incenter_blend"] == pytest.approx(
        baseline["placement_reference"]["inter_leg_transfer"][
            "support_incenter_blend"
        ]
    )
    assert transfer["target_offset_m"]["lateral"] == pytest.approx(0.0)
    assert transfer["post_transfer_weight_shift"]["lateral_m"] == pytest.approx(
        0.0
    )
    assert gains["lateral"] >= gains["forward"] >= 0.0
    assert gains["vertical"] == pytest.approx(0.0)
    assert stabilized["foot_contact_material"]["enabled"] is False


def test_v14_reverses_the_front_pair_and_mastery_gates_left_placement(
    v10_config: dict,
    v14_config: dict,
) -> None:
    single_right = v10_config["task"]
    front_pair = v14_config["task"]
    placement = front_pair["placement_reference"]
    levels = front_pair["placement_curriculum"]["levels"]

    assert front_pair["staircase"] == single_right["staircase"]
    assert front_pair["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert placement["sequence_legs"] == ["front_right", "front_left"]
    assert placement["success_mode"] == "tread_contact"
    assert placement["inter_leg_transfer"]["enabled"] is True
    assert placement["inter_leg_transfer"][
        "base_target_tolerance_m"
    ] == pytest.approx(0.065)
    assert placement["inter_leg_transfer"][
        "phase_snapshot_restore_zero_velocities"
    ] is True
    assert placement["inter_leg_transfer"][
        "phase_snapshot_restore_settle_control_steps"
    ] == 12
    left_timing = placement["timing_override_by_leg"]["front_left"]
    assert left_timing["lift_start_seconds"] == pytest.approx(0.50)
    assert left_timing["lower_start_seconds"] == pytest.approx(5.50)
    assert placement["level_override_by_leg"]["front_right"][
        "target_tread_fraction"
    ] == pytest.approx(0.24)
    assert front_pair["placement_curriculum"]["mode"] == "mastery"
    assert front_pair["placement_curriculum"][
        "mastery_successes_per_level"
    ] == 2
    assert [level["id"] for level in levels] == [
        "left-supported-015mm-lift",
        "left-supported-035mm-lift",
        "left-supported-060mm-lift",
        "left-supported-100mm-lift",
        "left-supported-140mm-lift",
        "left-supported-190mm-lift",
        "left-near-edge-force-touch",
        "left-quarter-tread-load",
        "left-center-tread-load",
    ]
    assert [level["minimum_lift_m"] for level in levels[:6]] == (
        pytest.approx([0.015, 0.035, 0.060, 0.100, 0.140, 0.190])
    )
    assert [level["success_mode"] for level in levels] == [
        "swing_lift_hold",
        "swing_lift_hold",
        "swing_lift_hold",
        "swing_lift_hold",
        "swing_lift_hold",
        "swing_lift_hold",
        "tread_contact",
        "tread_contact",
        "tread_contact",
    ]
    assert [level["contact_hold_seconds"] for level in levels[-3:]] == (
        pytest.approx([0.25, 0.50, 0.75])
    )
    assert levels[-1]["apex_lift_m"] == pytest.approx(0.205)
    assert front_pair["termination"][
        "maximum_lateral_deviation_m"
    ] == pytest.approx(0.30)
    assert front_pair["foot_contact_material"]["enabled"] is False


def test_v15_mirrors_the_proven_direct_lift_for_front_left(
    v13_config: dict,
    v15_config: dict,
) -> None:
    right = v13_config["task"]
    left = v15_config["task"]

    assert left["staircase"] == right["staircase"]
    assert left["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    assert left["placement_reference"]["sequence_legs"] == ["front_left"]
    assert left["placement_reference"]["success_mode_by_leg"] == {
        "front_left": "swing_lift_hold"
    }
    assert left["placement_reference"]["minimum_lift_m"] == pytest.approx(
        0.190
    )
    assert left["placement_reference"][
        "minimum_lift_support_margin_m"
    ] == pytest.approx(0.015)
    levels = left["placement_curriculum"]["levels"]
    assert [level["minimum_lift_m"] for level in levels] == pytest.approx(
        [0.015, 0.020, 0.035, 0.060, 0.100, 0.140, 0.190]
    )
    assert all(level["id"].startswith("front-left-") for level in levels)
    assert levels[-1]["lift_hold_seconds"] == pytest.approx(0.50)
    assert left["termination"] == right["termination"]
    assert left["foot_contact_material"]["enabled"] is False


def test_v16_rejects_post_transfer_base_drift_on_25cm_tread(
    v14_config: dict,
    v16_config: dict,
) -> None:
    baseline = v14_config["task"]
    stabilized = v16_config["task"]
    transfer = stabilized["placement_reference"]["inter_leg_transfer"]

    assert stabilized["staircase"] == baseline["staircase"]
    assert stabilized["staircase"]["tread_depth_m"] == pytest.approx(0.25)
    first_level = stabilized["placement_curriculum"]["levels"][0]
    assert first_level["minimum_lift_m"] == pytest.approx(0.015)
    assert first_level["apex_lift_m"] == pytest.approx(0.065)
    assert first_level["landing_lift_m"] == pytest.approx(0.040)
    lift_levels = stabilized["placement_curriculum"]["levels"][:6]
    assert [level["minimum_lift_m"] for level in lift_levels] == pytest.approx(
        [0.015, 0.035, 0.060, 0.100, 0.140, 0.190]
    )
    expected_lift_reach = pytest.approx(
        [0.020, 0.035, 0.050, 0.065, 0.080, 0.120]
    )
    assert [
        level["lift_forward_offset_m"] for level in lift_levels
    ] == expected_lift_reach
    assert [
        level["swing_forward_offset_m"] for level in lift_levels
    ] == expected_lift_reach
    assert [
        level["landing_forward_offset_m"] for level in lift_levels
    ] == expected_lift_reach
    assert placement_curriculum_level(0.0, lift_levels)["id"] == (
        "left-supported-015mm-lift"
    )
    assert transfer["post_transfer_swing_reference_mode"] == (
        "blend_to_nominal_stance"
    )
    left_timing = stabilized["placement_reference"][
        "timing_override_by_leg"
    ]["front_left"]
    assert left_timing["lift_duration_seconds"] == pytest.approx(2.0)
    assert left_timing["advance_start_seconds"] == pytest.approx(2.5)
    assert lift_levels[-1]["apex_lift_m"] == pytest.approx(0.205)
    invalid_contact = {
        **lift_levels[0],
        "success_mode": "tread_contact",
        "lift_forward_offset_m": 0.0,
        "swing_forward_offset_m": 0.0,
        "landing_forward_offset_m": 0.0,
    }
    with pytest.raises(ValueError, match="pure swing lift hold"):
        placement_curriculum_level(0.0, [invalid_contact])
    assert stabilized["placement_reference"]["sequence_legs"] == [
        "front_right",
        "front_left",
    ]
    assert transfer["post_transfer_weight_shift"] == {
        "forward_m": pytest.approx(0.0),
        "lateral_m": pytest.approx(0.0),
    }
    assert transfer["support_base_error_feedback_gain"] == {
        "forward": pytest.approx(1.0),
        "lateral": pytest.approx(0.0),
        "vertical": pytest.approx(1.0),
    }
    assert transfer["support_world_anchor_follow_gain"] == pytest.approx(0.0)
    assert transfer["support_world_anchor_follow_gain_xyz"] == {
        "forward": pytest.approx(0.0),
        "lateral": pytest.approx(0.0),
        "vertical": pytest.approx(0.0),
    }
    assert transfer["target_offset_m"] == {
        "forward": pytest.approx(0.0),
        "lateral": pytest.approx(0.0),
    }
    assert transfer["com_regulation"] == {
        "enabled": True,
        "balance_point": "composite_com",
        "target_incenter_blend": pytest.approx(1.0),
        "target_offset_m": {
            "forward": pytest.approx(0.0),
            "lateral": pytest.approx(0.0),
        },
        "target_offset_by_swing_leg": {
            "front_left": {
                "forward": pytest.approx(0.0),
                "lateral": pytest.approx(0.0),
            }
        },
        "hold_target_offset_by_swing_leg": {
            "front_left": {
                "forward": pytest.approx(0.0),
                "lateral": pytest.approx(0.0),
            }
        },
        "support_squat_thrust_by_swing_leg": {
            "front_left": {
                "legs": ["rear_left", "rear_right"],
                "crouch_m": pytest.approx(0.035),
                "release_lift_fraction": pytest.approx(0.25),
            },
        },
        "maximum_correction_m": {
            "forward": pytest.approx(0.080),
            "lateral": pytest.approx(0.120),
        },
        "maximum_feedback_correction_m": {
            "forward": pytest.approx(0.025),
            "lateral": pytest.approx(0.052),
            "vertical": pytest.approx(0.020),
        },
        "transfer_feedback_gain": {
            "forward": pytest.approx(1.0),
            "lateral": pytest.approx(1.20),
            "vertical": pytest.approx(1.0),
        },
        "feedback_gain": {
            "forward": pytest.approx(1.0),
            "lateral": pytest.approx(1.20),
            "vertical": pytest.approx(1.0),
        },
    }
    residual_scale = stabilized["placement_residual_action_scale_rad"]
    assert residual_scale["support"] == {
        "hip_abduction": pytest.approx(0.25),
        "hip_flexion": pytest.approx(0.22),
        "knee": pytest.approx(0.30),
    }
    assert residual_scale["override_by_swing_leg"]["front_left"][
        "support"
    ] == {
        "hip_abduction": pytest.approx(0.10),
        "hip_flexion": pytest.approx(0.22),
        "knee": pytest.approx(0.30),
    }
    assert stabilized["termination"][
        "maximum_lateral_deviation_m"
    ] == pytest.approx(0.50)
    assert stabilized["termination"][
        "lateral_deviation_tolerance_m"
    ] == pytest.approx(0.001)
    assert stabilized["termination"]["minimum_base_clearance_m"] == (
        baseline["termination"]["minimum_base_clearance_m"]
    )
    assert stabilized["foot_contact_material"]["enabled"] is True
    assert stabilized["foot_contact_material"]["static_friction"] == (
        pytest.approx(1.20)
    )
    assert stabilized["foot_contact_material"]["dynamic_friction"] == (
        pytest.approx(1.00)
    )
    assert stabilized["foot_contact_material"][
        "friction_combine_mode"
    ] == "average"


def test_placement_success_mode_prefers_level_then_leg_then_default() -> None:
    assert (
        placement_success_mode(
            swing_leg="front_left",
            default_mode="tread_contact",
            mode_by_leg={"front_left": "swing_lift_hold"},
            active_level={"success_mode": "tread_contact"},
        )
        == "tread_contact"
    )
    assert (
        placement_success_mode(
            swing_leg="front_left",
            mode_by_leg={"front_left": "swing_lift_hold"},
        )
        == "swing_lift_hold"
    )
    assert placement_success_mode(swing_leg="front_right") == "tread_contact"
    with pytest.raises(ValueError, match="success mode"):
        placement_success_mode(
            swing_leg="front_right",
            active_level={"success_mode": "unsupported"},
        )


def test_inter_leg_transfer_uses_smooth_weight_shift_and_support_incenter() -> None:
    transfer_options = {
        "duration_seconds": 4.0,
        "unload_duration_seconds": 1.5,
    }
    start = inter_leg_transfer_state(0.0, **transfer_options)
    halfway = inter_leg_transfer_state(2.0, **transfer_options)
    done = inter_leg_transfer_state(4.0, **transfer_options)
    unloaded = inter_leg_transfer_state(5.5, **transfer_options)
    held_loaded = inter_leg_transfer_state(
        5.5,
        **transfer_options,
        unload_elapsed_seconds=0.0,
    )
    gated_halfway = inter_leg_transfer_state(
        8.0,
        **transfer_options,
        unload_elapsed_seconds=0.75,
    )

    assert start["phase"] == "weight_shift"
    assert start["phase_one_hot"] == (1.0, 0.0, 0.0, 0.0, 0.0)
    assert start["transfer_fraction"] == pytest.approx(0.0)
    assert halfway["transfer_fraction"] == pytest.approx(0.5)
    assert done["transfer_fraction"] == pytest.approx(1.0)
    assert done["unload_fraction"] == pytest.approx(0.0)
    assert unloaded["unload_fraction"] == pytest.approx(1.0)
    assert held_loaded["unload_fraction"] == pytest.approx(0.0)
    assert held_loaded["transfer_stage"] == "pre_unload_settle"
    assert gated_halfway["unload_fraction"] == pytest.approx(0.5)
    assert gated_halfway["transfer_stage"] == "unload"
    assert done["desired_lift_m"] == pytest.approx(0.0)
    assert done["contact_expected"] is False

    incenter = support_triangle_incenter_xy(
        ((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
    )
    expected = 2.0 - np.sqrt(2.0)
    np.testing.assert_allclose(incenter, (expected, expected), atol=1e-6)

    with pytest.raises(ValueError, match="nonzero area"):
        support_triangle_incenter_xy(
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
        )


def test_pre_unload_gate_requires_stable_four_foot_support() -> None:
    ready = {
        "transfer_fraction": 1.0,
        "support_contact_fraction": 1.0,
        "completed_tread_loaded": True,
        "next_swing_total_load_n": 18.0,
        "minimum_next_swing_preload_n": 5.0,
        "support_margin_m": 0.08,
        "minimum_support_margin_m": 0.015,
        "balance_target_error_m": 0.008,
        "maximum_balance_target_error_m": 0.020,
        "base_speed_m_s": 0.008,
        "maximum_base_speed_m_s": 0.020,
        "body_rate_rad_s": 0.04,
        "maximum_body_rate_rad_s": 0.10,
        "upright_cosine": 0.998,
        "minimum_upright_cosine": 0.978,
    }
    assert inter_leg_pre_unload_gate_failures(**ready) == ()

    unstable = dict(ready)
    unstable.update(
        {
            "transfer_fraction": 0.9,
            "support_contact_fraction": 0.67,
            "completed_tread_loaded": False,
            "next_swing_total_load_n": 0.0,
            "support_margin_m": 0.010,
            "balance_target_error_m": 0.030,
            "base_speed_m_s": 0.030,
            "body_rate_rad_s": 0.20,
            "upright_cosine": 0.95,
        }
    )
    assert inter_leg_pre_unload_gate_failures(**unstable) == (
        "transfer_incomplete",
        "support_contact_lost",
        "placed_tread_unloaded",
        "next_swing_not_preloaded",
        "support_margin_low",
        "balance_target_error_high",
        "base_not_settled",
        "body_rate_high",
        "body_not_upright",
    )


def test_com_target_moves_toward_support_incenter_with_bounded_shift() -> None:
    target = bounded_support_incenter_target_xy(
        reference_point_xy_m=(0.90, 0.90),
        support_points_xy_m=((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)),
        incenter_blend=1.0,
        target_offset_xy_m=(0.01, -0.01),
        maximum_shift_xy_m=(0.20, 0.10),
    )
    np.testing.assert_allclose(target, (0.70, 0.80), atol=1e-6)

    with pytest.raises(ValueError, match="incenter_blend"):
        bounded_support_incenter_target_xy(
            reference_point_xy_m=(0.0, 0.0),
            support_points_xy_m=((0.0, 0.0), (2.0, 0.0), (0.0, 2.0)),
            incenter_blend=0.0,
        )


def test_balance_target_error_uses_the_com_target_frame() -> None:
    error = balance_target_error_xy(
        balance_position_xy_m=(0.42, -0.03),
        target_position_xy_m=(0.36, -0.09),
    )

    np.testing.assert_allclose(error, (0.06, 0.06), atol=1e-7)
    with pytest.raises(ValueError, match="balance_position_xy_m"):
        balance_target_error_xy(
            balance_position_xy_m=(np.nan, 0.0),
            target_position_xy_m=(0.0, 0.0),
        )


def test_com_target_is_clipped_to_requested_support_margin() -> None:
    support_points = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
    unchanged = support_margin_constrained_target_xy(
        desired_target_xy_m=(0.20, 0.20),
        support_points_xy_m=support_points,
        minimum_margin_m=0.10,
    )
    np.testing.assert_allclose(unchanged, (0.20, 0.20), atol=1e-9)

    constrained = support_margin_constrained_target_xy(
        desired_target_xy_m=(0.90, 0.90),
        support_points_xy_m=support_points,
        minimum_margin_m=0.10,
    )
    expected_edge_coordinate = (1.0 - np.sqrt(2.0) * 0.10) / 2.0
    np.testing.assert_allclose(
        constrained,
        (expected_edge_coordinate, expected_edge_coordinate),
        atol=1e-7,
    )

    with pytest.raises(ValueError, match="inradius"):
        support_margin_constrained_target_xy(
            desired_target_xy_m=(0.20, 0.20),
            support_points_xy_m=support_points,
            minimum_margin_m=0.30,
        )


def test_touchdown_load_feedback_retracts_only_above_target() -> None:
    arguments = {
        "target_tread_load_n": 15.0,
        "proportional_gain_m_per_n": 0.0005,
        "maximum_lift_correction_m": 0.035,
    }
    assert touchdown_load_lift_correction_m(
        measured_tread_load_n=10.0,
        **arguments,
    ) == pytest.approx(0.0)
    assert touchdown_load_lift_correction_m(
        measured_tread_load_n=35.0,
        **arguments,
    ) == pytest.approx(0.010)
    assert touchdown_load_lift_correction_m(
        measured_tread_load_n=300.0,
        **arguments,
    ) == pytest.approx(0.035)

    with pytest.raises(ValueError, match="target_tread_load_n"):
        touchdown_load_lift_correction_m(
            measured_tread_load_n=10.0,
            target_tread_load_n=0.0,
            proportional_gain_m_per_n=0.0005,
            maximum_lift_correction_m=0.035,
        )


def test_joint_effort_telemetry_reports_tracking_and_cap_utilization() -> None:
    target = np.linspace(-0.3, 0.3, 12)
    measured = target.copy()
    measured[4] -= 0.2
    velocities = np.zeros(12)
    velocities[4] = 0.25
    stiffness = np.full(12, 5.0)
    damping = np.full(12, 0.2)
    reported = np.zeros(12)
    reported[4] = 0.95
    projected = np.zeros(12)
    projected[7] = -1.2

    sample = joint_effort_telemetry_sample(
        target_joint_positions_rad=target,
        measured_joint_positions_rad=measured,
        joint_velocities_rad_s=velocities,
        drive_stiffness_nm_rad=stiffness,
        drive_damping_nm_s_rad=damping,
        effort_cap_nm=1.0,
        reported_actuation_effort_nm=reported,
        projected_joint_reaction_load_nm=projected,
    )

    assert sample["joint_tracking_error_rad"][4] == pytest.approx(0.2)
    assert sample["requested_pd_effort_nm"][4] == pytest.approx(0.95)
    assert sample["capped_pd_effort_nm"][4] == pytest.approx(0.95)
    assert sample["requested_pd_effort_nm_peak_to_cap_ratio"] == pytest.approx(
        0.95,
    )
    assert sample["requested_pd_effort_nm_95pct_cap_fraction"] == pytest.approx(
        1.0 / 12.0
    )
    assert sample["reported_actuation_effort_nm"][4] == pytest.approx(0.95)
    assert sample["projected_joint_reaction_load_nm"][7] == pytest.approx(-1.2)
    with pytest.raises(ValueError, match="effort_cap_nm"):
        joint_effort_telemetry_sample(
            target_joint_positions_rad=target,
            measured_joint_positions_rad=measured,
            effort_cap_nm=0.0,
        )
    with pytest.raises(ValueError, match="provided together"):
        joint_effort_telemetry_sample(
            target_joint_positions_rad=target,
            measured_joint_positions_rad=measured,
            joint_velocities_rad_s=velocities,
            effort_cap_nm=1.0,
        )


def test_support_reference_feedback_amplifies_body_drift_rejection() -> None:
    desired = (0.020, 0.000, 0.000)
    actual = (0.050, 0.060, -0.010)

    baseline = stabilized_support_reference_base_delta(
        desired_base_delta_m=desired,
        actual_base_delta_m=actual,
        anchor_follow_gain=0.0,
        error_feedback_gain_xyz=(0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(baseline, desired)

    stabilized = stabilized_support_reference_base_delta(
        desired_base_delta_m=desired,
        actual_base_delta_m=actual,
        anchor_follow_gain=0.0,
        error_feedback_gain_xyz=(0.25, 0.75, 0.0),
    )
    np.testing.assert_allclose(stabilized, (0.0125, -0.045, 0.0))

    followed = stabilized_support_reference_base_delta(
        desired_base_delta_m=desired,
        actual_base_delta_m=actual,
        anchor_follow_gain=1.0,
        error_feedback_gain_xyz=(0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(followed, actual)

    lateral_anchor = stabilized_support_reference_base_delta(
        desired_base_delta_m=desired,
        actual_base_delta_m=actual,
        anchor_follow_gain=(0.0, 1.0, 0.0),
        error_feedback_gain_xyz=(0.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(lateral_anchor, (0.020, 0.060, 0.000))

    with pytest.raises(ValueError, match="feedback gains"):
        stabilized_support_reference_base_delta(
            desired_base_delta_m=desired,
            actual_base_delta_m=actual,
            anchor_follow_gain=0.0,
            error_feedback_gain_xyz=(0.0, 2.1, 0.0),
        )


def test_support_pitch_feedback_levels_nose_down_attitude() -> None:
    corrections = support_pitch_vertical_corrections(
        support_legs=("front_right", "rear_left", "rear_right"),
        projected_gravity_x=-0.246,
        proportional_gain_m=0.120,
        maximum_correction_m=0.035,
    )
    assert corrections == {
        "front_right": pytest.approx(-0.02952),
        "rear_left": pytest.approx(0.02952),
        "rear_right": pytest.approx(0.02952),
    }

    saturated = support_pitch_vertical_corrections(
        support_legs=("front_right", "rear_left", "rear_right"),
        projected_gravity_x=0.5,
        proportional_gain_m=0.120,
        maximum_correction_m=0.035,
    )
    assert saturated["front_right"] == pytest.approx(0.035)
    assert saturated["rear_left"] == pytest.approx(-0.035)
    with pytest.raises(ValueError, match="known robot legs"):
        support_pitch_vertical_corrections(
            support_legs=("middle_left",),
            projected_gravity_x=0.0,
            proportional_gain_m=0.120,
            maximum_correction_m=0.035,
        )


def test_placement_completion_waits_for_proprioceptive_settling() -> None:
    assert placement_completion_settle_gate_failures(
        base_linear_velocity_xyz_m_s=(0.004, -0.003, 0.010),
        body_angular_velocity_xyz_rad_s=(0.03, 0.04, 0.02),
        upright_cosine=0.99,
        maximum_base_speed_m_s=0.025,
        maximum_body_rate_rad_s=0.20,
        minimum_upright_cosine=0.9781476,
    ) == ()

    failures = placement_completion_settle_gate_failures(
        base_linear_velocity_xyz_m_s=(0.066, 0.0, 0.066),
        body_angular_velocity_xyz_rad_s=(0.16, 0.16, 0.0),
        upright_cosine=0.97,
        maximum_base_speed_m_s=0.025,
        maximum_body_rate_rad_s=0.20,
        minimum_upright_cosine=0.9781476,
    )
    assert failures == (
        "base_not_settled",
        "body_rate_high",
        "body_not_upright",
    )


def test_staged_swing_reference_releases_only_during_advance() -> None:
    base_delta = (0.020, 0.040, -0.010)

    before_advance = staged_swing_reference_base_delta(
        base_delta_m=base_delta,
        advance_fraction=0.0,
        end_scale_xyz=(0.0, 1.0, 1.0),
    )
    halfway = staged_swing_reference_base_delta(
        base_delta_m=base_delta,
        advance_fraction=0.5,
        end_scale_xyz=(0.0, 1.0, 1.0),
    )
    fully_advanced = staged_swing_reference_base_delta(
        base_delta_m=base_delta,
        advance_fraction=1.0,
        end_scale_xyz=(0.0, 1.0, 1.0),
    )

    np.testing.assert_allclose(before_advance, base_delta)
    np.testing.assert_allclose(halfway, (0.010, 0.040, -0.010))
    np.testing.assert_allclose(fully_advanced, (0.0, 0.040, -0.010))

    with pytest.raises(ValueError, match="advance_fraction"):
        staged_swing_reference_base_delta(
            base_delta_m=base_delta,
            advance_fraction=1.01,
            end_scale_xyz=(0.0, 1.0, 1.0),
        )
    with pytest.raises(ValueError, match="end_scale_xyz"):
        staged_swing_reference_base_delta(
            base_delta_m=base_delta,
            advance_fraction=0.5,
            end_scale_xyz=(-0.1, 1.0, 1.0),
        )


def test_staged_swing_outward_offset_ramps_during_advance() -> None:
    assert staged_swing_outward_offset_m(
        maximum_offset_m=0.005,
        advance_fraction=0.0,
    ) == pytest.approx(0.0)
    assert staged_swing_outward_offset_m(
        maximum_offset_m=0.005,
        advance_fraction=0.5,
    ) == pytest.approx(0.0025)
    assert staged_swing_outward_offset_m(
        maximum_offset_m=-0.005,
        advance_fraction=1.0,
    ) == pytest.approx(-0.005)

    with pytest.raises(ValueError, match="maximum_offset_m"):
        staged_swing_outward_offset_m(
            maximum_offset_m=0.151,
            advance_fraction=0.5,
        )
    with pytest.raises(ValueError, match="advance_fraction"):
        staged_swing_outward_offset_m(
            maximum_offset_m=0.005,
            advance_fraction=-0.1,
        )


def test_post_landing_reposition_snapshot_rewinds_only_state_machine() -> None:
    references = {
        leg: {
            "forward_m": float(index),
            "vertical_m": 0.20 + index * 0.01,
            "outward_m": 0.02,
        }
        for index, leg in enumerate(
            ("front_left", "front_right", "rear_left", "rear_right")
        )
    }
    snapshot = {
        "placement_sequence_legs": (
            "front_right",
            "front_left",
            "rear_right",
            "rear_left",
        ),
        "placement_sequence_position": 3,
        "placement_swing_leg": "rear_left",
        "placement_transfer_active": True,
        "completed_placement_legs": [
            "front_right",
            "front_left",
            "rear_right",
        ],
        "completed_placement_joint_targets_by_leg": {
            "front_right": [1.0, 2.0, 3.0],
            "front_left": [4.0, 5.0, 6.0],
            "rear_right": [7.0, 8.0, 9.0],
        },
        "completed_placement_reference_by_leg": {
            "front_right": {"forward_m": 1.0},
            "front_left": {"forward_m": 2.0},
            "rear_right": {"forward_m": 3.0},
        },
        "placement_transfer_reference_by_leg": references,
        "placement_leg_baseline_reference_by_leg": {"stale": {}},
        "base_position_m": [0.48, 0.08, 0.36],
        "placement_transfer_start_balance_position_m": [0.49, 0.07, 0.34],
        "placement_transfer_start_base_position_m": [0.48, 0.08, 0.36],
        "placement_transfer_target_base_position_m": [0.56, -0.01, 0.36],
        "placement_transfer_target_balance_position_m": [0.57, -0.03, 0.34],
        "maximum_foot_lift_m": [0.24, 0.20, 0.0, 0.181],
        "previous_action": [0.1] * 12,
        "previous_residual_action": [-0.1] * 12,
    }

    rewound = post_landing_reposition_snapshot(snapshot, leg="rear_right")

    assert rewound["placement_sequence_position"] == 2
    assert rewound["placement_swing_leg"] == "rear_right"
    assert rewound["completed_placement_legs"] == [
        "front_right",
        "front_left",
    ]
    assert rewound["placement_transfer_active"] is False
    assert rewound["placement_transfer_reference_by_leg"] == {}
    assert rewound["placement_leg_baseline_reference_by_leg"] == references
    assert rewound["placement_leg_baseline_lift_offset_m"] == pytest.approx(
        0.181
    )
    assert "rear_right" not in rewound[
        "completed_placement_joint_targets_by_leg"
    ]
    assert "rear_right" not in rewound["completed_placement_reference_by_leg"]
    assert rewound["previous_action"] == [0.0] * 12
    assert rewound["previous_residual_action"] == [0.0] * 12
    assert snapshot["placement_transfer_active"] is True

    invalid = dict(snapshot)
    invalid["placement_transfer_active"] = False
    with pytest.raises(ValueError, match="active inter-leg transfer"):
        post_landing_reposition_snapshot(invalid, leg="rear_right")


def test_post_clearance_advance_shifts_body_before_swing() -> None:
    assert split_post_clearance_advance_fractions(
        advance_fraction=0.0,
        body_shift_fraction_of_advance=0.5,
    ) == pytest.approx((0.0, 0.0))
    assert split_post_clearance_advance_fractions(
        advance_fraction=0.5,
        body_shift_fraction_of_advance=0.5,
    ) == pytest.approx((1.0, 0.0))
    assert split_post_clearance_advance_fractions(
        advance_fraction=0.75,
        body_shift_fraction_of_advance=0.5,
    ) == pytest.approx((1.0, 0.5))
    assert split_post_clearance_advance_fractions(
        advance_fraction=1.0,
        body_shift_fraction_of_advance=0.5,
    ) == pytest.approx((1.0, 1.0))

    with pytest.raises(ValueError, match="body_shift_fraction_of_advance"):
        split_post_clearance_advance_fractions(
            advance_fraction=0.5,
            body_shift_fraction_of_advance=1.0,
        )


def test_post_clearance_advance_can_shift_body_after_swing() -> None:
    arguments = {
        "body_shift_fraction_of_advance": 0.5,
        "sequence": "swing_then_body",
    }
    assert split_post_clearance_advance_fractions(
        advance_fraction=0.0,
        **arguments,
    ) == pytest.approx((0.0, 0.0))
    assert split_post_clearance_advance_fractions(
        advance_fraction=0.25,
        **arguments,
    ) == pytest.approx((0.0, 0.5))
    assert split_post_clearance_advance_fractions(
        advance_fraction=0.5,
        **arguments,
    ) == pytest.approx((0.0, 1.0))
    assert split_post_clearance_advance_fractions(
        advance_fraction=0.75,
        **arguments,
    ) == pytest.approx((0.5, 1.0))
    assert split_post_clearance_advance_fractions(
        advance_fraction=1.0,
        **arguments,
    ) == pytest.approx((1.0, 1.0))

    with pytest.raises(ValueError, match="sequence"):
        split_post_clearance_advance_fractions(
            advance_fraction=0.5,
            body_shift_fraction_of_advance=0.5,
            sequence="simultaneous",
        )


def test_staged_support_rear_pitch_scale_holds_then_blends() -> None:
    assert staged_support_rear_pitch_scale(
        elapsed_seconds=1.5,
        front_only_seconds=2.0,
        blend_seconds=2.0,
    ) == pytest.approx(0.0)
    assert staged_support_rear_pitch_scale(
        elapsed_seconds=3.0,
        front_only_seconds=2.0,
        blend_seconds=2.0,
    ) == pytest.approx(0.5)
    assert staged_support_rear_pitch_scale(
        elapsed_seconds=4.5,
        front_only_seconds=2.0,
        blend_seconds=2.0,
    ) == pytest.approx(1.0)
    assert staged_support_rear_pitch_scale(
        elapsed_seconds=2.1,
        front_only_seconds=2.0,
        blend_seconds=0.0,
    ) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="front_only_seconds"):
        staged_support_rear_pitch_scale(
            elapsed_seconds=1.0,
            front_only_seconds=-0.1,
            blend_seconds=2.0,
        )


def test_swing_support_abduction_mask_preserves_lift_authority() -> None:
    dof_names = (
        "front_left_hip_abduction",
        "rear_left_hip_abduction",
        "front_right_hip_abduction",
        "rear_right_hip_abduction",
        "front_left_hip_flexion",
        "rear_left_hip_flexion",
        "front_right_hip_flexion",
        "rear_right_hip_flexion",
        "front_left_knee",
        "rear_left_knee",
        "front_right_knee",
        "rear_right_knee",
    )
    mask = placement_policy_action_mask(
        dof_names,
        target_leg="front_left",
        mode="swing_plus_support_abduction",
    )
    assert np.flatnonzero(mask).tolist() == [0, 1, 2, 3, 4, 8]
    swing_only = placement_policy_action_mask(
        dof_names,
        target_leg="front_left",
        mode="swing_only",
    )
    assert np.flatnonzero(swing_only).tolist() == [0, 4, 8]
    with pytest.raises(ValueError, match="unknown placement target leg"):
        placement_policy_action_mask(
            dof_names,
            target_leg="middle_left",
            mode="swing_plus_support_abduction",
        )


def test_lift_hold_requires_height_support_margin_and_upright_body() -> None:
    options = {
        "swing_tip_height_m": 0.201,
        "initial_swing_tip_height_m": 0.010,
        "support_normal_loads_n": (12.0, 10.0, 11.0),
        "support_margin_m": 0.020,
        "projected_gravity_xyz": (0.0, 0.0, -0.999),
        "minimum_lift_m": 0.190,
        "contact_on_threshold_n": 1.0,
        "minimum_support_margin_m": 0.015,
        "minimum_upright_cosine": 0.978,
    }
    assert placement_lift_hold_reached(**options) is True
    assert placement_lift_hold_reached(
        **{**options, "swing_tip_height_m": 0.199}
    ) is False
    assert placement_lift_hold_reached(
        **{**options, "support_normal_loads_n": (12.0, 0.5, 11.0)}
    ) is False
    assert placement_lift_hold_reached(
        **{**options, "support_margin_m": 0.014}
    ) is False


def test_phase_training_replays_verified_prefix_before_exposing_target() -> None:
    gym = pytest.importorskip("gymnasium")
    from _placement_phase_training import PlacementPhaseTrainingEnv

    class FakePlacementEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(3,),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(2,),
                dtype=np.float32,
            )
            self.placement_sequence_legs = ("front_left", "front_right")
            self.actions: list[np.ndarray] = []
            self.snapshot_restores = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.completed_placement_legs: list[str] = []
            self.placement_swing_leg = "front_left"
            self.placement_transfer_active = False
            self.actions.clear()
            return np.zeros(3, dtype=np.float32), {}

        def capture_placement_phase_snapshot(self):
            return {"ready": True}

        def restore_placement_phase_snapshot(self, snapshot, *, seed=None, options=None):
            assert snapshot == {"ready": True}
            self.completed_placement_legs = ["front_left"]
            self.placement_swing_leg = "front_right"
            self.placement_transfer_active = False
            self.snapshot_restores += 1
            return np.full(3, 9.0, dtype=np.float32), {}

        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
            if len(self.actions) == 1:
                self.completed_placement_legs.append("front_left")
                self.placement_transfer_active = True
            elif len(self.actions) == 2:
                self.placement_transfer_active = False
                self.placement_swing_leg = "front_right"
            terminated = len(self.actions) >= 3
            return (
                np.full(3, len(self.actions), dtype=np.float32),
                float(len(self.actions)),
                terminated,
                False,
                {
                    "maximum_foot_lift_m_by_leg": {"front_right": 0.020},
                    "base_clearance_m": 0.300,
                    "placement_support_margin_m": 0.010,
                    "placement_support_contact_fraction": 2.0 / 3.0,
                    "maximum_support_slip_m": 0.005,
                    "placement_swing_lift_m": 0.025,
                    "placement_upright_cosine": 0.990,
                    "placement_goal_hold_step_count": 4,
                    "placement_desired_lift_m": 0.030,
                    "placement_swing_reference_joint_positions_rad": (
                        np.asarray((0.1, 0.2, 0.3), dtype=np.float32)
                        * len(self.actions)
                    ),
                    "placement_swing_actual_joint_positions_rad": (
                        np.asarray((0.01, 0.02, 0.03), dtype=np.float32)
                        * len(self.actions)
                    ),
                },
            )

    class FixedPolicy:
        def predict(self, observation, *, deterministic):
            assert deterministic is True
            return np.full(2, 0.25, dtype=np.float32), None

    raw = FakePlacementEnv()
    wrapped = PlacementPhaseTrainingEnv(
        raw,
        target_leg="front_right",
        precursor_policies={"front_left": FixedPolicy()},
    )
    observation, info = wrapped.reset(seed=7)

    assert placement_phase_ready(
        sequence_legs=raw.placement_sequence_legs,
        completed_legs=raw.completed_placement_legs,
        active_leg=raw.placement_swing_leg,
        transfer_active=raw.placement_transfer_active,
        target_leg="front_right",
    ) is True
    np.testing.assert_allclose(observation, (2.0, 2.0, 2.0))
    np.testing.assert_allclose(raw.actions[0], (0.25, 0.25))
    np.testing.assert_allclose(raw.actions[1], (0.0, 0.0))
    assert info["phase_training_precursor_steps"] == 2
    assert wrapped.training_stats()["successful_precursor_attempts"] == 1

    _, reward, terminated, truncated, _ = wrapped.step(
        np.full(2, -0.5, dtype=np.float32)
    )
    assert reward == pytest.approx(3.0)
    assert terminated is True
    assert truncated is False
    np.testing.assert_allclose(raw.actions[2], (-0.5, -0.5))
    stats = wrapped.training_stats()
    assert stats["target_steps"] == 1
    assert stats["maximum_target_swing_lift_m"] == pytest.approx(0.025)
    assert stats["minimum_target_base_clearance_m"] == pytest.approx(0.300)
    assert stats["minimum_target_support_margin_m"] == pytest.approx(0.010)
    assert stats["minimum_target_support_contact_fraction"] == pytest.approx(
        2.0 / 3.0
    )
    assert stats["maximum_target_support_slip_m"] == pytest.approx(0.005)
    assert stats["minimum_target_upright_cosine"] == pytest.approx(0.990)
    assert stats["maximum_target_goal_hold_steps"] == 4
    assert stats["maximum_target_desired_lift_m"] == pytest.approx(0.030)
    assert stats["maximum_target_swing_reference_change_rad"] == (
        pytest.approx(0.0)
    )
    assert stats["maximum_target_swing_actual_change_rad"] == pytest.approx(0.0)
    assert stats["maximum_target_residual_action_abs"] == pytest.approx(0.5)
    assert stats["maximum_target_applied_action_abs"] == pytest.approx(0.5)

    observation, info = wrapped.reset(seed=8)
    np.testing.assert_allclose(observation, (9.0, 9.0, 9.0))
    assert info["phase_training_snapshot_restored"] is True
    assert raw.snapshot_restores == 1
    assert wrapped.training_stats()["cached_phase_restores"] == 1

    masked_raw = FakePlacementEnv()
    masked = PlacementPhaseTrainingEnv(
        masked_raw,
        target_leg="front_right",
        precursor_policies={"front_left": FixedPolicy()},
        target_residual_mask=np.asarray((1.0, 0.0), dtype=np.float32),
    )
    masked.reset(seed=9)
    _, _, _, _, masked_info = masked.step(
        np.full(2, -0.5, dtype=np.float32)
    )
    np.testing.assert_allclose(masked_raw.actions[2], (-0.5, 0.0))
    assert masked_info["phase_training_residual_action_max_abs"] == (
        pytest.approx(0.5)
    )
    assert masked_info["phase_training_applied_action_max_abs"] == (
        pytest.approx(0.5)
    )
    assert masked.training_stats()["target_action_mode"] == (
        "masked_direct_ppo_action"
    )
    assert masked.training_stats()[
        "target_residual_active_action_indices"
    ] == [0]

    compact_raw = FakePlacementEnv()
    compact = PlacementPhaseTrainingEnv(
        compact_raw,
        target_leg="front_right",
        precursor_policies={"front_left": FixedPolicy()},
        target_residual_mask=np.asarray((1.0, 0.0), dtype=np.float32),
        compact_residual_action=True,
    )
    assert compact.action_space.shape == (1,)
    assert compact.raw_action_space.shape == (2,)
    compact.reset(seed=10)
    compact.step(np.asarray((-0.75,), dtype=np.float32))
    np.testing.assert_allclose(compact_raw.actions[2], (-0.75, 0.0))
    compact_stats = compact.training_stats()
    assert compact_stats["compact_residual_action"] is True
    assert compact_stats["policy_action_size"] == 1
    assert compact_stats["raw_action_size"] == 2
    with pytest.raises(ValueError, match="action shape"):
        compact.step(np.asarray((-0.75, 0.0), dtype=np.float32))


def test_frozen_base_residual_policy_preserves_compact_swing_composition() -> None:
    gym = pytest.importorskip("gymnasium")
    from _placement_phase_training import FrozenBaseResidualPolicy

    class FixedPolicy:
        def __init__(self, action, observation_size) -> None:
            self.action = np.asarray(action, dtype=np.float32)
            self.observation_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(observation_size,),
                dtype=np.float32,
            )

        def predict(self, observation, *, deterministic):
            assert deterministic is True
            assert observation.shape == self.observation_space.shape
            return self.action.copy(), None

    action_space = gym.spaces.Box(
        -1.0,
        1.0,
        shape=(4,),
        dtype=np.float32,
    )
    policy = FrozenBaseResidualPolicy(
        base_policy=FixedPolicy((0.8, 0.7, -0.6, -0.5), 3),
        residual_policy=FixedPolicy((0.4, -0.8), 5),
        action_space=action_space,
        residual_scale=0.5,
        base_mask=np.asarray((1.0, 0.0, 1.0, 0.0), dtype=np.float32),
        residual_mask=np.asarray((1.0, 0.0, 1.0, 0.0), dtype=np.float32),
        compact_residual_action=True,
    )
    action, _ = policy.predict(
        np.arange(5, dtype=np.float32),
        deterministic=True,
    )
    np.testing.assert_allclose(action, (1.0, 0.0, -1.0, 0.0))
    assert policy.observation_space.shape == (5,)
    with pytest.raises(ValueError, match="requires residual_mask"):
        FrozenBaseResidualPolicy(
            base_policy=FixedPolicy((0.0,) * 4, 3),
            residual_policy=FixedPolicy((0.0,), 3),
            action_space=action_space,
            residual_scale=0.5,
            compact_residual_action=True,
        )


def test_transfer_training_controls_supports_and_ends_on_gate_acceptance() -> None:
    gym = pytest.importorskip("gymnasium")
    from _placement_phase_training import PlacementPhaseTrainingEnv

    class FakeTransferEnv(gym.Env):
        def __init__(self) -> None:
            self.observation_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(3,),
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Box(
                -1.0,
                1.0,
                shape=(2,),
                dtype=np.float32,
            )
            self.placement_sequence_legs = ("front_left", "front_right")
            self.control_hz = 10
            self.reward_config = {"success": 100.0}
            self.inter_leg_transfer_config = {
                "training_reward": {
                    "balance_target_error_progress_per_m": 1000.0,
                    "support_margin_progress_per_m": 2000.0,
                    "maximum_progress_m_per_step": 0.010,
                }
            }
            self.actions: list[np.ndarray] = []
            self.snapshot_restores = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.completed_placement_legs: list[str] = []
            self.placement_swing_leg = "front_left"
            self.placement_transfer_active = False
            self.actions.clear()
            return np.zeros(3, dtype=np.float32), {}

        def capture_placement_phase_snapshot(self):
            return {"transfer": self.placement_transfer_active}

        def restore_placement_phase_snapshot(
            self,
            snapshot,
            *,
            seed=None,
            options=None,
        ):
            assert snapshot in ({"transfer": True}, {"transfer": False})
            self.completed_placement_legs = ["front_left"]
            self.placement_swing_leg = "front_right"
            self.placement_transfer_active = bool(snapshot["transfer"])
            self.actions.clear()
            self.snapshot_restores += 1
            return np.full(3, 9.0, dtype=np.float32), {}

        def step(self, action):
            self.actions.append(np.asarray(action, dtype=np.float32).copy())
            transfer_event = None
            if not self.completed_placement_legs:
                self.completed_placement_legs = ["front_left"]
                self.placement_swing_leg = "front_right"
                self.placement_transfer_active = True
            elif self.placement_transfer_active:
                self.placement_transfer_active = False
                transfer_event = "front_left->front_right"
            return (
                np.full(3, len(self.actions), dtype=np.float32),
                2.0,
                False,
                False,
                {
                    "placement_transfer_completed_event": transfer_event,
                    "base_clearance_m": 0.31,
                    "placement_support_margin_m": (
                        0.018 if len(self.actions) == 2 else 0.012
                    ),
                    "placement_support_contact_fraction": 1.0,
                    "maximum_support_slip_m": 0.004,
                    "placement_upright_cosine": 0.985,
                    "placement_transfer_base_target_error_m": (
                        0.004 if len(self.actions) == 2 else 0.009
                    ),
                    "placement_transfer_body_rate_rad_s": 0.08,
                    "placement_transfer_swing_total_load_n": 0.2,
                },
            )

    class FixedPolicy:
        def predict(self, observation, *, deterministic):
            assert deterministic is True
            return np.full(2, 0.25, dtype=np.float32), None

    raw = FakeTransferEnv()
    wrapped = PlacementPhaseTrainingEnv(
        raw,
        target_leg="front_right",
        precursor_policies={"front_left": FixedPolicy()},
        target_residual_mask=np.asarray((1.0, 0.0), dtype=np.float32),
        compact_residual_action=True,
        train_transfer=True,
    )
    observation, info = wrapped.reset(seed=11)
    np.testing.assert_allclose(observation, (1.0, 1.0, 1.0))
    assert info["phase_training_target_mode"] == "inter_leg_transfer"
    np.testing.assert_allclose(raw.actions[0], (0.25, 0.25))

    _, reward, terminated, truncated, result = wrapped.step(
        np.asarray((-0.5,), dtype=np.float32)
    )
    np.testing.assert_allclose(raw.actions[1], (-0.5, 0.0))
    assert reward == pytest.approx(119.0)
    assert terminated is True
    assert truncated is False
    assert result["phase_training_transfer_completed"] is True
    stats = wrapped.training_stats()
    assert stats["target_mode"] == "inter_leg_transfer"
    assert stats["completed_target_transfers"] == 1
    assert stats["maximum_target_transfer_balance_error_m"] == pytest.approx(
        0.004
    )
    assert stats["minimum_target_transfer_swing_load_n"] == pytest.approx(0.2)
    assert stats["transfer_progress_reward"]["cumulative_reward"] == (
        pytest.approx(17.0)
    )
    assert result["phase_training_transfer_balance_error_progress_m"] == (
        pytest.approx(0.005)
    )
    assert result["phase_training_transfer_support_margin_progress_m"] == (
        pytest.approx(0.006)
    )

    observation, info = wrapped.reset(seed=12)
    np.testing.assert_allclose(observation, (9.0, 9.0, 9.0))
    assert info["phase_training_snapshot_restored"] is True
    assert raw.snapshot_restores == 1

    hold_raw = FakeTransferEnv()
    hold_wrapped = PlacementPhaseTrainingEnv(
        hold_raw,
        target_leg="front_right",
        precursor_policies={"front_left": FixedPolicy()},
        target_residual_mask=np.asarray((1.0, 0.0), dtype=np.float32),
        compact_residual_action=True,
        train_transfer=True,
        transfer_post_hold_seconds=0.2,
        train_post_transfer_hold_only=True,
    )
    hold_wrapped.reset(seed=13)
    assert hold_wrapped.phase_snapshot_mode == "inter_leg_transfer"
    _, reward, terminated, _, result = hold_wrapped.step(
        np.asarray((-0.4,), dtype=np.float32)
    )
    assert result["phase_training_transfer_completed"] is True
    assert result["phase_training_transfer_post_hold_completed"] is False
    assert result["phase_training_transfer_post_hold_remaining_steps"] == 2
    np.testing.assert_allclose(hold_raw.actions[1], (0.0, 0.0))
    assert reward == pytest.approx(2.0)
    assert terminated is False
    _, _, terminated, _, result = hold_wrapped.step(
        np.asarray((-0.3,), dtype=np.float32)
    )
    assert result["phase_training_transfer_post_hold_remaining_steps"] == 1
    assert terminated is False
    _, reward, terminated, _, result = hold_wrapped.step(
        np.asarray((-0.2,), dtype=np.float32)
    )
    assert result["phase_training_transfer_post_hold_completed"] is True
    assert result["phase_training_transfer_post_hold_remaining_steps"] == 0
    assert reward == pytest.approx(102.0)
    assert terminated is True
    hold_stats = hold_wrapped.training_stats()
    assert hold_stats["target_mode"] == "post_transfer_hold"
    assert hold_stats["train_post_transfer_hold_only"] is True
    assert hold_stats["transfer_post_hold_steps"] == 2
    assert hold_stats["completed_target_transfers"] == 1
    assert hold_stats["completed_target_transfer_holds"] == 1
    assert hold_stats["phase_snapshot_mode"] == "post_transfer_hold"

    observation, info = hold_wrapped.reset(seed=14)
    np.testing.assert_allclose(observation, (9.0, 9.0, 9.0))
    assert info["phase_training_snapshot_restored"] is True
    assert hold_wrapped.transfer_post_hold_steps_remaining == 2


def test_phase_training_ready_requires_every_earlier_leg() -> None:
    class RawState:
        placement_sequence_legs = ("front_left", "front_right")
        completed_placement_legs = ()
        placement_swing_leg = "front_right"
        placement_transfer_active = False

    state = RawState()
    assert placement_phase_ready(
        sequence_legs=state.placement_sequence_legs,
        completed_legs=state.completed_placement_legs,
        active_leg=state.placement_swing_leg,
        transfer_active=state.placement_transfer_active,
        target_leg="front_right",
    ) is False
    RawState.completed_placement_legs = ("front_left",)
    state = RawState()
    assert placement_phase_ready(
        sequence_legs=state.placement_sequence_legs,
        completed_legs=state.completed_placement_legs,
        active_leg=state.placement_swing_leg,
        transfer_active=state.placement_transfer_active,
        target_leg="front_right",
    ) is True
    with pytest.raises(ValueError, match="Unknown placement phase target"):
        placement_phase_ready(
            sequence_legs=state.placement_sequence_legs,
            completed_legs=state.completed_placement_legs,
            active_leg=state.placement_swing_leg,
            transfer_active=state.placement_transfer_active,
            target_leg="rear_left",
        )


def test_transfer_training_ready_requires_exact_prefix_and_active_transfer() -> None:
    options = {
        "sequence_legs": ("front_right", "front_left", "rear_right"),
        "completed_legs": ("front_right", "front_left"),
        "active_leg": "rear_right",
        "transfer_active": True,
        "target_leg": "rear_right",
    }
    assert placement_transfer_ready(**options) is True
    assert placement_transfer_ready(
        **{**options, "transfer_active": False}
    ) is False
    assert placement_transfer_ready(
        **{**options, "completed_legs": ("front_left", "front_right")}
    ) is False
    assert placement_transfer_ready(
        **{**options, "target_leg": "front_right"}
    ) is False
    with pytest.raises(ValueError, match="Unknown placement transfer target"):
        placement_transfer_ready(
            **{**options, "target_leg": "rear_left"}
        )


def test_bounded_residual_action_preserves_base_and_clips_correction() -> None:
    action = compose_bounded_residual_action(
        (0.90, -0.80, 0.10),
        (1.00, -1.00, 0.40),
        residual_scale=0.25,
    )
    np.testing.assert_allclose(action, (1.0, -1.0, 0.20))
    masked = compose_bounded_residual_action(
        (0.90, -0.80, 0.10),
        (1.00, -1.00, 0.40),
        residual_scale=0.25,
        residual_mask=(1.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(masked, (1.0, -0.80, 0.20))

    with pytest.raises(ValueError, match="matching vectors"):
        compose_bounded_residual_action((0.0,), (0.0, 1.0), residual_scale=0.25)
    with pytest.raises(ValueError, match="within"):
        compose_bounded_residual_action((0.0,), (0.0,), residual_scale=0.0)
    with pytest.raises(ValueError, match="residual_mask"):
        compose_bounded_residual_action(
            (0.0,),
            (0.0,),
            residual_scale=0.25,
            residual_mask=(1.0, 0.0),
        )


def test_compact_action_expands_onto_active_joint_mask() -> None:
    from _stair_rl_contract import expand_compact_masked_action

    expanded = expand_compact_masked_action(
        (0.25, -0.50, 0.75),
        (1.0, 0.0, 1.0, 0.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(expanded, (0.25, 0.0, -0.50, 0.0, 0.0, 0.75))
    with pytest.raises(ValueError, match="size"):
        expand_compact_masked_action((0.25,), (1.0, 0.0, 1.0))
    with pytest.raises(ValueError, match="binary"):
        expand_compact_masked_action((0.25,), (0.5, 0.0))


def test_masked_overlay_keeps_swing_action_and_replaces_support_action() -> None:
    composed = overlay_masked_action(
        (0.8, -0.6, 0.4, -0.2),
        (-0.9, 0.7, -0.5, 0.3),
        (0.0, 1.0, 0.0, 1.0),
    )
    np.testing.assert_allclose(composed, (0.8, 0.7, 0.4, 0.3))
    with pytest.raises(ValueError, match="binary"):
        overlay_masked_action((0.0,), (0.0,), (0.5,))


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
