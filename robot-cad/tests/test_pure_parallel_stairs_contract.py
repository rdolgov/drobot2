"""Static contracts for the external Isaac Lab pure-parallel stair task."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "simulation" / "isaac" / "rl" / "parallel_stairs"


def _source(name: str) -> str:
    return (PACKAGE / name).read_text(encoding="utf-8")


def _class_assignments(source: str, class_name: str) -> dict[str, object]:
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    values: dict[str, object] = {}
    for node in class_node.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            try:
                values[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def _method_source(source: str, class_name: str, method_name: str) -> str:
    tree = ast.parse(source)
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    segment = ast.get_source_segment(source, method)
    assert segment is not None
    return segment


def test_exact_stair_and_policy_dimensions_are_fixed() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    cfg_values = _class_assignments(cfg_source, "DrobotPureStairsEnvCfg")
    assert cfg_values["action_space"] == 12
    assert cfg_values["observation_space"] == 70
    assert cfg_values["stair_rise_m"] == 0.18
    assert cfg_values["stair_tread_depth_m"] == 0.25
    assert cfg_values["stair_step_count"] == 4
    assert "EFFORT_CAP_NM = 0.8825985" in cfg_source

    terrain_values = _class_assignments(
        _source("exact_stairs_terrain.py"), "ExactStairsTerrainCfg"
    )
    assert terrain_values["rise_m"] == 0.18
    assert terrain_values["tread_depth_m"] == 0.25
    assert terrain_values["step_count"] == 4


def test_policy_observation_uses_only_deployable_sensor_state() -> None:
    env_source = _source("pure_stairs_env.py")
    observation = _method_source(env_source, "DrobotPureStairsEnv", "_get_observations")

    required = (
        "root_ang_vel_b",
        "projected_gravity_b",
        "joint_pos",
        "joint_vel",
        "_previous_actions",
        "_foot_forces",
        "_depth_observation",
    )
    for field in required:
        assert field in observation

    privileged = ("root_pos_w", "body_pos_w", "env_origins", "_terrain_height")
    for field in privileged:
        assert field not in observation


def test_reward_has_no_scripted_phase_or_reference_action() -> None:
    reward = _method_source(_source("pure_stairs_env.py"), "DrobotPureStairsEnv", "_get_rewards")
    lowered = reward.lower()
    forbidden_terms = (
        "gait_phase",
        "leg_order",
        "reference_action",
        "trajectory",
        "inverse_kinematics",
    )
    for forbidden in forbidden_terms:
        assert forbidden not in lowered


def test_low_sideways_and_hip_variants_keep_pure_rl_contract() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    hip = _class_assignments(cfg_source, "DrobotPureStairsHipEnvCfg")
    low = _class_assignments(cfg_source, "DrobotPureStairsLowHipEnvCfg")
    sideways = _class_assignments(cfg_source, "DrobotPureStairsSidewaysHipEnvCfg")

    assert hip["action_scale_abduction_rad"] == 0.30
    assert hip["action_scale_hip_rad"] == 0.90
    assert hip["action_scale_knee_rad"] == 1.20
    assert low["initial_base_height_m"] == 0.30
    assert sideways["reset_yaw_deg"] == 90.0
    assert "(0.0, 0.0, 0.7071067812, 0.7071067812)" in cfg_source

    env_source = _source("pure_stairs_env.py")
    pre_step = _method_source(env_source, "DrobotPureStairsEnv", "_pre_physics_step")
    reward = _method_source(env_source, "DrobotPureStairsEnv", "_get_rewards")
    assert "soft_joint_pos_limits" in pre_step
    assert "placement_score" in reward
    assert "tread_potential" in reward


def test_first_step_curriculum_requires_supported_tread_hold() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    first_step = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepHipEnvCfg"
    )
    assert first_step["reset_forward_offset_m"] == 0.10
    assert first_step["first_step_curriculum"] is True
    assert first_step["reward_tread_count"] == 1
    assert first_step["support_reward_scale"] == 0.25
    assert first_step["supported_lift_reward_scale"] == 0.50

    env_source = _source("pure_stairs_env.py")
    dones = _method_source(env_source, "DrobotPureStairsEnv", "_get_dones")
    reward = _method_source(env_source, "DrobotPureStairsEnv", "_get_rewards")
    assert "support_count >= 3.0" in dones
    assert "_tread_hold_steps" in dones
    assert "first_step_min_base_gain_m" in dones
    assert "first_step_require_base_gain" in dones
    assert "base_contact" in dones
    assert "base_contact_failure" in dones
    assert "retained_support" in reward
    assert "tread_hold_fraction" in reward
    assert "tread_transfer" in reward
    assert "narrow_transfer = narrow_tread_potential * base_gain_fraction" in reward
    assert "narrow_transfer_reward_scale" in reward
    assert "first_step_completion = self._success.float()" in reward
    assert "first_step_completion_reward_scale" in reward
    assert "tread_height_delta_scale" in reward
    assert "new_tread_potential_reward_scale" in reward
    assert "tread_potential_reward_scale" in reward
    assert "narrow_tread_potential" in reward
    assert "new_narrow_tread_potential_reward_scale" in reward
    assert "tread_contact_reward_scale" in reward

    close_2 = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepClose2HipEnvCfg"
    )
    close_4 = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepClose4HipEnvCfg"
    )
    close_6 = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepClose6HipEnvCfg"
    )
    assert close_2["reset_forward_offset_m"] == 0.10
    assert close_2["first_step_min_base_gain_m"] == 0.02
    assert close_2["tread_potential_reward_scale"] == 0.40
    assert close_2["new_narrow_tread_potential_reward_scale"] == 8.0
    assert close_2["narrow_tread_potential_reward_scale"] == 1.00
    assert close_4["first_step_min_base_gain_m"] == 0.04
    assert close_6["first_step_min_base_gain_m"] == 0.06

    landing = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepLandingHipEnvCfg"
    )
    assert landing["first_step_require_base_gain"] is False
    assert landing["first_step_hold_steps"] == 3
    assert landing["tread_contact_reward_scale"] == 2.00
    assert landing["first_step_completion_reward_scale"] == 10.0

    close_1 = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepClose1HipEnvCfg"
    )
    assert close_1["first_step_min_base_gain_m"] == 0.01
    assert close_1["reset_forward_jitter_m"] == 0.03
    assert close_1["first_step_hold_steps"] == 4
    assert close_1["progress_delta_reward_scale"] == 10.0
    assert close_1["height_delta_reward_scale"] == 60.0
    assert close_1["narrow_transfer_reward_scale"] == 4.00
    assert close_1["first_step_completion_reward_scale"] == 15.0


def test_foot_lift_curriculum_is_symmetric_and_supported() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    foot_lift = _class_assignments(
        cfg_source, "DrobotPureStairsFootLiftHipEnvCfg"
    )
    assert foot_lift["foot_lift_curriculum"] is True
    assert foot_lift["reset_forward_offset_m"] == -0.10
    assert foot_lift["supported_lift_reward_scale"] == 1.50

    env_source = _source("pure_stairs_env.py")
    dones = _method_source(env_source, "DrobotPureStairsEnv", "_get_dones")
    assert "foot_clearance.max" in dones
    assert "support_count >= 3.0" in dones
    assert "foot_lift_height_m" in dones
    assert "_lift_hold_steps" in dones

    reward = _method_source(env_source, "DrobotPureStairsEnv", "_get_rewards")
    assert "supported_lift" in reward
    assert "supported_lift_reward_scale" in reward

    lift_10 = _class_assignments(cfg_source, "DrobotPureStairsFootLift10HipEnvCfg")
    lift_14 = _class_assignments(cfg_source, "DrobotPureStairsFootLift14HipEnvCfg")
    assert lift_10["foot_lift_height_m"] == 0.10
    assert lift_14["foot_lift_height_m"] == 0.14
