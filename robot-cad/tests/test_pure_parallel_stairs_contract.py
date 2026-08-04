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

    terrain_values = _class_assignments(_source("exact_stairs_terrain.py"), "ExactStairsTerrainCfg")
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


def test_full_fold_sideways_lift_curriculum_is_force_backed_and_staged() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    env_source = _source("pure_stairs_env.py")
    runner_source = _source("agents/rsl_rl_ppo_cfg.py")
    registration = _source("__init__.py")

    lift5 = _class_assignments(cfg_source, "DrobotPureStairsYaw90FullFoldFootLift5HipEnvCfg")
    lift10 = _class_assignments(cfg_source, "DrobotPureStairsYaw90FullFoldFootLift10HipEnvCfg")
    lift14 = _class_assignments(cfg_source, "DrobotPureStairsYaw90FullFoldFootLift14HipEnvCfg")
    lift19 = _class_assignments(cfg_source, "DrobotPureStairsYaw90FullFoldFootLift19HipEnvCfg")
    assert lift10["foot_lift_curriculum"] is True
    assert lift10["foot_lift_height_m"] == 0.10
    assert lift10["foot_lift_settle_steps"] == 60
    assert lift10["progress_delta_reward_scale"] == 0.0
    assert lift10["supported_lift_reward_scale"] == 1.50
    assert lift10["success_completion_reward_scale"] == 200.0
    assert lift5["foot_lift_height_m"] == 0.05
    assert lift5["foot_lift_hold_steps"] == 4
    assert lift14["foot_lift_height_m"] == 0.14
    assert lift19["foot_lift_height_m"] == 0.19
    assert "support_count >= 3.0" in env_source
    assert "self._steps_since_reset > self.cfg.foot_lift_settle_steps" in env_source
    assert "max(self.cfg.foot_lift_height_m, 1.0e-6)" in env_source
    assert "foot_lift_success = (" in env_source
    assert ") & ~self._failed" in env_source

    for height in ("5", "10", "14", "19"):
        assert f"DrobotPureStairsYaw90FullFoldFootLift{height}HipPPORunnerCfg" in runner_source
        assert f"Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift{height}-Hip-Direct" in registration
    assert "DrobotPureStairsYaw90FullFoldFootLift5ConsolidateHipPPORunnerCfg" in runner_source
    assert "DrobotPureStairsYaw90FullFoldFootLift10ConsolidateHipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-Consolidate-Hip-Direct" in registration

    bridge5 = _class_assignments(cfg_source, "DrobotPureStairsYaw90FoldBridgeFootLift5HipEnvCfg")
    bridge10 = _class_assignments(cfg_source, "DrobotPureStairsYaw90FoldBridgeFootLift10HipEnvCfg")
    assert bridge10["reset_fold_fraction_min"] == 0.0
    assert bridge10["reset_fold_fraction_max"] == 1.0
    assert bridge10["reset_alpha_power"] == 2.0
    assert bridge10["reset_base_height_min_m"] == 0.30
    assert bridge10["reset_base_height_max_m"] == 0.46
    assert bridge10["foot_unload_reward_scale"] == 0.75
    assert bridge5["foot_lift_height_m"] == 0.05
    assert "relative_candidate_force" in env_source
    assert "per_foot_unload" in env_source
    assert "foot_unload_reward_scale * foot_unload" in env_source
    assert "per_foot_lift * per_foot_unload" in env_source
    assert "unloaded_lift_reward_scale * unloaded_lift" in env_source
    for height in ("5", "10"):
        assert f"DrobotPureStairsYaw90FoldBridgeFootLift{height}HipPPORunnerCfg" in runner_source
        assert f"Drobot-Pure-Stairs-Yaw90-FoldBridge-Foot-Lift{height}-Hip-Direct" in registration
    assert "DrobotPureStairsYaw90FoldBridgeFootLift5ConsolidateHipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Yaw90-FoldBridge-Foot-Lift5-Consolidate-Hip-Direct" in registration
    assert "DrobotPureStairsYaw90FoldBridgeFootLift5Wide512HipPPORunnerCfg" in runner_source
    assert "self.actor.hidden_dims = [512, 512]" in runner_source
    assert "self.critic.hidden_dims = [512, 512]" in runner_source
    assert "Drobot-Pure-Stairs-Yaw90-FoldBridge-Foot-Lift5-Wide512-Hip-Direct" in registration
    assert "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-Wide512-Hip-Direct" in registration
    coupled = _class_assignments(
        cfg_source, "DrobotPureStairsYaw90FullFoldFootLift5CoupledHipEnvCfg"
    )
    assert coupled["foot_unload_reward_scale"] == 0.75
    assert coupled["unloaded_lift_reward_scale"] == 4.0
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-Coupled-Wide512-Hip-Direct"
        in registration
    )
    tail75 = _class_assignments(
        cfg_source, "DrobotPureStairsYaw90FoldTail75FootLift5HipEnvCfg"
    )
    assert tail75["reset_fold_fraction_min"] == 0.75
    assert tail75["reset_fold_fraction_max"] == 1.0
    assert tail75["reset_alpha_power"] == 0.5
    assert tail75["reset_base_height_min_m"] == 0.30
    assert tail75["reset_base_height_max_m"] == 0.34
    assert "DrobotPureStairsYaw90FoldTail75FootLift5Wide512HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Yaw90-FoldTail75-Foot-Lift5-Wide512-Hip-Direct" in registration
    assert "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift10-Consolidate-Hip-Direct" in registration
    assert "self.algorithm.entropy_coef = 0.0" in runner_source
    assert "self.algorithm.learning_rate = 2.0e-5" in runner_source


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
    first_step = _class_assignments(cfg_source, "DrobotPureStairsFirstStepHipEnvCfg")
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

    close_2 = _class_assignments(cfg_source, "DrobotPureStairsFirstStepClose2HipEnvCfg")
    close_4 = _class_assignments(cfg_source, "DrobotPureStairsFirstStepClose4HipEnvCfg")
    close_6 = _class_assignments(cfg_source, "DrobotPureStairsFirstStepClose6HipEnvCfg")
    assert close_2["reset_forward_offset_m"] == 0.10
    assert close_2["first_step_min_base_gain_m"] == 0.02
    assert close_2["tread_potential_reward_scale"] == 0.40
    assert close_2["new_narrow_tread_potential_reward_scale"] == 8.0
    assert close_2["narrow_tread_potential_reward_scale"] == 1.00
    assert close_4["first_step_min_base_gain_m"] == 0.04
    assert close_6["first_step_min_base_gain_m"] == 0.06

    landing = _class_assignments(cfg_source, "DrobotPureStairsFirstStepLandingHipEnvCfg")
    assert landing["first_step_require_base_gain"] is False
    assert landing["first_step_hold_steps"] == 3
    assert landing["tread_contact_reward_scale"] == 2.00
    assert landing["first_step_completion_reward_scale"] == 10.0

    close_1 = _class_assignments(cfg_source, "DrobotPureStairsFirstStepClose1HipEnvCfg")
    assert close_1["first_step_min_base_gain_m"] == 0.01
    assert close_1["reset_forward_jitter_m"] == 0.03
    assert close_1["first_step_hold_steps"] == 4
    assert close_1["progress_delta_reward_scale"] == 10.0
    assert close_1["height_delta_reward_scale"] == 60.0
    assert close_1["narrow_transfer_reward_scale"] == 4.00
    assert close_1["first_step_completion_reward_scale"] == 15.0


def test_foot_lift_curriculum_is_symmetric_and_supported() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    foot_lift = _class_assignments(cfg_source, "DrobotPureStairsFootLiftHipEnvCfg")
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


def test_long_landing_runner_retains_complete_rare_sequences() -> None:
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "class DrobotPureStairsFirstStepLandingLongHipPPORunnerCfg" in runner_source
    assert "num_steps_per_env = 64" in runner_source
    assert "self.algorithm.num_learning_epochs = 10" in runner_source
    assert "self.algorithm.num_mini_batches = 8" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Landing-Long-Hip-Direct" in registration


def test_lift_consolidation_uses_accurate_episode_counts_and_low_entropy() -> None:
    env_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env.py").read_text(
        encoding="utf-8"
    )
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    assert "self._completed_episode_count += completed_count" in env_source
    assert "self._successful_episode_count += successful_count" in env_source
    assert '"Metrics/reset_success_rate"' in env_source
    assert '"Metrics/completed_episodes"' in env_source
    assert '"[DROBOT_EPISODE_TOTALS] "' in env_source
    assert "class DrobotPureStairsFootLiftConsolidateHipEnvCfg" in cfg_source
    assert "success_completion_reward_scale = 100.0" in cfg_source
    assert "class DrobotPureStairsFootLiftConsolidateHipPPORunnerCfg" in runner_source
    assert "self.algorithm.entropy_coef = 0.0" in runner_source
    assert "self.algorithm.learning_rate = 5.0e-5" in runner_source
    assert "Drobot-Pure-Stairs-Foot-Lift-Consolidate-Hip-Direct" in registration

    anneal_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/anneal_rsl_rl_checkpoint.py"
    ).read_text(encoding="utf-8")
    assert 'std_key = "distribution.std_param"' in anneal_source
    assert 'infos["noise_anneal"]' in anneal_source
    assert '"mlp_weights_changed": False' in anneal_source

    widen_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/widen_rsl_rl_checkpoint.py"
    ).read_text(encoding="utf-8")
    assert 'widened["mlp.2.weight"] = second_weight[mapping]' in widen_source
    assert "output_weight[:, mapping].clone()" in widen_source
    assert "downstream_split.unsqueeze(0)" in widen_source
    assert '"symmetry_breaking_downstream_split": [0.49, 0.51]' in widen_source
    assert 'optimizer["state"] = {}' in widen_source
    assert '"function_preserving": True' in widen_source
    assert "max(actor_error, critic_error) > 1.0e-4" in widen_source


def test_landing_reset_comparison_keeps_full_fold_and_true_sideways_pose() -> None:
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    low = _class_assignments(cfg_source, "DrobotPureStairsFirstStepLandingLowHipEnvCfg")
    sideways = _class_assignments(cfg_source, "DrobotPureStairsFirstStepLandingSidewaysHipEnvCfg")
    assert low["initial_base_height_m"] == 0.30
    assert sideways["reset_yaw_deg"] == 90.0
    assert "LOW_FOLD_HIP_RAD" in cfg_source
    assert "LOW_FOLD_KNEE_RAD" in cfg_source
    assert "Drobot-Pure-Stairs-First-Step-Landing-Low-Hip-Direct" in registration
    assert "Drobot-Pure-Stairs-First-Step-Landing-Sideways-Hip-Direct" in registration


def test_landing_consolidation_reinforces_rare_exact_contacts() -> None:
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    assert "class DrobotPureStairsFirstStepLandingConsolidateHipEnvCfg" in cfg_source
    assert "success_completion_reward_scale = 100.0" in cfg_source
    assert "class DrobotPureStairsFirstStepLandingConsolidateHipPPORunnerCfg" in runner_source
    assert "self.algorithm.entropy_coef = 0.0" in runner_source
    assert "self.algorithm.learning_rate = 5.0e-5" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Landing-Consolidate-Hip-Direct" in registration


def test_contact_retention_requires_centered_touchdown_and_four_foot_support() -> None:
    env_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env.py").read_text(
        encoding="utf-8"
    )
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    assert "class DrobotPureStairsFirstStepContactRetentionHipEnvCfg" in cfg_source
    assert "first_step_min_support_count = 4" in cfg_source
    assert "first_step_require_centered_contact = True" in cfg_source
    assert "centered_tread_contact_reward_scale = 4.0" in cfg_source
    assert "descending_center_approach_reward_scale = 4.0" in cfg_source
    assert "retained_ground_support_reward_scale = 3.0" in cfg_source
    assert "three_other_supports" in env_source
    assert "descending_center_approach" in env_source
    assert "required_tread_contacts" in env_source
    assert "support_count >= self.cfg.first_step_min_support_count" in env_source
    assert "class DrobotPureStairsFirstStepContactRetentionHipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Contact-Retention-Hip-Direct" in registration


def test_broad_support_retention_is_a_single_constraint_bridge() -> None:
    env_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env.py").read_text(
        encoding="utf-8"
    )
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    broad = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepBroadSupportRetentionHipEnvCfg"
    )
    assert broad["first_step_min_support_count"] == 4
    assert broad["first_step_require_centered_contact"] is False
    assert broad["retained_ground_support_reward_scale"] == 3.0
    assert "required_tread_binary" in env_source
    assert "Drobot-Pure-Stairs-First-Step-Broad-Support-Retention-Hip-Direct" in registration


def test_width105_curriculum_narrows_only_the_valid_contact_band() -> None:
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    width105 = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepWidth105SupportRetentionHipEnvCfg"
    )
    assert width105["centered_tread_half_width_m"] == 0.105
    assert "DrobotPureStairsFirstStepContactRetentionHipEnvCfg" in cfg_source
    assert "DrobotPureStairsFirstStepWidth105SupportRetentionHipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Width105-Support-Retention-Hip-Direct" in registration


def test_width90_curriculum_continues_gradual_tread_narrowing() -> None:
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    width90 = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepWidth90SupportRetentionHipEnvCfg"
    )
    assert width90["centered_tread_half_width_m"] == 0.090
    assert "DrobotPureStairsFirstStepWidth90SupportRetentionHipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Width90-Support-Retention-Hip-Direct" in registration


def test_width105_body_rise_requires_contact_gated_height_transfer() -> None:
    env_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env.py").read_text(
        encoding="utf-8"
    )
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    rise10 = _class_assignments(cfg_source, "DrobotPureStairsFirstStepWidth105Rise10HipEnvCfg")
    assert rise10["first_step_min_base_gain_m"] == 0.01
    assert rise10["first_step_require_base_gain"] is True
    assert rise10["first_step_hold_steps"] == 4
    assert rise10["tread_transfer_reward_scale"] == 10.0
    assert rise10["narrow_transfer_reward_scale"] == 0.0
    assert "tread_transfer = required_tread_binary * base_gain_fraction" in env_source
    assert "tread_height_delta_scale * required_tread_binary * height_delta" in env_source
    assert "Drobot-Pure-Stairs-First-Step-Width105-Rise10-Hip-Direct" in registration


def test_width105_lower_reset_curriculum_reaches_verified_full_fold_gradually() -> None:
    cfg_source = (ROOT / "simulation/isaac/rl/parallel_stairs/pure_stairs_env_cfg.py").read_text(
        encoding="utf-8"
    )
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )

    expected = {
        "DrobotPureStairsFirstStepWidth105Low25HipEnvCfg": (0.42, 0.25),
        "DrobotPureStairsFirstStepWidth105Low50HipEnvCfg": (0.38, 0.50),
        "DrobotPureStairsFirstStepWidth105Low75HipEnvCfg": (0.34, 0.75),
        "DrobotPureStairsFirstStepWidth105Low100HipEnvCfg": (0.30, 1.0),
    }
    for class_name, (height, fraction) in expected.items():
        values = _class_assignments(cfg_source, class_name)
        assert values["initial_base_height_m"] == height
        assert values["reset_fold_fraction"] == fraction

    assert "_apply_folded_reset(self, self.reset_fold_fraction)" in cfg_source
    assert "LOW_FOLD_HIP_RAD - NORMAL_HIP_RAD" in cfg_source
    assert "LOW_FOLD_KNEE_RAD - NORMAL_KNEE_RAD" in cfg_source
    assert "DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg" in runner_source
    assert "self.algorithm.learning_rate = 5.0e-5" in runner_source
    for stage in ("Low25", "Low50", "Low75", "Low100"):
        assert f"Drobot-Pure-Stairs-First-Step-Width105-{stage}-Hip-Direct" in registration


def test_bounded_parallel_evaluator_disables_only_rgb_capture() -> None:
    source = _source("evaluate_pure_parallel_stairs.py")
    assert "gym.wrappers.RecordVideo = _bounded_without_rgb" in source
    assert "isaaclab_app.launch_simulation = _launch_headless_bounded" in source
    assert "entrypoint_common.create_isaaclab_env = _create_headless_bounded_env" in source
    assert "args_cli.video = False" in source
    assert "args_cli.enable_cameras = False" in source
    assert "OnPolicyRunner.export_policy_to_jit = _skip_policy_export" in source
    assert "OnPolicyRunner.export_policy_to_onnx = _skip_policy_export" in source
    assert "run_play_cli" in source
    assert "no video file is produced" in source.lower()


def test_two_mode_policy_preserves_pure_sensor_contract_and_commits_deterministically() -> None:
    distribution = _source("two_mode_gaussian.py")
    transplant = _source("transplant_two_mode_checkpoint.py")
    env_cfg = _source("pure_stairs_env_cfg.py")
    runner = _source("agents/rsl_rl_ppo_cfg.py")
    registration = _source("__init__.py")

    assert "self.num_modes + self.num_modes * self.output_dim" in distribution
    assert "torch.multinomial" in distribution
    assert "logsumexp" in distribution
    assert "logits.argmax" in distribution
    assert "means.gather" in distribution
    assert "selected leg" not in distribution.lower()
    assert "source_weight + direction" in transplant
    assert "source_weight - direction" in transplant
    assert "mode_average_max_weight_error" in transplant
    assert "DrobotTwoModeGaussianDistributionCfg" in runner
    assert "DrobotPureStairsYaw90FullFoldFootLift5TwoModeHipPPORunnerCfg" in runner
    assert "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-TwoMode-Hip-Direct" in registration

    forced_mode = _source("force_two_mode_checkpoint.py")
    assert 'actor["mlp.4.weight"][:2].zero_()' in forced_mode
    assert 'actor["mlp.4.bias"][:2].fill_(-20.0)' in forced_mode
    assert 'actor["mlp.4.bias"][args.mode] = 20.0' in forced_mode
    assert '"training_checkpoint": False' in forced_mode

    assert "DrobotPureStairsYaw90FullFoldFootLift5GruHipPPORunnerCfg" in runner
    assert "RslRlRNNModelCfg" in runner
    assert 'rnn_type="gru"' in runner
    assert "rnn_hidden_dim=128" in runner
    assert "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-GRU-Hip-Direct" in registration

    success_dominant = _class_assignments(
        env_cfg,
        "DrobotPureStairsYaw90FullFoldFootLift5SuccessDominantHipEnvCfg",
    )
    assert success_dominant["support_reward_scale"] == 0.0
    assert success_dominant["success_completion_reward_scale"] == 400.0
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-SuccessDominant-GRU-Hip-Direct"
        in registration
    )

    sensor_asym = _class_assignments(
        env_cfg,
        "DrobotPureStairsYaw90FullFoldFootLift5SensorAsymHipEnvCfg",
    )
    assert sensor_asym["reset_joint_position_noise_rad"] == 0.04
    assert sensor_asym["reset_lateral_jitter_m"] == 0.02
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-SensorAsym-GRU-Hip-Direct"
        in registration
    )

    cem_robust = _class_assignments(
        env_cfg,
        "DrobotPureStairsYaw90FullFoldFootLift5CemRobustHipEnvCfg",
    )
    assert cem_robust["reset_joint_position_noise_rad"] == 0.02
    assert cem_robust["reset_lateral_jitter_m"] == 0.015
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentBias-CEM-Robust-Hip-Direct"
        in registration
    )

    cem_robust_lift10 = _class_assignments(
        env_cfg,
        "DrobotPureStairsYaw90FullFoldFootLift10CemRobustHipEnvCfg",
    )
    assert cem_robust_lift10["foot_lift_height_m"] == 0.10
    assert cem_robust_lift10["foot_lift_hold_steps"] == 6
    assert cem_robust_lift10["support_reward_scale"] == 0.0
    assert cem_robust_lift10["foot_unload_reward_scale"] == 0.75
    assert cem_robust_lift10["unloaded_lift_reward_scale"] == 4.0
    assert cem_robust_lift10["success_completion_reward_scale"] == 400.0
    assert cem_robust_lift10["reset_joint_position_noise_rad"] == 0.02
    assert cem_robust_lift10["reset_lateral_jitter_m"] == 0.015
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift10-PersistentBias-CEM-Robust-Hip-Direct"
        in registration
    )

    cem_robust_lift7p5 = _class_assignments(
        env_cfg,
        "DrobotPureStairsYaw90FullFoldFootLift7p5CemRobustHipEnvCfg",
    )
    assert cem_robust_lift7p5["foot_lift_height_m"] == 0.075
    assert cem_robust_lift7p5["foot_lift_hold_steps"] == 5
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift7p5-PersistentBias-CEM-Robust-Hip-Direct"
        in registration
    )


def test_persistent_mode_is_sampled_once_and_replayed_by_ppo() -> None:
    policy = _source("persistent_mode_policy.py")
    runner = _source("agents/rsl_rl_ppo_cfg.py")
    registration = _source("__init__.py")

    assert "needs_mode = one_hot.sum(dim=-1) < 0.5" in policy
    assert "torch.multinomial" in policy
    assert "self._commitment_state[:, dones == 1, :] = 0.0" in policy
    assert "padded_new_mode[0]" in policy
    assert "gaussian + self._new_mode.to(gaussian.dtype) * selected_logit" in policy
    assert "self.transition.actions = self.actor(obs, stochastic_output=True)" in policy
    assert "self.transition.hidden_states = (" in policy
    assert "selected leg" not in policy.lower()
    assert "DrobotPersistentModeGaussianDistributionCfg" in runner
    assert "DrobotPureStairsYaw90FullFoldFootLift5PersistentModeHipPPORunnerCfg" in runner
    assert "PersistentModePPO" in runner
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentMode-Hip-Direct"
        in registration
    )

    transplant = _source("transplant_persistent_bias_checkpoint.py")
    assert "PersistentBiasGaussianDistribution" in policy
    assert "PersistentBiasActor" in policy
    assert "_TorchPersistentBiasActor" in policy
    assert "def as_jit(self) -> nn.Module:" in policy
    assert "proposed_bias = torch.normal" in policy
    assert "episode_bias = torch.where" in policy
    assert "bias_log_prob + categorical_log_prob" in policy
    assert "DrobotPersistentBiasGaussianDistributionCfg" in runner
    assert "DrobotPureStairsYaw90FullFoldFootLift5PersistentBiasHipPPORunnerCfg" in runner
    assert "PersistentBiasPPO" in runner
    assert "commitment_credit_scale" in policy
    assert "DrobotPersistentBiasConsolidateDistributionCfg" in runner
    assert "commitment_credit_scale: float = 32.0" in runner
    assert "DrobotPersistentBiasCemRobustDistributionCfg" in runner
    assert "init_action_std: float = 0.03" in runner
    assert "init_bias_std: float = 0.05" in runner
    assert "DrobotPureStairsYaw90FullFoldFootLift5PersistentBiasCemRobustHipPPORunnerCfg" in runner
    assert "DrobotPureStairsYaw90FullFoldFootLift10PersistentBiasCemRobustHipPPORunnerCfg" in runner
    assert (
        "DrobotPureStairsYaw90FullFoldFootLift7p5PersistentBiasCemRobustHipPPORunnerCfg"
        in runner
    )
    assert 'infos.get("persistent_bias_transplant") or from_cem' in policy
    assert "CEM_ACTION_STD_MAX = 0.03" in policy
    assert "CEM_BIAS_STD_MAX = 0.05" in policy
    assert (
        "DrobotPureStairsYaw90FullFoldFootLift5PersistentBiasConsolidateHipPPORunnerCfg"
        in runner
    )
    assert "widened_weight = output_weight.new_zeros(50" in transplant
    assert 'actor["distribution.action_std_param"]' in transplant
    assert 'actor["distribution.bias_std_param"]' in transplant
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentBias-Hip-Direct"
        in registration
    )
    assert (
        "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-PersistentBias-Consolidate-Hip-Direct"
        in registration
    )


def test_low25_consolidation_uses_long_low_entropy_batches() -> None:
    runner_source = (
        ROOT / "simulation/isaac/rl/parallel_stairs/agents/rsl_rl_ppo_cfg.py"
    ).read_text(encoding="utf-8")
    registration = (ROOT / "simulation/isaac/rl/parallel_stairs/__init__.py").read_text(
        encoding="utf-8"
    )
    assert "DrobotPureStairsFirstStepWidth105Low25ConsolidateHipPPORunnerCfg" in runner_source
    assert "num_steps_per_env = 64" in runner_source
    assert "self.algorithm.entropy_coef = 0.0" in runner_source
    assert "self.algorithm.learning_rate = 2.0e-5" in runner_source

    assert "self.algorithm.num_learning_epochs = 10" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Width105-Low25-Consolidate-Hip-Direct" in registration


def test_low25_to_37_bridge_randomizes_only_hardware_representable_reset_state() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    env_source = _source("pure_stairs_env.py")
    runner_source = _source("agents/rsl_rl_ppo_cfg.py")
    registration = _source("__init__.py")

    bridge = _class_assignments(cfg_source, "DrobotPureStairsFirstStepWidth105Low25To37HipEnvCfg")
    assert bridge["initial_base_height_m"] == 0.40
    assert bridge["reset_fold_fraction"] == 0.375
    assert bridge["reset_fold_fraction_min"] == 0.25
    assert bridge["reset_fold_fraction_max"] == 0.375
    assert bridge["reset_base_height_min_m"] == 0.40
    assert bridge["reset_base_height_max_m"] == 0.42
    assert "reset_fraction = fold_min + reset_alpha * (fold_max - fold_min)" in env_source
    assert "height_max\n                + reset_alpha * (height_min - height_max)" in env_source
    observation = _method_source(env_source, "DrobotPureStairsEnv", "_get_observations")
    assert "self._reset_fold_fraction" not in observation
    assert "hard_reset = self._reset_fold_fraction[env_ids] >= reset_midpoint" in env_source
    assert "hard_completed={hard_completed} hard_successful={hard_successful}" in env_source
    assert "DrobotPureStairsFirstStepWidth105Low25To37HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Width105-Low25-To37-Hip-Direct" in registration

    fixed = _class_assignments(cfg_source, "DrobotPureStairsFirstStepWidth105Low37HipEnvCfg")
    assert fixed["initial_base_height_m"] == 0.40
    assert fixed["reset_fold_fraction"] == 0.375
    assert "DrobotPureStairsFirstStepWidth105Low37HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Width105-Low37-Hip-Direct" in registration

    hard_bias = _class_assignments(
        cfg_source, "DrobotPureStairsFirstStepWidth105Low25To37HardBiasHipEnvCfg"
    )
    assert hard_bias["reset_alpha_power"] == 0.5
    assert "torch.rand(len(env_ids), device=self.device).pow(" in env_source
    assert "self.cfg.reset_alpha_power" in env_source
    assert "DrobotPureStairsFirstStepWidth105Low25To37HardBiasHipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-First-Step-Width105-Low25-To37-HardBias-Hip-Direct" in registration


def test_hard_bias_rise10_rewards_only_four_support_body_transfer() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    env_source = _source("pure_stairs_env.py")
    runner_source = _source("agents/rsl_rl_ppo_cfg.py")
    registration = _source("__init__.py")

    rise10 = _class_assignments(
        cfg_source,
        "DrobotPureStairsFirstStepWidth105Low25To37HardBiasRise10HipEnvCfg",
    )
    assert rise10["first_step_min_base_gain_m"] == 0.01
    assert rise10["first_step_require_base_gain"] is True
    assert rise10["first_step_hold_steps"] == 4
    assert rise10["height_delta_reward_scale"] == 0.0
    assert rise10["tread_transfer_reward_scale"] == 0.0
    assert rise10["supported_transfer_reward_scale"] == 12.0
    assert rise10["supported_tread_height_delta_scale"] == 220.0
    assert "full_support = torch.clamp(support_count - 3.0, 0.0, 1.0)" in env_source
    assert "supported_transfer_gate = required_tread_binary * full_support" in env_source
    assert "Metrics/max_supported_base_gain_m" in env_source
    assert "supported_gain_episodes={supported_gain_episodes}" in env_source
    assert "supported_gain_mean_m={supported_gain_mean_m:.8f}" in env_source
    assert (
        "DrobotPureStairsFirstStepWidth105Low25To37HardBiasRise10HipPPORunnerCfg" in runner_source
    )
    assert (
        "Drobot-Pure-Stairs-First-Step-Width105-Low25-To37-HardBias-Rise10-Hip-Direct"
        in registration
    )
    assert "DrobotPureStairsYaw90FullFoldFootLift5ConsolidateHipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Yaw90-FullFold-Foot-Lift5-Consolidate-Hip-Direct" in registration
    assert "self.algorithm.entropy_coef = 0.0" in runner_source
    assert "self.algorithm.learning_rate = 2.0e-5" in runner_source


def test_hard_bias_stand_precursor_is_pure_four_support_height_rl() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    env_source = _source("pure_stairs_env.py")
    runner_source = _source("agents/rsl_rl_ppo_cfg.py")
    registration = _source("__init__.py")

    stand = _class_assignments(cfg_source, "DrobotPureStairsLow25To37HardBiasStandRise10HipEnvCfg")
    assert stand["first_step_curriculum"] is False
    assert stand["support_rise_curriculum"] is True
    assert stand["support_rise_min_base_gain_m"] == 0.01
    assert stand["support_rise_hold_steps"] == 4
    assert stand["clearance_reward_scale"] == 0.0
    assert stand["support_rise_min_support_count"] == 4
    assert stand["support_rise_reward_scale"] == 10.0
    assert stand["support_rise_height_delta_scale"] == 180.0
    assert stand["tread_contact_reward_scale"] == 0.0
    assert "support_rise = support_rise_reward_gate * support_rise_gain_fraction" in env_source
    assert "Metrics/max_four_support_base_gain_m" in env_source
    assert "self.cfg.support_rise_curriculum" in env_source
    assert "DrobotPureStairsLow25To37HardBiasStandRise10HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Low25-To37-HardBias-Stand-Rise10-Hip-Direct" in registration

    three_support = _class_assignments(
        cfg_source,
        "DrobotPureStairsLow25To37HardBiasThreeSupportRise10HipEnvCfg",
    )
    assert three_support["support_rise_min_support_count"] == 3
    assert "DrobotPureStairsLow25To37HardBiasThreeSupportRise10HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Low25-To37-HardBias-ThreeSupport-Rise10-Hip-Direct" in registration

    two_support = _class_assignments(
        cfg_source,
        "DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipEnvCfg",
    )
    assert two_support["support_rise_min_base_gain_m"] == 0.005
    assert two_support["support_rise_min_support_count"] == 2
    assert two_support["support_rise_soft_support_reward"] is True
    assert two_support["support_reward_scale"] == 0.25
    assert "DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Low25-To37-HardBias-TwoSupport-Rise5-Hip-Direct" in registration
    assert (
        "DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipConsolidatePPORunnerCfg"
        in runner_source
    )
    assert (
        "Drobot-Pure-Stairs-Low25-To37-HardBias-TwoSupport-Rise5-Hip-Consolidate-Direct"
        in registration
    )
    assert "self.algorithm.entropy_coef = 0.0" in runner_source
    assert "self.algorithm.learning_rate = 2.0e-5" in runner_source

    full_fold = _class_assignments(
        cfg_source,
        "DrobotPureStairsFullFoldTwoSupportRise5HipEnvCfg",
    )
    assert full_fold["initial_base_height_m"] == 0.30
    assert full_fold["reset_fold_fraction"] == 1.0
    assert full_fold["reset_fold_fraction_min"] is None
    assert full_fold["reset_fold_fraction_max"] is None
    assert full_fold["support_rise_settle_steps"] == 60
    assert full_fold["support_rise_hold_steps"] == 12
    assert "self._steps_since_reset > self.cfg.support_rise_settle_steps" in env_source
    assert "support_rise_reward_gate * support_rise_active" in env_source
    assert "DrobotPureStairsFullFoldTwoSupportRise5HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-FullFold-TwoSupport-Rise5-Hip-Direct" in registration

    yaw45 = _class_assignments(
        cfg_source,
        "DrobotPureStairsYaw45FullFoldTwoSupportRise5HipEnvCfg",
    )
    assert yaw45["reset_yaw_deg"] == 45.0
    assert yaw45["action_scale_abduction_rad"] == 0.42
    assert "self.robot.init_state.rot = (0.0, 0.0, 0.3826834324, 0.9238795325)" in cfg_source
    assert "self.depth_sensor.offset.pos = (0.0809637264, -0.0809637264, 0.123)" in cfg_source
    assert "DrobotPureStairsYaw45FullFoldTwoSupportRise5HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Yaw45-FullFold-TwoSupport-Rise5-Hip-Direct" in registration

    yaw67p5 = _class_assignments(
        cfg_source,
        "DrobotPureStairsYaw67p5FullFoldTwoSupportRise5HipEnvCfg",
    )
    assert yaw67p5["reset_yaw_deg"] == 67.5
    assert yaw67p5["action_scale_abduction_rad"] == 0.42
    assert "self.robot.init_state.rot = (0.0, 0.0, 0.5555702330, 0.8314696123)" in cfg_source
    assert "self.depth_sensor.offset.pos = (0.0438172530, -0.1057842065, 0.123)" in cfg_source
    assert "DrobotPureStairsYaw67p5FullFoldTwoSupportRise5HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Yaw67p5-FullFold-TwoSupport-Rise5-Hip-Direct" in registration

    yaw90 = _class_assignments(
        cfg_source,
        "DrobotPureStairsYaw90FullFoldTwoSupportRise5HipEnvCfg",
    )
    assert yaw90["reset_yaw_deg"] == 90.0
    assert yaw90["action_scale_abduction_rad"] == 0.42
    assert "DrobotPureStairsYaw90FullFoldTwoSupportRise5HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Yaw90-FullFold-TwoSupport-Rise5-Hip-Direct" in registration

    sideways = _class_assignments(
        cfg_source,
        "DrobotPureStairsSidewaysTwoSupportRise5HipEnvCfg",
    )
    assert sideways["reset_yaw_deg"] == 90.0
    assert sideways["action_scale_abduction_rad"] == 0.42
    assert "self.depth_sensor.offset.pos = (0.0, -0.1145, 0.123)" in cfg_source
    assert "DrobotPureStairsSidewaysTwoSupportRise5HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Sideways-TwoSupport-Rise5-Hip-Direct" in registration

    upright = _class_assignments(
        cfg_source,
        "DrobotPureStairsLow25To37HardBiasUprightRise10HipEnvCfg",
    )
    assert upright["support_rise_min_support_count"] == 0
    assert upright["support_reward_scale"] == 0.0
    assert "strict_support_rise_gate = torch.ones_like(support_count)" in env_source
    assert "support_count / float(self.cfg.support_rise_min_support_count)" in env_source
    assert ") & ~self._failed" in env_source
    assert "max(self.cfg.first_step_min_base_gain_m, 1.0e-6)" in env_source
    assert "DrobotPureStairsLow25To37HardBiasUprightRise10HipPPORunnerCfg" in runner_source
    assert "Drobot-Pure-Stairs-Low25-To37-HardBias-Upright-Rise10-Hip-Direct" in registration


def test_episode_bias_cem_is_reward_only_and_bakes_deployable_centers() -> None:
    source = _source("optimize_episode_bias_cem.py")
    assert "multi_population_diagonal_cem" in source
    assert "winner_centered_multi_population_diagonal_cem" in source
    assert "episode_returns" in source
    assert "successes" in source
    assert "policy(obs) + candidates" in source
    assert '"selected_leg_input": False' in source
    assert '"reference_motion": False' in source
    assert "center_start = 2 + 2 * 12" in source
    assert "output_bias[start : start + 12] += bias" in source
    assert 'parser.add_argument("--num_envs", type=int, default=128)' in source
    assert '"--replicas_per_candidate"' in source
    assert '"--minimum_success_replicas"' in source
    assert "replicated_candidate_grid = candidate_grid[:, :, None, :].expand" in source
    assert "replica_return_grid.mean(dim=-1)" in source
    assert "replica_success_grid.float().mean(dim=-1)" in source
    assert "success_count_grid >= args_cli.minimum_success_replicas" in source
    assert "success_fraction_grid * 1.0e6" in source
    assert '"replicas_per_candidate": args_cli.replicas_per_candidate' in source
    assert '"--initial_report"' in source
    assert '"--winner_centered"' in source


def test_parallel_play_can_follow_a_reproducible_success_environment() -> None:
    source = _source("play_pure_parallel_stairs.py")
    assert 'flag = "--viewer_env_index"' in source
    assert 'HIDE_OTHER_ROBOTS = _consume_flag("--hide_other_robots")' in source
    assert "UsdGeom.Imageable(prim).MakeInvisible()" in source
    assert "[DROBOT_ENV_SPACING_AUDIT]" in source
    assert "min_origin_xy_m=" in source
    assert "min_root_xy_m=" in source
    assert "replicate_physics=" in source
    assert 'env_cfg.viewer.origin_type = "world"' in source
    assert "origin = origins[VIEWER_ENV_INDEX]" in source
    assert "set_kit_renderer_camera_view(" in source
    assert "[DROBOT_RECORDED_SUCCESS]" in source


def test_manual_stair_rl_workflow_separates_visual_and_headless_modes() -> None:
    source = _source("run_stair_rl_workflow.ps1")
    assert '[ValidateSet("test", "train-visible", "train-headless")]' in source
    assert '"--num_envs 1"' not in source
    assert "--viewer_env_index 0" in source
    assert "--hide_other_robots" in source
    assert '$visibleEnvs = if ($NumEnvs -gt 0) { $NumEnvs } else { 5 }' in source
    assert '$headlessEnvs = if ($NumEnvs -gt 0) { $NumEnvs } else { 128 }' in source
    assert '-Visualizer "kit"' in source
    assert '-Visualizer "none"' in source
    assert 'agent.algorithm.num_mini_batches=1' in source
    assert '-PreferWorkflowRun' in source
    assert '"--resume"' in source
    assert '"--max_iterations"' in source
    assert (
        "Drobot-Pure-Stairs-Yaw90-Neutral-Foot-Lift7p5-"
        "PersistentBias-CEM-Robust-Hip-Direct"
    ) in source
    assert "drobot_pure_stairs_yaw90_neutral_foot_lift7p5" in source


def test_manual_stair_rl_workflow_uses_a_neutral_non_crossed_reset() -> None:
    cfg_source = _source("pure_stairs_env_cfg.py")
    runner_source = _source("agents/rsl_rl_ppo_cfg.py")
    registration = _source("__init__.py")
    neutral = _class_assignments(
        cfg_source,
        "DrobotPureStairsYaw90NeutralFootLift7p5CemRobustHipEnvCfg",
    )

    assert neutral["initial_base_height_m"] == 0.46
    assert neutral["reset_fold_fraction"] == 0.0
    assert neutral["reset_fold_fraction_min"] is None
    assert neutral["reset_fold_fraction_max"] is None
    assert neutral["reset_base_height_min_m"] is None
    assert neutral["reset_base_height_max_m"] is None
    assert "_apply_separated_neutral_reset(self)" in cfg_source
    assert '"front_left_hip_flexion": -SEPARATED_NEUTRAL_HIP_RAD' in cfg_source
    assert '"rear_left_hip_flexion": SEPARATED_NEUTRAL_HIP_RAD' in cfg_source
    assert '"front_right_hip_flexion": SEPARATED_NEUTRAL_HIP_RAD' in cfg_source
    assert '"rear_right_hip_flexion": -SEPARATED_NEUTRAL_HIP_RAD' in cfg_source
    assert '"front_left_knee": -SEPARATED_NEUTRAL_KNEE_RAD' in cfg_source
    assert '"rear_left_knee": SEPARATED_NEUTRAL_KNEE_RAD' in cfg_source
    assert '"front_right_knee": SEPARATED_NEUTRAL_KNEE_RAD' in cfg_source
    assert '"rear_right_knee": -SEPARATED_NEUTRAL_KNEE_RAD' in cfg_source
    assert (
        "DrobotPureStairsYaw90NeutralFootLift7p5PersistentBiasCemRobustHipPPORunnerCfg"
        in runner_source
    )
    assert (
        "Drobot-Pure-Stairs-Yaw90-Neutral-Foot-Lift7p5-"
        "PersistentBias-CEM-Robust-Hip-Direct"
    ) in registration
