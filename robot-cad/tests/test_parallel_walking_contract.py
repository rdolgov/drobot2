"""Static contracts for the Isaac-only parallel walking workflow."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WALKING = ROOT / "simulation" / "isaac" / "rl" / "parallel_walking"


def _source(name: str) -> str:
    return (WALKING / name).read_text(encoding="utf-8")


def test_python_entrypoints_and_environment_parse() -> None:
    for relative in (
        "__init__.py",
        "commanded_walking_env_cfg.py",
        "commanded_walking_env.py",
        "train_commanded_walking.py",
        "play_commanded_walking.py",
        "preview_control.py",
        "bootstrap_walking_from_sb3.py",
        "evaluate_sustained_walking.py",
        "agents/rsl_rl_ppo_cfg.py",
    ):
        ast.parse(_source(relative), filename=relative)


def test_policy_contract_reserves_three_commands_and_uses_real_imu() -> None:
    cfg = _source("commanded_walking_env_cfg.py")
    env = _source("commanded_walking_env.py")
    play = _source("play_commanded_walking.py")
    assert "observation_space = 48" in cfg
    assert "self._commands," in env
    assert "self._imu_sensor.data.ang_vel_b.torch" in env
    assert "self._imu_sensor.data.lin_acc_b.torch / 9.81" in env
    assert "self._robot.data.projected_gravity_b.torch" in env
    assert "depth" not in env.lower()
    assert "from parallel_walking import preview_control" in play
    assert "commanded_walking_env import" not in play


def test_neutral_reset_is_symmetric_and_grounded() -> None:
    cfg = _source("commanded_walking_env_cfg.py")
    assert "decimation = 2" in cfg
    assert "pos=(0.0, 0.0, 0.3730)" in cfg
    assert '"front_.*_hip_flexion": STABLE_NEUTRAL_FRONT_HIP_RAD' in cfg
    assert '"rear_.*_hip_flexion": STABLE_NEUTRAL_REAR_HIP_RAD' in cfg
    assert '"front_.*_knee": STABLE_NEUTRAL_FRONT_KNEE_RAD' in cfg
    assert '"rear_.*_knee": STABLE_NEUTRAL_REAR_KNEE_RAD' in cfg
    assert "ImplicitActuatorCfg" in cfg
    assert "IdealPDActuatorCfg" not in cfg
    assert "velocity_limit_sim=SERVO_VELOCITY_LIMIT_RAD_S" in cfg


def test_forward_and_directional_tasks_share_policy_shape() -> None:
    registration = _source("__init__.py")
    cfg = _source("commanded_walking_env_cfg.py")
    agent = _source("agents/rsl_rl_ppo_cfg.py")
    assert "Drobot-Commanded-Walk-Forward-Direct" in registration
    assert "Drobot-Commanded-Walk-Directional-Direct" in registration
    assert 'command_profile = "forward"' in cfg
    assert 'command_profile = "directional"' in cfg
    assert "DrobotCommandedWalkingDirectionalPPORunnerCfg" in agent
    assert "hidden_dims=[256, 256]" in agent


def test_v16_policy_is_bounded_and_keeps_the_transfer_shape() -> None:
    agent = _source("agents/rsl_rl_ppo_cfg.py")
    common = _source("walking_workflow_common.ps1")
    assert "DrobotBoundedBetaDistributionCfg" in agent
    assert 'class_name: str = "BetaDistribution"' in agent
    assert "action_range: tuple[float, float] = (-1.0, 1.0)" in agent
    assert "DrobotGaussianDistributionCfg" not in agent
    assert "RslRlRNNModelCfg" not in agent
    assert 'activation="elu"' in agent
    assert 'obs_groups = {"actor": ["policy"], "critic": ["critic"]}' in agent
    assert "entropy_coef=0.001" in agent
    assert "value_loss_coef=0.5" in agent
    assert "max_grad_norm=0.5" in agent
    assert (
        'experiment_name = "drobot_commanded_walk_forward_v16_sustained_beta_direct"'
        in agent
    )
    assert "gamma=0.995" in agent
    assert "obs_normalization=False" in agent
    assert 'drobot_commanded_walk_${suffix}_v16_sustained_beta_direct' in common


def test_rl_transfer_bootstrap_maps_actor_and_resets_optimizer() -> None:
    bootstrap = _source("bootstrap_walking_from_sb3.py")
    assert 'source_policy["mlp_extractor.policy_net.0.weight"]' in bootstrap
    assert 'source_policy["mlp_extractor.policy_net.2.weight"]' in bootstrap
    assert 'source_policy["action_net.weight"]' in bootstrap
    assert '"--rsl-source"' in bootstrap
    assert "target_weight.shape[0] == 24" in bootstrap
    assert 'target_distribution = "bounded_beta"' in bootstrap
    assert 'actor["distribution.log_std_param"]' in bootstrap
    assert 'optimizer["state"] = {}' in bootstrap
    assert '"--actor-output-scale"' in bootstrap


def test_v16_reward_requires_sustained_forward_motion() -> None:
    cfg = _source("commanded_walking_env_cfg.py")
    env = _source("commanded_walking_env.py")
    assert "initial_forward_speed_min_m_s = 0.15" in cfg
    assert "initial_forward_speed_max_m_s = 0.15" in cfg
    assert "command_curriculum_steps = 1" in cfg
    assert "command_curriculum_offset_steps = 0" in cfg
    assert "minimum_base_height_m = 0.22" in cfg
    assert "minimum_upright_cosine = 0.78" in cfg
    assert "velocity_tracking_sigma_m_s = 0.10" in cfg
    assert "reward_signed_command_progress" not in cfg
    assert "reward_forward_velocity_tracking = 2.0" in cfg
    assert "episode_length_s = 32.0" in cfg
    assert "initial_training_horizon_s = 8.0" in cfg
    assert "final_training_horizon_s = 32.0" in cfg
    assert "episode_horizon_curriculum_steps = 64_000" in cfg
    assert "sustained_speed_window_s = 2.0" in cfg
    assert "minimum_sustained_speed_m_s = 0.04" in cfg
    assert "reward_sustained_progress = 0.75" in cfg
    assert "penalty_sustained_stall = 0.50" in cfg
    assert "penalty_normalized_forward_velocity_error" not in cfg
    assert "reward_body_height_tracking" not in cfg
    assert "reward_upright = 0.50" in cfg
    assert "reward_alive = 0.50" in cfg
    assert "penalty_termination = 100.0" in cfg
    assert "reward_scale = 0.10" in cfg
    assert "velocity_tracking = torch.exp(" in env
    assert "net_commanded_distance - self._episode_commanded_distance" not in env
    assert '"signed_command_progress"' not in env
    assert '"failed_progress_revoke"' not in env
    assert "self._episode_positive_progress_reward_sum" not in env
    assert '"forward_velocity_error"' not in env
    assert '"body_height_tracking"' not in env
    assert '"forward_velocity_tracking"' in env
    assert '"sustained_progress"' in env
    assert '"sustained_stall"' in env
    assert '"lateral_velocity"' in env
    assert '"termination"' in env
    assert "* self.cfg.reward_scale" in env
    assert "newly_reached_milestones" not in env
    assert "touchdown_quality" not in env
    assert "stationary" not in env
    assert "symmetric_swing" in env
    assert 'log["Metrics/mean_commanded_speed_m_s"]' in env
    assert 'log["Metrics/net_forward_displacement_m"]' in env
    assert 'log["Metrics/mean_base_height_m"]' in env
    assert 'log["Metrics/min_rolling_forward_speed_m_s"]' in env
    assert 'log["Metrics/sustained_stall_rate"]' in env
    assert 'log["Metrics/current_episode_horizon_s"]' in env
    assert 'log["Metrics/distance_success_rate"]' in env
    assert 'log[f"Metrics/failure_{label}_rate"]' in env
    assert 'log["Metrics/action_saturation_rate"]' in env
    assert 'log["Metrics/swing_step_rate"]' in env
    assert 'log["Metrics/touchdowns_per_episode"]' in env
    assert 'log[f"Reward/{name}"]' in env
    assert "if self.cfg.disable_time_limit:" in env
    assert "time_out = torch.zeros_like(self._failed)" in env
    assert "self._current_episode_horizon_steps() - 1" in env


def test_actor_observation_is_deployable_and_critic_gets_simulation_state() -> None:
    cfg = _source("commanded_walking_env_cfg.py")
    env = _source("commanded_walking_env.py")
    assert "state_space = 56" in cfg
    assert '"policy": torch.clamp(policy_observation' in env
    assert '"critic": torch.clamp(critic_observation' in env
    assert "self._robot.data.root_lin_vel_b.torch" in env
    assert "self._imu_sensor.data.ang_vel_b.torch," in env
    assert "ang_vel_b.torch * 0.25" not in env
    assert "foot_contact" in env


def test_directional_distribution_contains_all_requested_motion_classes() -> None:
    env = _source("commanded_walking_env.py")
    assert "50% forward, 15% backward, 15% left, 15% right, 5% stop" in env
    assert "self._commands[env_ids[backward], 0] = -(" in env
    assert "self._commands[env_ids[left], 2] = turn_rate[left]" in env
    assert "self._commands[env_ids[right], 2] = -turn_rate[right]" in env


def test_user_scripts_are_separate_and_resume_by_default() -> None:
    preview = _source("preview_walking.ps1")
    visible = _source("train_walking_visible.ps1")
    headless = _source("train_walking_headless.ps1")
    common = _source("walking_workflow_common.ps1")
    assert 'ValidateSet("forward", "backward", "left", "right", "stop")' in preview
    assert '$RecordSeconds * 60' in preview
    assert "[switch]$NoTimeLimit" in preview
    assert 'env.disable_time_limit=true' in preview
    assert '[int]$NumEnvs = 5' in visible
    assert '[int]$NumEnvs = 128' in headless
    assert "Find-LatestWalkingCheckpoint" in common
    assert 'Where-Object { $_.Name -notlike "_*" }' in common
    assert r"models\parallel-walking-v16\model_250.pt" in common
    assert "env.command_curriculum_offset_steps=$curriculumOffsetSteps" in common
    assert "([int64]$Matches[1] + 1) * 64" in common
    assert '"--resume"' in common
    assert '-Visualizer "none"' in headless
