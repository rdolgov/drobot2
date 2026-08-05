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
    assert "pos=(0.0, 0.0, 0.3305)" in cfg
    assert '"front_.*_hip_flexion": STABLE_NEUTRAL_HIP_RAD' in cfg
    assert '"rear_.*_hip_flexion": -STABLE_NEUTRAL_HIP_RAD' in cfg
    assert '"front_.*_knee": STABLE_NEUTRAL_KNEE_RAD' in cfg
    assert '"rear_.*_knee": -STABLE_NEUTRAL_KNEE_RAD' in cfg


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
    assert '[int]$NumEnvs = 5' in visible
    assert '[int]$NumEnvs = 128' in headless
    assert "Find-LatestWalkingCheckpoint" in common
    assert '"--resume"' in common
    assert '-Visualizer "none"' in headless
