"""Evaluate whether a walking checkpoint sustains motion beyond reset transients."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seconds", type=int, default=30)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--window-seconds", type=int, default=5)
parser.add_argument(
    "--task", default="Drobot-Commanded-Walk-Forward-Direct"
)
parser.add_argument("--forward-speed", type=float, default=0.15)
args = parser.parse_args()
if args.seconds < 10 or args.episodes < 1:
    parser.error("evaluation needs at least 10 seconds and one episode")
if not 1 <= args.window_seconds <= args.seconds:
    parser.error("--window-seconds must be within the evaluation duration")

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from importlib import metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

package_parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(package_parent))

import parallel_walking  # noqa: E402, F401
from parallel_walking import preview_control  # noqa: E402
from parallel_walking.commanded_walking_env import (  # noqa: E402
    LEG_NAMES,
    _yaw_from_wxyz,
)

task = args.task
env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
env_cfg.scene.num_envs = 1
env_cfg.seed = 4401
env_cfg.reset_joint_position_noise_rad = 0.0
env_cfg.reset_xy_jitter_m = 0.0
env_cfg.disable_time_limit = True
preview_control.COMMAND_OVERRIDE = (args.forward_speed, 0.0, 0.0)

env = gym.make(task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
checkpoint = str(Path(args.checkpoint).resolve())
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

results: list[dict[str, float | bool | list[float]]] = []
for _ in range(args.episodes):
    env.unwrapped._reset_idx(torch.tensor([0], device=env.unwrapped.device))
    env.unwrapped.episode_length_buf.zero_()
    obs = env.get_observations()
    start_x = float(env.unwrapped._robot.data.root_pos_w.torch[0, 0].item())
    start_y = float(env.unwrapped._robot.data.root_pos_w.torch[0, 1].item())
    start_yaw = float(
        _yaw_from_wxyz(env.unwrapped._robot.data.root_quat_w.torch)[0].item()
    )
    positions = [start_x]
    lateral_positions = [start_y]
    saturated_action_sum = 0.0
    action_rate_sum = 0.0
    action_acceleration_sum = 0.0
    joint_acceleration_squared_sum = 0.0
    body_linear_acceleration_squared_sum = 0.0
    body_angular_acceleration_squared_sum = 0.0
    target_limiter_gap_sum = 0.0
    three_foot_support_sum = 0.0
    excess_airborne_sum = 0.0
    absolute_yaw_travel_rad = 0.0
    previous_action = torch.zeros((12,), device=env.unwrapped.device)
    older_action = torch.zeros((12,), device=env.unwrapped.device)
    previous_joint_velocity = env.unwrapped._robot.data.joint_vel.torch[0].clone()
    previous_body_linear_velocity = (
        env.unwrapped._robot.data.root_lin_vel_w.torch[0].clone()
    )
    previous_body_angular_velocity = (
        env.unwrapped._robot.data.root_ang_vel_w.torch[0].clone()
    )
    foot_names = [
        next(leg for leg in LEG_NAMES if leg in name)
        for name in env.unwrapped._foot_sensor_names
    ]
    previous_contact = env.unwrapped._foot_forces()[0] > 1.0
    touchdown_counts = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_contact_matches = torch.zeros((4,), device=env.unwrapped.device)
    fell = False
    completed_steps = 0
    for step in range(args.seconds * 60):
        with torch.inference_mode():
            actions = policy(obs)
        # Keep environment state as ordinary tensors so another episode can
        # reset it in-place after deterministic inference completes.
        actions = actions.clone()
        saturated_action_sum += float(
            torch.mean((torch.abs(actions[0]) >= 0.98).float()).item()
        )
        action_rate_sum += float(torch.mean(torch.abs(actions[0] - previous_action)).item())
        action_acceleration_sum += float(
            torch.mean(torch.abs(actions[0] - 2.0 * previous_action + older_action)).item()
        )
        older_action.copy_(previous_action)
        previous_action.copy_(actions[0])
        obs, _, dones, _ = env.step(actions)
        joint_velocity = env.unwrapped._robot.data.joint_vel.torch[0]
        body_linear_velocity = env.unwrapped._robot.data.root_lin_vel_w.torch[0]
        body_angular_velocity = env.unwrapped._robot.data.root_ang_vel_w.torch[0]
        target_limiter_gap_sum += float(
            torch.mean(
                torch.abs(
                    env.unwrapped._desired_targets[0]
                    - env.unwrapped._processed_actions[0]
                )
            ).item()
        )
        joint_acceleration_squared_sum += float(
            torch.mean(torch.square((joint_velocity - previous_joint_velocity) * 60.0)).item()
        )
        body_linear_acceleration_squared_sum += float(
            torch.mean(
                torch.square((body_linear_velocity - previous_body_linear_velocity) * 60.0)
            ).item()
        )
        body_angular_acceleration_squared_sum += float(
            torch.mean(
                torch.square((body_angular_velocity - previous_body_angular_velocity) * 60.0)
            ).item()
        )
        previous_joint_velocity.copy_(joint_velocity)
        previous_body_linear_velocity.copy_(body_linear_velocity)
        previous_body_angular_velocity.copy_(body_angular_velocity)
        foot_contact = env.unwrapped._foot_forces()[0] > 1.0
        support_count = int(torch.count_nonzero(foot_contact).item())
        three_foot_support_sum += float(support_count >= 3)
        excess_airborne_sum += float(support_count <= 2)
        absolute_yaw_travel_rad += (
            abs(float(body_angular_velocity[2].item())) / 60.0
        )
        touchdown_counts += (foot_contact & ~previous_contact).float()
        previous_contact.copy_(foot_contact)
        _, scheduled_contact = env.unwrapped._gait_targets()
        scheduled_contact_matches += (foot_contact == scheduled_contact[0]).float()
        policy.reset(dones)
        completed_steps = step + 1
        if bool(dones[0].item()):
            fell = True
            break
        if completed_steps % 60 == 0:
            positions.append(
                float(env.unwrapped._robot.data.root_pos_w.torch[0, 0].item())
            )
            lateral_positions.append(
                float(env.unwrapped._robot.data.root_pos_w.torch[0, 1].item())
            )

    duration_s = completed_steps / 60.0
    distance_m = positions[-1] - positions[0]
    final_yaw = float(
        _yaw_from_wxyz(env.unwrapped._robot.data.root_quat_w.torch)[0].item()
    )
    heading_error_rad = math.atan2(
        math.sin(final_yaw - start_yaw), math.cos(final_yaw - start_yaw)
    )
    window_speeds = [
        (positions[index] - positions[index - args.window_seconds])
        / args.window_seconds
        for index in range(args.window_seconds, len(positions))
    ]
    stall_speed_threshold_m_s = max(0.001, 0.4 * args.forward_speed)
    results.append(
        {
            "duration_s": duration_s,
            "distance_m": distance_m,
            "lateral_displacement_m": lateral_positions[-1] - lateral_positions[0],
            "final_heading_error_rad": heading_error_rad,
            "mean_speed_m_s": distance_m / max(duration_s, 1.0 / 60.0),
            "minimum_window_speed_m_s": min(window_speeds, default=0.0),
            "final_window_speed_m_s": window_speeds[-1] if window_speeds else 0.0,
            "stall_window_fraction": (
                sum(speed < stall_speed_threshold_m_s for speed in window_speeds)
                / max(len(window_speeds), 1)
            ),
            "stall_speed_threshold_m_s": stall_speed_threshold_m_s,
            "action_saturation_fraction": saturated_action_sum
            / max(completed_steps, 1),
            "mean_abs_action_rate_per_step": action_rate_sum
            / max(completed_steps, 1),
            "mean_abs_action_acceleration_per_step2": action_acceleration_sum
            / max(completed_steps, 1),
            "rms_joint_acceleration_rad_s2": math.sqrt(
                joint_acceleration_squared_sum / max(completed_steps, 1)
            ),
            "rms_body_linear_acceleration_m_s2": math.sqrt(
                body_linear_acceleration_squared_sum / max(completed_steps, 1)
            ),
            "rms_body_angular_acceleration_rad_s2": math.sqrt(
                body_angular_acceleration_squared_sum / max(completed_steps, 1)
            ),
            "mean_target_limiter_gap_rad": target_limiter_gap_sum
            / max(completed_steps, 1),
            "three_or_four_foot_support_fraction": three_foot_support_sum
            / max(completed_steps, 1),
            "two_or_fewer_foot_support_fraction": excess_airborne_sum
            / max(completed_steps, 1),
            "absolute_yaw_travel_rad": absolute_yaw_travel_rad,
            "touchdowns_by_leg": {
                name: int(touchdown_counts[index].item())
                for index, name in enumerate(foot_names)
            },
            "scheduled_contact_match_by_leg": {
                name: float(scheduled_contact_matches[index].item())
                / max(completed_steps, 1)
                for index, name in enumerate(foot_names)
            },
            "fell": fell,
            "one_second_positions_m": positions,
            "one_second_lateral_positions_m": lateral_positions,
        }
    )

summary = {
    "checkpoint": checkpoint,
    "seconds": args.seconds,
    "window_seconds": args.window_seconds,
    "episodes": results,
    "task": task,
    "forward_speed_m_s": args.forward_speed,
    "mean_distance_m": sum(float(item["distance_m"]) for item in results)
    / len(results),
    "mean_abs_lateral_displacement_m": sum(
        abs(float(item["lateral_displacement_m"])) for item in results
    )
    / len(results),
    "mean_abs_final_heading_error_rad": sum(
        abs(float(item["final_heading_error_rad"])) for item in results
    )
    / len(results),
    "mean_final_window_speed_m_s": sum(
        float(item["final_window_speed_m_s"]) for item in results
    )
    / len(results),
    "mean_stall_window_fraction": sum(
        float(item["stall_window_fraction"]) for item in results
    )
    / len(results),
    "fall_rate": sum(bool(item["fell"]) for item in results) / len(results),
    "mean_rms_joint_acceleration_rad_s2": sum(
        float(item["rms_joint_acceleration_rad_s2"]) for item in results
    ) / len(results),
    "mean_rms_body_linear_acceleration_m_s2": sum(
        float(item["rms_body_linear_acceleration_m_s2"]) for item in results
    ) / len(results),
    "mean_rms_body_angular_acceleration_rad_s2": sum(
        float(item["rms_body_angular_acceleration_rad_s2"]) for item in results
    ) / len(results),
    "mean_target_limiter_gap_rad": sum(
        float(item["mean_target_limiter_gap_rad"]) for item in results
    ) / len(results),
    "mean_three_or_four_foot_support_fraction": sum(
        float(item["three_or_four_foot_support_fraction"]) for item in results
    ) / len(results),
    "mean_two_or_fewer_foot_support_fraction": sum(
        float(item["two_or_fewer_foot_support_fraction"]) for item in results
    ) / len(results),
    "mean_absolute_yaw_travel_rad": sum(
        float(item["absolute_yaw_travel_rad"]) for item in results
    ) / len(results),
}
print("SUSTAINED_EVALUATION_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
env.close()
simulation_app.close()
