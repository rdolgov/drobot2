"""Evaluate whether a walking checkpoint sustains motion beyond reset transients."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seconds", type=int, default=30)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--window-seconds", type=int, default=5)
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

task = "Drobot-Commanded-Walk-Forward-Direct"
env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
env_cfg.scene.num_envs = 1
env_cfg.seed = 4401
env_cfg.reset_joint_position_noise_rad = 0.0
env_cfg.reset_xy_jitter_m = 0.0
env_cfg.disable_time_limit = True
preview_control.COMMAND_OVERRIDE = (0.15, 0.0, 0.0)

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
    positions = [start_x]
    lateral_positions = [start_y]
    saturated_action_sum = 0.0
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
        obs, _, dones, _ = env.step(actions)
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
    window_speeds = [
        (positions[index] - positions[index - args.window_seconds])
        / args.window_seconds
        for index in range(args.window_seconds, len(positions))
    ]
    results.append(
        {
            "duration_s": duration_s,
            "distance_m": distance_m,
            "lateral_displacement_m": lateral_positions[-1] - lateral_positions[0],
            "mean_speed_m_s": distance_m / max(duration_s, 1.0 / 60.0),
            "minimum_window_speed_m_s": min(window_speeds, default=0.0),
            "final_window_speed_m_s": window_speeds[-1] if window_speeds else 0.0,
            "stall_window_fraction": (
                sum(speed < 0.02 for speed in window_speeds)
                / max(len(window_speeds), 1)
            ),
            "action_saturation_fraction": saturated_action_sum
            / max(completed_steps, 1),
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
    "mean_distance_m": sum(float(item["distance_m"]) for item in results)
    / len(results),
    "mean_abs_lateral_displacement_m": sum(
        abs(float(item["lateral_displacement_m"])) for item in results
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
}
print("SUSTAINED_EVALUATION_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
env.close()
simulation_app.close()
