"""Evaluate or view a trained Drobot PPO walking policy in Isaac Sim."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
ISAAC_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(ISAAC_DIR) not in sys.path:
    sys.path.insert(0, str(ISAAC_DIR))

parser = argparse.ArgumentParser(
    description="Evaluate a trained Drobot quadruped PPO policy."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_walk_v1.yaml"),
)
parser.add_argument("--world", default=None)
parser.add_argument(
    "--model",
    default="simulation/isaac/output/rl/ppo-walk-v1/drobot_walk_ppo_final.zip",
)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--device", default="cpu")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--screenshot", default=None)
parser.add_argument(
    "--camera-view",
    choices=("external", "onboard"),
    default="external",
)
parser.add_argument("--report", default=None)
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if args.episodes <= 0:
    parser.error("--episodes must be positive")
config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
task_config = dict(config["task"])
world_path = _resolve_project_path(args.world or task_config["world"])
model_path = _resolve_project_path(args.model)
report_path = (
    _resolve_project_path(args.report)
    if args.report
    else model_path.parent / "evaluation_report.json"
)
screenshot_path = (
    _resolve_project_path(args.screenshot) if args.screenshot else None
)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1280,
        "height": 720,
    }
)

from _quadruped_rl_env import QuadrupedWalkEnv  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import (  # noqa: E402
    capture_viewport_to_file,
    get_active_viewport,
)
from stable_baselines3 import PPO  # noqa: E402


def _capture_screenshot(path: Path, raw_env: QuadrupedWalkEnv) -> None:
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim has no active viewport")
    if args.camera_view == "onboard":
        viewport.camera_path = str(task_config["camera_prim"])
    else:
        position = raw_env.robot.get_world_poses()[0]
        if hasattr(position, "numpy"):
            position = position.numpy()
        position = np.asarray(position).reshape(-1, 3)[0]
        set_camera_view(
            eye=[
                float(position[0] + 0.75),
                float(position[1] - 0.90),
                float(position[2] + 0.38),
            ],
            target=[
                float(position[0]),
                float(position[1]),
                float(position[2] - 0.15),
            ],
            camera_prim_path="/OmniverseKit_Persp",
        )
    for _ in range(45):
        simulation_app.update()
    path.parent.mkdir(parents=True, exist_ok=True)
    task = asyncio.ensure_future(
        capture_viewport_to_file(
            viewport,
            file_path=str(path),
            is_hdr=False,
        ).wait_for_result()
    )
    for _ in range(300):
        simulation_app.update()
        if task.done():
            break
    if not task.done():
        raise TimeoutError("Timed out waiting for Isaac viewport capture")
    task.result()
    for _ in range(120):
        if path.is_file() and path.stat().st_size > 0:
            break
        simulation_app.update()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Isaac Sim did not create a usable PNG: {path}")


report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "model": str(model_path),
    "world": str(world_path),
    "episodes_requested": args.episodes,
    "seed": args.seed,
    "device": args.device,
    "camera_view": args.camera_view,
    "screenshot": str(screenshot_path) if screenshot_path else None,
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
}
exit_code = 1
raw_env: QuadrupedWalkEnv | None = None

try:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    raw_env = QuadrupedWalkEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    model = PPO.load(str(model_path), env=raw_env, device=args.device)
    observation, _ = raw_env.reset(seed=args.seed)
    episode_metrics: list[dict[str, object]] = []
    maximum_steps = args.episodes * raw_env.max_episode_steps * 2
    for _ in range(maximum_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = raw_env.step(action)
        if terminated or truncated:
            metrics = dict(info["episode_metrics"])
            episode_metrics.append(metrics)
            if len(episode_metrics) >= args.episodes:
                break
            observation, _ = raw_env.reset()
    if len(episode_metrics) != args.episodes:
        raise RuntimeError(
            f"Expected {args.episodes} completed episodes, got {len(episode_metrics)}"
        )
    if screenshot_path is not None:
        _capture_screenshot(screenshot_path, raw_env)
    report.update(
        {
            "status": "PASS",
            "episodes": episode_metrics,
            "mean_return": float(
                sum(float(item["return"]) for item in episode_metrics)
                / len(episode_metrics)
            ),
            "mean_forward_displacement_m": float(
                sum(
                    float(item["forward_displacement_m"])
                    for item in episode_metrics
                )
                / len(episode_metrics)
            ),
            "termination_count": sum(
                bool(item["terminated"]) for item in episode_metrics
            ),
            "environment_contract": raw_env.contract,
            "screenshot_bytes": (
                screenshot_path.stat().st_size if screenshot_path else None
            ),
        }
    )
    exit_code = 0
    if args.gui:
        print("Policy evaluation complete; close Isaac Sim to exit.")
        while simulation_app.is_running():
            simulation_app.update()
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    if raw_env is not None:
        raw_env.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_RL_PLAY_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
