"""Deterministically evaluate a Drobot single-foot-lift PPO policy."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # noqa: F401
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
RL_DIR = SCRIPT_DIR.parent
STAIRS_DIR = RL_DIR / "stairs"
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(STAIRS_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _run_support import (  # noqa: E402
    validate_model_manifest,
    validate_ppo_algorithm_contract,
)

parser = argparse.ArgumentParser(
    description="Evaluate the Drobot 190 mm single-foot-lift PPO policy."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_foot_lift_v1.yaml"),
)
parser.add_argument("--world", default=None)
parser.add_argument(
    "--model",
    default=("simulation/isaac/output/rl/ppo-foot-lift-v1-190mm/drobot_foot_lift_ppo_final.zip"),
)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=191)
parser.add_argument("--device", default="cpu")
parser.add_argument("--gui", action="store_true")
parser.add_argument("--screenshot", default=None)
parser.add_argument("--report", default=None)
parser.add_argument("--allow-unverified-model", action="store_true")
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
world_dependency_paths = tuple(
    _resolve_project_path(value) for value in task_config.get("world_dependencies", ())
)
model_path = _resolve_project_path(args.model)
report_path = (
    _resolve_project_path(args.report)
    if args.report
    else model_path.parent / "evaluation_report.json"
)
screenshot_path = _resolve_project_path(args.screenshot) if args.screenshot else None

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720})

from _quadruped_foot_lift_env import QuadrupedFootLiftEnv  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import (  # noqa: E402
    capture_viewport_to_file,
    get_active_viewport,
)
from stable_baselines3 import PPO  # noqa: E402


def _capture_screenshot(path: Path, raw_env: QuadrupedFootLiftEnv) -> None:
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim has no active viewport")
    position = raw_env.robot.get_world_poses()[0]
    if hasattr(position, "numpy"):
        position = position.numpy()
    position = np.asarray(position).reshape(-1, 3)[0]
    set_camera_view(
        eye=[float(position[0] + 0.28), float(position[1] + 1.35), 0.72],
        target=[float(position[0] + 0.15), float(position[1]), 0.24],
        camera_prim_path="/OmniverseKit_Persp",
    )
    viewport.camera_path = "/OmniverseKit_Persp"
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
    "screenshot": str(screenshot_path) if screenshot_path else None,
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
}
exit_code = 1
raw_env: QuadrupedFootLiftEnv | None = None

try:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    raw_env = QuadrupedFootLiftEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    verification = validate_model_manifest(
        model_path=model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=world_dependency_paths,
        environment_contract=raw_env.contract,
        allow_unverified=args.allow_unverified_model,
    )
    model = PPO.load(str(model_path), env=raw_env, device=args.device)
    algorithm_verification = (
        validate_ppo_algorithm_contract(
            model,
            verification["algorithm_contract"],
        )
        if verification["status"] == "PASS"
        else {"status": "SKIPPED", "reason": "model_verification_skipped"}
    )
    observation, _ = raw_env.reset(seed=args.seed)
    episodes: list[dict[str, object]] = []
    maximum_steps = args.episodes * raw_env.max_episode_steps * 2
    for _ in range(maximum_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = raw_env.step(action)
        if terminated or truncated:
            episodes.append(dict(info["episode_metrics"]))
            if len(episodes) >= args.episodes:
                break
            observation, _ = raw_env.reset()
    if len(episodes) != args.episodes:
        raise RuntimeError(f"Expected {args.episodes} completed episodes, got {len(episodes)}")
    if screenshot_path is not None:
        _capture_screenshot(screenshot_path, raw_env)
    failure_counts = Counter(
        reason for episode in episodes for reason in episode["failure_reasons"]
    )
    success_count = sum(bool(episode["skill_completed"]) for episode in episodes)
    report.update(
        {
            "status": "PASS",
            "model_contract_verification": verification,
            "ppo_algorithm_verification": algorithm_verification,
            "episodes": episodes,
            "success_count": success_count,
            "success_rate": success_count / len(episodes),
            "maximum_swing_foot_lift_m": max(
                float(episode["maximum_swing_foot_lift_m"]) for episode in episodes
            ),
            "maximum_body_tilt_deg": max(
                float(episode["maximum_body_tilt_deg"]) for episode in episodes
            ),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "mean_return": float(np.mean([float(episode["return"]) for episode in episodes])),
            "environment_contract": raw_env.contract,
            "screenshot_bytes": (screenshot_path.stat().st_size if screenshot_path else None),
        }
    )
    exit_code = 0
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
    print("DROBOT_FOOT_LIFT_EVAL_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
