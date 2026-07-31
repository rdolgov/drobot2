"""Deterministically evaluate a separate Drobot stair-climbing PPO model."""

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
ISAAC_DIR = SCRIPT_DIR.parents[1]
PROJECT_ROOT = SCRIPT_DIR.parents[3]
for module_dir in (str(ISAAC_DIR), str(RL_DIR), str(SCRIPT_DIR)):
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

from _run_support import (  # noqa: E402
    validate_model_manifest,
    validate_ppo_algorithm_contract,
)
from _stair_rl_contract import config_for_height_stage  # noqa: E402

parser = argparse.ArgumentParser(
    description="Evaluate a trained Drobot stairs PPO policy."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v1.yaml"),
)
parser.add_argument("--world", default=None)
parser.add_argument(
    "--height-stage",
    default=None,
    help="Apply one stair_height_stages entry declared by the config.",
)
parser.add_argument(
    "--model",
    default=(
        "simulation/isaac/output/rl/ppo-stairs-v1/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--seed", type=int, default=143)
parser.add_argument("--device", default="cpu")
parser.add_argument("--active-steps", type=int, default=None)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--screenshot", default=None)
parser.add_argument(
    "--camera-view",
    choices=("external", "onboard"),
    default="external",
)
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
if int(config.get("schema_version", 0)) != 1:
    parser.error(f"Unsupported stairs config schema: {config.get('schema_version')}")
try:
    config = config_for_height_stage(config, args.height_stage)
except ValueError as exc:
    parser.error(str(exc))
task_config = dict(config["task"])
staircase = dict(task_config["staircase"])
active_steps = (
    int(staircase["step_count"])
    if args.active_steps is None
    else args.active_steps
)
if active_steps < 1 or active_steps > int(staircase["step_count"]):
    parser.error("--active-steps is outside the configured staircase")
world_path = _resolve_project_path(args.world or task_config["world"])
world_dependency_paths = tuple(
    _resolve_project_path(value)
    for value in task_config.get("world_dependencies", ())
)
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

from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from omni.kit.viewport.utility import (  # noqa: E402
    capture_viewport_to_file,
    get_active_viewport,
)
from stable_baselines3 import PPO  # noqa: E402


def _set_external_camera() -> None:
    approach_start_x = min(
        float(value) for value in task_config["reset_start_x_range_m"]
    )
    landing_center_x = (
        float(staircase["start_x_m"])
        + int(staircase["step_count"]) * float(staircase["tread_depth_m"])
        + float(staircase["top_platform_depth_m"]) / 2.0
    )
    camera_center_x = (approach_start_x + landing_center_x) / 2.0
    top_height = int(staircase["step_count"]) * float(staircase["rise_m"])
    set_camera_view(
        eye=[camera_center_x, -2.65, top_height + 0.82],
        target=[camera_center_x, 0.0, top_height + 0.10],
        camera_prim_path="/OmniverseKit_Persp",
    )


def _capture_screenshot(path: Path) -> None:
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim has no active viewport")
    if args.camera_view == "onboard":
        viewport.camera_path = str(task_config["camera_prim"])
    else:
        _set_external_camera()
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
    "active_steps": active_steps,
    "height_stage": args.height_stage,
    "camera_view": args.camera_view,
    "screenshot": str(screenshot_path) if screenshot_path else None,
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
}
exit_code = 1
raw_env: QuadrupedStairsEnv | None = None

try:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not world_path.is_file():
        raise FileNotFoundError(world_path)
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    raw_env.set_evaluation_level(active_steps)
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
        else {
            "status": "SKIPPED",
            "reason": "model_manifest_verification_was_skipped",
        }
    )
    observation, _ = raw_env.reset(seed=args.seed)
    episode_metrics: list[dict[str, object]] = []
    maximum_steps = args.episodes * raw_env.max_episode_steps * 2
    for _ in range(maximum_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = raw_env.step(action)
        if terminated or truncated:
            episode_metrics.append(dict(info["episode_metrics"]))
            if len(episode_metrics) >= args.episodes:
                break
            observation, _ = raw_env.reset()
    if len(episode_metrics) != args.episodes:
        raise RuntimeError(
            f"Expected {args.episodes} completed episodes, "
            f"got {len(episode_metrics)}"
        )
    if screenshot_path is not None:
        _capture_screenshot(screenshot_path)
    failure_counts = Counter(
        reason
        for item in episode_metrics
        for reason in item["failure_reasons"]
    )
    success_count = sum(bool(item["stairs_completed"]) for item in episode_metrics)
    report.update(
        {
            "status": "PASS",
            "model_contract_verification": verification,
            "ppo_algorithm_verification": algorithm_verification,
            "episodes": episode_metrics,
            "success_count": success_count,
            "success_rate": success_count / len(episode_metrics),
            "failure_reason_counts": dict(sorted(failure_counts.items())),
            "mean_return": float(
                np.mean([float(item["return"]) for item in episode_metrics])
            ),
            "mean_highest_step_reached": float(
                np.mean(
                    [
                        float(item["highest_step_reached"])
                        for item in episode_metrics
                    ]
                )
            ),
            "mean_forward_displacement_m": float(
                np.mean(
                    [
                        float(item["forward_displacement_m"])
                        for item in episode_metrics
                    ]
                )
            ),
            "mean_elevation_gain_m": float(
                np.mean(
                    [float(item["elevation_gain_m"]) for item in episode_metrics]
                )
            ),
            "minimum_base_clearance_m": float(
                min(
                    float(item["minimum_base_clearance_m"])
                    for item in episode_metrics
                )
            ),
            "maximum_body_tilt_deg": float(
                max(
                    float(item["maximum_body_tilt_deg"])
                    for item in episode_metrics
                )
            ),
            "environment_contract": raw_env.contract,
            "screenshot_bytes": (
                screenshot_path.stat().st_size if screenshot_path else None
            ),
        }
    )
    exit_code = 0
    if args.gui:
        print("Stair policy evaluation complete; close Isaac Sim to exit.")
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
    print("DROBOT_STAIRS_EVAL_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
