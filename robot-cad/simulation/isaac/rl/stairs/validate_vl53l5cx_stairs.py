"""Validate the VL53L5CX PhysX observation against the authored stair world."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

import cv2
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

from _stair_rl_contract import config_for_height_stage  # noqa: E402
from _vl53l5cx_contract import vl53l5cx_observation_fields  # noqa: E402

parser = argparse.ArgumentParser(
    description="Validate VL53L5CX ray geometry, cadence, latency, and hits."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v7_vl53l5cx.yaml"),
)
parser.add_argument("--world", default=None)
parser.add_argument("--height-stage", default="180mm")
parser.add_argument("--control-steps", type=int, default=24)
parser.add_argument("--seed", type=int, default=147)
parser.add_argument(
    "--report",
    default="reviews/vl53l5cx-stairs-validation.json",
)
parser.add_argument(
    "--heatmap",
    default="reviews/vl53l5cx-stairs-depth.png",
)
parser.add_argument("--gui", action="store_true")
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


if args.control_steps < 16:
    parser.error("--control-steps must be at least 16 to verify cadence and latency")
config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
try:
    config = config_for_height_stage(config, args.height_stage)
except ValueError as exc:
    parser.error(str(exc))
task_config = dict(config["task"])
world_path = _resolve_project_path(args.world or task_config["world"])
report_path = _resolve_project_path(args.report)
heatmap_path = _resolve_project_path(args.heatmap)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1280,
        "height": 720,
    }
)

from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402


def _json_grid(grid: np.ndarray) -> list[list[float | None]]:
    return [
        [float(value) if np.isfinite(value) else None for value in row]
        for row in np.asarray(grid)
    ]


def _write_heatmap(path: Path, grid: np.ndarray, useful_range_m: float) -> None:
    normalized = np.clip(np.asarray(grid, dtype=np.float32) / useful_range_m, 0.0, 1.0)
    valid = np.isfinite(normalized)
    image = np.zeros(normalized.shape, dtype=np.uint8)
    image[valid] = np.rint((1.0 - normalized[valid]) * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    colored[~valid] = (0, 0, 0)
    colored = cv2.resize(colored, (640, 640), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((720, 760, 3), 24, dtype=np.uint8)
    canvas[55:695, 60:700] = colored
    cv2.putText(
        canvas,
        "VL53L5CX simulated depth (near = warm)",
        (60, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "LEFT", (60, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, "RIGHT", (640, 715), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, "TOP", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, "BOTTOM", (5, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Could not write depth heatmap: {path}")


report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "world": str(world_path),
    "seed": args.seed,
    "control_steps": args.control_steps,
    "report": str(report_path),
    "heatmap": str(heatmap_path),
}
exit_code = 1
raw_env: QuadrupedStairsEnv | None = None

try:
    if not world_path.is_file():
        raise FileNotFoundError(world_path)
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    raw_env.set_evaluation_level(int(task_config["staircase"]["step_count"]))
    observation, reset_info = raw_env.reset(seed=args.seed)
    sensor = raw_env.vl53l5cx_sensor
    if sensor is None:
        raise RuntimeError("VL53L5CX sensor runtime was not constructed")
    sensor_fields = vl53l5cx_observation_fields(task_config["terrain_perception"])
    sensor_start = raw_env.observation_fields.index(sensor_fields[0])
    samples = [observation[sensor_start : sensor_start + len(sensor_fields)].copy()]
    terminated_early = False
    for _ in range(args.control_steps):
        observation, _, terminated, truncated, _ = raw_env.step(
            np.zeros(12, dtype=np.float32)
        )
        samples.append(
            observation[sensor_start : sensor_start + len(sensor_fields)].copy()
        )
        if terminated or truncated:
            terminated_early = True
            break

    changed_control_frames = [
        index
        for index in range(1, len(samples))
        if not np.array_equal(samples[index], samples[index - 1])
    ]
    paths = tuple(sensor.latest_hit_paths)
    path_counts = Counter(path for path in paths if path)
    valid_count = int(np.count_nonzero(np.isfinite(sensor.latest_noisy_depth_grid_m)))
    _write_heatmap(
        heatmap_path,
        sensor.latest_noisy_depth_grid_m,
        float(task_config["terrain_perception"]["useful_normalization_range_m"]),
    )
    checks = {
        "runtime_observation_shape_84": tuple(observation.shape) == (84,),
        "ray_count_64": sensor.ray_directions_from_base.shape == (8, 8, 3),
        "update_period_8_control_frames": (
            sensor.control_frames_per_measurement == 8
        ),
        "one_sensor_frame_latency": sensor.latency_frames == 1,
        "first_depth_delivery_after_latency": (
            len(samples) > 8
            and np.all(samples[0] == 1.0)
            and not np.array_equal(samples[8], samples[0])
        ),
        "held_between_measurements": all(
            np.array_equal(samples[index], samples[0]) for index in range(1, 8)
        ),
        "finite_policy_depth_values": all(
            bool(np.all(np.isfinite(sample))) for sample in samples
        ),
        "physx_depth_hits_present": valid_count > 0,
        "completed_requested_control_steps": not terminated_early,
    }
    report.update(
        {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "reset_info": {
                "terrain_perception_mode": reset_info["terrain_perception_mode"],
                "observation_size": int(observation.shape[0]),
            },
            "sensor_metrics": sensor.metrics,
            "changed_control_frames": changed_control_frames,
            "hit_prim_counts": dict(path_counts),
            "true_depth_grid_m": _json_grid(sensor.latest_true_depth_grid_m),
            "noisy_depth_grid_m": _json_grid(sensor.latest_noisy_depth_grid_m),
            "compressed_observation": sensor.latest_observation.tolist(),
            "terminated_early": terminated_early,
        }
    )
    exit_code = 0 if report["status"] == "PASS" else 2
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
    print("DROBOT_VL53L5CX_VALIDATION_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
