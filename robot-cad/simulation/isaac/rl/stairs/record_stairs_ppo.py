"""Record one deterministic Drobot stair-climbing episode as H.264 MP4."""

from __future__ import annotations

import argparse
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

parser = argparse.ArgumentParser(
    description="Record one deterministic Drobot stair-climbing episode."
)
parser.add_argument(
    "--config",
    default=str(SCRIPT_DIR / "quadruped_stairs_v1.yaml"),
)
parser.add_argument("--world", default=None)
parser.add_argument(
    "--model",
    default=(
        "simulation/isaac/output/rl/ppo-stairs-v1/"
        "drobot_stairs_ppo_final.zip"
    ),
)
parser.add_argument("--seed", type=int, default=143)
parser.add_argument("--device", default="cpu")
parser.add_argument("--active-steps", type=int, default=None)
parser.add_argument(
    "--camera-view",
    choices=("external", "onboard"),
    default="external",
)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument(
    "--video",
    default="reviews/ppo-stairs-v1-evaluation.mp4",
)
parser.add_argument(
    "--thumbnail",
    default="reviews/ppo-stairs-v1-evaluation.png",
)
parser.add_argument(
    "--report",
    default=(
        "simulation/isaac/output/rl/ppo-stairs-v1/"
        "recording_report.json"
    ),
)
parser.add_argument("--allow-unverified-model", action="store_true")
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


if args.fps <= 0 or args.width <= 0 or args.height <= 0:
    parser.error("--fps, --width, and --height must be positive")
config_path = _resolve_project_path(args.config)
with config_path.open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if int(config.get("schema_version", 0)) != 1:
    parser.error(f"Unsupported stairs config schema: {config.get('schema_version')}")
task_config = dict(config["task"])
staircase = dict(task_config["staircase"])
active_steps = (
    int(staircase["step_count"])
    if args.active_steps is None
    else args.active_steps
)
if active_steps < 1 or active_steps > int(staircase["step_count"]):
    parser.error("--active-steps is outside the configured staircase")
control_hz = int(task_config["control_hz"])
if control_hz % args.fps:
    parser.error("--fps must divide the configured control_hz exactly")
control_steps_per_frame = control_hz // args.fps
world_path = _resolve_project_path(args.world or task_config["world"])
world_dependency_paths = tuple(
    _resolve_project_path(value)
    for value in task_config.get("world_dependencies", ())
)
model_path = _resolve_project_path(args.model)
video_path = _resolve_project_path(args.video)
thumbnail_path = _resolve_project_path(args.thumbnail)
report_path = _resolve_project_path(args.report)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": True,
        "width": args.width,
        "height": args.height,
    }
)

from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from PIL import Image  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from video_encoding import get_video_encoding_interface  # noqa: E402

report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "model": str(model_path),
    "world": str(world_path),
    "seed": args.seed,
    "device": args.device,
    "active_steps": active_steps,
    "camera_view": args.camera_view,
    "video": str(video_path),
    "thumbnail": str(thumbnail_path),
    "fps": args.fps,
    "resolution_wh": [args.width, args.height],
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
}
exit_code = 1
raw_env: QuadrupedStairsEnv | None = None
encoding_interface = None
encoding_started = False

try:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not world_path.is_file():
        raise FileNotFoundError(world_path)
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human",
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

    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("Isaac Sim has no active viewport")
    if args.camera_view == "onboard":
        camera_path = str(task_config["camera_prim"])
        viewport.camera_path = camera_path
    else:
        approach_start_x = min(
            float(value) for value in task_config["reset_start_x_range_m"]
        )
        landing_center_x = (
            float(staircase["start_x_m"])
            + int(staircase["step_count"])
            * float(staircase["tread_depth_m"])
            + float(staircase["top_platform_depth_m"]) / 2.0
        )
        camera_center_x = (approach_start_x + landing_center_x) / 2.0
        top_height = (
            int(staircase["step_count"]) * float(staircase["rise_m"])
        )
        camera_path = "/OmniverseKit_Persp"
        set_camera_view(
            eye=[camera_center_x, -2.65, top_height + 0.82],
            target=[camera_center_x, 0.0, top_height + 0.10],
            camera_prim_path=camera_path,
        )
        viewport.camera_path = camera_path

    camera_prim = raw_env.stage.GetPrimAtPath(camera_path)
    if not camera_prim.IsValid():
        raise RuntimeError(f"Recording camera prim is missing: {camera_path}")
    if "OmniSensorAPI" not in camera_prim.GetAppliedSchemas():
        camera_prim.ApplyAPI("OmniSensorAPI")
    rtx_camera = RtxCamera(
        camera_path,
        tick_rate=None,
        reset_xform_op_properties=False,
    )
    camera_sensor = CameraSensor(
        rtx_camera,
        resolution=(args.height, args.width),
        annotators=["rgb"],
    )
    rgb_data = None
    for _ in range(30):
        simulation_app.update()
        candidate, _ = camera_sensor.get_data("rgb")
        if candidate is not None:
            rgb_data = candidate
    if rgb_data is None:
        raise RuntimeError("Recording camera produced no RGB warmup frame")

    observation, _ = raw_env.reset(seed=args.seed)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
    encoding_interface = get_video_encoding_interface()
    if encoding_interface is None:
        raise RuntimeError("Isaac H.264 video encoding interface is unavailable")
    if not encoding_interface.start_encoding(
        str(video_path),
        float(args.fps),
        0,
        True,
    ):
        raise RuntimeError(f"Could not initialize video encoder for {video_path}")
    encoding_started = True

    recorded_frames = 0
    episode_metrics: dict[str, object] | None = None
    thumbnail_rgb: np.ndarray | None = None
    for control_step in range(raw_env.max_episode_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = raw_env.step(action)
        if (control_step + 1) % control_steps_per_frame == 0:
            rgb_data, _ = camera_sensor.get_data("rgb")
            if rgb_data is None:
                raise RuntimeError(
                    f"Recording camera returned no frame at step {control_step + 1}"
                )
            rgb = _numpy(rgb_data)
            if rgb.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"Unexpected recording frame shape: {rgb.shape}")
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                raise RuntimeError(
                    f"Unexpected recording channel layout: {rgb.shape}"
                )
            rgb = np.ascontiguousarray(rgb[..., :3])
            alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
            if not encoding_interface.encode_next_frame_from_buffer(
                rgba,
                args.width,
                args.height,
            ):
                raise RuntimeError(
                    f"Video encoder rejected frame {recorded_frames}"
                )
            recorded_frames += 1
            thumbnail_rgb = rgb.copy()
        if terminated or truncated:
            episode_metrics = dict(info["episode_metrics"])
            break

    encoding_interface.finalize_encoding()
    encoding_started = False
    if episode_metrics is None:
        raise RuntimeError("The recorded stair episode did not finish")
    if thumbnail_rgb is None:
        raise RuntimeError("No recording frame was available for the thumbnail")
    Image.fromarray(thumbnail_rgb, mode="RGB").save(thumbnail_path)
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Isaac Sim did not create a usable MP4: {video_path}")
    report.update(
        {
            "status": "PASS",
            "model_contract_verification": verification,
            "ppo_algorithm_verification": algorithm_verification,
            "recorded_frames": recorded_frames,
            "episode": episode_metrics,
            "environment_contract": raw_env.contract,
            "video_bytes": video_path.stat().st_size,
            "thumbnail_bytes": thumbnail_path.stat().st_size,
        }
    )
    exit_code = 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
finally:
    if encoding_started and encoding_interface is not None:
        encoding_interface.finalize_encoding()
    if raw_env is not None:
        raw_env.close()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_STAIRS_RECORD_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
