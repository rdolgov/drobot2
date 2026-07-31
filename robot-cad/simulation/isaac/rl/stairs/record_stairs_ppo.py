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
from _stair_rl_contract import config_for_height_stage  # noqa: E402

parser = argparse.ArgumentParser(
    description="Record one deterministic Drobot stair-climbing episode."
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
parser.add_argument("--seed", type=int, default=143)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--stochastic",
    action="store_true",
    help="Sample the PPO action distribution instead of using its mean.",
)
parser.add_argument(
    "--policy-seed",
    type=int,
    default=None,
    help="Torch/action RNG seed; defaults to --seed.",
)
parser.add_argument(
    "--skip-policy-samples",
    type=int,
    default=0,
    help=(
        "Advance this many stochastic policy samples before recording, for "
        "exact replay of a reported rollout sequence."
    ),
)
parser.add_argument(
    "--skip-episodes",
    type=int,
    default=0,
    help=(
        "Replay this many seeded episodes without storing frames before "
        "recording, preserving policy RNG and physics reset history."
    ),
)
parser.add_argument(
    "--search-success-episodes",
    type=int,
    default=0,
    help=(
        "Search up to this many seeded episodes and encode only the first "
        "physical success; zero records the first episode regardless."
    ),
)
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
parser.add_argument(
    "--trajectory",
    default=None,
    help="Optional compressed NumPy observation/action trajectory output.",
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
if args.skip_policy_samples < 0:
    parser.error("--skip-policy-samples cannot be negative")
if args.skip_episodes < 0:
    parser.error("--skip-episodes cannot be negative")
if args.search_success_episodes < 0:
    parser.error("--search-success-episodes cannot be negative")
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
trajectory_path = (
    _resolve_project_path(args.trajectory)
    if args.trajectory is not None
    else None
)
policy_seed = args.seed if args.policy_seed is None else args.policy_seed

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
    "deterministic": not args.stochastic,
    "policy_seed": policy_seed,
    "skip_policy_samples": args.skip_policy_samples,
    "skip_episodes": args.skip_episodes,
    "search_success_episodes": args.search_success_episodes,
    "device": args.device,
    "active_steps": active_steps,
    "height_stage": args.height_stage,
    "camera_view": args.camera_view,
    "video": str(video_path),
    "thumbnail": str(thumbnail_path),
    "trajectory": str(trajectory_path) if trajectory_path else None,
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
    model.set_random_seed(policy_seed)
    if args.skip_policy_samples:
        dummy_observation = np.zeros(
            raw_env.observation_space.shape,
            dtype=np.float32,
        )
        for _ in range(args.skip_policy_samples):
            model.predict(dummy_observation, deterministic=False)
    skipped_episode_metrics: list[dict[str, object]] = []
    for skipped_index in range(args.skip_episodes):
        skipped_seed = args.seed + skipped_index
        observation, _ = raw_env.reset(seed=skipped_seed)
        skipped_metrics: dict[str, object] | None = None
        for _ in range(raw_env.max_episode_steps):
            action, _ = model.predict(
                observation,
                deterministic=not args.stochastic,
            )
            observation, _, terminated, truncated, info = raw_env.step(action)
            if terminated or truncated:
                skipped_metrics = dict(info["episode_metrics"])
                break
        if skipped_metrics is None:
            raise RuntimeError(
                f"Skipped episode {skipped_index + 1} did not finish"
            )
        skipped_episode_metrics.append(skipped_metrics)
        if (skipped_index + 1) % 10 == 0:
            print(
                "DROBOT_STAIRS_RECORD_SKIP_PROGRESS="
                + json.dumps(
                    {
                        "episodes_replayed": skipped_index + 1,
                        "latest_seed": skipped_seed,
                        "latest_highest_step": int(
                            skipped_metrics["highest_step_reached"]
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    attempt_limit = args.search_success_episodes or 1
    search_episode_metrics: list[dict[str, object]] = []
    selected_episode_index: int | None = None
    selected_episode_seed: int | None = None
    episode_metrics: dict[str, object] | None = None
    selected_frames: list[np.ndarray] | None = None
    trajectory_observations: list[np.ndarray] = []
    trajectory_actions: list[np.ndarray] = []
    for attempt_index in range(attempt_limit):
        episode_seed = args.seed + args.skip_episodes + attempt_index
        observation, _ = raw_env.reset(seed=episode_seed)
        candidate_frames: list[np.ndarray] = []
        candidate_observations: list[np.ndarray] = []
        candidate_actions: list[np.ndarray] = []
        candidate_metrics: dict[str, object] | None = None
        for control_step in range(raw_env.max_episode_steps):
            action, _ = model.predict(
                observation,
                deterministic=not args.stochastic,
            )
            candidate_observations.append(observation.copy())
            candidate_actions.append(
                np.asarray(action, dtype=np.float32).reshape(12).copy()
            )
            observation, _, terminated, truncated, info = raw_env.step(action)
            if (control_step + 1) % control_steps_per_frame == 0:
                rgb_data, _ = camera_sensor.get_data("rgb")
                if rgb_data is None:
                    raise RuntimeError(
                        "Recording camera returned no frame at step "
                        f"{control_step + 1}"
                    )
                rgb = _numpy(rgb_data)
                if rgb.shape[:2] != (args.height, args.width):
                    raise RuntimeError(
                        f"Unexpected recording frame shape: {rgb.shape}"
                    )
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
                    raise RuntimeError(
                        f"Unexpected recording channel layout: {rgb.shape}"
                    )
                candidate_frames.append(
                    np.ascontiguousarray(rgb[..., :3]).copy()
                )
            if terminated or truncated:
                candidate_metrics = dict(info["episode_metrics"])
                break
        if candidate_metrics is None:
            raise RuntimeError(
                f"Search episode {attempt_index + 1} did not finish"
            )
        search_episode_metrics.append(candidate_metrics)
        progress = {
            "episode": attempt_index + 1,
            "seed": episode_seed,
            "stairs_completed": bool(candidate_metrics["stairs_completed"]),
            "highest_step_reached": int(
                candidate_metrics["highest_step_reached"]
            ),
            "maximum_base_elevation_gain_m": float(
                candidate_metrics["maximum_base_elevation_gain_m"]
            ),
            "forward_displacement_m": float(
                candidate_metrics["forward_displacement_m"]
            ),
            "failure_reasons": list(candidate_metrics["failure_reasons"]),
        }
        print(
            "DROBOT_STAIRS_RECORD_SEARCH_EPISODE="
            + json.dumps(progress, sort_keys=True),
            flush=True,
        )
        require_success = args.search_success_episodes > 0
        if require_success and not bool(candidate_metrics["stairs_completed"]):
            continue
        selected_episode_index = attempt_index + 1
        selected_episode_seed = episode_seed
        episode_metrics = candidate_metrics
        selected_frames = candidate_frames
        trajectory_observations = candidate_observations
        trajectory_actions = candidate_actions
        break

    if episode_metrics is None or selected_frames is None:
        raise RuntimeError(
            "No physically successful stair episode found within "
            f"{attempt_limit} attempts"
        )
    if not selected_frames:
        raise RuntimeError("No recording frame was available for the episode")

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
    for frame_index, rgb in enumerate(selected_frames):
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        rgba = np.ascontiguousarray(np.concatenate((rgb, alpha), axis=2))
        if not encoding_interface.encode_next_frame_from_buffer(
            rgba,
            args.width,
            args.height,
        ):
            raise RuntimeError(f"Video encoder rejected frame {frame_index}")
    encoding_interface.finalize_encoding()
    encoding_started = False
    recorded_frames = len(selected_frames)
    Image.fromarray(selected_frames[-1], mode="RGB").save(thumbnail_path)
    if trajectory_path is not None:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            trajectory_path,
            observations=np.asarray(
                trajectory_observations,
                dtype=np.float32,
            ),
            actions=np.asarray(trajectory_actions, dtype=np.float32),
        )
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"Isaac Sim did not create a usable MP4: {video_path}")
    report.update(
        {
            "status": "PASS",
            "model_contract_verification": verification,
            "ppo_algorithm_verification": algorithm_verification,
            "recorded_frames": recorded_frames,
            "episode": episode_metrics,
            "selected_episode_index": selected_episode_index,
            "selected_episode_seed": selected_episode_seed,
            "search_episodes_attempted": len(search_episode_metrics),
            "search_episode_metrics": search_episode_metrics,
            "skipped_episode_metrics": skipped_episode_metrics,
            "environment_contract": raw_env.contract,
            "video_bytes": video_path.stat().st_size,
            "thumbnail_bytes": thumbnail_path.stat().st_size,
            "trajectory_bytes": (
                trajectory_path.stat().st_size
                if trajectory_path is not None
                else None
            ),
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
