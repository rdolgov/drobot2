"""Deterministically evaluate a separate Drobot stair-climbing PPO model."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
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
from _stair_rl_contract import (  # noqa: E402
    compose_bounded_residual_action,
    config_for_height_stage,
)

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
parser.add_argument(
    "--maximum-lateral-deviation-m",
    type=float,
    default=None,
    help="Override only the evaluation corridor without changing training config.",
)
parser.add_argument("--gui", action="store_true")
parser.add_argument("--screenshot", default=None)
parser.add_argument(
    "--camera-view",
    choices=("external", "onboard"),
    default="external",
)
parser.add_argument("--report", default=None)
parser.add_argument("--allow-unverified-model", action="store_true")
parser.add_argument(
    "--leg-model",
    action="append",
    default=[],
    metavar="LEG=MODEL",
    help=(
        "Use a force-verified per-leg PPO policy outside inter-leg transfer; "
        "repeat for each composed skill."
    ),
)
parser.add_argument(
    "--leg-base-model",
    action="append",
    default=[],
    metavar="LEG=MODEL",
    help="Frozen base policy for a bounded per-leg residual model.",
)
parser.add_argument(
    "--leg-residual-scale",
    action="append",
    default=[],
    metavar="LEG=SCALE",
    help="Residual action scale for a leg that also has --leg-base-model.",
)
parser.add_argument(
    "--leg-residual-support-only",
    action="append",
    default=[],
    metavar="LEG",
    help="Mask a bounded residual off the named swing leg's joints.",
)
parser.add_argument(
    "--leg-residual-support-abduction-only",
    action="append",
    default=[],
    metavar="LEG",
    help="Apply a leg residual only to support hip-abduction joints.",
)
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_leg_models(values: list[str], option_name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        leg, separator, model = str(value).partition("=")
        if not separator or not leg or not model:
            parser.error(f"{option_name} must use LEG=MODEL syntax")
        if leg in result:
            parser.error(f"duplicate {option_name} for {leg}")
        result[leg] = _resolve_project_path(model)
    return result


def _parse_leg_scales(values: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        leg, separator, scale_text = str(value).partition("=")
        if not separator or not leg or not scale_text:
            parser.error("--leg-residual-scale must use LEG=SCALE syntax")
        if leg in result:
            parser.error(f"duplicate --leg-residual-scale for {leg}")
        try:
            scale = float(scale_text)
        except ValueError:
            parser.error(f"invalid residual scale for {leg}: {scale_text}")
        if scale <= 0.0 or scale > 1.0:
            parser.error("leg residual scales must be within (0, 1]")
        result[leg] = scale
    return result


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
if args.maximum_lateral_deviation_m is not None:
    if args.maximum_lateral_deviation_m <= 0.0:
        parser.error("--maximum-lateral-deviation-m must be positive")
    termination = dict(task_config["termination"])
    termination["maximum_lateral_deviation_m"] = float(
        args.maximum_lateral_deviation_m
    )
    task_config["termination"] = termination
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
leg_model_paths = _parse_leg_models(args.leg_model, "--leg-model")
leg_base_model_paths = _parse_leg_models(
    args.leg_base_model,
    "--leg-base-model",
)
leg_residual_scales = _parse_leg_scales(args.leg_residual_scale)
leg_residual_support_only = set(args.leg_residual_support_only)
if len(leg_residual_support_only) != len(args.leg_residual_support_only):
    parser.error("duplicate --leg-residual-support-only leg")
leg_residual_support_abduction_only = set(
    args.leg_residual_support_abduction_only
)
if len(leg_residual_support_abduction_only) != len(
    args.leg_residual_support_abduction_only
):
    parser.error("duplicate --leg-residual-support-abduction-only leg")
if set(leg_base_model_paths) != set(leg_residual_scales):
    parser.error(
        "--leg-base-model and --leg-residual-scale must select the same legs"
    )
if not set(leg_base_model_paths).issubset(leg_model_paths):
    parser.error("each leg base model requires a residual --leg-model")
if not leg_residual_support_only.issubset(leg_base_model_paths):
    parser.error("support-only residual legs require --leg-base-model")
if not leg_residual_support_abduction_only.issubset(leg_base_model_paths):
    parser.error("support-abduction residual legs require --leg-base-model")
if leg_residual_support_only & leg_residual_support_abduction_only:
    parser.error("leg residual joint masks are mutually exclusive")
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
    if bool(task_config.get("placement_reference", {}).get("enabled", False)):
        target_x = float(staircase["start_x_m"]) + 0.35 * float(
            staircase["tread_depth_m"]
        )
        camera_center_x = (approach_start_x + target_x) / 2.0
        set_camera_view(
            eye=[camera_center_x, -1.45, 0.72],
            target=[camera_center_x, 0.0, 0.18],
            camera_prim_path="/OmniverseKit_Persp",
        )
        return
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
    "leg_models": {
        leg: str(path) for leg, path in leg_model_paths.items()
    },
    "leg_base_models": {
        leg: str(path) for leg, path in leg_base_model_paths.items()
    },
    "leg_residual_scales": leg_residual_scales,
    "leg_residual_support_only": sorted(leg_residual_support_only),
    "leg_residual_support_abduction_only": sorted(
        leg_residual_support_abduction_only
    ),
    "world": str(world_path),
    "episodes_requested": args.episodes,
    "seed": args.seed,
    "device": args.device,
    "active_steps": active_steps,
    "maximum_lateral_deviation_override_m": (
        args.maximum_lateral_deviation_m
    ),
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
    known_placement_legs = set(raw_env.placement_sequence_legs)
    unknown_leg_models = sorted(
        (set(leg_model_paths) | set(leg_base_model_paths))
        - known_placement_legs
    )
    if unknown_leg_models:
        raise ValueError(
            "Leg-model mapping is outside the placement sequence: "
            f"{unknown_leg_models}"
        )
    leg_models: dict[str, PPO] = {}
    leg_model_verification: dict[str, dict[str, object]] = {}
    for leg, path in leg_model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_path = Path(str(path) + ".contract.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        model_sha256 = _sha256_file(path)
        if str(manifest.get("model_sha256")) != model_sha256:
            raise RuntimeError(
                f"Per-leg model hash mismatch for {leg}: {path}"
            )
        leg_model = PPO.load(str(path), device=args.device)
        if (
            tuple(leg_model.observation_space.shape)
            != tuple(raw_env.observation_space.shape)
            or tuple(leg_model.action_space.shape)
            != tuple(raw_env.action_space.shape)
        ):
            raise RuntimeError(
                f"Per-leg model spaces do not match the sequence for {leg}"
            )
        leg_models[leg] = leg_model
        leg_model_verification[leg] = {
            "status": "PASS",
            "model": str(path),
            "model_sha256": model_sha256,
            "manifest": str(manifest_path),
            "source_task_id": manifest.get("task_id"),
            "observation_shape": list(leg_model.observation_space.shape),
            "action_shape": list(leg_model.action_space.shape),
        }
    leg_base_models: dict[str, PPO] = {}
    leg_base_model_verification: dict[str, dict[str, object]] = {}
    for leg, path in leg_base_model_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_path = Path(str(path) + ".contract.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        model_sha256 = _sha256_file(path)
        if str(manifest.get("model_sha256")) != model_sha256:
            raise RuntimeError(
                f"Per-leg base model hash mismatch for {leg}: {path}"
            )
        base_model = PPO.load(str(path), device=args.device)
        if (
            tuple(base_model.observation_space.shape)
            != tuple(raw_env.observation_space.shape)
            or tuple(base_model.action_space.shape)
            != tuple(raw_env.action_space.shape)
        ):
            raise RuntimeError(
                f"Per-leg base model spaces do not match for {leg}"
            )
        leg_base_models[leg] = base_model
        leg_base_model_verification[leg] = {
            "status": "PASS",
            "model": str(path),
            "model_sha256": model_sha256,
            "manifest": str(manifest_path),
            "source_task_id": manifest.get("task_id"),
            "residual_scale": leg_residual_scales[leg],
        }
    observation, _ = raw_env.reset(seed=args.seed)
    episode_metrics: list[dict[str, object]] = []
    maximum_steps = args.episodes * raw_env.max_episode_steps * 2
    for _ in range(maximum_steps):
        if raw_env.placement_transfer_active:
            action = np.zeros(raw_env.action_space.shape, dtype=np.float32)
        else:
            active_leg = raw_env.placement_swing_leg
            active_model = leg_models.get(active_leg, model)
            action, _ = active_model.predict(observation, deterministic=True)
            base_model = leg_base_models.get(active_leg)
            if base_model is not None:
                base_action, _ = base_model.predict(
                    observation,
                    deterministic=True,
                )
                action = compose_bounded_residual_action(
                    base_action,
                    action,
                    residual_scale=leg_residual_scales[active_leg],
                    residual_mask=(
                        [
                            1.0
                            if (
                                not name.startswith(f"{active_leg}_")
                                and name.endswith("_hip_abduction")
                            )
                            else 0.0
                            for name in raw_env.dof_names
                        ]
                        if active_leg in leg_residual_support_abduction_only
                        else (
                            [
                                0.0
                                if name.startswith(f"{active_leg}_")
                                else 1.0
                                for name in raw_env.dof_names
                            ]
                            if active_leg in leg_residual_support_only
                            else None
                        )
                    ),
                )
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
            "leg_model_verification": leg_model_verification,
            "leg_base_model_verification": leg_base_model_verification,
            "policy_composition": (
                "per_leg_frozen_base_plus_bounded_residual"
                if leg_base_models
                else (
                    "per_leg_models_with_zero_residual_inter_leg_transfer"
                    if leg_models
                    else "single_model"
                )
            ),
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
