"""Record one deterministic Drobot stair-climbing episode as H.264 MP4."""

from __future__ import annotations

import argparse
import hashlib
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

from _placement_phase_training import FrozenBaseResidualPolicy  # noqa: E402
from _policy_transfer import (  # noqa: E402
    policy_observation_prefix_compatibility,
    predict_with_observation_prefix,
)
from _run_support import (  # noqa: E402
    validate_model_manifest,
    validate_ppo_algorithm_contract,
)
from _stair_rl_contract import (  # noqa: E402
    compose_bounded_residual_action,
    config_for_height_stage,
    expand_compact_masked_action,
    overlay_masked_action,
    placement_policy_action_mask,
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
    "--increment-reset-seeds",
    action="store_true",
    help=(
        "Use base_seed + episode_index on every reset. By default only the "
        "first reset is seeded and later resets continue the same RNG stream, "
        "matching evaluate_stairs_ppo.py."
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
    "--placement-level",
    default=None,
    help="Record one named placement curriculum level instead of the final level.",
)
parser.add_argument(
    "--maximum-lateral-deviation-m",
    type=float,
    default=None,
    help="Override only the recording corridor without changing training config.",
)
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
parser.add_argument(
    "--leg-model",
    action="append",
    default=[],
    metavar="LEG=MODEL",
    help="Use a verified per-leg PPO outside inter-leg transfer.",
)
parser.add_argument(
    "--transfer-model",
    action="append",
    default=[],
    metavar="LEG=MODEL",
    help=(
        "Use a compact nine-output support-joint PPO during the inter-leg "
        "transfer into LEG."
    ),
)
parser.add_argument(
    "--post-transfer-model",
    action="append",
    default=[],
    metavar="LEG=MODEL",
    help=(
        "Use a compact nine-output support-joint PPO only during the "
        "configured post-transfer hold into LEG."
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
    "--leg-base-swing-only",
    action="append",
    default=[],
    metavar="LEG",
    help="Mask a frozen per-leg base model to the swing joints.",
)
parser.add_argument(
    "--leg-base-residual-model",
    action="append",
    default=[],
    metavar="LEG=MODEL",
    help=(
        "Second frozen compact swing policy composed over --leg-base-model "
        "before the mapped support policy."
    ),
)
parser.add_argument(
    "--leg-base-residual-scale",
    action="append",
    default=[],
    metavar="LEG=SCALE",
    help="Bounded scale for each --leg-base-residual-model.",
)
parser.add_argument(
    "--leg-base-residual-swing-only",
    action="append",
    default=[],
    metavar="LEG",
    help="Mask the second frozen base residual to the named swing leg.",
)
parser.add_argument(
    "--leg-base-residual-compact-action",
    action="append",
    default=[],
    metavar="LEG",
    help="Expand the second frozen base residual from three swing outputs.",
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
    help="Mask the mapped policy action off the named swing leg's joints.",
)
parser.add_argument(
    "--leg-residual-swing-only",
    action="append",
    default=[],
    metavar="LEG",
    help="Apply a mapped policy only to the named swing leg's joints.",
)
parser.add_argument(
    "--leg-residual-support-abduction-only",
    action="append",
    default=[],
    metavar="LEG",
    help="Apply a mapped policy only to support hip-abduction joints.",
)
parser.add_argument(
    "--leg-residual-swing-support-abduction",
    action="append",
    default=[],
    metavar="LEG",
    help="Apply a mapped policy to the swing leg and support hip abduction.",
)
parser.add_argument(
    "--leg-compact-action",
    action="append",
    default=[],
    metavar="LEG",
    help="Expand a mapped policy's compact outputs onto its residual mask.",
)
parser.add_argument(
    "--zero-action-leg",
    action="append",
    default=[],
    metavar="LEG",
    help="Use the analytic placement reference without a PPO residual for LEG.",
)
args, _ = parser.parse_known_args()


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _numpy(value) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


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


def _parse_leg_scales(
    values: list[str],
    option_name: str = "--leg-residual-scale",
) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        leg, separator, scale_text = str(value).partition("=")
        if not separator or not leg or not scale_text:
            parser.error(f"{option_name} must use LEG=SCALE syntax")
        if leg in result:
            parser.error(f"duplicate {option_name} for {leg}")
        try:
            scale = float(scale_text)
        except ValueError:
            parser.error(f"invalid residual scale for {leg}: {scale_text}")
        if scale <= 0.0 or scale > 1.0:
            parser.error("leg residual scales must be within (0, 1]")
        result[leg] = scale
    return result


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
transfer_policy_post_hold_seconds = float(
    task_config.get("placement_reference", {})
    .get("inter_leg_transfer", {})
    .get("policy_post_hold_seconds", 0.0)
)
if transfer_policy_post_hold_seconds < 0.0:
    parser.error("transfer policy_post_hold_seconds cannot be negative")
transfer_policy_post_hold_steps = int(
    round(
        transfer_policy_post_hold_seconds
        * float(task_config["control_hz"])
    )
)
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
leg_model_paths = _parse_leg_models(args.leg_model, "--leg-model")
transfer_model_paths = _parse_leg_models(
    args.transfer_model,
    "--transfer-model",
)
post_transfer_model_paths = _parse_leg_models(
    args.post_transfer_model,
    "--post-transfer-model",
)
leg_base_model_paths = _parse_leg_models(
    args.leg_base_model,
    "--leg-base-model",
)
leg_base_residual_model_paths = _parse_leg_models(
    args.leg_base_residual_model,
    "--leg-base-residual-model",
)
leg_base_residual_scales = _parse_leg_scales(
    args.leg_base_residual_scale,
    "--leg-base-residual-scale",
)
leg_base_residual_swing_only = set(args.leg_base_residual_swing_only)
leg_base_residual_compact_actions = set(
    args.leg_base_residual_compact_action
)
leg_base_swing_only = set(args.leg_base_swing_only)
if len(leg_base_swing_only) != len(args.leg_base_swing_only):
    parser.error("duplicate --leg-base-swing-only leg")
leg_residual_scales = _parse_leg_scales(args.leg_residual_scale)
leg_residual_support_only = set(args.leg_residual_support_only)
if len(leg_residual_support_only) != len(args.leg_residual_support_only):
    parser.error("duplicate --leg-residual-support-only leg")
leg_residual_swing_only = set(args.leg_residual_swing_only)
if len(leg_residual_swing_only) != len(args.leg_residual_swing_only):
    parser.error("duplicate --leg-residual-swing-only leg")
leg_residual_support_abduction_only = set(
    args.leg_residual_support_abduction_only
)
leg_residual_swing_support_abduction = set(
    args.leg_residual_swing_support_abduction
)
zero_action_legs = set(args.zero_action_leg)
if len(zero_action_legs) != len(args.zero_action_leg):
    parser.error("duplicate --zero-action-leg leg")
leg_compact_actions = set(args.leg_compact_action)
if len(leg_compact_actions) != len(args.leg_compact_action):
    parser.error("duplicate --leg-compact-action leg")
if set(leg_base_model_paths) != set(leg_residual_scales):
    parser.error(
        "--leg-base-model and --leg-residual-scale must select the same legs"
    )
if not leg_base_swing_only.issubset(leg_base_model_paths):
    parser.error("swing-only base masks require --leg-base-model")
if set(leg_base_residual_model_paths) != set(leg_base_residual_scales):
    parser.error(
        "--leg-base-residual-model and --leg-base-residual-scale must "
        "select the same legs"
    )
if set(leg_base_residual_model_paths) != leg_base_residual_swing_only:
    parser.error(
        "each leg base residual requires --leg-base-residual-swing-only"
    )
if not leg_base_residual_compact_actions.issubset(
    leg_base_residual_model_paths
):
    parser.error(
        "base residual compact expansion requires a base residual model"
    )
if not set(leg_base_residual_model_paths).issubset(leg_base_model_paths):
    parser.error("each leg base residual requires --leg-base-model")
if not set(leg_base_model_paths).issubset(leg_model_paths):
    parser.error("each leg base model requires a residual --leg-model")
if not leg_residual_support_only.issubset(leg_model_paths):
    parser.error("support-only action masks require --leg-model")
if not leg_residual_swing_only.issubset(leg_model_paths):
    parser.error("swing-only action masks require --leg-model")
if not leg_residual_support_abduction_only.issubset(leg_model_paths):
    parser.error("support-abduction action masks require --leg-model")
if not leg_residual_swing_support_abduction.issubset(leg_model_paths):
    parser.error("swing/support-abduction action masks require --leg-model")
if not leg_compact_actions.issubset(leg_model_paths):
    parser.error("compact action expansion requires --leg-model")
mask_sets = (
    leg_residual_support_only,
    leg_residual_swing_only,
    leg_residual_support_abduction_only,
    leg_residual_swing_support_abduction,
)
if not leg_compact_actions.issubset(set().union(*mask_sets)):
    parser.error("compact action expansion requires a residual joint mask")
if any(
    left & right
    for index, left in enumerate(mask_sets)
    for right in mask_sets[index + 1 :]
):
    parser.error("leg residual joint masks are mutually exclusive")
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
    "leg_models": {leg: str(path) for leg, path in leg_model_paths.items()},
    "transfer_models": {
        leg: str(path) for leg, path in transfer_model_paths.items()
    },
    "post_transfer_models": {
        leg: str(path) for leg, path in post_transfer_model_paths.items()
    },
    "transfer_policy_post_hold_seconds": (
        transfer_policy_post_hold_seconds
    ),
    "transfer_policy_post_hold_steps": transfer_policy_post_hold_steps,
    "leg_base_models": {
        leg: str(path) for leg, path in leg_base_model_paths.items()
    },
    "leg_base_swing_only": sorted(leg_base_swing_only),
    "leg_base_residual_models": {
        leg: str(path)
        for leg, path in leg_base_residual_model_paths.items()
    },
    "leg_base_residual_scales": leg_base_residual_scales,
    "leg_base_residual_swing_only": sorted(
        leg_base_residual_swing_only
    ),
    "leg_base_residual_compact_actions": sorted(
        leg_base_residual_compact_actions
    ),
    "leg_residual_scales": leg_residual_scales,
    "leg_residual_support_only": sorted(leg_residual_support_only),
    "leg_residual_swing_only": sorted(leg_residual_swing_only),
    "leg_residual_support_abduction_only": sorted(
        leg_residual_support_abduction_only
    ),
    "leg_residual_swing_support_abduction": sorted(
        leg_residual_swing_support_abduction
    ),
    "zero_action_legs": sorted(zero_action_legs),
    "leg_compact_actions": sorted(leg_compact_actions),
    "world": str(world_path),
    "seed": args.seed,
    "deterministic": not args.stochastic,
    "policy_seed": policy_seed,
    "skip_policy_samples": args.skip_policy_samples,
    "skip_episodes": args.skip_episodes,
    "reset_seed_mode": (
        "increment_each_episode"
        if args.increment_reset_seeds
        else "seed_once_then_continue"
    ),
    "search_success_episodes": args.search_success_episodes,
    "device": args.device,
    "active_steps": active_steps,
    "placement_level": args.placement_level,
    "maximum_lateral_deviation_override_m": (
        args.maximum_lateral_deviation_m
    ),
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
    if args.placement_level is not None:
        raw_env.set_placement_level(
            args.placement_level,
            activate_immediately=True,
        )
    verification = validate_model_manifest(
        model_path=model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=world_dependency_paths,
        environment_contract=raw_env.contract,
        allow_unverified=args.allow_unverified_model,
    )
    # Keep the default model unbound so a compact phase policy can provide
    # the exact manifest/config contract while explicit per-leg mappings
    # expand its outputs before the raw 12-joint environment is stepped.
    model = PPO.load(str(model_path), device=args.device)
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
        (
            set(leg_model_paths)
            | set(leg_base_model_paths)
            | set(leg_base_residual_model_paths)
            | set(transfer_model_paths)
            | set(post_transfer_model_paths)
            | zero_action_legs
        )
        - known_placement_legs
    )
    if unknown_leg_models:
        raise ValueError(
            "Leg-model mapping is outside the placement sequence: "
            f"{unknown_leg_models}"
        )
    default_policy_legs = (
        known_placement_legs - set(leg_model_paths) - zero_action_legs
    )
    if (
        tuple(model.action_space.shape) != tuple(raw_env.action_space.shape)
        and default_policy_legs
    ):
        raise RuntimeError(
            "Compact default model cannot control unmapped placement legs: "
            f"{sorted(default_policy_legs)}"
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
            raise RuntimeError(f"Per-leg model hash mismatch for {leg}: {path}")
        leg_model = PPO.load(str(path), device=args.device)
        if leg in leg_residual_swing_support_abduction:
            compact_mode = "swing_plus_support_abduction"
        elif leg in leg_residual_swing_only:
            compact_mode = "swing_only"
        elif leg in leg_residual_support_abduction_only:
            compact_mode = "support_abduction_only"
        else:
            compact_mode = "support_only"
        expected_action_size = raw_env.action_space.shape[0]
        if leg in leg_compact_actions:
            expected_action_size = int(
                np.count_nonzero(
                    placement_policy_action_mask(
                        raw_env.dof_names,
                        target_leg=leg,
                        mode=compact_mode,
                    )
                )
            )
        if tuple(leg_model.action_space.shape) != (expected_action_size,):
            raise RuntimeError(
                f"Per-leg action space does not match the sequence for {leg}"
            )
        observation_compatibility = policy_observation_prefix_compatibility(
            leg_model,
            manifest,
            raw_env.contract,
        )
        leg_models[leg] = leg_model
        leg_model_verification[leg] = {
            "status": "PASS",
            "model": str(path),
            "model_sha256": model_sha256,
            "manifest": str(manifest_path),
            "source_task_id": manifest.get("task_id"),
            "observation_compatibility": observation_compatibility,
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
        if tuple(base_model.action_space.shape) != tuple(raw_env.action_space.shape):
            raise RuntimeError(
                f"Per-leg base model action space does not match for {leg}"
            )
        observation_compatibility = policy_observation_prefix_compatibility(
            base_model,
            manifest,
            raw_env.contract,
        )
        leg_base_models[leg] = base_model
        leg_base_model_verification[leg] = {
            "status": "PASS",
            "model": str(path),
            "model_sha256": model_sha256,
            "manifest": str(manifest_path),
            "source_task_id": manifest.get("task_id"),
            "residual_scale": leg_residual_scales[leg],
            "observation_compatibility": observation_compatibility,
        }
    leg_composed_base_policies: dict[str, FrozenBaseResidualPolicy] = {}
    leg_base_residual_model_verification: dict[str, dict[str, object]] = {}
    for leg, path in leg_base_residual_model_paths.items():
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
                f"Per-leg base residual model hash mismatch for {leg}: {path}"
            )
        residual_model = PPO.load(str(path), device=args.device)
        residual_mask = placement_policy_action_mask(
            raw_env.dof_names,
            target_leg=leg,
            mode="swing_only",
        )
        expected_action_size = (
            int(np.count_nonzero(residual_mask))
            if leg in leg_base_residual_compact_actions
            else raw_env.action_space.shape[0]
        )
        if tuple(residual_model.action_space.shape) != (expected_action_size,):
            raise RuntimeError(
                "Per-leg base residual action space does not match for "
                f"{leg}: {residual_model.action_space.shape}"
            )
        observation_compatibility = policy_observation_prefix_compatibility(
            residual_model,
            manifest,
            raw_env.contract,
        )
        base_mask = (
            placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=leg,
                mode="swing_only",
            )
            if leg in leg_base_swing_only
            else None
        )
        leg_composed_base_policies[leg] = FrozenBaseResidualPolicy(
            base_policy=leg_base_models[leg],
            residual_policy=residual_model,
            action_space=raw_env.action_space,
            residual_scale=leg_base_residual_scales[leg],
            base_mask=base_mask,
            residual_mask=residual_mask,
            compact_residual_action=(
                leg in leg_base_residual_compact_actions
            ),
        )
        leg_base_residual_model_verification[leg] = {
            "status": "PASS",
            "model": str(path),
            "model_sha256": model_sha256,
            "manifest": str(manifest_path),
            "source_task_id": manifest.get("task_id"),
            "residual_scale": leg_base_residual_scales[leg],
            "residual_mask": "swing_only",
            "compact_action": leg in leg_base_residual_compact_actions,
            "observation_compatibility": observation_compatibility,
        }
    transfer_models: dict[str, PPO] = {}
    transfer_model_verification: dict[str, dict[str, object]] = {}
    for leg, path in transfer_model_paths.items():
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
                f"Transfer model hash mismatch for {leg}: {path}"
            )
        transfer_model = PPO.load(str(path), device=args.device)
        support_mask = placement_policy_action_mask(
            raw_env.dof_names,
            target_leg=leg,
            mode="support_only",
        )
        expected_action_size = int(np.count_nonzero(support_mask))
        if tuple(transfer_model.action_space.shape) != (expected_action_size,):
            raise RuntimeError(
                f"Transfer action space does not match support joints for {leg}"
            )
        observation_compatibility = policy_observation_prefix_compatibility(
            transfer_model,
            manifest,
            raw_env.contract,
        )
        transfer_models[leg] = transfer_model
        transfer_model_verification[leg] = {
            "status": "PASS",
            "model": str(path),
            "model_sha256": model_sha256,
            "manifest": str(manifest_path),
            "source_task_id": manifest.get("task_id"),
            "observation_shape": list(transfer_model.observation_space.shape),
            "action_shape": list(transfer_model.action_space.shape),
            "residual_mask": "support_only",
            "policy_post_hold_seconds": (
                transfer_policy_post_hold_seconds
            ),
            "observation_compatibility": observation_compatibility,
        }

    post_transfer_models: dict[str, PPO] = {}
    post_transfer_model_verification: dict[str, dict[str, object]] = {}
    for leg, path in post_transfer_model_paths.items():
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
                f"Post-transfer model hash mismatch for {leg}: {path}"
            )
        support_model = PPO.load(str(path), device=args.device)
        support_mask = placement_policy_action_mask(
            raw_env.dof_names,
            target_leg=leg,
            mode="support_only",
        )
        expected_action_size = int(np.count_nonzero(support_mask))
        if tuple(support_model.action_space.shape) != (expected_action_size,):
            raise RuntimeError(
                f"Post-transfer action space does not match supports for {leg}"
            )
        observation_compatibility = policy_observation_prefix_compatibility(
            support_model,
            manifest,
            raw_env.contract,
        )
        post_transfer_models[leg] = support_model
        post_transfer_model_verification[leg] = {
            "status": "PASS",
            "model": str(path),
            "model_sha256": model_sha256,
            "manifest": str(manifest_path),
            "source_task_id": manifest.get("task_id"),
            "observation_shape": list(support_model.observation_space.shape),
            "action_shape": list(support_model.action_space.shape),
            "residual_mask": "support_only",
            "policy_post_hold_seconds": transfer_policy_post_hold_seconds,
            "observation_compatibility": observation_compatibility,
        }

    transfer_hold_state: dict[str, object] = {
        "leg": None,
        "steps_remaining": 0,
        "applied": False,
    }

    def transfer_support_action(
        support_models: dict[str, PPO],
        leg: str,
        policy_observation: np.ndarray,
    ) -> np.ndarray:
        transfer_model = support_models[leg]
        compact_action, _ = predict_with_observation_prefix(
            transfer_model,
            policy_observation,
            deterministic=not args.stochastic,
        )
        return expand_compact_masked_action(
            compact_action,
            placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=leg,
                mode="support_only",
            ),
        )

    def apply_transfer_hold(
        action: np.ndarray,
        policy_observation: np.ndarray,
        active_leg: str,
    ) -> np.ndarray:
        hold_applied = bool(
            transfer_hold_state["leg"] == active_leg
            and int(transfer_hold_state["steps_remaining"]) > 0
        )
        transfer_hold_state["applied"] = hold_applied
        if not hold_applied:
            return np.asarray(action, dtype=np.float32)
        return overlay_masked_action(
            action,
            transfer_support_action(
                post_transfer_models,
                active_leg,
                policy_observation,
            ),
            placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=active_leg,
                mode="support_only",
            ),
        )

    def advance_transfer_hold(info: dict) -> None:
        transfer_event = info.get("placement_transfer_completed_event")
        if (
            transfer_event
            and raw_env.placement_swing_leg in post_transfer_models
            and transfer_policy_post_hold_steps > 0
        ):
            transfer_hold_state["leg"] = raw_env.placement_swing_leg
            transfer_hold_state["steps_remaining"] = (
                transfer_policy_post_hold_steps
            )
        elif bool(transfer_hold_state["applied"]):
            remaining = int(transfer_hold_state["steps_remaining"]) - 1
            transfer_hold_state["steps_remaining"] = remaining
            if remaining <= 0:
                transfer_hold_state["leg"] = None
        transfer_hold_state["applied"] = False

    def policy_action(observation: np.ndarray) -> np.ndarray:
        if raw_env.placement_transfer_active:
            transfer_hold_state["applied"] = False
            active_leg = raw_env.placement_swing_leg
            transfer_model = transfer_models.get(active_leg)
            if transfer_model is None:
                return np.zeros(raw_env.action_space.shape, dtype=np.float32)
            return transfer_support_action(
                transfer_models,
                active_leg,
                observation,
            )
        active_leg = raw_env.placement_swing_leg
        if active_leg in zero_action_legs:
            action = np.zeros(raw_env.action_space.shape, dtype=np.float32)
        else:
            active_model = leg_models.get(active_leg, model)
            action, _ = predict_with_observation_prefix(
                active_model,
                observation,
                deterministic=not args.stochastic,
            )
        residual_mask = None
        if active_leg in leg_residual_swing_support_abduction:
            residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=active_leg,
                mode="swing_plus_support_abduction",
            )
        elif active_leg in leg_residual_swing_only:
            residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=active_leg,
                mode="swing_only",
            )
        elif active_leg in leg_residual_support_abduction_only:
            residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=active_leg,
                mode="support_abduction_only",
            )
        elif active_leg in leg_residual_support_only:
            residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=active_leg,
                mode="support_only",
            )
        if active_leg in leg_compact_actions:
            if residual_mask is None:
                raise RuntimeError(
                    f"Compact action for {active_leg} has no residual mask"
                )
            action = expand_compact_masked_action(action, residual_mask)
        base_model = leg_composed_base_policies.get(
            active_leg,
            leg_base_models.get(active_leg),
        )
        if base_model is None:
            direct_action = np.asarray(action, dtype=np.float32)
            if residual_mask is not None:
                direct_action = direct_action * np.asarray(
                    residual_mask,
                    dtype=np.float32,
                )
            return apply_transfer_hold(
                direct_action,
                observation,
                active_leg,
            )
        base_action, _ = predict_with_observation_prefix(
            base_model,
            observation,
            deterministic=True,
        )
        if (
            active_leg in leg_base_swing_only
            and active_leg not in leg_composed_base_policies
        ):
            base_action = np.asarray(
                base_action,
                dtype=np.float32,
            ) * placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=active_leg,
                mode="swing_only",
            )
        return apply_transfer_hold(
            compose_bounded_residual_action(
                base_action,
                action,
                residual_scale=leg_residual_scales[active_leg],
                residual_mask=residual_mask,
            ),
            observation,
            active_leg,
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
        placement_view = bool(
            task_config.get("placement_reference", {}).get("enabled", False)
        )
        if placement_view:
            target_x = float(staircase["start_x_m"]) + 0.35 * float(
                staircase["tread_depth_m"]
            )
            camera_center_x = (approach_start_x + target_x) / 2.0
            camera_eye = [camera_center_x, -1.45, 0.72]
            camera_target = [camera_center_x, 0.0, 0.18]
        else:
            landing_center_x = (
                float(staircase["start_x_m"])
                + int(staircase["step_count"])
                * float(staircase["tread_depth_m"])
                + float(staircase["top_platform_depth_m"]) / 2.0
            )
            camera_center_x = (approach_start_x + landing_center_x) / 2.0
            top_height = (
                int(staircase["step_count"])
                * float(staircase["rise_m"])
            )
            camera_eye = [camera_center_x, -2.65, top_height + 0.82]
            camera_target = [camera_center_x, 0.0, top_height + 0.10]
        camera_path = "/OmniverseKit_Persp"
        set_camera_view(
            eye=camera_eye,
            target=camera_target,
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

    def reset_for_episode(reset_index: int) -> tuple[np.ndarray, dict]:
        reset_seed = (
            args.seed + reset_index
            if args.increment_reset_seeds
            else (args.seed if reset_index == 0 else None)
        )
        transfer_hold_state["leg"] = None
        transfer_hold_state["steps_remaining"] = 0
        transfer_hold_state["applied"] = False
        return raw_env.reset(seed=reset_seed)

    for skipped_index in range(args.skip_episodes):
        skipped_seed = (
            args.seed + skipped_index
            if args.increment_reset_seeds
            else (args.seed if skipped_index == 0 else None)
        )
        observation, _ = reset_for_episode(skipped_index)
        skipped_metrics: dict[str, object] | None = None
        for _ in range(raw_env.max_episode_steps):
            action = policy_action(observation)
            observation, _, terminated, truncated, info = raw_env.step(action)
            advance_transfer_hold(info)
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
    selected_reset_index: int | None = None
    selected_episode_seed: int | None = None
    episode_metrics: dict[str, object] | None = None
    selected_frames: list[np.ndarray] | None = None
    trajectory_observations: list[np.ndarray] = []
    trajectory_actions: list[np.ndarray] = []
    for attempt_index in range(attempt_limit):
        reset_index = args.skip_episodes + attempt_index
        episode_seed = (
            args.seed + reset_index
            if args.increment_reset_seeds
            else (args.seed if reset_index == 0 else None)
        )
        observation, _ = reset_for_episode(reset_index)
        candidate_frames: list[np.ndarray] = []
        candidate_observations: list[np.ndarray] = []
        candidate_actions: list[np.ndarray] = []
        candidate_metrics: dict[str, object] | None = None
        for control_step in range(raw_env.max_episode_steps):
            action = policy_action(observation)
            candidate_observations.append(observation.copy())
            candidate_actions.append(
                np.asarray(action, dtype=np.float32).reshape(12).copy()
            )
            observation, _, terminated, truncated, info = raw_env.step(action)
            advance_transfer_hold(info)
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
        selected_reset_index = reset_index + 1
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
            "leg_model_verification": leg_model_verification,
            "leg_base_model_verification": leg_base_model_verification,
            "transfer_model_verification": transfer_model_verification,
            "post_transfer_model_verification": (
                post_transfer_model_verification
            ),
            "leg_base_residual_model_verification": (
                leg_base_residual_model_verification
            ),
            "policy_composition": (
                "per_leg_models_with_bounded_residuals"
                if leg_base_models
                else (
                    "per_leg_models_with_compact_support_transfer_policy"
                    if transfer_models or post_transfer_models
                    else (
                        "per_leg_models_with_zero_inter_leg_transfer"
                        if leg_models
                        else "single_model"
                    )
                )
            ),
            "recorded_frames": recorded_frames,
            "episode": episode_metrics,
            "selected_episode_index": selected_episode_index,
            "selected_reset_index": selected_reset_index,
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
