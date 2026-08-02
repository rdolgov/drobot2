"""Train a separate PPO policy for climbing the Drobot stair world."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import deque
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

from _policy_transfer import (  # noqa: E402
    EXPANDABLE_INPUT_WEIGHTS,
    physical_action_output_ratios,
    policy_observation_prefix_compatibility,
    transfer_policy_state,
)
from _run_support import (  # noqa: E402
    build_model_manifest,
    expected_ppo_algorithm_contract,
    file_hash_records,
    model_manifest_path,
    read_model_manifest,
    sha256_file,
    validate_ppo_algorithm_contract,
    write_model_manifest,
)
from _stair_rl_contract import (  # noqa: E402
    config_for_height_stage,
    placement_policy_action_mask,
    progress_gate_failures,
    stair_observation_fields,
)
from _vl53l5cx_contract import (  # noqa: E402
    VL53L5CX_MODE,
    validate_vl53l5cx_config,
    vl53l5cx_observation_fields,
)

parser = argparse.ArgumentParser(
    description="Train the separate Drobot stair-climbing PPO policy."
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
    "--fixed-active-steps",
    type=int,
    default=None,
    help="Pin training to one declared stair count instead of advancing mastery.",
)
parser.add_argument(
    "--placement-start-level",
    default=None,
    help="Start a mastery run at one named placement curriculum level.",
)
parser.add_argument(
    "--fixed-placement-level",
    default=None,
    help="Pin every training episode to one placement curriculum level.",
)
parser.add_argument(
    "--output-dir",
    default="simulation/isaac/output/rl/ppo-stairs-v1",
)
parser.add_argument("--total-timesteps", type=int, default=None)
parser.add_argument(
    "--curriculum-total-timesteps",
    type=int,
    default=None,
    help=(
        "Optional timestep horizon for curriculum fractions. Use the explicit "
        "small-run budget to exercise every stage without changing PPO defaults."
    ),
)
parser.add_argument("--seed", type=int, default=142)
parser.add_argument("--device", default="cpu")
parser.add_argument(
    "--resume",
    default=None,
    help="Resume a stairs PPO model with a matching .contract.json manifest.",
)
parser.add_argument(
    "--initialize-from-flat",
    default=None,
    help=(
        "Initialize a new 57-input stairs policy from a 48-input flat-walk "
        "policy. Policy weights transfer; optimizer state does not."
    ),
)
parser.add_argument(
    "--initialize-from-stairs",
    default=None,
    help=(
        "Initialize a new same-shape stair policy from another stair model. "
        "Policy parameters transfer; optimizer state does not."
    ),
)
parser.add_argument(
    "--initialize-from-balance",
    default=None,
    help=(
        "Initialize from the 56-input unsupported foot-balance policy. Only "
        "the shared 48-value proprioceptive prefix transfers; skill/terrain "
        "columns start at zero."
    ),
)
parser.add_argument(
    "--allow-unverified-resume",
    action="store_true",
    help="Allow a deliberate resume from a stairs model without its manifest.",
)
parser.add_argument(
    "--smoke-test",
    action="store_true",
    help="Run a 512-step end-to-end pipeline check; not convergence evidence.",
)
parser.add_argument(
    "--initialize-only",
    action="store_true",
    help=(
        "Save the initialized/transferred policy and a current manifest "
        "without collecting a PPO rollout."
    ),
)
parser.add_argument(
    "--ppo-learning-rate",
    type=float,
    default=None,
    help="Optional run-local PPO learning-rate override.",
)
parser.add_argument(
    "--ppo-initial-log-std",
    type=float,
    default=None,
    help="Optional run-local initial policy log-standard-deviation override.",
)
parser.add_argument(
    "--ppo-entropy-coefficient",
    type=float,
    default=None,
    help="Optional run-local PPO entropy coefficient override.",
)
parser.add_argument(
    "--phase-train-leg",
    default=None,
    help=(
        "Train only this placement leg after replaying every earlier leg and "
        "inter-leg transfer with deterministic verified policies."
    ),
)
parser.add_argument(
    "--phase-train-transfer",
    action="store_true",
    help=(
        "Train the inter-leg COM/support transfer into --phase-train-leg "
        "instead of the target leg's swing/landing phase."
    ),
)
parser.add_argument(
    "--phase-transfer-post-hold-seconds",
    type=float,
    default=None,
    help=(
        "After a learned transfer gate opens, keep the support policy active "
        "for this many target-leg seconds before awarding success. Defaults "
        "to placement_reference.inter_leg_transfer.policy_post_hold_seconds."
    ),
)
parser.add_argument(
    "--phase-post-transfer-hold-only",
    action="store_true",
    help=(
        "Keep PPO actions at zero during the analytic transfer and train the "
        "support policy only after the transfer gate opens."
    ),
)
parser.add_argument(
    "--precursor-leg-model",
    action="append",
    default=[],
    metavar="LEG=MODEL",
    help=(
        "Deterministic policy used to replay an earlier placement leg; repeat "
        "for every leg before --phase-train-leg."
    ),
)
parser.add_argument(
    "--phase-base-model",
    default=None,
    help=(
        "Frozen target-leg base policy; PPO learns a bounded corrective "
        "residual instead of replacing its action."
    ),
)
parser.add_argument(
    "--phase-base-swing-only",
    action="store_true",
    help="Mask the frozen phase base policy to the target swing leg.",
)
parser.add_argument(
    "--phase-base-residual-model",
    default=None,
    help=(
        "Second frozen policy composed as a bounded residual over "
        "--phase-base-model before the trainable PPO correction."
    ),
)
parser.add_argument(
    "--phase-base-residual-scale",
    type=float,
    default=0.5,
)
parser.add_argument(
    "--phase-base-residual-mode",
    choices=(
        "swing_only",
        "support_only",
        "support_abduction_only",
        "swing_plus_support_abduction",
    ),
    default=None,
    help="Joint mask used by --phase-base-residual-model.",
)
parser.add_argument(
    "--phase-base-residual-compact-action",
    action="store_true",
    help="Expand the second frozen policy from its compact masked action.",
)
parser.add_argument("--phase-residual-scale", type=float, default=0.25)
parser.add_argument(
    "--phase-residual-support-only",
    action="store_true",
    help=(
        "Mask PPO action off the target swing leg's three joints, with or "
        "without a frozen base policy."
    ),
)
parser.add_argument(
    "--phase-residual-swing-only",
    action="store_true",
    help="Apply PPO corrections only to the target swing leg's three joints.",
)
parser.add_argument(
    "--phase-residual-support-abduction-only",
    action="store_true",
    help="Apply PPO corrections only to support-leg hip-abduction joints.",
)
parser.add_argument(
    "--phase-residual-swing-support-abduction",
    action="store_true",
    help=(
        "Apply PPO to all three swing-leg joints and only hip abduction on "
        "the support legs."
    ),
)
parser.add_argument(
    "--phase-compact-residual-action",
    action="store_true",
    help=(
        "Expose only the active residual-mask joints to PPO, then expand the "
        "compact action into the robot's 12-joint command."
    ),
)
parser.add_argument("--phase-reset-attempts", type=int, default=8)
parser.add_argument("--phase-precursor-max-steps", type=int, default=1800)
parser.add_argument(
    "--phase-disable-snapshot-cache",
    action="store_true",
    help=(
        "Replay the physical precursor on every phase reset so PPO trains "
        "across live transfer-state variation."
    ),
)
parser.add_argument(
    "--phase-snapshot",
    default=None,
    help=(
        "Verified JSON simulator boundary used instead of replaying phase "
        "precursor policies."
    ),
)
parser.add_argument("--gui", action="store_true")
args, _ = parser.parse_known_args()
if args.placement_start_level and args.fixed_placement_level:
    parser.error(
        "--placement-start-level and --fixed-placement-level are mutually exclusive"
    )


def _resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _parse_leg_model_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        leg, separator, model = str(value).partition("=")
        if not separator or not leg or not model:
            parser.error("--precursor-leg-model must use LEG=MODEL syntax")
        if leg in result:
            parser.error(f"duplicate precursor model for {leg}")
        result[leg] = _resolve_project_path(model)
    return result


initialization_options = (
    args.resume,
    args.initialize_from_flat,
    args.initialize_from_stairs,
    args.initialize_from_balance,
)
if sum(value is not None for value in initialization_options) > 1:
    parser.error(
        "--resume, --initialize-from-flat, --initialize-from-stairs, and "
        "--initialize-from-balance are mutually exclusive"
    )
if args.total_timesteps is not None and args.total_timesteps <= 0:
    parser.error("--total-timesteps must be positive")
if args.ppo_learning_rate is not None and args.ppo_learning_rate <= 0.0:
    parser.error("--ppo-learning-rate must be positive")
if (
    args.ppo_entropy_coefficient is not None
    and args.ppo_entropy_coefficient < 0.0
):
    parser.error("--ppo-entropy-coefficient cannot be negative")
if args.smoke_test and args.initialize_only:
    parser.error("--smoke-test and --initialize-only are mutually exclusive")
if args.phase_reset_attempts < 1:
    parser.error("--phase-reset-attempts must be positive")
if args.phase_precursor_max_steps < 1:
    parser.error("--phase-precursor-max-steps must be positive")
precursor_leg_model_paths = _parse_leg_model_paths(args.precursor_leg_model)
phase_snapshot_path = (
    _resolve_project_path(args.phase_snapshot) if args.phase_snapshot else None
)
if phase_snapshot_path is not None and not args.phase_train_leg:
    parser.error("--phase-snapshot requires --phase-train-leg")
if phase_snapshot_path is not None and args.phase_disable_snapshot_cache:
    parser.error("--phase-snapshot requires phase snapshot caching")
if precursor_leg_model_paths and not args.phase_train_leg:
    parser.error("--precursor-leg-model requires --phase-train-leg")
phase_base_model_path = (
    _resolve_project_path(args.phase_base_model)
    if args.phase_base_model
    else None
)
phase_base_residual_model_path = (
    _resolve_project_path(args.phase_base_residual_model)
    if args.phase_base_residual_model
    else None
)
if phase_base_model_path is not None and not args.phase_train_leg:
    parser.error("--phase-base-model requires --phase-train-leg")
if args.phase_base_swing_only and phase_base_model_path is None:
    parser.error("--phase-base-swing-only requires --phase-base-model")
if phase_base_residual_model_path is not None and phase_base_model_path is None:
    parser.error(
        "--phase-base-residual-model requires --phase-base-model"
    )
if phase_base_residual_model_path is not None and (
    args.phase_base_residual_mode is None
):
    parser.error(
        "--phase-base-residual-model requires --phase-base-residual-mode"
    )
if phase_base_residual_model_path is None and (
    args.phase_base_residual_mode is not None
    or args.phase_base_residual_compact_action
):
    parser.error(
        "phase base residual options require --phase-base-residual-model"
    )
if not 0.0 < args.phase_base_residual_scale <= 1.0:
    parser.error("--phase-base-residual-scale must be within (0, 1]")
if args.phase_train_transfer and not args.phase_train_leg:
    parser.error("--phase-train-transfer requires --phase-train-leg")
if args.phase_post_transfer_hold_only and not args.phase_train_transfer:
    parser.error(
        "--phase-post-transfer-hold-only requires --phase-train-transfer"
    )
if (
    args.phase_transfer_post_hold_seconds is not None
    and args.phase_transfer_post_hold_seconds < 0.0
):
    parser.error("--phase-transfer-post-hold-seconds cannot be negative")
if (
    args.phase_transfer_post_hold_seconds is not None
    and not args.phase_train_transfer
):
    parser.error(
        "--phase-transfer-post-hold-seconds requires --phase-train-transfer"
    )
if not 0.0 < args.phase_residual_scale <= 1.0:
    parser.error("--phase-residual-scale must be within (0, 1]")
phase_mask_flags = (
    args.phase_residual_support_only,
    args.phase_residual_swing_only,
    args.phase_residual_support_abduction_only,
    args.phase_residual_swing_support_abduction,
)
if sum(phase_mask_flags) > 1:
    parser.error("phase residual joint masks are mutually exclusive")
if any(phase_mask_flags) and not args.phase_train_leg:
    parser.error("phase residual joint masks require --phase-train-leg")
if args.phase_compact_residual_action and not any(phase_mask_flags):
    parser.error(
        "--phase-compact-residual-action requires a phase residual joint mask"
    )

phase_policy_action_size = 12
if args.phase_compact_residual_action:
    if args.phase_residual_swing_support_abduction:
        phase_policy_action_size = 6
    elif args.phase_residual_support_only:
        phase_policy_action_size = 9
    else:
        # Both swing-only and support-abduction-only expose three joints.
        phase_policy_action_size = 3
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
configured_transfer_post_hold_seconds = float(
    task_config.get("placement_reference", {})
    .get("inter_leg_transfer", {})
    .get("policy_post_hold_seconds", 0.0)
)
phase_transfer_post_hold_seconds = (
    configured_transfer_post_hold_seconds
    if args.phase_transfer_post_hold_seconds is None
    else float(args.phase_transfer_post_hold_seconds)
)
if phase_transfer_post_hold_seconds < 0.0:
    parser.error(
        "placement transfer policy_post_hold_seconds cannot be negative"
    )
if (
    args.phase_post_transfer_hold_only
    and phase_transfer_post_hold_seconds <= 0.0
):
    parser.error(
        "--phase-post-transfer-hold-only requires a positive post hold"
    )
if args.fixed_active_steps is not None:
    maximum_steps = int(task_config["staircase"]["step_count"])
    if args.fixed_active_steps < 1 or args.fixed_active_steps > maximum_steps:
        parser.error(
            f"--fixed-active-steps must be within 1..{maximum_steps}"
        )
    curriculum = dict(task_config["curriculum"])
    curriculum["levels"] = [
        {"start_fraction": 0.0, "active_steps": args.fixed_active_steps}
    ]
    task_config["curriculum"] = curriculum
ppo_config = dict(config["ppo"])
if args.ppo_learning_rate is not None:
    ppo_config["learning_rate"] = float(args.ppo_learning_rate)
if args.ppo_initial_log_std is not None:
    ppo_config["initial_log_std"] = float(args.ppo_initial_log_std)
if args.ppo_entropy_coefficient is not None:
    ppo_config["entropy_coefficient"] = float(
        args.ppo_entropy_coefficient
    )
initial_action_bias = ppo_config.get("initial_action_bias")
if initial_action_bias is not None:
    if not isinstance(initial_action_bias, list):
        parser.error("ppo.initial_action_bias must be a list")
    initial_action_bias = [float(value) for value in initial_action_bias]
    if len(initial_action_bias) != phase_policy_action_size:
        parser.error(
            "ppo.initial_action_bias length must match the policy action size: "
            f"{len(initial_action_bias)} != {phase_policy_action_size}"
        )
    if any(abs(value) > 1.0 for value in initial_action_bias):
        parser.error("ppo.initial_action_bias values must be within [-1, 1]")
residual_policy_config = dict(task_config.get("residual_policy", {}))
if bool(residual_policy_config.get("enabled", False)) and args.initialize_from_flat:
    parser.error(
        "Residual stair policies use their configured frozen base model and "
        "cannot also use --initialize-from-flat"
    )
terrain_perception_config = dict(
    task_config.get(
        "terrain_perception",
        {"mode": "analytic_height_profile"},
    )
)
terrain_perception_mode = str(
    terrain_perception_config.get("mode", "analytic_height_profile")
)
terrain_observation_fields = None
if terrain_perception_mode == VL53L5CX_MODE:
    try:
        validate_vl53l5cx_config(
            terrain_perception_config,
            control_hz=int(task_config["control_hz"]),
        )
    except ValueError as exc:
        parser.error(str(exc))
    terrain_observation_fields = vl53l5cx_observation_fields(
        terrain_perception_config
    )
elif terrain_perception_mode != "analytic_height_profile":
    parser.error(
        f"Unsupported terrain perception mode: {terrain_perception_mode}"
    )
policy_observation_size = len(
    stair_observation_fields(
        task_config["staircase"]["terrain_sample_offsets_m"],
        include_navigation_observation=bool(
            task_config.get("include_navigation_observation", False)
        ),
        include_foot_progress_observation=bool(
            task_config.get("include_foot_progress_observation", False)
        ),
        include_placement_reference_observation=bool(
            task_config.get("placement_reference", {}).get("enabled", False)
            and task_config.get(
                "include_placement_reference_observation",
                True,
            )
        ),
        include_support_regulation_observation=bool(
            task_config.get("placement_reference", {}).get("enabled", False)
            and task_config.get(
                "include_placement_reference_observation",
                True,
            )
            and task_config.get(
                "include_support_regulation_observation",
                False,
            )
        ),
        terrain_observation_fields=terrain_observation_fields,
    )
)
world_path = _resolve_project_path(args.world or task_config["world"])
world_dependency_paths = tuple(
    _resolve_project_path(value)
    for value in task_config.get("world_dependencies", ())
)
phase_snapshot_payload: dict[str, object] | None = None
initial_phase_snapshot: dict[str, object] | None = None
initial_phase_snapshot_mode: str | None = None
if phase_snapshot_path is not None:
    if not phase_snapshot_path.is_file():
        raise FileNotFoundError(phase_snapshot_path)
    with phase_snapshot_path.open("r", encoding="utf-8") as stream:
        loaded_snapshot = json.load(stream)
    if not isinstance(loaded_snapshot, dict):
        parser.error("--phase-snapshot must contain a JSON object")
    phase_snapshot_payload = loaded_snapshot
    if int(phase_snapshot_payload.get("schema_version", 0)) != 1:
        parser.error("unsupported phase snapshot schema")
    if str(phase_snapshot_payload.get("target_leg")) != str(
        args.phase_train_leg
    ):
        parser.error("phase snapshot target_leg does not match --phase-train-leg")
    expected_snapshot_mode = (
        "inter_leg_transfer" if args.phase_train_transfer else "placement"
    )
    initial_phase_snapshot_mode = str(
        phase_snapshot_payload.get("phase_snapshot_mode", "")
    )
    if initial_phase_snapshot_mode != expected_snapshot_mode:
        parser.error(
            "phase snapshot mode does not match the requested training mode"
        )
    snapshot_sequence = tuple(
        phase_snapshot_payload.get("placement_sequence_legs", ())
    )
    configured_sequence = tuple(
        task_config["placement_reference"]["sequence_legs"]
    )
    if snapshot_sequence != configured_sequence:
        parser.error("phase snapshot placement sequence does not match config")
    immutable_snapshot_values = {
        "stair_rise_m": float(task_config["staircase"]["rise_m"]),
        "stair_tread_depth_m": float(
            task_config["staircase"]["tread_depth_m"]
        ),
        "effort_cap_nm": float(
            task_config["robot_hardware_profile"]["effort_cap_nm"]
        ),
    }
    for label, expected_value in immutable_snapshot_values.items():
        if not np.isclose(
            float(phase_snapshot_payload.get(label, float("nan"))),
            expected_value,
            rtol=0.0,
            atol=1e-9,
        ):
            parser.error(f"phase snapshot {label} does not match config")
    stored_snapshot = phase_snapshot_payload.get("snapshot")
    if not isinstance(stored_snapshot, dict):
        parser.error("phase snapshot payload has no simulator snapshot object")
    initial_phase_snapshot = stored_snapshot
output_dir = _resolve_project_path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
report_path = output_dir / "training_report.json"

if args.smoke_test:
    total_timesteps = args.total_timesteps or 512
    rollout_steps = min(128, total_timesteps)
    batch_size = min(64, rollout_steps)
    epochs = min(2, int(ppo_config["epochs"]))
    checkpoint_frequency = max(128, rollout_steps)
    curriculum_total_timesteps = total_timesteps
else:
    total_timesteps = args.total_timesteps or int(ppo_config["total_timesteps"])
    rollout_steps = int(ppo_config["rollout_steps"])
    batch_size = int(ppo_config["batch_size"])
    epochs = int(ppo_config["epochs"])
    checkpoint_frequency = int(ppo_config["checkpoint_frequency_steps"])
    curriculum_total_timesteps = (
        int(args.curriculum_total_timesteps)
        if args.curriculum_total_timesteps is not None
        else int(ppo_config["total_timesteps"])
    )
if curriculum_total_timesteps <= 0:
    parser.error("--curriculum-total-timesteps must be positive")
if rollout_steps < 2 or batch_size < 2 or rollout_steps % batch_size:
    parser.error("PPO rollout_steps must be divisible by batch_size and both >= 2")
training_mode = "smoke" if args.smoke_test else "full"
algorithm_contract = expected_ppo_algorithm_contract(
    ppo_config,
    training_mode=training_mode,
    rollout_steps=rollout_steps,
    batch_size=batch_size,
    epochs=epochs,
    observation_size=policy_observation_size,
    action_size=phase_policy_action_size,
)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": not args.gui,
        "width": 1280,
        "height": 720,
    }
)

import stable_baselines3  # noqa: E402
from _placement_phase_training import (  # noqa: E402
    FrozenBaseResidualPolicy,
    PlacementPhaseTrainingEnv,
)
from _quadruped_stairs_env import QuadrupedStairsEnv  # noqa: E402
from _run_support import validate_model_manifest  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import (  # noqa: E402
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor  # noqa: E402


class TrainingControlCallback(BaseCallback):
    def __init__(
        self,
        *,
        task_config: dict[str, object],
        total_for_curriculum: int,
        report_path: Path,
        fixed_placement_level: str | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.task_config = task_config
        self.total_for_curriculum = max(1, int(total_for_curriculum))
        self.curriculum_config = dict(task_config["curriculum"])
        self.curriculum_mode = str(
            self.curriculum_config.get("mode", "timesteps")
        )
        if self.curriculum_mode not in {"timesteps", "mastery"}:
            raise ValueError(
                f"Unsupported curriculum mode: {self.curriculum_mode}"
            )
        self.mastery_window = int(
            self.curriculum_config.get("mastery_window_episodes", 40)
        )
        self.mastery_minimum = int(
            self.curriculum_config.get("mastery_minimum_episodes", 20)
        )
        self.mastery_success_rate = float(
            self.curriculum_config.get("mastery_success_rate", 0.70)
        )
        self.level_outcomes: deque[bool] = deque(maxlen=self.mastery_window)
        self.placement_curriculum_config = dict(
            task_config.get("placement_curriculum", {})
        )
        self.placement_curriculum_mode = str(
            self.placement_curriculum_config.get("mode", "timesteps")
        )
        if self.placement_curriculum_mode not in {"timesteps", "mastery"}:
            raise ValueError(
                "Unsupported placement curriculum mode: "
                f"{self.placement_curriculum_mode}"
            )
        self.placement_mastery_successes_required = int(
            self.placement_curriculum_config.get(
                "mastery_successes_per_level",
                2,
            )
        )
        if self.placement_mastery_successes_required < 1:
            raise ValueError(
                "placement mastery_successes_per_level must be positive"
            )
        self.placement_mastery_level_id: str | None = None
        self.placement_mastery_successes = 0
        self.fixed_placement_level = (
            None
            if fixed_placement_level is None
            else str(fixed_placement_level)
        )
        self.watchdog = dict(task_config.get("progress_watchdog", {}))
        self.watchdog_enabled = bool(self.watchdog.get("enabled", False))
        self.report_path = report_path
        self.report_interval_steps = int(
            self.watchdog.get("report_interval_steps", 10000)
        )
        self.completed_episodes = 0
        self.successful_episodes = 0
        self.first_step_climb_episodes = 0
        self.maximum_step_reached = 0
        self.maximum_base_elevation_gain_m = 0.0
        self.last_progress_step = 0
        self.last_report_step = -self.report_interval_steps
        self.initial_gate_evaluated = False
        self.initial_gate_passed = False
        self.aborted = False
        self.abort_reason: str | None = None

    def _raw_env(self) -> QuadrupedStairsEnv:
        return self.training_env.envs[0].unwrapped

    def snapshot(self) -> dict[str, object]:
        climb_rate = (
            self.first_step_climb_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        raw = self._raw_env()
        return {
            "status": "ABORTED" if self.aborted else "RUNNING",
            "timesteps": int(self.num_timesteps),
            "curriculum_mode": self.curriculum_mode,
            "active_step_count": int(raw.active_step_count),
            "pending_active_step_count": int(raw.pending_active_step_count),
            "placement_curriculum_mode": self.placement_curriculum_mode,
            "placement_mastery_level_id": self.placement_mastery_level_id,
            "placement_mastery_successes": self.placement_mastery_successes,
            "placement_mastery_successes_required": (
                self.placement_mastery_successes_required
            ),
            "fixed_placement_level": self.fixed_placement_level,
            "completed_episodes": self.completed_episodes,
            "successful_episodes": self.successful_episodes,
            "first_step_climb_episodes": self.first_step_climb_episodes,
            "first_step_climb_rate": climb_rate,
            "maximum_step_reached": self.maximum_step_reached,
            "maximum_base_elevation_gain_m": (
                self.maximum_base_elevation_gain_m
            ),
            "last_progress_step": self.last_progress_step,
            "initial_gate_evaluated": self.initial_gate_evaluated,
            "initial_gate_passed": self.initial_gate_passed,
            "abort_reason": self.abort_reason,
            "watchdog_config": self.watchdog,
            "curriculum_transitions": list(raw.curriculum_transitions),
        }

    def _write_report(self) -> None:
        climb_rate = (
            self.first_step_climb_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        success_rate = (
            self.successful_episodes / self.completed_episodes
            if self.completed_episodes
            else 0.0
        )
        self.logger.record("stair/completed_episodes", self.completed_episodes)
        self.logger.record("stair/success_rate", success_rate)
        self.logger.record("stair/first_step_climb_rate", climb_rate)
        self.logger.record("stair/maximum_step_reached", self.maximum_step_reached)
        self.logger.record(
            "stair/maximum_base_elevation_gain_m",
            self.maximum_base_elevation_gain_m,
        )
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as stream:
            json.dump(self.snapshot(), stream, indent=2)
            stream.write("\n")

    def _record_episode(self, metrics: dict[str, object]) -> None:
        self.completed_episodes += 1
        active_steps = int(metrics["active_step_count"])
        succeeded = bool(metrics["stairs_completed"])
        highest_step = int(metrics["highest_step_reached"])
        elevation_gain = float(metrics["maximum_base_elevation_gain_m"])
        minimum_elevation = min(
            float(self.watchdog.get("minimum_base_elevation_gain_m", 0.02)),
            (
                active_steps
                * float(self.task_config["staircase"]["rise_m"])
                * 0.5
            ),
        )
        physically_reached_first_step = (
            highest_step >= 1 and elevation_gain >= minimum_elevation
        )
        if succeeded:
            self.successful_episodes += 1
        if physically_reached_first_step:
            self.first_step_climb_episodes += 1
        if highest_step > self.maximum_step_reached or succeeded:
            self.last_progress_step = int(self.num_timesteps)
        self.maximum_step_reached = max(self.maximum_step_reached, highest_step)
        self.maximum_base_elevation_gain_m = max(
            self.maximum_base_elevation_gain_m,
            elevation_gain,
        )

        raw = self._raw_env()
        placement_level_id = metrics.get("placement_curriculum_level")
        if (
            self.fixed_placement_level is None
            and
            self.placement_curriculum_mode == "mastery"
            and placement_level_id is not None
        ):
            placement_level_id = str(placement_level_id)
            if placement_level_id != self.placement_mastery_level_id:
                self.placement_mastery_level_id = placement_level_id
                self.placement_mastery_successes = 0
            if succeeded:
                self.placement_mastery_successes += 1
            if (
                self.placement_mastery_successes
                >= self.placement_mastery_successes_required
            ):
                levels = list(raw.placement_curriculum_levels)
                current_index = next(
                    index
                    for index, level in enumerate(levels)
                    if str(level["id"]) == placement_level_id
                )
                if current_index + 1 < len(levels):
                    next_level = levels[current_index + 1]
                    raw.set_placement_curriculum_progress(
                        float(next_level["start_fraction"])
                    )
                    self.placement_mastery_level_id = str(next_level["id"])
                    self.placement_mastery_successes = 0
                    self.last_progress_step = int(self.num_timesteps)
        elif self.fixed_placement_level is not None:
            raw.set_placement_level(self.fixed_placement_level)
            self.placement_mastery_level_id = self.fixed_placement_level
        if active_steps == raw.active_step_count:
            self.level_outcomes.append(succeeded)
        if self.curriculum_mode != "mastery":
            return
        maximum_steps = int(self.task_config["staircase"]["step_count"])
        if (
            raw.active_step_count < maximum_steps
            and len(self.level_outcomes) >= self.mastery_minimum
        ):
            success_rate = sum(self.level_outcomes) / len(self.level_outcomes)
            if success_rate >= self.mastery_success_rate:
                raw.set_training_level(
                    raw.active_step_count + 1,
                    reason="mastery",
                    evidence={
                        "episodes": len(self.level_outcomes),
                        "success_rate": success_rate,
                        "timesteps": int(self.num_timesteps),
                    },
                )
                self.level_outcomes.clear()
                self.last_progress_step = int(self.num_timesteps)

    def _check_watchdog(self) -> None:
        if not self.watchdog_enabled or self.aborted:
            return
        gate_steps = int(self.watchdog["initial_gate_steps"])
        if not self.initial_gate_evaluated and self.num_timesteps >= gate_steps:
            self.initial_gate_evaluated = True
            minimum_episodes = int(self.watchdog["minimum_completed_episodes"])
            minimum_climbs = int(
                self.watchdog["minimum_first_step_climb_episodes"]
            )
            minimum_rate = float(
                self.watchdog["minimum_first_step_climb_rate"]
            )
            failures = progress_gate_failures(
                completed_episodes=self.completed_episodes,
                first_step_climb_episodes=self.first_step_climb_episodes,
                minimum_completed_episodes=minimum_episodes,
                minimum_first_step_climb_episodes=minimum_climbs,
                minimum_first_step_climb_rate=minimum_rate,
            )
            if failures:
                self.aborted = True
                self.abort_reason = (
                    "No measurable stair/elevation progress at the "
                    f"{gate_steps}-step gate: " + "; ".join(failures)
                )
            else:
                self.initial_gate_passed = True

        if self.initial_gate_passed:
            stagnation_steps = int(self.watchdog["stagnation_abort_steps"])
            if self.num_timesteps - self.last_progress_step >= stagnation_steps:
                self.aborted = True
                self.abort_reason = (
                    "No new stair level or successful episode for "
                    f"{stagnation_steps} training steps"
                )

    def _on_step(self) -> bool:
        raw = self._raw_env()
        if self.curriculum_mode == "timesteps":
            progress = min(
                1.0,
                self.num_timesteps / self.total_for_curriculum,
            )
            raw.set_training_progress(progress)
        for info in self.locals.get("infos", ()):
            metrics = info.get("episode_metrics")
            if isinstance(metrics, dict):
                self._record_episode(metrics)
        self._check_watchdog()
        if (
            self.aborted
            or self.num_timesteps - self.last_report_step
            >= self.report_interval_steps
        ):
            self._write_report()
            self.last_report_step = int(self.num_timesteps)
        return not self.aborted

    def _on_training_end(self) -> None:
        self._write_report()


class CheckpointManifestCallback(BaseCallback):
    def __init__(
        self,
        *,
        save_frequency: int,
        save_path: Path,
        config_path: Path,
        world_path: Path,
        world_dependencies: tuple[Path, ...],
        raw_env: QuadrupedStairsEnv,
        algorithm_contract: dict[str, object],
        seed: int,
        transferred_from: Path | None,
        resumed_from: Path | None,
        inherited_transfer: dict[str, object] | None,
    ) -> None:
        super().__init__(verbose=0)
        self.save_frequency = int(save_frequency)
        self.save_path = save_path
        self.config_path = config_path
        self.world_path = world_path
        self.world_dependencies = world_dependencies
        self.raw_env = raw_env
        self.algorithm_contract = algorithm_contract
        self.seed = seed
        self.transferred_from = transferred_from
        self.resumed_from = resumed_from
        self.inherited_transfer = inherited_transfer

    def _on_step(self) -> bool:
        if self.n_calls % self.save_frequency:
            return True
        model_path = (
            self.save_path
            / f"drobot_stairs_ppo_{self.num_timesteps}_steps.zip"
        )
        if model_path.is_file():
            manifest = build_model_manifest(
                model_path=model_path,
                config_path=self.config_path,
                world_path=self.world_path,
                world_dependencies=self.world_dependencies,
                environment_contract=self.raw_env.contract,
                algorithm_contract=self.algorithm_contract,
                training_seed=self.seed,
                transferred_from=self.transferred_from,
                resumed_from=self.resumed_from,
                inherited_transfer=self.inherited_transfer,
            )
            write_model_manifest(model_manifest_path(model_path), manifest)
        return True


report: dict[str, object] = {
    "status": "FAIL",
    "task_id": task_config["id"],
    "config": str(config_path),
    "config_sha256": sha256_file(config_path),
    "world": str(world_path),
    "world_sha256": sha256_file(world_path) if world_path.is_file() else None,
    "world_dependencies": [
        {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for path in world_dependency_paths
    ],
    "output_dir": str(output_dir),
    "smoke_test": args.smoke_test,
    "initialize_only": args.initialize_only,
    "training_mode": training_mode,
    "algorithm_contract_requested": algorithm_contract,
    "requested_total_timesteps": total_timesteps,
    "ppo_run_overrides": {
        "learning_rate": args.ppo_learning_rate,
        "initial_log_std": args.ppo_initial_log_std,
        "entropy_coefficient": args.ppo_entropy_coefficient,
    },
    "curriculum_total_timesteps": curriculum_total_timesteps,
    "requested_seed": args.seed,
    "height_stage": args.height_stage,
    "fixed_active_steps": args.fixed_active_steps,
    "placement_start_level": args.placement_start_level,
    "fixed_placement_level": args.fixed_placement_level,
    "phase_train_leg": args.phase_train_leg,
    "phase_train_transfer": args.phase_train_transfer,
    "phase_transfer_post_hold_seconds": (
        phase_transfer_post_hold_seconds
    ),
    "phase_post_transfer_hold_only": args.phase_post_transfer_hold_only,
    "precursor_leg_models": {
        leg: str(path) for leg, path in precursor_leg_model_paths.items()
    },
    "phase_base_model": (
        str(phase_base_model_path)
        if phase_base_model_path is not None
        else None
    ),
    "phase_base_swing_only": args.phase_base_swing_only,
    "phase_base_residual_model": (
        str(phase_base_residual_model_path)
        if phase_base_residual_model_path is not None
        else None
    ),
    "phase_base_residual_scale": args.phase_base_residual_scale,
    "phase_base_residual_mode": args.phase_base_residual_mode,
    "phase_base_residual_compact_action": (
        args.phase_base_residual_compact_action
    ),
    "phase_residual_scale": args.phase_residual_scale,
    "phase_residual_support_only": args.phase_residual_support_only,
    "phase_residual_swing_only": args.phase_residual_swing_only,
    "phase_residual_support_abduction_only": (
        args.phase_residual_support_abduction_only
    ),
    "phase_residual_swing_support_abduction": (
        args.phase_residual_swing_support_abduction
    ),
    "phase_compact_residual_action": args.phase_compact_residual_action,
    "phase_snapshot_cache_enabled": not args.phase_disable_snapshot_cache,
    "phase_snapshot": (
        {
            "path": str(phase_snapshot_path),
            "sha256": sha256_file(phase_snapshot_path),
            "source_task_id": phase_snapshot_payload.get("source_task_id"),
            "target_leg": phase_snapshot_payload.get("target_leg"),
            "mode": phase_snapshot_payload.get("phase_snapshot_mode"),
        }
        if phase_snapshot_path is not None
        and phase_snapshot_payload is not None
        else None
    ),
    "initial_action_bias_requested": initial_action_bias,
    "terrain_perception_mode": terrain_perception_mode,
    "terrain_perception": terrain_perception_config,
    "device_request": args.device,
    "isaac_sim_version": "6.0.1",
    "torch_version": torch.__version__,
    "stable_baselines3_version": stable_baselines3.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
}
exit_code = 1
raw_env: QuadrupedStairsEnv | None = None
phase_training_env: PlacementPhaseTrainingEnv | None = None
monitored_env = None
start_time = time.perf_counter()

try:
    if not world_path.is_file():
        raise FileNotFoundError(
            f"{world_path}. Generate it with create_stairs_world.py first."
        )
    file_hash_records(world_dependency_paths)
    raw_env = QuadrupedStairsEnv(
        simulation_app,
        world_path=str(world_path),
        task_config=task_config,
        render_mode="human" if args.gui else None,
    )
    if args.fixed_placement_level is not None:
        raw_env.set_placement_level(args.fixed_placement_level)
    elif args.placement_start_level is not None:
        raw_env.set_placement_level(args.placement_start_level)
    training_env = raw_env
    precursor_model_verification: dict[str, dict[str, object]] = {}
    if args.phase_train_leg:
        precursor_models: dict[str, PPO] = {}
        for leg, precursor_path in precursor_leg_model_paths.items():
            if not precursor_path.is_file():
                raise FileNotFoundError(precursor_path)
            precursor_manifest_path = model_manifest_path(precursor_path)
            if not precursor_manifest_path.is_file():
                raise FileNotFoundError(precursor_manifest_path)
            precursor_manifest = read_model_manifest(precursor_path)
            precursor_hash = sha256_file(precursor_path)
            if precursor_manifest.get("model_sha256") != precursor_hash:
                raise RuntimeError(
                    f"Precursor model hash mismatch for {leg}: {precursor_path}"
                )
            precursor_model = PPO.load(str(precursor_path), device=args.device)
            if tuple(precursor_model.action_space.shape) != tuple(
                raw_env.action_space.shape
            ):
                raise RuntimeError(
                    f"Precursor action space does not match the target for {leg}"
                )
            observation_compatibility = (
                policy_observation_prefix_compatibility(
                    precursor_model,
                    precursor_manifest,
                    raw_env.contract,
                )
            )
            precursor_models[leg] = precursor_model
            precursor_model_verification[leg] = {
                "status": "PASS",
                "model": str(precursor_path),
                "model_sha256": precursor_hash,
                "manifest": str(precursor_manifest_path),
                "source_task_id": precursor_manifest.get("task_id"),
                "observation_compatibility": observation_compatibility,
            }
        target_base_model: PPO | None = None
        if phase_base_model_path is not None:
            if not phase_base_model_path.is_file():
                raise FileNotFoundError(phase_base_model_path)
            base_manifest_path = model_manifest_path(phase_base_model_path)
            if not base_manifest_path.is_file():
                raise FileNotFoundError(base_manifest_path)
            base_manifest = read_model_manifest(phase_base_model_path)
            base_hash = sha256_file(phase_base_model_path)
            if base_manifest.get("model_sha256") != base_hash:
                raise RuntimeError(
                    f"Phase base model hash mismatch: {phase_base_model_path}"
                )
            target_base_model = PPO.load(
                str(phase_base_model_path),
                device=args.device,
            )
            if tuple(target_base_model.action_space.shape) != tuple(
                raw_env.action_space.shape
            ):
                raise RuntimeError("Phase base model action space does not match")
            observation_compatibility = policy_observation_prefix_compatibility(
                target_base_model,
                base_manifest,
                raw_env.contract,
            )
            report["phase_base_model_verification"] = {
                "status": "PASS",
                "model": str(phase_base_model_path),
                "model_sha256": base_hash,
                "manifest": str(base_manifest_path),
                "source_task_id": base_manifest.get("task_id"),
                "observation_compatibility": observation_compatibility,
            }
        target_residual_mask = None
        target_base_mask = None
        if args.phase_base_swing_only:
            target_base_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=str(args.phase_train_leg),
                mode="swing_only",
            )
        if args.phase_residual_swing_support_abduction:
            target_residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=str(args.phase_train_leg),
                mode="swing_plus_support_abduction",
            )
        elif args.phase_residual_swing_only:
            target_residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=str(args.phase_train_leg),
                mode="swing_only",
            )
        elif args.phase_residual_support_abduction_only:
            target_residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=str(args.phase_train_leg),
                mode="support_abduction_only",
            )
        elif args.phase_residual_support_only:
            target_residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=str(args.phase_train_leg),
                mode="support_only",
            )
        if phase_base_residual_model_path is not None:
            if not phase_base_residual_model_path.is_file():
                raise FileNotFoundError(phase_base_residual_model_path)
            residual_manifest_path = model_manifest_path(
                phase_base_residual_model_path
            )
            if not residual_manifest_path.is_file():
                raise FileNotFoundError(residual_manifest_path)
            residual_manifest = read_model_manifest(
                phase_base_residual_model_path
            )
            residual_hash = sha256_file(phase_base_residual_model_path)
            if residual_manifest.get("model_sha256") != residual_hash:
                raise RuntimeError(
                    "Phase base residual model hash mismatch: "
                    f"{phase_base_residual_model_path}"
                )
            base_residual_model = PPO.load(
                str(phase_base_residual_model_path),
                device=args.device,
            )
            base_residual_mask = placement_policy_action_mask(
                raw_env.dof_names,
                target_leg=str(args.phase_train_leg),
                mode=str(args.phase_base_residual_mode),
            )
            expected_residual_action_size = int(
                np.count_nonzero(base_residual_mask)
                if args.phase_base_residual_compact_action
                else raw_env.action_space.shape[0]
            )
            if tuple(base_residual_model.action_space.shape) != (
                expected_residual_action_size,
            ):
                raise RuntimeError(
                    "Phase base residual action space does not match: "
                    f"{base_residual_model.action_space.shape} != "
                    f"({expected_residual_action_size},)"
                )
            observation_compatibility = (
                policy_observation_prefix_compatibility(
                    base_residual_model,
                    residual_manifest,
                    raw_env.contract,
                )
            )
            report["phase_base_residual_model_verification"] = {
                "status": "PASS",
                "model": str(phase_base_residual_model_path),
                "model_sha256": residual_hash,
                "manifest": str(residual_manifest_path),
                "source_task_id": residual_manifest.get("task_id"),
                "residual_scale": args.phase_base_residual_scale,
                "residual_mode": args.phase_base_residual_mode,
                "compact_action": args.phase_base_residual_compact_action,
                "observation_compatibility": observation_compatibility,
            }
            if target_base_model is None:
                raise RuntimeError(
                    "Phase base residual composition has no base policy"
                )
            target_base_model = FrozenBaseResidualPolicy(
                base_policy=target_base_model,
                residual_policy=base_residual_model,
                action_space=raw_env.action_space,
                residual_scale=args.phase_base_residual_scale,
                base_mask=target_base_mask,
                residual_mask=base_residual_mask,
                compact_residual_action=(
                    args.phase_base_residual_compact_action
                ),
            )
            # The adapter already applies the base mask before composing V35.
            target_base_mask = None
        phase_training_env = PlacementPhaseTrainingEnv(
            raw_env,
            target_leg=str(args.phase_train_leg),
            precursor_policies=precursor_models,
            target_base_policy=target_base_model,
            target_base_mask=target_base_mask,
            target_residual_scale=args.phase_residual_scale,
            target_residual_mask=target_residual_mask,
            compact_residual_action=args.phase_compact_residual_action,
            train_transfer=args.phase_train_transfer,
            transfer_post_hold_seconds=(
                phase_transfer_post_hold_seconds
                if args.phase_train_transfer
                else 0.0
            ),
            train_post_transfer_hold_only=(
                args.phase_post_transfer_hold_only
            ),
            maximum_reset_attempts=args.phase_reset_attempts,
            maximum_precursor_steps=args.phase_precursor_max_steps,
            cache_phase_snapshot=not args.phase_disable_snapshot_cache,
            initial_phase_snapshot=initial_phase_snapshot,
            initial_phase_snapshot_mode=initial_phase_snapshot_mode,
        )
        training_env = phase_training_env
        report["precursor_model_verification"] = precursor_model_verification
    monitored_env = Monitor(training_env, filename=str(output_dir / "monitor.csv"))
    policy_kwargs = {
        "activation_fn": torch.nn.ELU,
        "net_arch": list(ppo_config["policy_hidden_layers"]),
    }
    if "initial_log_std" in ppo_config:
        policy_kwargs["log_std_init"] = float(ppo_config["initial_log_std"])
    transferred_from: Path | None = None
    resume_path: Path | None = None
    inherited_transfer: dict[str, object] | None = None
    reset_num_timesteps = True
    if args.resume:
        resume_path = _resolve_project_path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_verification = validate_model_manifest(
            model_path=resume_path,
            config_path=config_path,
            world_path=world_path,
            world_dependencies=world_dependency_paths,
            environment_contract=raw_env.contract,
            allow_unverified=args.allow_unverified_resume,
            expected_algorithm_contract=algorithm_contract,
        )
        if resume_verification["status"] == "PASS":
            resume_manifest = read_model_manifest(resume_path)
            saved_transfer = resume_manifest.get("transferred_from")
            if isinstance(saved_transfer, dict):
                inherited_transfer = dict(saved_transfer)
        model = PPO.load(
            str(resume_path),
            env=monitored_env,
            device=args.device,
        )
        algorithm_verification = validate_ppo_algorithm_contract(
            model,
            algorithm_contract,
        )
        model.set_random_seed(args.seed)
        reset_num_timesteps = False
        report["resume_checkpoint"] = str(resume_path)
        report["resume_contract_verification"] = resume_verification
    else:
        model = PPO(
            "MlpPolicy",
            monitored_env,
            learning_rate=float(ppo_config["learning_rate"]),
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=epochs,
            gamma=float(ppo_config["gamma"]),
            gae_lambda=float(ppo_config["gae_lambda"]),
            clip_range=float(ppo_config["clip_range"]),
            ent_coef=float(ppo_config["entropy_coefficient"]),
            vf_coef=float(ppo_config["value_coefficient"]),
            max_grad_norm=float(ppo_config["max_gradient_norm"]),
            target_kl=(
                float(ppo_config["target_kl"])
                if ppo_config.get("target_kl") is not None
                else None
            ),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(output_dir / "tensorboard"),
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        if initial_action_bias is not None:
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.copy_(
                    torch.as_tensor(
                        initial_action_bias,
                        dtype=model.policy.action_net.bias.dtype,
                        device=model.policy.action_net.bias.device,
                    )
                )
            report["action_mean_initialization"] = {
                "weight": "all_zero",
                "bias": initial_action_bias,
                "source": "ppo.initial_action_bias",
            }
        elif bool(ppo_config.get("zero_action_mean_init", False)):
            with torch.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.zero_()
            report["zero_action_mean_initialization"] = {
                "weight": "all_zero",
                "bias": "all_zero",
            }
        if args.initialize_from_balance:
            transferred_from = _resolve_project_path(
                args.initialize_from_balance
            )
            if not transferred_from.is_file():
                raise FileNotFoundError(transferred_from)
            source_model = PPO.load(str(transferred_from), device=args.device)
            source_observation_size = int(source_model.observation_space.shape[0])
            target_observation_size = int(model.observation_space.shape[0])
            if source_observation_size != 56:
                raise RuntimeError(
                    "Balance initializer must use the 56-value foot-lift input"
                )
            if tuple(source_model.action_space.shape) != tuple(
                model.action_space.shape
            ):
                raise RuntimeError("Balance source and stair action shapes differ")
            source_manifest = read_model_manifest(transferred_from)
            if source_manifest.get("model_sha256") != sha256_file(transferred_from):
                raise RuntimeError("Balance initializer manifest/model hash mismatch")
            if source_manifest.get("task_id") != (
                "Drobot-Quadruped-Foot-Lift-v2-190mm-Unsupported-Balance"
            ):
                raise RuntimeError("Initializer is not the reviewed unsupported balance task")
            source_contract = dict(source_manifest["environment_contract"])
            source_fields = tuple(source_contract["observation_fields"])
            target_fields = tuple(raw_env.contract["observation_fields"])
            shared_prefix_size = 48
            if source_fields[:shared_prefix_size] != target_fields[:shared_prefix_size]:
                raise RuntimeError(
                    "Balance/stair proprioceptive observation prefixes differ"
                )
            source_action_scale = tuple(
                float(value) for value in source_contract["action_scale_rad"]
            )
            target_action_scale = tuple(
                float(value) for value in raw_env.contract["action_scale_rad"]
            )
            if len(source_action_scale) != 12 or len(target_action_scale) != 12:
                raise RuntimeError("Balance/stair action scale contracts must have 12 values")
            action_output_ratios = tuple(
                source / target
                for source, target in zip(
                    source_action_scale,
                    target_action_scale,
                    strict=True,
                )
            )
            transferred_state, balance_transfer_report = transfer_policy_state(
                source_model.policy.state_dict(),
                model.policy.state_dict(),
                source_observation_size=source_observation_size,
                shared_observation_prefix_size=shared_prefix_size,
                action_output_ratios=action_output_ratios,
            )
            expected_exact_count = len(model.policy.state_dict()) - len(
                EXPANDABLE_INPUT_WEIGHTS
            )
            if (
                balance_transfer_report["expanded_inputs"]
                != list(EXPANDABLE_INPUT_WEIGHTS)
                or balance_transfer_report["copied_exact_count"]
                != expected_exact_count
                or balance_transfer_report["skipped"]
            ):
                raise RuntimeError(
                    "Balance policy transfer was not exact: "
                    f"{balance_transfer_report}"
                )
            model.policy.load_state_dict(transferred_state, strict=True)
            if "initial_log_std" in ppo_config:
                model.policy.log_std.data.fill_(
                    float(ppo_config["initial_log_std"])
                )
                balance_transfer_report["log_std_overridden_after_transfer"] = float(
                    ppo_config["initial_log_std"]
                )
            report["balance_policy_transfer"] = {
                "source_model": str(transferred_from),
                "source_model_sha256": sha256_file(transferred_from),
                "source_task_id": source_manifest["task_id"],
                "source_observation_size": source_observation_size,
                "target_observation_size": target_observation_size,
                **balance_transfer_report,
            }
            del source_model
        if args.initialize_from_stairs:
            transferred_from = _resolve_project_path(
                args.initialize_from_stairs
            )
            if not transferred_from.is_file():
                raise FileNotFoundError(transferred_from)
            source_model = PPO.load(str(transferred_from), device=args.device)
            source_observation_size = int(source_model.observation_space.shape[0])
            target_observation_size = int(model.observation_space.shape[0])
            if tuple(source_model.action_space.shape) != tuple(
                model.action_space.shape
            ):
                raise RuntimeError("Stair source and target action shapes differ")
            source_state = source_model.policy.state_dict()
            target_state = model.policy.state_dict()
            if source_observation_size == target_observation_size:
                mismatched = [
                    name
                    for name, tensor in target_state.items()
                    if name not in source_state
                    or tuple(source_state[name].shape) != tuple(tensor.shape)
                ]
                if set(source_state) != set(target_state) or mismatched:
                    raise RuntimeError(
                        "Stair policy parameter contracts differ: "
                        f"mismatched={mismatched}"
                    )
                transferred_state = source_state
                stair_transfer_report = {
                    "mode": "exact_same_shape",
                    "expanded_inputs": [],
                }
            elif source_observation_size < target_observation_size:
                transferred_state, stair_transfer_report = transfer_policy_state(
                    source_state,
                    target_state,
                    source_observation_size=source_observation_size,
                )
                stair_transfer_report["mode"] = "expanded_observation"
                if stair_transfer_report["skipped"]:
                    raise RuntimeError(
                        "Expanded stair policy transfer skipped parameters: "
                        f"{stair_transfer_report['skipped']}"
                    )
            else:
                raise RuntimeError(
                    "Stair source observation is larger than the target: "
                    f"{source_observation_size}>{target_observation_size}"
                )
            model.policy.load_state_dict(transferred_state, strict=True)
            if "initial_log_std" in ppo_config:
                model.policy.log_std.data.fill_(
                    float(ppo_config["initial_log_std"])
                )
                stair_transfer_report["log_std_overridden_after_transfer"] = (
                    float(ppo_config["initial_log_std"])
                )
            report["stair_policy_transfer"] = {
                "source_model": str(transferred_from),
                "source_model_sha256": sha256_file(transferred_from),
                "parameter_count": len(source_state),
                "optimizer_transferred": False,
                "source_observation_size": source_observation_size,
                "target_observation_size": target_observation_size,
                **stair_transfer_report,
            }
            del source_model
        if args.initialize_from_flat:
            transferred_from = _resolve_project_path(args.initialize_from_flat)
            if not transferred_from.is_file():
                raise FileNotFoundError(transferred_from)
            source_model = PPO.load(str(transferred_from), device=args.device)
            source_observation_shape = tuple(source_model.observation_space.shape)
            source_action_shape = tuple(source_model.action_space.shape)
            source_activation = getattr(
                source_model.policy.activation_fn,
                "__name__",
                str(source_model.policy.activation_fn),
            )
            if source_observation_shape != (48,) or source_action_shape != (12,):
                raise RuntimeError(
                    "Flat source must have observation/action shapes (48,)/(12,), "
                    f"got {source_observation_shape}/{source_action_shape}"
                )
            if source_activation != "ELU":
                raise RuntimeError(
                    f"Flat source activation must be ELU, got {source_activation}"
                )
            transfer_config = dict(
                task_config.get("flat_policy_transfer", {})
            )
            output_ratios = None
            if bool(
                transfer_config.get("preserve_physical_action_mean", False)
            ):
                output_ratios = physical_action_output_ratios(
                    raw_env.dof_names,
                    dict(transfer_config["source_action_scale_rad"]),
                    dict(task_config["action_scale_rad"]),
                )
            transferred_state, transfer_report = transfer_policy_state(
                source_model.policy.state_dict(),
                model.policy.state_dict(),
                source_observation_size=48,
                action_output_ratios=output_ratios,
            )
            expected_exact_count = len(model.policy.state_dict()) - len(
                EXPANDABLE_INPUT_WEIGHTS
            )
            if (
                transfer_report["expanded_inputs"]
                != list(EXPANDABLE_INPUT_WEIGHTS)
                or transfer_report["copied_exact_count"] != expected_exact_count
                or transfer_report["skipped"]
            ):
                raise RuntimeError(
                    f"Flat policy transfer was not exact: {transfer_report}"
                )
            model.policy.load_state_dict(transferred_state, strict=True)
            if "initial_log_std" in ppo_config:
                model.policy.log_std.data.fill_(
                    float(ppo_config["initial_log_std"])
                )
                transfer_report["log_std_overridden_after_transfer"] = float(
                    ppo_config["initial_log_std"]
                )
            report["flat_policy_transfer"] = {
                "source_model": str(transferred_from),
                "source_model_sha256": sha256_file(transferred_from),
                **transfer_report,
            }
            del source_model
        algorithm_verification = validate_ppo_algorithm_contract(
            model,
            algorithm_contract,
        )

    actual_training_seed = int(model.seed) if model.seed is not None else args.seed
    report["training_seed"] = actual_training_seed
    report["ppo_algorithm_verification"] = algorithm_verification
    initial_curriculum_progress = min(
        1.0,
        float(model.num_timesteps) / max(1, curriculum_total_timesteps),
    )
    if str(task_config["curriculum"].get("mode", "timesteps")) == "timesteps":
        raw_env.set_training_progress(initial_curriculum_progress)
    training_control_callback: TrainingControlCallback | None = None
    if not args.initialize_only:
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_callback = CheckpointCallback(
            save_freq=checkpoint_frequency,
            save_path=str(checkpoint_dir),
            name_prefix="drobot_stairs_ppo",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
        training_control_callback = TrainingControlCallback(
            task_config=task_config,
            total_for_curriculum=curriculum_total_timesteps,
            report_path=output_dir / "progress_watchdog.json",
            fixed_placement_level=args.fixed_placement_level,
        )
        callbacks = CallbackList(
            [
                checkpoint_callback,
                CheckpointManifestCallback(
                    save_frequency=checkpoint_frequency,
                    save_path=checkpoint_dir,
                    config_path=config_path,
                    world_path=world_path,
                    world_dependencies=world_dependency_paths,
                    raw_env=raw_env,
                    algorithm_contract=algorithm_contract,
                    seed=actual_training_seed,
                    transferred_from=transferred_from,
                    resumed_from=resume_path,
                    inherited_transfer=inherited_transfer,
                ),
                training_control_callback,
            ]
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
    aborted_no_progress = bool(
        training_control_callback is not None
        and training_control_callback.aborted
    )
    final_model_base = output_dir / (
        "drobot_stairs_ppo_aborted"
        if aborted_no_progress
        else (
            "drobot_stairs_ppo_initialized"
            if args.initialize_only
            else "drobot_stairs_ppo_final"
        )
    )
    model.save(str(final_model_base))
    final_model_path = final_model_base.with_suffix(".zip")
    if not final_model_path.is_file():
        raise RuntimeError(f"Stable-Baselines3 did not save {final_model_path}")
    final_manifest = build_model_manifest(
        model_path=final_model_path,
        config_path=config_path,
        world_path=world_path,
        world_dependencies=world_dependency_paths,
        environment_contract=raw_env.contract,
        algorithm_contract=algorithm_contract,
        training_seed=actual_training_seed,
        transferred_from=transferred_from,
        resumed_from=resume_path,
        inherited_transfer=inherited_transfer,
    )
    final_manifest_path = model_manifest_path(final_model_path)
    write_model_manifest(final_manifest_path, final_manifest)
    report.update(
        {
            "status": (
                "ABORTED_NO_PROGRESS" if aborted_no_progress else "PASS"
            ),
            "actual_total_timesteps": int(model.num_timesteps),
            "model": str(final_model_path),
            "model_bytes": final_model_path.stat().st_size,
            "model_manifest": str(final_manifest_path),
            "model_parameter_count": int(
                sum(parameter.numel() for parameter in model.policy.parameters())
            ),
            "ppo": {
                **algorithm_contract,
            },
            "initial_curriculum_progress": initial_curriculum_progress,
            "environment_contract": raw_env.contract,
            "curriculum_transitions": raw_env.curriculum_transitions,
            "progress_watchdog": (
                training_control_callback.snapshot()
                if training_control_callback is not None
                else {
                    "status": "NOT_RUN",
                    "reason": "initialize_only",
                }
            ),
            "recent_completed_episodes": raw_env.completed_episode_metrics,
            "phase_training": (
                phase_training_env.training_stats()
                if phase_training_env is not None
                else None
            ),
            "elapsed_seconds": time.perf_counter() - start_time,
            "scope": (
                "Policy initialization without PPO updates"
                if args.initialize_only
                else (
                    (
                        "Target-transfer PPO after deterministic placement-prefix replay"
                        if args.phase_train_transfer
                        else "Target-leg PPO after deterministic placement-prefix replay"
                    )
                    if phase_training_env is not None
                    else (
                        "Pipeline validation only"
                        if args.smoke_test
                        else (
                            "Automatically aborted stair PPO training"
                            if aborted_no_progress
                            else "Single-environment stair PPO training"
                        )
                    )
                )
            ),
            "terrain_input": (
                "VL53L5CX-style 8 x 8 noisy raycast depth at 15 Hz; RGB "
                "camera pixels are not policy inputs."
                if terrain_perception_mode == VL53L5CX_MODE
                else (
                    "Analytic height profile used for simulation learning; "
                    "not a hardware sensor pipeline."
                )
            ),
        }
    )
    exit_code = 3 if aborted_no_progress else 0
except Exception as exc:
    report["error"] = repr(exc)
    report["traceback"] = traceback.format_exc()
    report["elapsed_seconds"] = time.perf_counter() - start_time
finally:
    if phase_training_env is not None:
        report["phase_training"] = phase_training_env.training_stats()
    if monitored_env is not None:
        monitored_env.close()
    elif raw_env is not None:
        raw_env.close()
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print("DROBOT_STAIRS_TRAIN_RESULT=" + json.dumps(report, sort_keys=True))
    simulation_app.close()

sys.exit(exit_code)
