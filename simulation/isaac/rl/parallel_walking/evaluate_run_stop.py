"""Evaluate versioned run/stop policies as separate deterministic phases."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import statistics
import sys
from pathlib import Path
from uuid import uuid4

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--rear-pitch-candidate-binding-json", type=Path, default=None)
parser.add_argument("--rear-pitch-candidate-binding-sha256", default=None)
parser.add_argument("--speed-response-v32", choices=("low", "high"), default=None,
    help="Bound candidate31 speed comparison only; never a production or general protocol override.")
parser.add_argument(
    "--task",
    default=(
        "Drobot-Commanded-Run-Integrated-Target-Velocity-Stop-"
        "External-Rear-Payload"
    ),
)
parser.add_argument("--speed", type=float, default=0.06)
parser.add_argument("--run-seconds", type=float, default=4.0)
parser.add_argument("--ramp-seconds", type=float, default=0.60)
parser.add_argument("--hold-seconds", type=float, default=3.0)
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--seed", type=int, default=3101)
parser.add_argument(
    "--domain-mode", choices=("task", "nominal", "randomized"), default="nominal"
)
parser.add_argument(
    "--fixed-start-phase-zero",
    action="store_true",
    help=(
        "Disable randomized quarter-cycle starts and reproduce the onboard "
        "runtime's phase-zero startup."
    ),
)
parser.add_argument(
    "--ramp-start-phase-cycles",
    type=float,
    default=None,
    help=(
        "Optionally delay the requested run duration until the gait reaches "
        "this phase before beginning the stop ramp."
    ),
)
parser.add_argument(
    "--ramp-start-phase-tolerance-cycles",
    type=float,
    default=0.0,
)
parser.add_argument("--maximum-phase-alignment-wait-s", type=float, default=0.0)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--diagnostic-trace-jsonl", type=Path, default=None,
    help="V44 only: buffer every nonterminal step, then write a new diagnostic JSONL file (never overwrite).",
)
parser.add_argument(
    "--env-overrides-json", type=Path, default=None,
    help="Explicit, whitelisted V44 ablation overrides; production task defaults stay unchanged.",
)
parser.add_argument(
    "--zero-policy-actions", action="store_true",
    help="V44 ablation only: evaluate the reference with zero residual actions, not learned policy performance.",
)
parser.add_argument(
    "--actor-joint-velocity-fd", action="store_true",
    help="UNQUALIFIED V44 observation diagnostic: replace only actor qdot with causal clipped 60 Hz pre-position FD; physics, critic, rewards and gates remain raw.",
)
parser.add_argument(
    "--schedule-aware-crawl", action="store_true",
    help="V44 ablation only: qualify a sequential crawl foundation with actual three-foot support gates, never running qualification.",
)
parser.add_argument(
    "--physx-effort-readback", action="store_true",
    help="V44 ablation only: read live PhysX max-force limits after construction, episode resets and first steps; never change them.",
)
parser.add_argument(
    "--physx-step-diagnostic-jsonl", type=Path, default=None,
    help="V44 ablation only: passively capture existing 120 Hz physics steps during each scheduled final one-second hold; write a new buffered JSONL, never overwrite.",
)
args = parser.parse_args()
if args.speed <= 0.0:
    parser.error("--speed must be positive")
if args.run_seconds < 1.0 or args.ramp_seconds < 0.0 or args.hold_seconds < 1.0:
    parser.error("run/hold must be at least one second and ramp must be non-negative")
if args.episodes < 1:
    parser.error("--episodes must be positive")
if args.ramp_start_phase_cycles is not None and not (
    0.0 <= args.ramp_start_phase_cycles < 1.0
):
    parser.error("--ramp-start-phase-cycles must be in [0, 1)")
if not 0.0 <= args.ramp_start_phase_tolerance_cycles <= 0.25:
    parser.error("--ramp-start-phase-tolerance-cycles must be in [0, 0.25]")
if not 0.0 <= args.maximum_phase_alignment_wait_s <= 5.0:
    parser.error("--maximum-phase-alignment-wait-s must be in [0, 5]")
if args.ramp_start_phase_cycles is not None and (
    args.ramp_start_phase_tolerance_cycles <= 0.0
    or args.maximum_phase_alignment_wait_s <= 0.0
):
    parser.error(
        "phase-aligned stopping requires a positive tolerance and maximum wait"
    )

from v44_ablation_contract import (  # noqa: E402
    apply_v44_overrides, file_sha256, load_v44_overrides,
)
from v44_training_contract import TASK as V44_TASK  # noqa: E402

if args.diagnostic_trace_jsonl is not None:
    if args.task != V44_TASK:
        parser.error("--diagnostic-trace-jsonl is supported only for the exact V44 task")
    args.diagnostic_trace_jsonl = args.diagnostic_trace_jsonl.resolve()
    if args.diagnostic_trace_jsonl.exists():
        parser.error("--diagnostic-trace-jsonl must be a new path; existing files are never overwritten")
    if args.output is not None and args.output.resolve() == args.diagnostic_trace_jsonl:
        parser.error("The diagnostic trace and evaluation report must use different paths")

ablation_values, ablation_provenance = {}, None
if args.env_overrides_json is not None:
    try:
        ablation_values, ablation_provenance = load_v44_overrides(args.env_overrides_json, args.task)
    except (OSError, ValueError, TypeError) as exc:
        parser.error(f"Invalid V44 environment overrides: {exc}")
rear_pitch_evaluation = "gait_rear_pitch_lift_m" in ablation_values
rear_projection_evaluation = ablation_values.get("gait_rear_pitch_projection_enabled", False)
if args.speed_response_v32 is not None and (
    not rear_projection_evaluation or args.rear_pitch_candidate_binding_json is None
    or args.rear_pitch_candidate_binding_sha256 is None
):
    parser.error("Speed-response32 requires its projected candidate binding")
selected_rear_evaluation_speed = (
    .02227272727272727 if args.speed_response_v32 == "high" else .01909090909090909
)


def _sustained_credit_metadata():
    value = (ablation_provenance or {}).get("sustained_progress_credit_ablation")
    if value is None:
        return {}
    result = {"sustained_progress_credit_ablation": value}
    if value["enabled"]:
        result.update(qualification_status="UNQUALIFIED_SUSTAINED_PROGRESS_CREDIT_ABLATION",
            production_export_allowed=False, qualifies_hardware=False, qualifies_running_goal=False,
            passes_all_gates_scope="unchanged_numeric_gates_not_policy_or_hardware_qualification")
    return result


if rear_pitch_evaluation:
    if (args.task != V44_TASK or args.diagnostic_trace_jsonl is None or args.output is None
        or args.domain_mode != "nominal" or not args.fixed_start_phase_zero
        or args.episodes != 2 or args.seed != 3101
        or abs(args.speed-selected_rear_evaluation_speed) > 1.e-12
        or args.run_seconds != 25.0 or args.ramp_seconds != 0.9 or args.hold_seconds != 3.0
        or args.zero_policy_actions or args.actor_joint_velocity_fd or args.schedule_aware_crawl
        or args.physx_effort_readback or args.physx_step_diagnostic_jsonl is not None):
        parser.error("Rear pitch lift requires its selected2-episode nominal actor50 protocol, exact command/durations and new report/trace paths")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("Rear pitch evaluation report must be exclusive; never overwrite a previous result")
if args.zero_policy_actions and (args.task != V44_TASK or ablation_provenance is None):
    parser.error("--zero-policy-actions requires the V44 task and --env-overrides-json")
if args.actor_joint_velocity_fd:
    if args.task != V44_TASK or ablation_provenance is None or args.diagnostic_trace_jsonl is None:
        parser.error("--actor-joint-velocity-fd requires exact V44, --env-overrides-json and a new --diagnostic-trace-jsonl")
    if args.zero_policy_actions:
        parser.error("Actor FD diagnostics cannot be combined with --zero-policy-actions")
    if any(value != 1.0 for value in ablation_values.get("gait_swing_residual_scale_by_leg", [1.0] * 4)):
        parser.error("Actor FD diagnostics require identity swing residual scales for one-variable attribution")
    print("UNQUALIFIED_OBSERVATION_DIAGNOSTIC: actor-only 60 Hz position FD; no production or hardware qualification.")
if args.schedule_aware_crawl and (args.task != V44_TASK or ablation_provenance is None):
    parser.error("--schedule-aware-crawl requires the V44 task and --env-overrides-json")
if args.physx_effort_readback and (args.task != V44_TASK or ablation_provenance is None):
    parser.error("--physx-effort-readback requires the V44 task and --env-overrides-json")
if args.physx_step_diagnostic_jsonl is not None:
    if args.task != V44_TASK or ablation_provenance is None:
        parser.error("--physx-step-diagnostic-jsonl requires the V44 task and --env-overrides-json")
    if args.episodes > 100:
        parser.error("The passive physics diagnostic supports at most 100 episodes")
    args.physx_step_diagnostic_jsonl = args.physx_step_diagnostic_jsonl.resolve()
    if args.physx_step_diagnostic_jsonl.exists():
        parser.error("--physx-step-diagnostic-jsonl must be a new path; existing files are never overwritten")
    if not args.physx_step_diagnostic_jsonl.parent.is_dir():
        parser.error("The passive physics diagnostic output parent directory must already exist")
    if args.physx_step_diagnostic_jsonl in (
        args.output.resolve() if args.output is not None else None,
        args.diagnostic_trace_jsonl,
    ):
        parser.error("The passive physics diagnostic, control trace and evaluation report must use different paths")
checkpoint_source_path = Path(args.checkpoint).resolve()
evaluation_source_provenance = {
    "checkpoint_sha256": file_sha256(checkpoint_source_path),
    "checkpoint_training_config_sha256": {
        name: file_sha256(checkpoint_source_path.parent / "params" / name)
        for name in ("env.yaml", "agent.yaml")
        if (checkpoint_source_path.parent / "params" / name).is_file()
    },
    "evaluation_environment_source_sha256": file_sha256(Path(__file__).with_name("commanded_walking_env.py")),
    "evaluation_environment_config_source_sha256": file_sha256(Path(__file__).with_name("commanded_walking_env_cfg.py")),
    "evaluation_script_sha256": file_sha256(Path(__file__)),
}
candidate_binding_requested = (args.rear_pitch_candidate_binding_json is not None
                               or args.rear_pitch_candidate_binding_sha256 is not None)
if candidate_binding_requested:
    if (not rear_projection_evaluation or args.rear_pitch_candidate_binding_json is None
        or args.rear_pitch_candidate_binding_sha256 is None
        or not math.isclose(args.speed, selected_rear_evaluation_speed, rel_tol=0., abs_tol=1.e-15)
        or args.ramp_start_phase_cycles != .05
        or args.ramp_start_phase_tolerance_cycles != .03 or args.maximum_phase_alignment_wait_s != .5):
        parser.error("A trained rear-pitch candidate requires both binding arguments and its exact projected canonical protocol")
    import importlib.util

    try:
        binding_input_path = Path(args.rear_pitch_candidate_binding_json).resolve(strict=True)
        if file_sha256(binding_input_path) != args.rear_pitch_candidate_binding_sha256:
            raise ValueError("Explicit candidate binding SHA differs before version dispatch")
        binding_attempt = json.loads(binding_input_path.read_text(encoding="utf-8-sig")).get("attempt_id")
        if (binding_attempt in {"speed-response-32-v2", "swing-tilt-continuation33-v1", "swing-tilt-credit34-v1", "front-stance35-v1", "normalized-slip36-v1", "beta-concentration37-v1", "rear-feedback38-v1", "constant-stance39-v1", "constant-stance40-v1", "long-stroke41-v1", "body-tilt42-v1"}) != (args.speed_response_v32 is not None):
            raise ValueError("Speed-response32 scope must match the exact binding route")
        validator_names = {
            None: "rear_pitch_candidate_binding.py",
            "q-clamp-attempt02": "rear_pitch_candidate_binding_attempt02.py",
            "credit-horizon-lambda-v1": "credit_horizon_candidate_binding.py",
            "sustained-progress-v1": "sustained_progress_candidate_binding.py",
            "raw-smoothness-v1": "raw_smoothness_candidate_binding.py",
            "pitch-feedback-v1": "pitch_feedback_candidate_binding.py",
            "swing-tilt-v1": "swing_tilt_candidate_binding.py",
            "swing-tilt-dose-v1": "swing_tilt_dose_candidate_binding.py",
            "speed-response-32-v2": "speed_response32_candidate_binding.py",
            "swing-tilt-continuation33-v1": "swing_tilt_continuation33_binding.py",
            "swing-tilt-credit34-v1": "swing_tilt_credit34_binding.py",
            "front-stance35-v1": "front_stance35_binding.py",
            "normalized-slip36-v1": "normalized_slip36_binding.py",
            "beta-concentration37-v1": "beta_concentration37_binding.py",
            "rear-feedback38-v1": "rear_feedback38_binding.py",
            "constant-stance39-v1": "constant_stance39_binding.py",
            "constant-stance40-v1": "constant_stance40_binding.py",
            "long-stroke41-v1": "long_stroke41_binding.py",
            "body-tilt42-v1": "body_tilt42_binding.py",
        }
        if binding_attempt not in validator_names:
            raise ValueError("Unknown projected-candidate attempt")
        binding_validator_path = (Path(__file__).resolve().parents[4]
            / "simulation/reviews/v44-actuator-20260905" / validator_names[binding_attempt])
        binding_spec = importlib.util.spec_from_file_location("_drobot_rear_pitch_candidate_binding", binding_validator_path)
        if binding_spec is None or binding_spec.loader is None:
            raise ValueError("Cannot load the study-only candidate ancestry validator")
        binding_module = importlib.util.module_from_spec(binding_spec)
        binding_spec.loader.exec_module(binding_module)
        evaluation_source_provenance["rear_pitch_candidate_binding"] = binding_module.validate_candidate_binding(
            args.rear_pitch_candidate_binding_json, args.rear_pitch_candidate_binding_sha256,
            checkpoint_source_path, args.env_overrides_json,
            ablation_values["gait_rear_pitch_lift_m"], args.task,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(f"Invalid trained rear-pitch candidate ancestry: {exc}")
elif rear_pitch_evaluation and evaluation_source_provenance["checkpoint_sha256"] != "f624e14294f83947d46a433c0eb27747921313ed6c9cac489c47e57fe45e9a89":
    parser.error("Rear pitch comparison requires the exact original Control09 model298 checkpoint")

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
from parallel_walking.commanded_walking_env import RearPitchLiftMappingError  # noqa: E402
from parallel_walking.commanded_walking_env import (  # noqa: E402
    _rotate_vectors_xyzw,
    _yaw_from_xyzw,
)


INITIALIZATION_VERSION = (
    "v44.finalized-reset-fresh-observation.v1"
    if args.task == V44_TASK else "legacy.manual-reset-cached-observation.v1"
)


def _finalized_initial_observations(vec_env, episode):
    """Finish the existing V44 manual reset after its final schedule changes.

    Mirror DirectRLEnv.reset's kinematics/observation tail, not another reset.
    Exactly one _get_observations call; no policy call or physics time advance.
    """
    robot_env = vec_env.unwrapped
    if args.task != V44_TASK:
        return vec_env.get_observations(), None
    from isaaclab.envs import DirectRLEnv

    if not isinstance(robot_env, DirectRLEnv) or robot_env.num_envs != 1:
        raise ValueError("V44 observation initialization requires the audited single DirectRLEnv")
    if robot_env.has_rtx_sensors or robot_env.cfg.observation_noise_model is not None:
        raise ValueError("V44 observation initialization does not support unaudited RTX or external observation-noise lifecycle")
    counters_before = (int(robot_env._sim_step_counter), int(robot_env.common_step_counter),
                       int(robot_env._steps_since_reset[0].item()), int(robot_env.episode_length_buf[0].item()))
    # Installed DirectRLEnv.reset uses this same order. forward() updates only
    # kinematics; do not step, resample/reset the domain or advance sensor time.
    robot_env.scene.write_data_to_sim()
    robot_env.sim.forward()
    robot_env.obs_buf = robot_env._get_observations()
    obs = vec_env.get_observations()  # TensorDict wrapper of the now-fresh obs_buf.
    counters_after = (int(robot_env._sim_step_counter), int(robot_env.common_step_counter),
                      int(robot_env._steps_since_reset[0].item()), int(robot_env.episode_length_buf[0].item()))
    if counters_before != counters_after:
        raise RuntimeError("Initial observation refresh unexpectedly advanced environment counters")
    policy_obs = obs["policy"]
    if policy_obs.shape != (1, 50) or not bool(torch.isfinite(policy_obs).all()):
        raise ValueError("Fresh V44 actor observation must be finite [1,50]")
    data = robot_env._robot.data
    from parallel_walking.commanded_walking_env_cfg import SERVO_VELOCITY_LIMIT_RAD_S

    expected = {
        "command_scaled": ((0, 3), robot_env._commands * robot_env._command_observation_scale),
        "joint_position_minus_default_rad": ((14, 26), data.joint_pos.torch - data.default_joint_pos.torch),
        "joint_velocity_normalized": ((26, 38), data.joint_vel.torch / SERVO_VELOCITY_LIMIT_RAD_S),
    }
    gaps = {name: float(torch.max(torch.abs(policy_obs[:, start:end] - value.clamp(-20., 20.))).item())
            for name, ((start, end), value) in expected.items()}
    if any(gap > 2.e-6 for gap in gaps.values()):
        raise RuntimeError(f"Fresh V44 initial observation does not match finalized state: {gaps}")
    return obs, {
        "episode": episode, "initialization_version": INITIALIZATION_VERSION,
        "fresh_observation_generation_count": 1,
        "physics_counter": counters_after[0], "common_control_counter": counters_after[1],
        "steps_since_reset": counters_after[2], "episode_length_buf": counters_after[3],
        "counters_unchanged_by_refresh": True,
        "actor_command_scaled": policy_obs[0, :3].detach().cpu().tolist(),
        "actor_gait_sin_cos": policy_obs[0, 3:5].detach().cpu().tolist(),
        "gait_phase_cycles": float(robot_env._gait_phase()[0].item()),
        "max_abs_feature_prestate_gaps": gaps,
    }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _rms(values: list[float]) -> float:
    return math.sqrt(_mean([value * value for value in values]))


def _named_first_row(names: tuple[str, ...], values: torch.Tensor) -> dict[str, float]:
    row = values.detach().cpu()
    if row.ndim > 1:
        row = row[0]
    return {
        name: float(value)
        for name, value in zip(names, row.tolist(), strict=True)
    }


def _target_ceiling_measurement(robot_env, target_delta, packet_elapsed_s):
    """Measure delivered packets; V44 shares the controller's actual bounds."""
    if args.task == V44_TASK:
        allowed_delta = (
            robot_env._target_velocity_limit_by_joint
            * packet_elapsed_s
            * robot_env._target_velocity_scale_by_joint[0]
        )
        if robot_env._target_packet_step_limit_rad > 0.0:
            allowed_delta = torch.clamp(
                allowed_delta, max=robot_env._target_packet_step_limit_rad
            )
        ceiling_active = (
            (target_delta >= 0.98 * allowed_delta)
            & ~robot_env._target_packet_held[0]
        )
    else:
        # Explicit historical report compatibility, not a missing-V44-data fallback.
        allowed_delta = (
            robot_env.cfg.target_velocity_limit_rad_s
            * robot_env.step_dt
            * robot_env._target_velocity_scale_by_joint[0]
        )
        ceiling_active = target_delta >= 0.98 * allowed_delta
    return allowed_delta, ceiling_active


def _write_passive_physics_diagnostic(recorder, *, status, error=None):
    """Only called after a selected env.step has unwound and restored sim.step."""
    recorder.metadata["evaluation_capture_status"] = status
    recorder.metadata["evaluation_error"] = error
    recorder.metadata["completed_episode_outcomes"] = [
        {key: episode[key] for key in ("episode", "fell", "fall_phase", "completed_seconds")}
        for episode in episode_results
    ]
    metadata_record = recorder.write_jsonl(args.physx_step_diagnostic_jsonl)
    return {
        "path": str(args.physx_step_diagnostic_jsonl),
        "sha256": file_sha256(args.physx_step_diagnostic_jsonl),
        "trace_id": metadata_record["trace_id"],
        "capture_status": status,
        "recorded_physics_steps": metadata_record["recorded_physics_steps"],
        "eligible_physics_steps": metadata_record["eligible_physics_steps"],
        "capture_truncated": metadata_record["capture_truncated"],
        "lifecycle_errors": metadata_record["lifecycle_errors"],
        "offline_summary_location": {
            "path": str(args.physx_step_diagnostic_jsonl),
            "jsonl_line_one_based": 1,
            "json_pointer": "/offline_summary",
        },
        "qualification": "passive_state_diagnostic_only_not_a_gate_or_timing_benchmark",
    }


def _optional_mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"samples": 0, "mean": None, "rms": None, "min": None, "p05": None, "p95": None, "max": None}
    ordered = sorted(values)

    def quantile(fraction):
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

    return {"samples": len(values), "mean": _mean(values), "rms": _rms(values),
            "min": ordered[0], "p05": quantile(.05), "p95": quantile(.95), "max": ordered[-1]}


def _physx_effort_readback(robot_env, stage, episode=None, terminal_autoreset=False):
    """Independent physics-view getter, with explicit failure and no cache fallback."""
    import warp as wp

    robot = robot_env._robot
    row = {"stage": stage, "episode": episode, "terminal_autoreset_already_occurred": terminal_autoreset,
           "articulation_type": f"{type(robot).__module__}.{type(robot).__qualname__}",
           "getter": "robot.root_view.get_dof_max_forces()", "units": "Nm",
           "expected_user_order_nm": None, "cached_user_order_nm": None,
           "physics_backend_order_nm": None, "physics_user_order_nm": None,
           "max_abs_physics_minus_expected_nm": None, "max_abs_physics_minus_cache_nm": None,
           "max_abs_cache_minus_expected_nm": None, "matches_expected_within_1e_5_nm": None,
           "status": "unavailable", "errors": {}}
    arrays = {}
    for name, getter in (
        ("expected_user_order_nm", lambda: robot_env._actuator_effort_limit_by_joint),
        ("cached_user_order_nm", lambda: robot.data.joint_effort_limits.torch),
    ):
        try:
            value = getter().detach().cpu().clone()
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite values; not serialized as valid limits")
            arrays[name], row[name] = value, value.tolist()
        except Exception as exc:
            row["errors"][name] = f"{type(exc).__name__}: {exc}"
    try:
        row["joint_names"] = list(robot.joint_names)
        row["backend_joint_names"] = list(robot.backend_joint_names)
        if len(set(row["backend_joint_names"])) != len(row["backend_joint_names"]):
            raise ValueError("Backend joint names are not unique")
        order = [row["backend_joint_names"].index(name) for name in row["joint_names"]]
        # This calls the physics view, not ArticulationData's cached ProxyArray.
        backend = wp.to_torch(robot.root_view.get_dof_max_forces()).detach().cpu().clone()
        if backend.ndim != 2 or backend.shape[1] != len(row["backend_joint_names"]):
            raise ValueError("Unexpected backend max-force shape")
        if not bool(torch.isfinite(backend).all()):
            raise ValueError("Nonfinite PhysX max-force readback")
        actual = backend[:, order]
        row["backend_indices_in_user_joint_order"] = order
        row["physics_backend_order_nm"], row["physics_user_order_nm"] = backend.tolist(), actual.tolist()
        arrays["physics_user_order_nm"] = actual
        row["status"] = "physics_readback_available_comparison_incomplete"
    except Exception as exc:
        row["errors"]["physics_getter"] = f"{type(exc).__name__}: {exc}"
    if len(arrays) == 3:
        actual, expected, cached = (arrays[key] for key in (
            "physics_user_order_nm", "expected_user_order_nm", "cached_user_order_nm"))
        if actual.shape != expected.shape or actual.shape != cached.shape:
            row["errors"]["comparison"] = "Physics/expected/cache shapes differ; no inferred broadcast"
        else:
            row["max_abs_physics_minus_expected_nm"] = float(torch.max(torch.abs(actual - expected)).item())
            row["max_abs_physics_minus_cache_nm"] = float(torch.max(torch.abs(actual - cached)).item())
            row["max_abs_cache_minus_expected_nm"] = float(torch.max(torch.abs(cached - expected)).item())
            row["matches_expected_within_1e_5_nm"] = row["max_abs_physics_minus_expected_nm"] <= 1.e-5
            row["status"] = "matches_expected" if row["matches_expected_within_1e_5_nm"] else "physics_limit_mismatch"
    return row


def _effort_observable_contract(robot_env):
    """Describe the opt-in learning signal without changing numeric gates."""
    enabled = robot_env.cfg.native_effort_telemetry_enabled
    return {
        "schema": "drobot.v44-active-effort-observable.v1",
        "enabled": enabled,
        "reward_observable": "native_active_generalized_effort" if robot_env.cfg.native_effort_reward_enabled else "retained_clipped_approximate_pd",
        "isolated_ablation_task_marker": robot_env.cfg.native_effort_ablation_task,
        "getter": "root_view.get_dof_projected_joint_forces" if enabled else None,
        "native_semantics": "Post-solve active generalized joint effort; not isolated drive torque, motor current, electrical power, heating or an interval peak",
        "sampling": "One getter at existing reward boundary after the last120Hz solve of each60Hz control step. Trace copies retained snapshots, with no second getter, step, observation or actor call.",
        "ordering": "Exactly joint_names, checked against native DOF names; reversed joints rejected",
        "qualification_effort_observable": "unchanged retained clipped approximate PD",
        "all_numeric_gates_unchanged": True,
        "production_export_allowed": False,
    }


def _critic_context_contract(robot_env):
    """Reporting only: the expanded critic is not a deployed actor input."""
    from parallel_walking.commanded_walking_env import (
        CRITIC_CONTEXT_FEATURE_NAMES, CRITIC_CONTEXT_SCHEMA,
    )

    mode = robot_env.cfg.critic_context_mode
    enabled = mode != "legacy58"
    return {
        "schema": CRITIC_CONTEXT_SCHEMA if enabled else None,
        "mode": mode,
        "actor_observation_count": 50,
        "critic_observation_count": 90 if enabled else 58,
        "isolated_ablation_task_marker": robot_env.cfg.critic_context_ablation_task,
        "context_count": 32 if enabled else 0,
        "appended_feature_names": list(CRITIC_CONTEXT_FEATURE_NAMES) if enabled else [],
        "context_input": "all_zeros_after_same_causal_builder" if mode == "context90_zero"
                         else "causal_reward_and_controller_summaries" if enabled else "none",
        "interpretation": "Training-only critic information; deterministic actor still receives the unchanged50 inputs. Compact summaries are not a full Markov state. Evaluation motion cannot establish value accuracy.",
        "all_numeric_gates_unchanged": True,
        "production_export_allowed": False,
    }


def _phase_reference_sampling_contract(robot_env):
    """Read-only additive metadata; no policy call, phase advance, or tensor math."""
    table = robot_env._distributed_reference_joint_pos
    enabled = robot_env.cfg.gait_phase_reference_linear_interpolation_enabled
    sample_count = int(table.shape[0]) if table is not None else None
    return {
        "schema": "drobot.v44-phase-joint-interpolation.v1",
        "enabled": enabled,
        "mode": "linear_adjacent" if enabled else "floor" if table is not None else "not_phase_table",
        "nondefault_mode_simulation_only": enabled,
        "isolated_ablation_task_marker": robot_env.cfg.gait_phase_reference_interpolation_ablation_task,
        "sample_count": sample_count,
        "table_dtype": str(table.dtype) if table is not None else None,
        "phase_source": "actual_target_gait_phase_cycles copied before target application, not legacy next-phase alias",
        "periodic_index_math": "s=phase*N; k=floor(s); i0=k%N; i1=(i0+1)%N; u=s-k",
        "position_sampling": "lerp adjacent rows within each low/high table, then existing low-to-high command-speed lerp" if enabled else "original floor row lookup then existing low-to-high command-speed lerp",
        "contact_sampling": "original floor contact table plus unchanged startup/park overrides",
        "body_transfer_metadata_supported_when_enabled": False,
        "default_floor_arithmetic_preserved": True,
        "legacy_phase_alias_unchanged": True,
        "startup_stop_rate_packet_response_semantics_unchanged": True,
        "qualification_note": "Sampling metadata is not qualification. Nondefault has no Pi/runtime or production export parity.",
    }


class _DiagnosticTrace:
    """Opt-in snapshots only: no actuator writes and no JSON/disk I/O in the step loop."""

    SCHEMA = "drobot.v44-step-diagnostic-trace.v1"

    def __init__(self, robot_env, path, actor, vec_env):
        from tensordict import TensorDictBase
        from parallel_walking.commanded_walking_env_cfg import SERVO_VELOCITY_LIMIT_RAD_S

        self.env, self.path = robot_env, path
        self.phase_reference_sampling = _phase_reference_sampling_contract(robot_env)
        self.trace_id = str(uuid4())
        self.rows = []
        self.tensor_dict_type = TensorDictBase
        if (
            type(actor).__module__ != "rsl_rl.models.mlp_model"
            or type(actor).__name__ != "MLPModel"
            or actor.is_recurrent
            or actor.training
            or list(actor.obs_groups) != ["policy"]
            or actor.obs_dim != 50
        ):
            raise ValueError("Exact actor tracing requires the evaluated nonrecurrent MLPModel with ordered ['policy'] 50-feature input")
        normalizer = actor.obs_normalizer
        normalizer_name = f"{type(normalizer).__module__}.{type(normalizer).__name__}"
        supported_normalizers = {
            "torch.nn.modules.linear.Identity": False,
            "rsl_rl.modules.normalization.EmpiricalNormalization": True,
        }
        if normalizer_name not in supported_normalizers or normalizer.training:
            raise ValueError("Exact actor tracing requires a known evaluation-mode observation normalizer")
        if bool(actor.obs_normalization) != supported_normalizers[normalizer_name]:
            raise ValueError("Actor normalization flag and actual normalizer disagree")
        distribution = actor.distribution
        if (
            type(distribution).__module__ != "rsl_rl.modules.distribution"
            or type(distribution).__name__ != "BetaDistribution"
            or distribution.output_dim != 12
        ):
            raise ValueError("Exact actor tracing requires the current 12-action BetaDistribution contract")
        self.actor_contract = {
            "schema": "drobot.exact-actor-input-output-trace.v1",
            "actor_class": f"{type(actor).__module__}.{type(actor).__name__}",
            "actor_obs_groups_in_concatenation_order": list(actor.obs_groups),
            "actor_input_shape_including_batch": [1, 50],
            "actor_output_shape_including_batch": [1, 12],
            "actor_eval_mode": not actor.training,
            "actor_recurrent": bool(actor.is_recurrent),
            "actor_mlp_architecture": repr(actor.mlp),
            "normalizer_class": normalizer_name,
            "normalization_enabled": bool(actor.obs_normalization),
            "normalizer_state_dict": {
                name: self._values(value) for name, value in normalizer.state_dict().items()
            },
            "normalizer_eps": float(normalizer.eps) if actor.obs_normalization else None,
            "normalizer_semantics": "Identity when disabled; otherwise (x - _mean) / (_std + eps). Statistics are checkpoint state, not updated or recomputed for this trace.",
            "distribution_class": f"{type(distribution).__module__}.{type(distribution).__name__}",
            "distribution_action_range": list(distribution.action_range),
            "raw_actor_output_semantics": "The existing policy(obs) result: deterministic Beta mean after softplus(alpha/beta logits)+1 and action-range rescaling. Raw means before zero override/wrapper clipping, NOT the 24 MLP logits and NOT a sampled action.",
            "wrapper_action_clip_absolute": vec_env.clip_actions,
            "environment_action_clip_absolute": 1.0,
            "simulation_swing_residual_mapping": {
                "schema": "drobot.v44-swing-residual-envelope.v1",
                "simulation_only_not_pi_or_export_parity": True,
                "enabled": bool(robot_env._swing_residual_scaling_enabled),
                "leg_order": ["front_left", "rear_left", "front_right", "rear_right"],
                "scale_by_leg": list(robot_env.cfg.gait_swing_residual_scale_by_leg),
                "isolated_ablation_task_marker": robot_env.cfg.gait_swing_residual_ablation_task,
                "hip_joint_ids_by_leg": self._values(robot_env._leg_hip_joint_ids),
                "knee_joint_ids_by_leg": self._values(robot_env._leg_knee_joint_ids),
                "phase_offsets_by_leg": self._values(robot_env._leg_phase_offsets),
                "duty_factor": float(robot_env.cfg.gait_duty_factor),
                "clearance_transition_fraction": float(robot_env.cfg.sole_clearance_transition_fraction),
                "semantics": "Hip/knee residual multiplier 1-(1-scale_leg)*E using applied phase and symmetric quintic swing envelope; abduction/stance/park unit multiplier. Raw actor and stored actions unchanged. Unit scales bypass original arithmetic exactly. Not an actual-contact-conditioned mask; loaded scheduled swing feet are also attenuated.",
                "phase_sampling": "actual_target_gait_phase_cycles is copied before target/table/mask evaluation. Legacy applied_gait_phase_cycles aliases the integrated accumulator, so after stepping it reports the next phase; this historical reward/trace behavior is deliberately unchanged for the paired experiment. At nominal 0.30-0.35 Hz, one control step is 0.005-0.0058333 cycles.",
            },
            "checkpoint_state_key": "actor_state_dict",
            "phase_reference_sampling": self.phase_reference_sampling,
            "source_files": {
                label: {"path": str(Path(inspect.getfile(cls)).resolve()),
                        "sha256": file_sha256(Path(inspect.getfile(cls)))}
                for label, cls in (("actor", type(actor)), ("normalizer", type(normalizer)),
                                   ("distribution", type(distribution)), ("vector_wrapper", type(vec_env)))
            },
            "policy_feature_slices_zero_based_end_exclusive": {
                "command_vx_vy_yaw_scaled": [0, 3],
                "gait_sin_cos": [3, 5],
                "imu_angular_velocity_body_rad_s": [5, 8],
                "projected_gravity_body": [8, 11],
                "imu_linear_acceleration_body_g": [11, 14],
                "joint_position_minus_default_rad": [14, 26],
                "joint_velocity_normalized": [26, 38],
                "action_history": [38, 50],
            },
            "command_observation_scale": self._values(robot_env._command_observation_scale),
            "joint_velocity_normalization_rad_s": float(SERVO_VELOCITY_LIMIT_RAD_S),
            "default_joint_position_rad": self._values(robot_env._robot.data.default_joint_pos.torch[0]),
            "action_history_mode": robot_env.cfg.action_history_mode,
            "policy_feature_clamp_absolute": 20.0,
            "initialization_version": INITIALIZATION_VERSION,
            "initial_observation_semantics": "For exact V44, the existing manual reset and final phase/command schedule are followed by kinematics forward and exactly one fresh _get_observations assignment to obs_buf before the wrapper getter. No extra physics step or policy call. Older captures used a cached getter and may contain stale first inputs; versioned reports must not be silently pooled.",
            **({"actor_velocity_observation_source": "causal_pre_position_fd_60hz.v1",
                "original_observation_field": "original_actor_observation_before_fd",
                "velocity_alignment_warning": "actor_observation_before_normalizer is the ACTUAL modified input. Its qdot channels are NOT expected to equal raw pre-step PhysX qdot; the original full observation is retained separately. Do not run a raw-qdot-only replay parser without checking this source contract."}
               if args.actor_joint_velocity_fd else {}),
        }
        self.reference_sources = {
            field: {"path": str(Path(value).resolve()), "sha256": file_sha256(Path(value))}
            for field in ("gait_phase_reference_path", "gait_phase_reference_low_path")
            if (value := getattr(robot_env.cfg, field, None)) is not None
        }

    @staticmethod
    def _values(tensor):
        return tensor.detach().cpu().tolist()

    def _exact_vector(self, tensor, size, name):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1, size):
            raise ValueError(f"{name} must be the actual [1,{size}] tensor")
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{name} must contain finite float32 values")
        return self._values(tensor[0])

    def capture_pre_inference(self, obs, episode, step):
        if not isinstance(obs, self.tensor_dict_type):
            raise ValueError("Exact actor input tracing requires the actual TensorDict input")
        data = self.env._robot.data
        return {
            "actor_observation_before_normalizer": self._exact_vector(obs["policy"], 50, "Actor policy observation"),
            "actor_observation_dtype": str(obs["policy"].dtype),
            "actor_observation_container_class": f"{type(obs).__module__}.{type(obs).__name__}",
            "evaluation_reset_identity": f"{self.trace_id}:episode:{episode}",
            "pre_step_control_step_index": step,
            "pre_step_episode_sim_time_s": step * self.env.step_dt,
            "pre_step_steps_since_reset": int(self.env._steps_since_reset[0].item()),
            "pre_step_episode_length_buf": int(self.env.episode_length_buf[0].item()),
            "pre_step_common_control_counter": int(self.env.common_step_counter),
            "pre_step_physics_counter": int(self.env._sim_step_counter),
            "pre_step_physics_counter_time_s": int(self.env._sim_step_counter) * float(self.env.physics_dt),
            "pre_step_articulation_cache_timestamp_s": float(data._sim_timestamp),
            "pre_step_joint_position_rad": self._exact_vector(data.joint_pos.torch, 12, "Pre-step joint position"),
            "pre_step_joint_velocity_rad_s": self._exact_vector(data.joint_vel.torch, 12, "Pre-step joint velocity"),
        }

    def _rear_pitch_capture(self):
        env, saved = self.env, self.env._rear_pitch_lift_last
        if saved is None or saved["physics_counter_before_step"] + env.cfg.decimation != int(env._sim_step_counter):
            raise ValueError("Rear-pitch mapping snapshot is absent or stale")
        if env._rear_tracking_feedback_gain and (
            env._rear_tracking_feedback_last is None or
            env._rear_tracking_feedback_last["physics_counter_before_step"] != saved["physics_counter_before_step"]
        ):
            raise ValueError("Rear tracking feedback snapshot is absent or stale")
        if env._front_stance_tracking_feedback_gain and (
            env._front_stance_tracking_feedback_last is None or
            env._front_stance_tracking_feedback_last["physics_counter_before_step"] != saved["physics_counter_before_step"]
        ):
            raise ValueError("Front stance feedback snapshot is absent or stale")
        pre = saved["pre_overlay_requested_target_rad"]
        mapped = saved["mapped_pre_bounds_target_rad"]
        delivered = env._processed_actions
        geometry = {name: env._rear_pitch_target_geometry(target) for name, target in (
            ("pre_overlay", pre), ("mapped_requested", mapped), ("delivered", delivered))}
        return {
            **({"tracking_feedback": {
                key: self._values(value) if isinstance(value, torch.Tensor) else value
                for key,value in env._rear_tracking_feedback_last.items()
            }} if env._rear_tracking_feedback_gain else {}),
            **({"front_stance_feedback": {
                key: self._values(value) if isinstance(value, torch.Tensor) else value
                for key,value in env._front_stance_tracking_feedback_last.items()
            }} if env._front_stance_tracking_feedback_gain else {}),
            "amplitude_m": float(env.cfg.gait_rear_pitch_lift_m),
            "mapping_status": "identity_bypass" if not env._rear_pitch_lift_enabled else (
                "mapped" if bool(torch.any(saved["requested_added_height_m"] > 1.e-12).item()) else "inactive_exact_bypass"),
            "physics_counter_before_step": saved["physics_counter_before_step"],
            "pre_step_age": int(saved["pre_step_age"][0].item()),
            "actual_target_phase_cycles": float(saved["actual_target_phase_cycles"][0].item()),
            "pre_overlay_requested_target_rad": self._exact_vector(pre, 12, "Pre-overlay target"),
            "mapped_pre_bounds_target_rad": self._exact_vector(mapped, 12, "Mapped target"),
            "requested_added_height_m": self._values(saved["requested_added_height_m"][0]),
            "geometry_by_target": {name: {key: self._values(value[0]) for key, value in values.items()}
                                   for name, values in geometry.items()},
            "mapped_minus_pre_body_X_m": self._values((geometry["mapped_requested"]["body_tread_center_m"] - geometry["pre_overlay"]["body_tread_center_m"])[0,:,0]),
            "mapped_minus_pre_hip_knee_sum_rad": self._values((geometry["mapped_requested"]["hip_knee_sum_rad"] - geometry["pre_overlay"]["hip_knee_sum_rad"])[0]),
            "delivered_minus_mapped_body_X_m": self._values((geometry["delivered"]["body_tread_center_m"] - geometry["mapped_requested"]["body_tread_center_m"])[0,:,0]),
            "delivered_minus_mapped_hip_knee_sum_rad": self._values((geometry["delivered"]["hip_knee_sum_rad"] - geometry["mapped_requested"]["hip_knee_sum_rad"])[0]),
            "delivery_difference_semantics": "Within this trajectory, includes original bounds/delay/packet/rate history; not a paired overlay-only effect and not a delivered-pitch invariant.",
            "analytic_evidence": {key: self._values(value) for key, value in saved.items() if key in (
                "startup_overlay_weight", "command_overlay_weight", "positive_lift_env_ids",
                "positive_lift_rear_leg_indices", "signed_requested_body_X_change_m", "achieved_lift_m",
                "original_level_root_tilt_rad", "branch_sign", "minimum_joint_bound_margin_rad",
                "requested_hip_knee_sum_error_rad")},
        }

    def _rear_projection_capture(self):
        saved = self.env._rear_pitch_lift_last
        projection = None if saved is None else saved.get("feasible_projection")
        if projection is None or projection["physics_counter_before_step"] + self.env.cfg.decimation != int(self.env._sim_step_counter):
            raise ValueError("Common-F5 projection snapshot is absent or stale")
        if projection["schema"] != "drobot.v44-rear-pitch-common-F5-projection.v1":
            raise ValueError("Unknown common-F5 projection diagnostic schema")
        self._exact_vector(projection["raw_requested_target_rad"],12,"Raw pre-projection target")
        self._exact_vector(projection["projected_requested_target_rad"],12,"Projected target")
        def snapshot(value):
            if isinstance(value,torch.Tensor):
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError("Nonfinite common-F5 diagnostic; inactive intervals must be explicit zeros")
                return self._values(value)
            if isinstance(value,dict):
                return {key:snapshot(item) for key,item in value.items()}
            if isinstance(value,(list,tuple)):
                return [snapshot(item) for item in value]
            return value
        return snapshot(projection)

    def rear_projection_summary(self):
        """Buffered records only; no additional actor, observation or physics calls."""
        legs = ("rear_left","rear_right")
        groups = {stage:{leg:{"samples":0,"active_samples":0,"changed_samples":0,
                    "maximum_joint_projection_rad":0.,"minimum_sum_interval_width_rad":None,
                    "minimum_conditional_hip_interval_width_rad":None} for leg in legs}
                  for stage in ("run","ramp","hold")}
        untouched = True
        ids = tuple(tuple(int(v) for v in values.tolist()) for values in self.env._rear_pitch_joint_ids)
        mutable = set(ids[1]+ids[2])
        for row in self.rows:
            data = row["rear_pitch_projection"]
            raw,projected = data["raw_requested_target_rad"][0],data["projected_requested_target_rad"][0]
            untouched = untouched and all(raw[i]==projected[i] for i in range(12) if i not in mutable)
            for index,leg in enumerate(legs):
                item = groups[row["stage"]][leg]
                item["samples"]+=1
                item["active_samples"]+=int(data["active_by_leg"][0][index])
                item["changed_samples"]+=int(data["changed_by_leg"][0][index])
                item["maximum_joint_projection_rad"] = max(item["maximum_joint_projection_rad"],
                    *(abs(projected[i]-raw[i]) for i in (ids[1][index],ids[2][index])))
                if data["active_by_leg"][0][index]:
                    for field,key in (("sum_interval_rad","minimum_sum_interval_width_rad"),
                                      ("conditional_hip_interval_rad","minimum_conditional_hip_interval_width_rad")):
                        lo,hi = data[field][0][index]
                        item[key] = hi-lo if item[key] is None else min(item[key],hi-lo)
        for group in groups.values():
            for item in group.values():
                item["changed_fraction_of_active_samples"] = (item["changed_samples"]/item["active_samples"]
                                                               if item["active_samples"] else None)
        return {"by_phase_and_leg":groups,"all_front_and_abduction_targets_unchanged":untouched,
                "semantics":"Nonterminal recorded controller ticks, includes startup. Positive common envelope defines active; this is proposed-target projection, not motor tracking/effort or qualification."}

    def write_mapping_failure(self, error, episode, step, phase, actor_capture,
                              completed_episodes, initializations):
        """Publish exclusive partial evidence before cleanup; never invent a post-step sample."""
        report = {
            "schema": "drobot.run-stop-evaluation.v2", "status": "mapping_rejected",
            "qualification_status": "UNQUALIFIED_REAR_PITCH_PROJECTION_ABLATION" if rear_projection_evaluation else "UNQUALIFIED_REAR_PITCH_LIFT_ABLATION",
            "rear_pitch_lift": self.env._rear_pitch_lift_contract,
            **({"rear_pitch_projection": self.env._rear_pitch_projection_contract} if rear_projection_evaluation else {}),
            "production_export_allowed": False, "qualifies_hardware": False,
            "qualifies_running_goal": False, "gates": None, "gates_evaluated": False,
            "passes_all_gates": False, "task": args.task, "checkpoint": checkpoint,
            **evaluation_source_provenance,
            "seed": args.seed, "environment_ablation": ablation_provenance,
            "initialization_version": INITIALIZATION_VERSION,
            "episode_observation_initializations": initializations,
            "episodes": completed_episodes,
            "completed_nonterminal_sample_count": len(self.rows),
            "mapping_failure": {"reason": error.reason, "details": error.details},
            "failed_attempted_step": {
                "episode_index": episode, "step_index": step,
                "stage": "ramp" if phase == "deceleration" else phase,
                "actor_capture": actor_capture,
                "physics_counter_after_exception": int(self.env._sim_step_counter),
                "physics_did_not_advance": int(self.env._sim_step_counter) == actor_capture["pre_step_physics_counter"],
                "not_a_completed_step_or_observed_physical_fall": True,
            },
            "lifecycle_note": "Evidence is published before env/Kit cleanup. External process exit and artifact audit remain required; Kit fast shutdown may bypass later Python code.",
            **_sustained_credit_metadata(),
        }
        metadata = {
            "record_type": "metadata", "schema": self.SCHEMA, "trace_id": self.trace_id,
            "status": "partial_mapping_rejected", "sample_count": len(self.rows),
            "evaluation_report_path": str(args.output.resolve()),
            "checkpoint": checkpoint, **evaluation_source_provenance,
            "task": args.task, "seed": args.seed,
            "rear_pitch_lift": self.env._rear_pitch_lift_contract,
            **({"rear_pitch_projection": self.env._rear_pitch_projection_contract} if rear_projection_evaluation else {}),
            "qualification_status": report["qualification_status"],
            "environment_ablation": ablation_provenance,
            "reference_sources_at_trace_start": self.reference_sources,
            "actor_input_output_contract": self.actor_contract,
            "joint_names": list(self.env._robot.joint_names),
            "foot_names": list(self.env._foot_leg_names),
            "step_dt_s": self.env.step_dt, "physics_dt_s": self.env.cfg.sim.dt,
            "initialization_version": INITIALIZATION_VERSION,
            "episode_observation_initializations": initializations,
            "mapping_failure": report["mapping_failure"],
            "failed_attempted_step": report["failed_attempted_step"],
            "semantics": "Only completed nonterminal steps follow. The rejected attempt did not receive a synthetic/fallback target or post-physics row; no numeric qualification was evaluated.",
            **_sustained_credit_metadata(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("x", encoding="utf-8") as stream:
                for row in (metadata, *self.rows):
                    stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            report["diagnostic_trace"] = {"path": str(self.path), "sha256": file_sha256(self.path),
                                          "sample_count": len(self.rows), "status": "partial_mapping_rejected"}
        except Exception as write_error:
            report["partial_trace_publication_error"] = f"{type(write_error).__name__}: {write_error}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, allow_nan=False, indent=2)
            stream.write("\n")

    def record(self, episode, step, phase, command_applied, packet_elapsed_s,
               foot_forces, foot_contact, scheduled_contact, run_warmup_steps,
               target_delta_limit, target_ceiling_active, actor_capture):
        env, data = self.env, self.env._robot.data
        effort_capture = {}
        if env.cfg.native_effort_telemetry_enabled:
            counters = env._native_effort_last_physics_counter
            if tuple(counters.shape) != (1,) or counters.dtype != torch.int64:
                raise ValueError("Native-effort trace requires one integer per-environment physics counter")
            counter = int(counters[0].item())
            if counter != int(env._sim_step_counter):
                raise ValueError("Native-effort reward snapshot is stale for this trace step")
            native = self._exact_vector(env._native_effort_last_nm, 12, "Fresh active generalized effort")
            retained_pd = self._exact_vector(env._native_effort_last_pd_nm, 12, "Retained pre-solve approximate PD")
            if retained_pd != self._exact_vector(data.applied_torque.torch, 12, "Current approximate PD"):
                raise ValueError("Approximate-PD snapshot changed after the reward boundary")
            effort_capture = {
                "native_active_generalized_effort_nm_NOT_drive_only": native,
                "reward_boundary_retained_clipped_pd_nm": retained_pd,
                "effort_observable_physics_counter": counter,
                "effort_reward_observable": "native_active_generalized_effort" if env.cfg.native_effort_reward_enabled else "retained_clipped_approximate_pd",
            }
        response = getattr(env, "_actuator_response_targets", None)
        response_enabled = bool(getattr(env, "_actuator_target_response_enabled", False))
        response_status = "available" if response_enabled and response is not None else (
            "filter_disabled" if not response_enabled else "filter_target_attribute_unavailable"
        )
        body_masses = data.body_mass.torch[0]
        body_com_positions = data.body_com_pos_w.torch[0]
        whole_robot_com = torch.sum(body_masses.unsqueeze(-1) * body_com_positions, dim=0) / torch.sum(body_masses)
        feet = None
        if bool(getattr(env, "_sole_diagnostics_enabled", False)):
            ids = env._sole_body_ids
            sole_quaternion = data.body_link_quat_w.torch[0, ids]
            offsets = _rotate_vectors_xyzw(sole_quaternion.unsqueeze(1), env._sole_local_corners.unsqueeze(0))
            corners_w = data.body_link_pos_w.torch[0, ids].unsqueeze(1) + offsets
            minimum_z = torch.amin(corners_w[..., 2], dim=1)
            normal = _rotate_vectors_xyzw(sole_quaternion, torch.tensor((1., 0., 0.), device=env.device))
            tilt_rad = torch.acos(torch.clamp(-normal[:, 2], -1., 1.))
            feet = {
                "names": list(env._foot_leg_names),
                "sole_corners_world_m": self._values(corners_w),
                "minimum_corner_world_z_m": self._values(minimum_z),
                "minimum_corner_height_above_terrain_origin_m": self._values(minimum_z - env._terrain.env_origins[0, 2]),
                "sole_tilt_from_level_rad": self._values(tilt_rad),
            }
        actual_target_phase = float(env._applied_reference_phase_for_diagnostics[0].item())
        phase_indices = None
        if self.phase_reference_sampling["sample_count"] == 512:
            if not math.isfinite(actual_target_phase):
                raise ValueError("Nonfinite saved target phase in interpolation trace")
            # Binary multiplication by512 is exact for the saved float32 phase.
            # This derives indices without reading the advanced accumulator or
            # overwriting metadata during subsequent reward/observation calls.
            scaled_phase = actual_target_phase * 512
            floor_phase = math.floor(scaled_phase)
            phase_indices = {
                "source": "derived_from_recorded_float32_actual_target_phase",
                "lower_row": floor_phase % 512,
                "upper_row": (floor_phase + 1) % 512,
                "within_cell_fraction": scaled_phase - floor_phase,
                "position_interpolation_fraction": scaled_phase - floor_phase
                if self.phase_reference_sampling["enabled"] else 0.0,
            }
        self.rows.append({
            "record_type": "step", "episode_index": episode, "step_index": step,
            **actor_capture,
            **({"rear_pitch_lift": self._rear_pitch_capture()} if rear_pitch_evaluation else {}),
            **({"rear_pitch_projection": self._rear_projection_capture()} if rear_projection_evaluation else {}),
            "environment_received_action": self._exact_vector(env._actions, 12, "Environment action"),
            "applied_swing_residual_multiplier_by_joint": self._exact_vector(
                env._applied_swing_residual_multiplier, 12, "Applied swing residual multiplier"
            ),
            "post_step_physics_counter": int(env._sim_step_counter),
            "post_step_articulation_cache_timestamp_s": float(data._sim_timestamp),
            "steps_since_reset": int(env._steps_since_reset[0].item()),
            "episode_sim_time_s": (step + 1) * env.step_dt,
            "interval_start_sim_time_s": step * env.step_dt,
            "stage": "ramp" if phase == "deceleration" else phase,
            "included_in_steady_run_metrics": phase == "run" and step >= run_warmup_steps,
            "command_applied_vx_vy_yawrate": command_applied,
            "command_next_observation_vx_vy_yawrate": self._values(env._commands[0]),
            "applied_gait_phase_cycles": float(env._applied_gait_phase[0].item()),
            "actual_target_gait_phase_cycles": actual_target_phase,
            "phase_reference_sampling_indices": phase_indices,
            "reported_minus_target_phase_cycles_mod1": float(torch.remainder(
                env._applied_gait_phase[0] - env._applied_reference_phase_for_diagnostics[0], 1.0
            ).item()),
            "packet_elapsed_budget_s": packet_elapsed_s,
            "packet_held": bool(env._target_packet_held[0].item()),
            "target_packet_allowed_delta_rad": self._values(target_delta_limit),
            "target_velocity_ceiling_active": self._values(target_ceiling_active),
            "requested_target_rad": self._values(env._desired_targets[0]),
            "delivered_target_rad": self._values(env._processed_actions[0]),
            "response_filtered_target_rad": self._values(response[0]) if response_status == "available" else None,
            "response_filtered_target_status": response_status,
            "response_tau_s": self._values(env._actuator_response_tau_s[0]) if response_status == "available" else None,
            "response_gain": self._values(env._actuator_response_gain[0]) if response_status == "available" else None,
            "joint_position_rad": self._values(data.joint_pos.torch[0]),
            "joint_velocity_rad_s": self._values(data.joint_vel.torch[0]),
            "applied_torque_nm": self._values(data.applied_torque.torch[0]),
            **effort_capture,
            "effort_limit_nm": self._values(env._actuator_effort_limit_by_joint[0]),
            "root_position_world_m": self._values(data.root_pos_w.torch[0]),
            "root_quaternion_world_xyzw": self._values(data.root_quat_w.torch[0]),
            "root_linear_velocity_world_m_s": self._values(data.root_lin_vel_w.torch[0]),
            "root_angular_velocity_world_rad_s": self._values(data.root_ang_vel_w.torch[0]),
            "whole_robot_com_world_m": self._values(whole_robot_com),
            "body_com_positions_world_m": self._values(body_com_positions),
            "body_masses_kg": self._values(body_masses),
            "episode_start_position_world_m": self._values(env._episode_start_position[0]),
            "episode_lateral_direction_world_xy": self._values(env._episode_lateral_direction_w[0]),
            "applied_expected_lateral_displacement_m": float(env._applied_expected_lateral_displacement[0].item()),
            "applied_expected_lateral_velocity_m_s": float(env._applied_expected_lateral_velocity[0].item()),
            "applied_expected_forward_displacement_m": float(env._applied_expected_forward_displacement[0].item()),
            "applied_expected_forward_velocity_m_s": float(env._applied_expected_forward_velocity[0].item()),
            "foot_force_norm_n": self._values(foot_forces),
            **({"foot_net_force_world_xyz_n": self._values(env._contact_sensor.data.net_forces_w.torch[0, env._foot_sensor_ids])}
               if args.schedule_aware_crawl else {}),
            "foot_contact": self._values(foot_contact),
            "scheduled_contact": self._values(scheduled_contact),
            "scheduled_release": self._values((~scheduled_contact) & (~foot_contact)),
            "feet": feet,
        })

    def write(self, report):
        metadata_record = {
            "record_type": "metadata", "schema": self.SCHEMA, "trace_id": self.trace_id,
            "evaluation_report_path": str(args.output.resolve()) if args.output is not None else None,
            "checkpoint": report["checkpoint"], **evaluation_source_provenance,
            "task": report["task"], "seed": report["seed"], "command": report["command"],
            "domain_mode": report["domain_mode"], "static_domain_parameters": report["static_domain_parameters"],
            "policy_execution_mode": report["policy_execution_mode"],
            **({"rear_pitch_lift": report["rear_pitch_lift"],
                "qualification_status": report["qualification_status"],
                "production_export_allowed": False} if rear_pitch_evaluation else {}),
            **({"rear_pitch_projection": report["rear_pitch_projection"]} if rear_projection_evaluation else {}),
            **({"rear_pitch_projection_summary": report["rear_pitch_projection_summary"]} if rear_projection_evaluation else {}),
            **_sustained_credit_metadata(),
            **({"qualification_status": report["qualification_status"],
                "actor_joint_velocity_fd_diagnostic": report["actor_joint_velocity_fd_diagnostic"]}
               if args.actor_joint_velocity_fd else {}),
            "fixed_start_phase_zero": report["fixed_start_phase_zero"], "phase_alignment": report["phase_alignment"],
            "environment_ablation": ablation_provenance, "reference_sources_at_trace_start": self.reference_sources,
            "phase_reference_sampling": self.phase_reference_sampling,
            "joint_names": list(self.env._robot.joint_names), "foot_names": list(self.env._foot_leg_names),
            "body_names": list(self.env._robot.body_names),
            "gait_phase_offsets_cycles": self._values(self.env._foot_phase_offsets),
            "gait_duty_factor": self.env.cfg.gait_duty_factor,
            "gait_diagnostic_config": {
                "gait_start_ramp_s": float(self.env.cfg.gait_start_ramp_s),
                "sole_clearance_transition_fraction": float(self.env._sole_clearance_transition_fraction),
                "gait_standstill_deadband_m_s": float(self.env.cfg.gait_standstill_deadband_m_s),
                "sole_clearance_target_m": float(self.env._sole_clearance_target_m),
            },
            "controller_measurement_contract": controller_measurement_contract,
            "effort_observable": report["effort_observable"],
            "actor_input_output_contract": self.actor_contract,
            "initialization_version": report["initialization_version"],
            "episode_observation_initializations": report["episode_observation_initializations"],
            "step_dt_s": self.env.step_dt, "physics_dt_s": self.env.cfg.sim.dt,
            "sample_count": len(self.rows),
            "episode_outcomes": [{key: episode[key] for key in (
                "episode", "fell", "fall_phase", "completed_seconds", "domain_parameters"
            )} for episode in report["episodes"]],
            **({"sequential_crawl_evaluation": report["sequential_crawl_evaluation"]}
               if args.schedule_aware_crawl else {}),
            **({"physx_effort_readback": report["physx_effort_readback"]}
               if args.physx_effort_readback else {}),
            "semantics": {
                "sampling": "Post-physics, after each nonterminal env.step; includes startup. Terminal auto-reset frames omitted.",
                "actor_capture": (
                    "Exact modified actor TensorDict policy input copied before the single inference call; original_actor_observation_before_fd preserves the unmodified environment input. Only26:38 differ. Raw actor output is the output actually executed on this FD-driven trajectory, not a second raw-input counterfactual. No extra observation generation, policy call or random draw."
                    if args.actor_joint_velocity_fd else
                    "Exact existing TensorDict policy input copied before the single inference call; raw actor output copied before zero override; submitted_action is passed to vector-wrapper env.step. environment_received_action is the task's stored action after wrapper and environment clipping. No extra observation generation, policy call or random draw."
                ),
                "pre_step_state": (
                    "Actor FD uses current PRE q minus preceding consecutive PRE q over actual physics-counter interval. First sample/reset uses zero. Current row POST-minus-PRE is a future outcome relative to this actor input and is never fed into it. Raw pre_step_joint_velocity_rad_s remains PhysX velocity; actual actor qdot has separate FD provenance."
                    if args.actor_joint_velocity_fd else
                    "Copied before inference without advancing physics. For finite differences use this row's post q minus pre q over actual physics-counter delta times physics_dt; never bridge reset identities or assume cached initial observation equals reset state."
                ),
                "time": "Episode-local simulated seconds, not wall time. Applied command/delivered target identify the interval; filtered target and state are last-substep snapshots, not interval averages.",
                "ordering": "All joint vectors follow joint_names; force/contact vectors and feet arrays follow foot_names. Indices are zero-based.",
                "commands": "[vx m/s, vy m/s, yaw rate rad/s]; applied captured before env.step, next_observation may advance the stop schedule/heading correction.",
                "requested_target_rad": "env._desired_targets, after action mapping/joint bounds/control-delay selection, before packet limiting/holding.",
                "delivered_target_rad": "env._processed_actions, controller-delivered packet before actuator response filtering.",
                "target_packet_allowed_delta_rad": "Per-joint randomized target rate times the elapsed budget captured before this control step, capped by the packet step limit. This is the potential delivery allowance; held packets deliver no change.",
                "target_velocity_ceiling_active": "Delivered target change >=98% of target_packet_allowed_delta_rad AND packet not held. Run aggregate counts all joints and all selected run ticks; held ticks remain zero-valued denominator samples.",
                "response_filtered_target_rad": "env._actuator_response_targets after the final physics-substep filter update. Null when disabled/unavailable; never synthesized from q.",
                "applied_torque_nm": "Isaac Lab data.applied_torque telemetry; implicit-actuator torque is an approximation, not measured motor/PhysX joint torque.",
                "effort_limit_nm": "env._actuator_effort_limit_by_joint, the per-joint randomized limit written to both actuator telemetry and the PhysX drive.",
                "feet": "World-space sole corners and their minimum height; null if sole diagnostics unavailable. Force is the norm of sensor net world force, not vertical force.",
                "whole_robot_com_world_m": "Mass-weighted sum of every simulated rigid body's world COM, including the combined rear payload. Body vectors follow body_names. This is not the root-link origin or an observed hardware COM; projected support margin alone is not a dynamic stability certificate.",
                "applied_expected_lateral_displacement_m": "Cached planned body sway for the applied reference, relative to episode start and projected along episode_lateral_direction_world_xy; a training reference, not measured body displacement or guaranteed kinematics during joint-space blending.",
                "contact": f"foot_contact is force norm > {self.env.cfg.scheduled_release_force_threshold_n} N; scheduled_release means scheduled swing AND no force-confirmed contact.",
                "buffering": "Snapshots buffered in memory; JSON serialization and file writes occur only after evaluation. Not a host/controller timing benchmark.",
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also catches another process creating this path after startup.
        with self.path.open("x", encoding="utf-8") as stream:
            for row in (metadata_record, *self.rows):
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        return {"schema": self.SCHEMA, "trace_id": self.trace_id, "path": str(self.path),
                "sha256": file_sha256(self.path), "sample_count": len(self.rows)}


class _ActorJointVelocityFD:
    """Evaluator-only causal actor-input intervention; never mutate environment state."""

    def __init__(self, robot_env, trace):
        from parallel_walking.commanded_walking_env import EXPECTED_ACTION_ORDER
        from parallel_walking.commanded_walking_env_cfg import SERVO_VELOCITY_LIMIT_RAD_S

        if (args.task != V44_TASK or ablation_provenance is None or trace is None
            or INITIALIZATION_VERSION != "v44.finalized-reset-fresh-observation.v1"
            or robot_env.num_envs != 1 or tuple(robot_env._robot.joint_names) != EXPECTED_ACTION_ORDER
            or robot_env.cfg.action_mode != "gait_residual" or robot_env.cfg.gait_reference_mode != "phase_table"
            or any(value != 1.0 for value in robot_env.cfg.gait_swing_residual_scale_by_leg)
            or robot_env.cfg.gait_swing_residual_ablation_task is not None):
            raise ValueError("Actor FD requires the audited single V44 fresh-reset, identity-mask phase-table/gait-residual context")
        self.env, self.trace = robot_env, trace
        self.physics_dt = float(robot_env.physics_dt)
        self.decimation = int(robot_env.cfg.decimation)
        self.velocity_limit = float(SERVO_VELOCITY_LIMIT_RAD_S)
        if (self.decimation != 2 or not math.isclose(self.physics_dt, 1.0 / 120.0, rel_tol=0.0, abs_tol=1.e-12)
            or not math.isclose(float(robot_env.step_dt), self.decimation * self.physics_dt, rel_tol=0.0, abs_tol=1.e-12)
            or not math.isclose(self.velocity_limit, 4.5836625, rel_tol=0.0, abs_tol=1.e-9)):
            raise ValueError("Actor FD diagnostic requires the audited 120 Hz physics / 60 Hz control and velocity normalization")
        self.episode = None
        self.previous_q = None
        self.previous_counter = None
        self.previous_step = None
        self.previous_cache_timestamp = None
        self.metadata = {
            "schema": "drobot.actor-only-joint-velocity-fd.v1",
            "qualification_status": "UNQUALIFIED_OBSERVATION_DIAGNOSTIC",
            "replaced_policy_slice_zero_based_end_exclusive": [26, 38],
            "velocity_limit_and_normalizer_rad_s": self.velocity_limit,
            "expected_physics_counter_delta": self.decimation,
            "physics_dt_s": self.physics_dt,
            "first_sample_after_explicit_reset": "zero velocity; no interval bridged across reset",
            "estimator": "(current PRE-inference q - previous PRE-inference q) / (actual physics-counter delta * physics_dt), clipped before normalization; never use the current action's future post-step response",
            "mutation_scope": "Shallow clone TensorDict, clone only policy tensor, replace only qdot slice. Original obs_buf, critic, physics qdot, reward histories and normalizer state untouched. One existing actor call and one existing env.step per tick.",
            "hardware_parity_limit": "60 Hz simulation position derivative only; NOT the Pi's asynchronous 100 Hz quantized USB-feedback/sample-and-hold process",
            "comparison_limit": "Closed-loop trajectories diverge; no second actor call supplies a same-state raw-input counterfactual action. Gate definitions remain unchanged, but passing numeric gates is not policy qualification.",
        }

    def reset(self, episode, initialization):
        if (initialization is None or initialization.get("episode") != episode
            or initialization.get("initialization_version") != INITIALIZATION_VERSION
            or initialization.get("fresh_observation_generation_count") != 1
            or initialization.get("counters_unchanged_by_refresh") is not True
            or int(self.env._steps_since_reset[0].item()) != 0):
            raise ValueError("Actor FD history requires the completed, explicit fresh episode initialization")
        self.episode = episode
        self.previous_q = self.previous_counter = self.previous_step = self.previous_cache_timestamp = None

    def prepare(self, obs, episode, step):
        if episode != self.episode or not isinstance(obs, self.trace.tensor_dict_type):
            raise ValueError("Actor FD requires the current reset identity and actual TensorDict observation")
        original = obs["policy"]
        original_values = self.trace._exact_vector(original, 50, "Original actor observation before FD")
        data = self.env._robot.data
        current_q = data.joint_pos.torch.clone()
        current_q_values = self.trace._exact_vector(current_q, 12, "Actor FD current PRE joint position")
        raw_velocity = data.joint_vel.torch
        self.trace._exact_vector(raw_velocity, 12, "Actor FD raw PRE physics velocity")
        expected_q = (current_q - data.default_joint_pos.torch).clamp(-20.0, 20.0)
        expected_raw_velocity = (raw_velocity / self.velocity_limit).clamp(-20.0, 20.0)
        if (not torch.allclose(original[:, 14:26], expected_q, rtol=0.0, atol=2.e-6)
            or not torch.allclose(original[:, 26:38], expected_raw_velocity, rtol=0.0, atol=2.e-6)):
            raise RuntimeError("Original FD source observation is not aligned to current raw PRE q/qdot")
        counter = int(self.env._sim_step_counter)
        cache_timestamp = float(data._sim_timestamp)
        if not math.isfinite(cache_timestamp) or counter < 0 or int(self.env._steps_since_reset[0].item()) != step:
            raise RuntimeError("Actor FD PRE counter/reset alignment is invalid")
        first = self.previous_q is None
        if first:
            if step != 0:
                raise RuntimeError("Actor FD cannot initialize mid-episode")
            delta_counter, interval_s = None, None
            fd = torch.zeros_like(current_q)
        else:
            delta_counter = counter - self.previous_counter
            if step != self.previous_step + 1 or delta_counter != self.decimation:
                raise RuntimeError("Actor FD requires consecutive PRE boundaries separated by exactly two physics steps")
            interval_s = delta_counter * self.physics_dt
            if not math.isclose(cache_timestamp - self.previous_cache_timestamp, interval_s, rel_tol=0.0, abs_tol=1.e-6):
                raise RuntimeError("Actor FD articulation timestamps disagree with the actual physics-counter interval")
            fd = (current_q - self.previous_q) / interval_s
        fd_values = self.trace._exact_vector(fd, 12, "Unclipped actor PRE-position FD velocity")
        clipped = fd.clamp(-self.velocity_limit, self.velocity_limit)
        actor_policy = original.clone()
        actor_policy[:, 26:38] = clipped / self.velocity_limit
        actor_obs = obs.clone(recurse=False)
        actor_obs["policy"] = actor_policy
        if (actor_policy.data_ptr() == original.data_ptr()
            or not torch.equal(actor_policy[:, :26], original[:, :26])
            or not torch.equal(actor_policy[:, 38:], original[:, 38:])):
            raise RuntimeError("Actor FD intervention changed or aliased an unrequested observation channel")
        provenance = {
            "original_actor_observation_before_fd": original_values,
            "actor_joint_velocity_fd": {
                "schema": self.metadata["schema"], "episode_index": episode,
                "first_sample_after_reset": first,
                "current_pre_joint_position_rad": current_q_values,
                "previous_pre_joint_position_rad": self.trace._values(self.previous_q[0]) if not first else None,
                "current_physics_counter": counter, "previous_physics_counter": self.previous_counter,
                "physics_counter_delta": delta_counter, "interval_s": interval_s,
                "current_cache_timestamp_s": cache_timestamp, "previous_cache_timestamp_s": self.previous_cache_timestamp,
                "unclipped_velocity_rad_s": fd_values,
                "clipped_velocity_rad_s": self.trace._exact_vector(clipped, 12, "Clipped actor FD velocity"),
                "clipped_components": self.trace._values((fd.abs() > self.velocity_limit)[0]),
                "unchanged_non_velocity_channels_exact": True,
            },
        }
        self.previous_q = current_q
        self.previous_counter, self.previous_step, self.previous_cache_timestamp = counter, step, cache_timestamp
        return actor_obs, provenance


def _sequential_crawl_schedule(robot_env):
    """Explicit conservative classification, not a user-provided gait label."""
    offsets = [float(value) % 1. for value in robot_env._foot_phase_offsets.detach().cpu().tolist()]
    duty = float(robot_env.cfg.gait_duty_factor)
    ordered = sorted(offsets)
    gaps = [(ordered[(index + 1) % 4] - ordered[index]) % 1. for index in range(4)]
    if len(offsets) != 4 or any(not math.isfinite(value) for value in offsets) or not (.75 <= duty < 1.) or any(abs(gap - .25) > 1.e-6 for gap in gaps):
        raise ValueError("Crawl qualification requires four quarter-spaced phases and duty in [0.75,1)")
    if float(robot_env.cfg.gait_contact_transition_fraction) != 0.:
        raise ValueError("Crawl qualification currently requires zero contact-transition relabeling")
    # Compare the actual stored phase-table contact schedule when available.
    contact_table = getattr(robot_env, "_distributed_reference_contact", None)
    if contact_table is not None and bool(torch.any((~contact_table).sum(dim=1) > 1)):
        raise ValueError("Stored reference schedule contains simultaneous swing legs")
    return {
        "classification": "quarter_spaced_sequential_crawl",
        "phase_offsets_cycles_by_leg": dict(zip(robot_env._foot_leg_names, offsets, strict=True)),
        "duty_factor": duty, "maximum_scheduled_swing_legs": 1,
        "qualification_scope": "commanded_speed_crawl_foundation_only_not_running",
        "qualifies_running_goal": False,
    }


class _SequentialCrawlDiagnostics:
    """Simultaneous physical support/release observations, not schedule labels."""

    def __init__(self, robot_env):
        self.env = robot_env
        self.names = tuple(robot_env._foot_leg_names)
        self.dt = float(robot_env.step_dt)
        total_mass = float(robot_env._robot.data.body_mass.torch[0].sum().item())
        gravity = abs(float(robot_env.cfg.sim.gravity[2]))
        self.load_threshold_n = max(float(robot_env.cfg.scheduled_release_force_threshold_n), .03 * total_mass * gravity)
        self.count = self.three_loaded_count = self.invalid_schedule_count = self.nonfinite_force_count = 0
        self.deficit_streak = self.maximum_deficit_streak = 0
        self.legs = {name: {"scheduled_support_samples": 0, "loaded_support_samples": 0,
                            "core_samples": 0, "coordinated_core_samples": 0, "complete_core_bouts": [],
                            "active_core_bout": None, "previous_core": False}
                     for name in self.names}

    def record(self, scheduled, physical_contact, lift_envelope, corner_height, startup):
        forces = self.env._contact_sensor.data.net_forces_w.torch[0, self.env._foot_sensor_ids].detach().cpu().tolist()
        finite_force = [all(math.isfinite(value) for value in force) for force in forces]
        loaded = [finite and force[2] > self.load_threshold_n for finite, force in zip(finite_force, forces, strict=True)]
        self.nonfinite_force_count += int(not all(finite_force))
        swing = [not value for value in scheduled]
        valid_schedule = sum(swing) <= 1
        three_loaded = sum(loaded) >= 3
        self.invalid_schedule_count += int(not valid_schedule)
        self.three_loaded_count += int(three_loaded)
        self.deficit_streak = 0 if three_loaded else self.deficit_streak + 1
        self.maximum_deficit_streak = max(self.maximum_deficit_streak, self.deficit_streak)
        for index, name in enumerate(self.names):
            leg = self.legs[name]
            leg["scheduled_support_samples"] += int(scheduled[index])
            leg["loaded_support_samples"] += int(scheduled[index] and loaded[index])
            core = swing[index] and lift_envelope[index] >= .90 and startup >= .99
            coordinated = (valid_schedule and all(finite_force) and sum(swing) == 1 and not physical_contact[index]
                           and math.isfinite(corner_height[index]) and corner_height[index] >= .005
                           and all(loaded[j] for j in range(4) if scheduled[j]))
            if core:
                if not leg["previous_core"]:
                    leg["active_core_bout"] = {"samples": 0, "coordinated_samples": 0,
                                                "observed_entry": self.count > 0}
                bout = leg["active_core_bout"]
                bout["samples"] += 1
                bout["coordinated_samples"] += int(coordinated)
                leg["core_samples"] += 1
                leg["coordinated_core_samples"] += int(coordinated)
            elif leg["previous_core"]:
                bout = leg["active_core_bout"]
                if bout["observed_entry"]:
                    leg["complete_core_bouts"].append(bout)
                leg["active_core_bout"] = None
            leg["previous_core"] = core
        self.count += 1

    def report(self):
        per_leg = {}
        for name, leg in self.legs.items():
            bouts = leg["complete_core_bouts"]
            per_leg[name] = {
                "scheduled_support_samples": leg["scheduled_support_samples"],
                "loaded_support_fraction": leg["loaded_support_samples"] / leg["scheduled_support_samples"] if leg["scheduled_support_samples"] else None,
                "core_samples": leg["core_samples"],
                "coordinated_core_fraction": leg["coordinated_core_samples"] / leg["core_samples"] if leg["core_samples"] else None,
                "complete_core_bout_count": len(bouts),
                "minimum_samples_per_complete_core_bout": min((bout["samples"] for bout in bouts), default=None),
                "worst_complete_core_bout_coordinated_fraction": min(
                    (bout["coordinated_samples"] / bout["samples"] for bout in bouts), default=None),
                "partial_bout_at_end_excluded_from_complete_count": leg["active_core_bout"] is not None,
            }
        return {
            "steady_run_samples": self.count,
            "loaded_vertical_force_threshold_n": self.load_threshold_n,
            "at_least_three_loaded_feet_fraction": self.three_loaded_count / self.count if self.count else None,
            "maximum_continuous_fewer_than_three_loaded_feet_s": self.maximum_deficit_streak * self.dt,
            "unexpected_multi_swing_schedule_samples": self.invalid_schedule_count,
            "nonfinite_foot_force_samples": self.nonfinite_force_count,
            "by_leg": per_leg,
            "definition": "After startup warmup. Loaded means positive world-Z net foot force above max(existing contact threshold,3% simulated weight). Coordinated core means exactly one scheduled swing, that foot force-released and lowest corner>=5mm, with all three scheduled stance feet loaded at the same sample. Core uses lift envelope>=.90. Complete bouts require observed entry/exit; boundary partial bouts are not coverage. No COM/support-polygon or hardware load claim.",
        }


class _FlatContactDiagnostics:
    """Read-only V44 geometry/packet observations; never call reward accumulators."""

    def __init__(self, robot_env):
        self.env = robot_env
        self.leg_names = tuple(robot_env._foot_leg_names)
        self.joint_names = tuple(robot_env._robot.joint_names)
        self.phases = {
            phase: {
                leg: {key: [] for key in (
                    "support_sole_tilt_deg", "force_confirmed_sole_tilt_deg",
                    "core_swing_lowest_corner_height_m", "swing_clearance_deficit_m",
                    "contact_patch_slip_speed_m_s",
                )}
                for leg in self.leg_names
            }
            for phase in ("run", "deceleration", "hold")
        }
        self.targets = []
        self.positions = []
        self.packet_held = []
        self.corrected_ceiling = []
        self.commanded_speeds = []
        self.actual_speeds = []
        self.hold_origin = None
        self.hold_drifts = []
        self.crawl = _SequentialCrawlDiagnostics(robot_env) if args.schedule_aware_crawl else None

    def record(self, phase, scheduled_contact, foot_contact, target_ceiling_active,
               commanded_speed, actual_speed, root_position):
        env = self.env
        data = env._robot.data
        ids = env._sole_body_ids
        position = data.body_link_pos_w.torch[0, ids]
        quaternion = data.body_link_quat_w.torch[0, ids]
        offsets = _rotate_vectors_xyzw(quaternion.unsqueeze(1), env._sole_local_corners.unsqueeze(0))
        heights = (position.unsqueeze(1) + offsets)[..., 2] - env._terrain.env_origins[0, 2]
        lowest = torch.amin(heights, dim=1)
        normal = _rotate_vectors_xyzw(quaternion, torch.tensor((1., 0., 0.), device=env.device))
        tilt = torch.rad2deg(torch.acos(torch.clamp(-normal[:, 2], -1., 1.)))
        velocity = data.body_link_lin_vel_w.torch[0, ids].unsqueeze(1) + torch.cross(
            data.body_link_ang_vel_w.torch[0, ids].unsqueeze(1).expand_as(offsets), offsets, dim=-1
        )
        near_lowest = heights <= lowest.unsqueeze(1) + env._sole_corner_soft_contact_height_m
        point_speed_squared = torch.sum(velocity[..., :2] ** 2, dim=-1)
        slip = torch.sqrt(torch.sum(point_speed_squared * near_lowest, dim=1)
                          / torch.clamp(near_lowest.sum(dim=1), min=1))
        leg_phase = torch.remainder(env._applied_gait_phase[0] + env._foot_phase_offsets, 1.).reshape(-1)
        swing_u = torch.clamp((leg_phase - env.cfg.gait_duty_factor) / (1. - env.cfg.gait_duty_factor), 0., 1.)
        transition = env._sole_clearance_transition_fraction
        enter = torch.clamp(swing_u / transition, 0., 1.)
        leave = torch.clamp((1. - swing_u) / transition, 0., 1.)
        envelope = (enter**3 * (10. - 15. * enter + 6. * enter**2)
                    * leave**3 * (10. - 15. * leave + 6. * leave**2))
        startup = min(1., float(env._steps_since_reset[0].item()) * env.step_dt / env.cfg.gait_start_ramp_s)
        desired_clearance = env._sole_clearance_target_m * envelope * startup
        values = torch.stack((tilt, lowest, slip, envelope, desired_clearance), dim=1).detach().cpu().tolist()
        scheduled = scheduled_contact.detach().cpu().tolist()
        physical = foot_contact.detach().cpu().tolist()
        if phase == "run" and self.crawl is not None:
            self.crawl.record(scheduled, physical, [value[3] for value in values], [value[1] for value in values], startup)
        for index, leg in enumerate(self.leg_names):
            row = self.phases[phase][leg]
            angle, height, point_speed, lift_fraction, desired_height = values[index]
            if scheduled[index] or physical[index]:
                row["support_sole_tilt_deg"].append(angle)
            if physical[index]:
                row["force_confirmed_sole_tilt_deg"].append(angle)
                row["contact_patch_slip_speed_m_s"].append(point_speed)
            if not scheduled[index] and commanded_speed > env.cfg.gait_standstill_deadband_m_s:
                row["swing_clearance_deficit_m"].append(max(0., desired_height - height))
                if lift_fraction >= .90 and startup >= .99:
                    row["core_swing_lowest_corner_height_m"].append(height)
        if phase == "run":
            held = bool(env._target_packet_held[0].item())
            self.corrected_ceiling.append(target_ceiling_active.detach().cpu())
            self.packet_held.append(float(held))
            self.targets.append(env._processed_actions[0].detach().cpu().clone())
            self.positions.append(data.joint_pos.torch[0].detach().cpu().clone())
            self.commanded_speeds.append(commanded_speed)
            self.actual_speeds.append(actual_speed)
        if phase == "hold":
            if self.hold_origin is None:
                self.hold_origin = root_position.clone()
            self.hold_drifts.append(float(torch.linalg.norm(root_position - self.hold_origin).item()))

    def report(self):
        lag_by_joint = {}
        if self.targets:
            targets, positions = torch.stack(self.targets), torch.stack(self.positions)
            ceiling = torch.stack(self.corrected_ceiling).float().mean(dim=0).tolist()
            max_lag_steps = min(round(.75 / self.env.step_dt), len(self.targets) // 4)
            for index, name in enumerate(self.joint_names):
                target_range = math.degrees(float((targets[:, index].max() - targets[:, index].min()).item()))
                best_corr, best_lag = -1., 0
                for lag in range(max_lag_steps + 1):
                    target = targets[:len(targets) - lag, index]
                    measured = positions[lag:, index]
                    target, measured = target - target.mean(), measured - measured.mean()
                    denominator = torch.linalg.norm(target) * torch.linalg.norm(measured)
                    correlation = float((torch.dot(target, measured) / torch.clamp(denominator, min=1e-12)).item())
                    if correlation > best_corr:
                        best_corr, best_lag = correlation, lag
                credible = target_range >= 1. and best_corr >= .60 and best_lag < max_lag_steps
                lag_by_joint[name] = {
                    "lag_s": best_lag * self.env.step_dt if credible else None,
                    "peak_correlation": best_corr, "target_range_deg": target_range,
                    "search_boundary_reached": best_lag == max_lag_steps,
                }
        else:
            ceiling = [None] * len(self.joint_names)
        return {
            "by_phase_and_leg": {phase: {leg: {key: _distribution(values) for key, values in metrics.items()}
                                          for leg, metrics in legs.items()} for phase, legs in self.phases.items()},
            "run_commanded_speed_m_s": _distribution(self.commanded_speeds),
            "run_actual_forward_speed_m_s": _distribution(self.actual_speeds),
            "run_corrected_target_ceiling_fraction_by_joint": dict(zip(self.joint_names, ceiling, strict=True)),
            "run_packet_held_fraction": _optional_mean(self.packet_held),
            "run_target_to_position_lag_by_joint": lag_by_joint,
            "hold_drift_from_first_hold_sample_m": _distribution(self.hold_drifts),
            **({"sequential_crawl": self.crawl.report()} if self.crawl is not None else {}),
        }


env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
if ablation_provenance is not None:
    apply_v44_overrides(env_cfg, ablation_values)
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(
    agent_cfg, metadata.version("rsl-rl-lib")
)
env_cfg.scene.num_envs = 1
env_cfg.seed = args.seed
env_cfg.disable_time_limit = True
env_cfg.reset_joint_position_noise_rad = 0.0
env_cfg.reset_xy_jitter_m = 0.0
if args.fixed_start_phase_zero:
    env_cfg.randomize_gait_start_phase_quarters = False
if args.domain_mode == "nominal":
    env_cfg.physical_randomization_nominal_fraction = 1.0
    env_cfg.rear_payload_combined_mass_scale_range = (1.0, 1.0)
    env_cfg.rear_payload_combined_com_jitter_m = (0.0, 0.0, 0.0)
    env_cfg.robot_mass_scale_range = (1.0, 1.0)
    env_cfg.actuator_effort_scale_range = (1.0, 1.0)
    env_cfg.target_velocity_scale_range = (1.0, 1.0)
elif args.domain_mode == "randomized":
    env_cfg.physical_randomization_nominal_fraction = 0.0

if ablation_provenance is not None:
    forced_fields = (
        ("physical_randomization_nominal_fraction", "rear_payload_combined_mass_scale_range",
         "rear_payload_combined_com_jitter_m", "robot_mass_scale_range",
         "actuator_effort_scale_range", "target_velocity_scale_range")
        if args.domain_mode == "nominal"
        else ("physical_randomization_nominal_fraction",) if args.domain_mode == "randomized" else ()
    )
    ablation_provenance = {
        **ablation_provenance,
        "domain_mode_applied_after_json": args.domain_mode,
        "forced_domain_values": {key: getattr(env_cfg, key) for key in forced_fields},
        "effective_values_after_domain_mode": {key: getattr(env_cfg, key) for key in ablation_values},
        "zero_policy_actions": args.zero_policy_actions,
    }

preview_control.COMMAND_OVERRIDE = None
preview_control.RUN_STOP_OVERRIDE = (
    args.speed,
    args.run_seconds,
    args.ramp_seconds,
)

env = gym.make(args.task, cfg=env_cfg)
effort_limit_readbacks = []
if args.physx_effort_readback:
    effort_limit_readbacks.append(_physx_effort_readback(env.unwrapped, "after_construction_before_vec_wrapper"))
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
checkpoint = str(Path(args.checkpoint).resolve())
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

unwrapped = env.unwrapped
joint_names = tuple(unwrapped._robot.joint_names)
foot_leg_names = tuple(unwrapped._foot_leg_names)
base_body_id = int(unwrapped._base_body_ids[0].item())
body_masses = unwrapped._robot.data.body_mass.torch[0]
static_domain_parameters = {
    "physical_randomization_active": bool(
        unwrapped._physical_randomization_active[0].item()
    ),
    "total_robot_mass_kg": float(torch.sum(body_masses).item()),
    "base_mass_kg": float(body_masses[base_body_id].item()),
    "base_center_of_mass_m": [
        float(value)
        for value in unwrapped._robot.data.body_com_pose_b.torch[
            0, base_body_id, :3
        ].detach().cpu().tolist()
    ],
    "common_actuator_effort_scale": float(
        unwrapped._actuator_effort_scale[0, 0].item()
    ),
    "common_target_velocity_scale": float(
        unwrapped._target_velocity_scale[0, 0].item()
    ),
    "actuator_effort_scale_by_joint": _named_first_row(
        joint_names, unwrapped._actuator_effort_scale_by_joint
    ),
    "target_velocity_scale_by_joint": _named_first_row(
        joint_names, unwrapped._target_velocity_scale_by_joint
    ),
    "actuator_stiffness_scale_by_joint": _named_first_row(
        joint_names, unwrapped._actuator_stiffness_scale_by_joint
    ),
    "actuator_damping_scale_by_joint": _named_first_row(
        joint_names, unwrapped._actuator_damping_scale_by_joint
    ),
    "effective_drive_stiffness_nm_rad_by_joint": _named_first_row(
        joint_names, unwrapped._robot.data.joint_stiffness.torch
    ),
    "effective_drive_damping_nm_s_rad_by_joint": _named_first_row(
        joint_names, unwrapped._robot.data.joint_damping.torch
    ),
    "implicit_torque_gain_telemetry_synchronized": bool(env_cfg.actuator_sync_drive_gain_telemetry),
    "joint_target_bias_deg": {
        name: math.degrees(value)
        for name, value in _named_first_row(
            joint_names, unwrapped._joint_target_bias
        ).items()
    },
    "foot_static_friction_by_leg": _named_first_row(
        foot_leg_names, unwrapped._foot_static_friction
    ),
    "foot_dynamic_friction_by_leg": _named_first_row(
        foot_leg_names, unwrapped._foot_dynamic_friction
    ),
    "control_delay_steps": int(unwrapped._control_delay_steps[0].item()),
}
controller_measurement_contract = {
    "target_ceiling_mode": (
        "v44_per_joint_elapsed_packet_capped.v1"
        if args.task == V44_TASK else "legacy_scalar_control_step.v1"
    ),
    "target_ceiling_threshold_fraction": 0.98,
    "run_target_ceiling_denominator": "All joints at all selected nonterminal steady-run control ticks; held ticks count as zero. Episode means are averaged, not pooled across episodes.",
    "target_ceiling_gate_maximum_joint_tick_fraction": 0.15,
    "nominal_target_rate_limit_rad_s_by_joint": _named_first_row(
        joint_names, unwrapped._target_velocity_limit_by_joint
    ),
    "effective_target_rate_limit_rad_s_by_joint": _named_first_row(
        joint_names,
        unwrapped._target_velocity_limit_by_joint
        * unwrapped._target_velocity_scale_by_joint[0],
    ),
    "target_packet_step_limit_rad": float(unwrapped._target_packet_step_limit_rad),
    "target_packet_hold_probability": float(unwrapped._target_packet_hold_probability),
    "target_packet_max_hold_steps": int(unwrapped._target_packet_max_hold_steps),
    "effort_fraction_denominator_nm_by_joint": _named_first_row(
        joint_names, unwrapped._actuator_effort_limit_by_joint
    ),
    "effort_soft_limit_fraction_of_transient_capacity": float(env_cfg.effort_soft_limit_fraction),
    "effort_soft_threshold_nm_by_joint": _named_first_row(
        joint_names,
        unwrapped._actuator_effort_limit_by_joint * env_cfg.effort_soft_limit_fraction,
    ),
    "effort_semantics": "The applied-effort fraction uses randomized transient capacity, not the sustainable soft threshold. Soft-limit occupancy includes all nonterminal run/ramp/hold control ticks; implicit-actuator torque is an estimate, not measured physical motor torque.",
}

control_hz = int(round(1.0 / env.unwrapped.step_dt))
ramp_steps = int(round(args.ramp_seconds * control_hz))
hold_steps = int(round(args.hold_seconds * control_hz))
settle_window_steps = max(1, int(round(0.50 * control_hz)))
flat_contact_enabled = bool(getattr(unwrapped, "_sole_diagnostics_enabled", False))
if args.schedule_aware_crawl and not flat_contact_enabled:
    raise ValueError("Crawl qualification requires enabled per-leg sole geometry diagnostics")
crawl_schedule = _sequential_crawl_schedule(unwrapped) if args.schedule_aware_crawl else None
# Leave historical V43 selection windows unchanged. V44 must finish its full
# 1.5-second startup ramp before steady-run comparisons begin.
run_warmup_steps = max(control_hz, math.ceil(env_cfg.gait_start_ramp_s * control_hz)) if flat_contact_enabled else control_hz

episode_results: list[dict[str, object]] = []
episode_observation_initializations: list[dict[str, object]] = []
diagnostic_trace = _DiagnosticTrace(unwrapped, args.diagnostic_trace_jsonl, policy, env) if args.diagnostic_trace_jsonl is not None else None
actor_velocity_fd = _ActorJointVelocityFD(unwrapped, diagnostic_trace) if args.actor_joint_velocity_fd else None
passive_physics_diagnostic = None
passive_physics_diagnostic_result = None
if args.physx_step_diagnostic_jsonl is not None:
    from parallel_walking.passive_physx_step_diagnostic import PassivePhysxStepDiagnostic

    passive_physics_diagnostic = PassivePhysxStepDiagnostic(
        unwrapped,
        task=args.task,
        provenance={
            **evaluation_source_provenance,
            "checkpoint": str(checkpoint_source_path),
            "evaluation_report_path": str(args.output.resolve()) if args.output is not None else None,
            "environment_ablation": ablation_provenance,
            "static_domain_parameters": static_domain_parameters,
            "controller_measurement_contract": controller_measurement_contract,
            "seed": args.seed,
            "domain_mode": args.domain_mode,
            "capture_window": "Scheduled final one second of each hold; no earlier sliding-window replacement if an episode falls. Includes terminal-step physics before any automatic reset; episode outcomes must be inspected.",
        },
        max_physics_steps=args.episodes * control_hz * int(env_cfg.decimation),
    )
    passive_physics_diagnostic.metadata["scheduled_capture_windows"] = []
for episode_index in range(args.episodes):
    env.unwrapped._reset_idx(torch.tensor([0], device=env.unwrapped.device))
    if args.physx_effort_readback:
        effort_limit_readbacks.append(_physx_effort_readback(env.unwrapped, "after_episode_reset", episode_index))
    env.unwrapped.episode_length_buf.zero_()
    alignment_delay_s = 0.0
    if args.ramp_start_phase_cycles is not None:
        current_phase = float(env.unwrapped._gait_phase()[0].item())
        frequency_hz = float(env.unwrapped._gait_schedule()[0][0].item())
        nominal_ramp_phase = (
            current_phase + frequency_hz * args.run_seconds
        ) % 1.0
        forward_cycles = (
            args.ramp_start_phase_cycles - nominal_ramp_phase
        ) % 1.0
        circular_error = min(forward_cycles, 1.0 - forward_cycles)
        if circular_error > args.ramp_start_phase_tolerance_cycles:
            alignment_delay_s = min(
                forward_cycles / frequency_hz,
                args.maximum_phase_alignment_wait_s,
            )
        env.unwrapped._initialize_run_stop_schedule(
            torch.tensor([0], device=env.unwrapped.device),
            torch.tensor([args.speed], device=env.unwrapped.device),
            run_duration_s=torch.tensor(
                [args.run_seconds + alignment_delay_s],
                device=env.unwrapped.device,
            ),
            ramp_duration_s=torch.tensor(
                [args.ramp_seconds], device=env.unwrapped.device
            ),
        )
    run_steps = int(
        round((args.run_seconds + alignment_delay_s) * control_hz)
    )
    total_steps = run_steps + ramp_steps + hold_steps
    passive_capture_start_step = max(run_steps + ramp_steps, total_steps - control_hz)
    if passive_physics_diagnostic is not None:
        passive_physics_diagnostic.metadata["scheduled_capture_windows"].append({
            "episode_index": episode_index,
            "start_control_step_inclusive": passive_capture_start_step,
            "end_control_step_exclusive": total_steps,
            "start_sim_time_s": passive_capture_start_step * unwrapped.step_dt,
            "end_sim_time_s": total_steps * unwrapped.step_dt,
        })
    episode_domain_parameters = {
        "base_force_body_frame_n": [
            float(value)
            for value in env.unwrapped._base_force_b[0]
            .detach()
            .cpu()
            .flatten()
            .tolist()
        ],
        "base_torque_body_frame_nm": [
            float(value)
            for value in env.unwrapped._base_torque_b[0]
            .detach()
            .cpu()
            .flatten()
            .tolist()
        ],
        "gait_start_phase_cycles": float(
            env.unwrapped._gait_phase_offset[0].item()
        ),
        "stop_phase_alignment_delay_s": alignment_delay_s,
    }
    obs, initialization = _finalized_initial_observations(env, episode_index)
    if initialization is not None:
        episode_observation_initializations.append(initialization)
    if actor_velocity_fd is not None:
        actor_velocity_fd.reset(episode_index, initialization)
    start_position = env.unwrapped._robot.data.root_pos_w.torch[0, :2].clone()
    start_yaw = float(
        _yaw_from_xyzw(env.unwrapped._robot.data.root_quat_w.torch)[0].item()
    )
    forward_direction = torch.tensor(
        (math.cos(start_yaw), math.sin(start_yaw)), device=env.unwrapped.device
    )
    lateral_direction = torch.tensor(
        (-math.sin(start_yaw), math.cos(start_yaw)), device=env.unwrapped.device
    )
    previous_action = torch.zeros(12, device=env.unwrapped.device)
    older_action = torch.zeros_like(previous_action)
    # No processed action exists until the first policy step.  The reset path
    # seeds ``_previous_targets`` with the joint target that was actually
    # written to simulation, so it is the correct baseline for first-packet
    # target-velocity occupancy as well as every subsequent processed target.
    previous_processed_target = env.unwrapped._previous_targets[0].clone()
    previous_joint_velocity = env.unwrapped._robot.data.joint_vel.torch[0].clone()
    previous_linear_velocity = env.unwrapped._robot.data.root_lin_vel_w.torch[0].clone()
    previous_angular_velocity = env.unwrapped._robot.data.root_ang_vel_w.torch[0].clone()

    flat_contact = _FlatContactDiagnostics(env.unwrapped) if flat_contact_enabled else None
    run_speeds: list[float] = []
    run_tracking_errors: list[float] = []
    run_lateral_speeds: list[float] = []
    deceleration_speeds: list[float] = []
    hold_speeds: list[float] = []
    hold_position_errors: list[float] = []
    hold_contacts: list[float] = []
    hold_stable: list[float] = []
    hold_foot_forces: list[torch.Tensor] = []
    hold_foot_contacts_sum = torch.zeros(4, device=env.unwrapped.device)
    hold_normalized_target_rms: list[float] = []
    hold_abs_normalized_target_sum = torch.zeros(
        12, device=env.unwrapped.device
    )
    hold_target_sample_count = 0
    action_accelerations: list[float] = []
    joint_accelerations: list[float] = []
    body_linear_accelerations: list[float] = []
    body_angular_accelerations: list[float] = []
    run_swing_opportunities = torch.zeros(4, device=env.unwrapped.device)
    run_swing_successes = torch.zeros(4, device=env.unwrapped.device)
    run_swing_release_quality_sum = torch.zeros(
        4, device=env.unwrapped.device
    )
    run_pair_release_qualities: list[float] = []
    run_pair_release_mismatches: list[float] = []
    run_action_saturation_rates: list[float] = []
    run_target_limiter_rates: list[float] = []
    run_target_velocity_ceiling_rates: list[float] = []
    run_abs_joint_tracking_errors: list[torch.Tensor] = []
    run_action_saturation_count_by_joint = torch.zeros(
        12, device=env.unwrapped.device
    )
    run_target_limiter_count_by_joint = torch.zeros(
        12, device=env.unwrapped.device
    )
    run_target_velocity_ceiling_count_by_joint = torch.zeros(
        12, device=env.unwrapped.device
    )
    run_feasibility_sample_count = 0
    effort_soft_limit_rates: list[float] = []
    effort_fraction_sum_by_phase = {
        phase: torch.zeros(12, device=env.unwrapped.device)
        for phase in ("run", "deceleration", "hold")
    }
    effort_soft_limit_count_by_phase = {
        phase: torch.zeros(12, device=env.unwrapped.device)
        for phase in ("run", "deceleration", "hold")
    }
    effort_sample_count_by_phase = {
        phase: 0 for phase in ("run", "deceleration", "hold")
    }
    maximum_effort_fraction = 0.0
    consecutive_settle_steps = 0
    settle_time_s: float | None = None
    gait_phase_at_ramp_start_cycles: float | None = None
    gait_phase_at_homing_start_cycles: float | None = None
    stop_origin = start_position.clone()
    stop_target = start_position.clone()
    final_position = start_position.clone()
    final_speed = 0.0
    final_heading_error = 0.0
    fell = False
    fall_phase = None
    completed_steps = 0

    for step in range(total_steps):
        trace_command_applied = _DiagnosticTrace._values(env.unwrapped._commands[0]) if diagnostic_trace is not None else None
        if actor_velocity_fd is not None:
            actor_obs, fd_provenance = actor_velocity_fd.prepare(obs, episode_index, step)
            actor_capture = diagnostic_trace.capture_pre_inference(actor_obs, episode_index, step)
            actor_capture.update(fd_provenance)
        else:
            actor_capture = diagnostic_trace.capture_pre_inference(obs, episode_index, step) if diagnostic_trace is not None else None
        packet_elapsed_s = (
            float(env.unwrapped._target_packet_elapsed_s[0].item()) + env.unwrapped.step_dt
            if args.task == V44_TASK else env.unwrapped.step_dt
        )
        with torch.inference_mode():
            if actor_velocity_fd is None:
                actions = policy(obs).clone()
            else:
                actions = policy(actor_obs).clone()
            if diagnostic_trace is not None:
                actor_capture["raw_actor_output"] = diagnostic_trace._exact_vector(actions, 12, "Raw actor output")
            if args.zero_policy_actions:
                actions.zero_()
            if diagnostic_trace is not None:
                actor_capture["submitted_action"] = diagnostic_trace._exact_vector(actions, 12, "Submitted action")
                actor_capture["zero_policy_override_applied"] = bool(args.zero_policy_actions)
        action_acceleration = torch.mean(
            torch.abs(actions[0] - 2.0 * previous_action + older_action)
        )
        older_action.copy_(previous_action)
        previous_action.copy_(actions[0])
        if passive_physics_diagnostic is None:
            try:
                obs, _, dones, _ = env.step(actions)
            except RearPitchLiftMappingError as mapping_error:
                if not rear_pitch_evaluation or diagnostic_trace is None:
                    raise
                rejected_phase = "run" if step < run_steps else "deceleration" if step < run_steps + ramp_steps else "hold"
                try:
                    diagnostic_trace.write_mapping_failure(
                        mapping_error, episode_index, step, rejected_phase, actor_capture,
                        episode_results, episode_observation_initializations,
                    )
                except BaseException as publication_error:
                    print(f"Partial mapping evidence publication failed: {type(publication_error).__name__}: {publication_error}", file=sys.stderr)
                print(f"Mapping rejected before physics: {mapping_error}", file=sys.stderr, flush=True)
                # Publish before cleanup: Kit may terminate the process without
                # returning. Neither exit0 nor partial files mean qualification.
                try:
                    env.close()
                finally:
                    simulation_app.close()
                raise
        else:
            try:
                if step >= passive_capture_start_step:
                    # Install only across this existing step; both contexts unwind
                    # before reset handling, disk I/O, or the next control tick.
                    with passive_physics_diagnostic:
                        with passive_physics_diagnostic.control_step(
                            episode_index=episode_index,
                            control_step_index=step,
                            stage="hold",
                        ):
                            obs, _, dones, _ = env.step(actions)
                else:
                    obs, _, dones, _ = env.step(actions)
            except BaseException as step_error:
                try:
                    _write_passive_physics_diagnostic(
                        passive_physics_diagnostic,
                        status="partial_env_step_failed_or_interrupted",
                        error=f"{type(step_error).__name__}: {step_error}"[:1000],
                    )
                except BaseException as write_error:
                    print(
                        "Could not preserve partial passive physics diagnostic: "
                        f"{type(write_error).__name__}: {write_error}",
                        file=sys.stderr,
                    )
                raise
        if args.physx_effort_readback and step == 0:
            effort_limit_readbacks.append(_physx_effort_readback(
                env.unwrapped, "after_first_step", episode_index, terminal_autoreset=bool(dones[0].item())))
        policy.reset(dones)
        if bool(dones[0].item()):
            fell = True
            fall_phase = "run" if step < run_steps else "deceleration" if step < run_steps + ramp_steps else "hold"
            break

        root_position = env.unwrapped._robot.data.root_pos_w.torch[0, :2]
        linear_velocity = env.unwrapped._robot.data.root_lin_vel_w.torch[0]
        angular_velocity = env.unwrapped._robot.data.root_ang_vel_w.torch[0]
        joint_velocity = env.unwrapped._robot.data.joint_vel.torch[0]
        planar_speed = float(torch.linalg.norm(linear_velocity[:2]).item())
        forward_speed = float(torch.dot(linear_velocity[:2], forward_direction).item())
        lateral_speed = float(torch.dot(linear_velocity[:2], lateral_direction).item())
        commanded_speed = float(env.unwrapped._commands[0, 0].item())
        applied_gait_phase = float(
            env.unwrapped._applied_gait_phase[0].item()
        )
        if (
            gait_phase_at_ramp_start_cycles is None
            and commanded_speed < args.speed - 1.0e-6
        ):
            gait_phase_at_ramp_start_cycles = applied_gait_phase
        if (
            gait_phase_at_homing_start_cycles is None
            and commanded_speed
            <= float(env_cfg.gait_standstill_deadband_m_s)
        ):
            gait_phase_at_homing_start_cycles = applied_gait_phase
        current_yaw = float(
            _yaw_from_xyzw(env.unwrapped._robot.data.root_quat_w.torch)[0].item()
        )
        final_heading_error = math.atan2(
            math.sin(current_yaw - start_yaw), math.cos(current_yaw - start_yaw)
        )
        foot_forces = env.unwrapped._foot_forces()[0]
        foot_contact = foot_forces > env_cfg.scheduled_release_force_threshold_n
        all_feet_contact = bool(torch.all(foot_contact).item())
        scheduled_contact = env.unwrapped._applied_scheduled_contact[0]
        scheduled_swing = ~scheduled_contact
        scheduled_release_quality = torch.exp(
            -torch.square(
                foot_forces / env.unwrapped._scheduled_release_force_sigma_n
            )
        ) * scheduled_swing.float()
        upright_cosine = float(
            (-env.unwrapped._robot.data.projected_gravity_b.torch[0, 2]).item()
        )
        applied_effort_fraction = torch.abs(
            env.unwrapped._robot.data.applied_torque.torch[0]
        ) / torch.clamp(
            env.unwrapped._actuator_effort_limit_by_joint[0], min=1.0e-6
        )
        effort_soft_limit_rates.append(
            float(
                torch.mean(
                    (
                        applied_effort_fraction
                        > env_cfg.effort_soft_limit_fraction
                    ).float()
                ).item()
            )
        )
        maximum_effort_fraction = max(
            maximum_effort_fraction,
            float(torch.amax(applied_effort_fraction).item()),
        )
        target_limiter_active = torch.abs(
            env.unwrapped._desired_targets[0]
            - env.unwrapped._processed_actions[0]
        ) > math.radians(0.5)
        processed_target_delta = torch.abs(
            env.unwrapped._processed_actions[0] - previous_processed_target
        )
        target_delta_limit, target_velocity_ceiling_active = _target_ceiling_measurement(
            env.unwrapped, processed_target_delta, packet_elapsed_s
        )
        previous_processed_target.copy_(env.unwrapped._processed_actions[0])
        applied_normalized_target = torch.clamp(
            (
                env.unwrapped._processed_actions[0]
                - env.unwrapped._robot.data.default_joint_pos.torch[0]
                - env.unwrapped._joint_target_bias[0]
            )
            / env.unwrapped._joint_scale,
            -1.0,
            1.0,
        )
        if step < run_steps:
            effort_phase = "run"
        elif step < run_steps + ramp_steps:
            effort_phase = "deceleration"
        else:
            effort_phase = "hold"
        if diagnostic_trace is not None:
            diagnostic_trace.record(
                episode_index, step, effort_phase, trace_command_applied, packet_elapsed_s,
                foot_forces, foot_contact, scheduled_contact, run_warmup_steps,
                target_delta_limit, target_velocity_ceiling_active,
                actor_capture,
            )
        if flat_contact is not None and (effort_phase != "run" or step >= run_warmup_steps):
            flat_contact.record(
                effort_phase, scheduled_contact, foot_contact, target_velocity_ceiling_active,
                commanded_speed, forward_speed, root_position,
            )
        effort_fraction_sum_by_phase[effort_phase] += applied_effort_fraction
        effort_soft_limit_count_by_phase[effort_phase] += (
            applied_effort_fraction > env_cfg.effort_soft_limit_fraction
        ).float()
        effort_sample_count_by_phase[effort_phase] += 1
        joint_acceleration = torch.sqrt(
            torch.mean(
                torch.square(
                    (joint_velocity - previous_joint_velocity) / env.unwrapped.step_dt
                )
            )
        )
        body_linear_acceleration = torch.sqrt(
            torch.mean(
                torch.square(
                    (linear_velocity - previous_linear_velocity)
                    / env.unwrapped.step_dt
                )
            )
        )
        body_angular_acceleration = torch.sqrt(
            torch.mean(
                torch.square(
                    (angular_velocity - previous_angular_velocity)
                    / env.unwrapped.step_dt
                )
            )
        )
        previous_joint_velocity.copy_(joint_velocity)
        previous_linear_velocity.copy_(linear_velocity)
        previous_angular_velocity.copy_(angular_velocity)
        action_accelerations.append(float(action_acceleration.item()))
        joint_accelerations.append(float(joint_acceleration.item()))
        body_linear_accelerations.append(float(body_linear_acceleration.item()))
        body_angular_accelerations.append(float(body_angular_acceleration.item()))

        if step < run_steps:
            # Historical tasks ignore one second; V44 ignores its whole ramp.
            if step >= run_warmup_steps:
                run_speeds.append(forward_speed)
                run_tracking_errors.append(forward_speed - commanded_speed)
                run_lateral_speeds.append(lateral_speed)
                run_swing_opportunities += scheduled_swing.float()
                run_swing_successes += (
                    scheduled_swing & ~foot_contact
                ).float()
                run_swing_release_quality_sum += scheduled_release_quality
                if int(torch.sum(scheduled_swing).item()) >= 2:
                    pair_quality = scheduled_release_quality[scheduled_swing]
                    run_pair_release_qualities.append(
                        float(torch.amin(pair_quality).item())
                    )
                    run_pair_release_mismatches.append(
                        float(torch.var(pair_quality, unbiased=False).item())
                    )
                run_action_saturation_rates.append(
                    float(torch.mean((torch.abs(actions[0]) >= 0.90).float()).item())
                )
                run_target_limiter_rates.append(
                    float(torch.mean(target_limiter_active.float()).item())
                )
                run_target_velocity_ceiling_rates.append(
                    float(
                        torch.mean(target_velocity_ceiling_active.float()).item()
                    )
                )
                run_abs_joint_tracking_errors.append(
                    torch.abs(
                        env.unwrapped._processed_actions[0]
                        - env.unwrapped._robot.data.joint_pos.torch[0]
                    ).clone()
                )
                run_action_saturation_count_by_joint += (
                    torch.abs(actions[0]) >= 0.90
                ).float()
                run_target_limiter_count_by_joint += target_limiter_active.float()
                run_target_velocity_ceiling_count_by_joint += (
                    target_velocity_ceiling_active.float()
                )
                run_feasibility_sample_count += 1
        elif step < run_steps + ramp_steps:
            if step == run_steps:
                stop_origin = root_position.clone()
                stop_target = env.unwrapped._run_stop_target_w[0].clone()
            deceleration_speeds.append(planar_speed)
        else:
            if ramp_steps == 0 and step == run_steps:
                stop_origin = env.unwrapped._run_stop_origin_w[0].clone()
                stop_target = env.unwrapped._run_stop_target_w[0].clone()
            position_error = float(torch.linalg.norm(root_position - stop_target).item())
            stable = (
                planar_speed <= env_cfg.stop_velocity_sigma_m_s
                and position_error <= env_cfg.stop_position_tolerance_m
                and all_feet_contact
                and upright_cosine >= env_cfg.minimum_upright_cosine
            )
            hold_speeds.append(planar_speed)
            hold_position_errors.append(position_error)
            hold_contacts.append(float(all_feet_contact))
            hold_stable.append(float(stable))
            hold_foot_forces.append(foot_forces.clone())
            hold_foot_contacts_sum += foot_contact.float()
            hold_normalized_target_rms.append(
                float(torch.sqrt(torch.mean(torch.square(applied_normalized_target))).item())
            )
            hold_abs_normalized_target_sum += torch.abs(applied_normalized_target)
            hold_target_sample_count += 1
            if stable:
                consecutive_settle_steps += 1
                if (
                    settle_time_s is None
                    and consecutive_settle_steps >= settle_window_steps
                ):
                    first_stable_step = step - settle_window_steps + 1
                    settle_time_s = (first_stable_step - run_steps) / control_hz
            else:
                consecutive_settle_steps = 0

        final_position = root_position.clone()
        final_speed = planar_speed
        completed_steps = step + 1

    path_delta = final_position - start_position
    forward_distance = float(torch.dot(path_delta, forward_direction).item())
    lateral_displacement = float(torch.dot(path_delta, lateral_direction).item())
    stop_delta = final_position - stop_origin
    stop_distance = float(torch.dot(stop_delta, forward_direction).item())
    expected_stop_distance = 0.5 * args.speed * args.ramp_seconds
    total_effort_samples = sum(effort_sample_count_by_phase.values())
    effort_fraction_sum = sum(effort_fraction_sum_by_phase.values())
    effort_soft_limit_count = sum(effort_soft_limit_count_by_phase.values())
    joint_names = tuple(env.unwrapped._robot.joint_names)
    foot_leg_names = tuple(
        next(leg for leg in ("front_left", "rear_left", "front_right", "rear_right") if leg in sensor_name)
        for sensor_name in env.unwrapped._foot_sensor_names
    )
    run_release_rates = run_swing_successes / torch.clamp(
        run_swing_opportunities, min=1.0
    )
    run_release_qualities = run_swing_release_quality_sum / torch.clamp(
        run_swing_opportunities, min=1.0
    )
    run_abs_tracking_error = (
        torch.stack(run_abs_joint_tracking_errors)
        if run_abs_joint_tracking_errors
        else torch.zeros((1, 12), device=env.unwrapped.device)
    )
    hold_force_samples = (
        torch.stack(hold_foot_forces)
        if hold_foot_forces
        else torch.zeros((1, 4), device=env.unwrapped.device)
    )
    episode_results.append(
        {
            "episode": episode_index,
            "domain_parameters": episode_domain_parameters,
            "fell": fell,
            "fall_phase": fall_phase,
            "completed_seconds": completed_steps / control_hz,
            "gait_phase_at_ramp_start_cycles": gait_phase_at_ramp_start_cycles,
            "gait_phase_at_homing_start_cycles": (
                gait_phase_at_homing_start_cycles
            ),
            "mean_run_speed_m_s": _mean(run_speeds),
            "peak_run_speed_m_s": max(run_speeds, default=0.0),
            "run_speed_tracking_rmse_m_s": _rms(run_tracking_errors),
            "mean_abs_run_lateral_speed_m_s": _mean(
                [abs(value) for value in run_lateral_speeds]
            ),
            "run_scheduled_release_rate_by_leg": {
                name: float(value)
                for name, value in zip(
                    foot_leg_names, run_release_rates.tolist(), strict=True
                )
            },
            "run_scheduled_release_quality_by_leg": {
                name: float(value)
                for name, value in zip(
                    foot_leg_names, run_release_qualities.tolist(), strict=True
                )
            },
            "mean_run_pair_release_quality": _mean(
                run_pair_release_qualities
            ),
            "mean_run_pair_release_mismatch": _mean(
                run_pair_release_mismatches
            ),
            "run_action_saturation_joint_fraction": _mean(
                run_action_saturation_rates
            ),
            "run_target_limiter_joint_fraction": _mean(
                run_target_limiter_rates
            ),
            "run_target_velocity_ceiling_joint_fraction": _mean(
                run_target_velocity_ceiling_rates
            ),
            "run_action_saturation_fraction_by_joint": {
                name: float(value)
                for name, value in zip(
                    joint_names,
                    (
                        run_action_saturation_count_by_joint
                        / max(run_feasibility_sample_count, 1)
                    ).tolist(),
                    strict=True,
                )
            },
            "run_target_limiter_fraction_by_joint": {
                name: float(value)
                for name, value in zip(
                    joint_names,
                    (
                        run_target_limiter_count_by_joint
                        / max(run_feasibility_sample_count, 1)
                    ).tolist(),
                    strict=True,
                )
            },
            "run_target_velocity_ceiling_fraction_by_joint": {
                name: float(value)
                for name, value in zip(
                    joint_names,
                    (
                        run_target_velocity_ceiling_count_by_joint
                        / max(run_feasibility_sample_count, 1)
                    ).tolist(),
                    strict=True,
                )
            },
            "run_joint_tracking_error_rms_deg": math.degrees(
                float(torch.sqrt(torch.mean(torch.square(run_abs_tracking_error))).item())
            ),
            "run_joint_tracking_error_p95_deg": math.degrees(
                float(torch.quantile(run_abs_tracking_error.flatten(), 0.95).item())
            ),
            "run_joint_tracking_error_max_deg": math.degrees(
                float(torch.amax(run_abs_tracking_error).item())
            ),
            "run_joint_tracking_error_rms_deg_by_joint": {
                name: math.degrees(float(value))
                for name, value in zip(
                    joint_names,
                    torch.sqrt(
                        torch.mean(torch.square(run_abs_tracking_error), dim=0)
                    ).tolist(),
                    strict=True,
                )
            },
            "run_joint_tracking_error_p95_deg_by_joint": {
                name: math.degrees(float(value))
                for name, value in zip(
                    joint_names,
                    torch.quantile(run_abs_tracking_error, 0.95, dim=0).tolist(),
                    strict=True,
                )
            },
            "forward_distance_m": forward_distance,
            "lateral_displacement_m": lateral_displacement,
            "absolute_lateral_per_forward": abs(lateral_displacement)
            / max(abs(forward_distance), 0.01),
            "final_heading_error_rad": final_heading_error,
            "stop_distance_m": stop_distance,
            "expected_command_ramp_distance_m": expected_stop_distance,
            "stop_distance_error_m": stop_distance - expected_stop_distance,
            "settle_time_s": settle_time_s,
            "mean_deceleration_speed_m_s": _mean(deceleration_speeds),
            "mean_hold_speed_m_s": _mean(hold_speeds),
            "final_speed_m_s": final_speed,
            "mean_hold_position_error_m": _mean(hold_position_errors),
            "final_stop_position_error_m": float(
                torch.linalg.norm(final_position - stop_target).item()
            ),
            "hold_all_feet_contact_fraction": _mean(hold_contacts),
            "hold_stable_fraction": _mean(hold_stable),
            "hold_contact_fraction_by_leg": {
                name: float(value)
                for name, value in zip(
                    foot_leg_names,
                    (
                        hold_foot_contacts_sum
                        / max(len(hold_foot_forces), 1)
                    ).tolist(),
                    strict=True,
                )
            },
            "mean_hold_foot_force_n_by_leg": {
                name: float(value)
                for name, value in zip(
                    foot_leg_names,
                    torch.mean(hold_force_samples, dim=0).tolist(),
                    strict=True,
                )
            },
            "p05_hold_foot_force_n_by_leg": {
                name: float(value)
                for name, value in zip(
                    foot_leg_names,
                    torch.quantile(hold_force_samples, 0.05, dim=0).tolist(),
                    strict=True,
                )
            },
            "mean_hold_normalized_target_rms": _mean(
                hold_normalized_target_rms
            ),
            "mean_abs_hold_normalized_target_by_joint": {
                name: float(value)
                for name, value in zip(
                    joint_names,
                    (
                        hold_abs_normalized_target_sum
                        / max(hold_target_sample_count, 1)
                    ).tolist(),
                    strict=True,
                )
            },
            "mean_abs_action_acceleration_per_step2": _mean(action_accelerations),
            "rms_joint_acceleration_rad_s2": _rms(joint_accelerations),
            "rms_body_linear_acceleration_m_s2": _rms(
                body_linear_accelerations
            ),
            "rms_body_angular_acceleration_rad_s2": _rms(
                body_angular_accelerations
            ),
            "effort_soft_limit_joint_fraction": _mean(effort_soft_limit_rates),
            "mean_applied_effort_fraction_by_phase": {
                phase: float(
                    torch.mean(
                        effort_fraction_sum_by_phase[phase]
                        / max(effort_sample_count_by_phase[phase], 1)
                    ).item()
                )
                for phase in effort_fraction_sum_by_phase
            },
            "mean_applied_effort_fraction_by_phase_and_joint": {
                phase: {
                    name: float(value)
                    for name, value in zip(
                        joint_names,
                        (
                            effort_fraction_sum_by_phase[phase]
                            / max(effort_sample_count_by_phase[phase], 1)
                        ).tolist(),
                        strict=True,
                    )
                }
                for phase in effort_fraction_sum_by_phase
            },
            "effort_soft_limit_joint_fraction_by_phase": {
                phase: float(
                    torch.mean(
                        effort_soft_limit_count_by_phase[phase]
                        / max(effort_sample_count_by_phase[phase], 1)
                    ).item()
                )
                for phase in effort_soft_limit_count_by_phase
            },
            "effort_soft_limit_fraction_by_phase_and_joint": {
                phase: {
                    name: float(value)
                    for name, value in zip(
                        joint_names,
                        (
                            effort_soft_limit_count_by_phase[phase]
                            / max(effort_sample_count_by_phase[phase], 1)
                        ).tolist(),
                        strict=True,
                    )
                }
                for phase in effort_soft_limit_count_by_phase
            },
            "mean_applied_effort_fraction_by_joint": {
                name: float(value)
                for name, value in zip(
                    joint_names,
                    (
                        effort_fraction_sum / max(total_effort_samples, 1)
                    ).tolist(),
                    strict=True,
                )
            },
            "effort_soft_limit_fraction_by_joint": {
                name: float(value)
                for name, value in zip(
                    joint_names,
                    (
                        effort_soft_limit_count / max(total_effort_samples, 1)
                    ).tolist(),
                    strict=True,
                )
            },
            "maximum_applied_effort_fraction": maximum_effort_fraction,
            **({"flat_contact_diagnostics": flat_contact.report()} if flat_contact is not None else {}),
        }
    )


if passive_physics_diagnostic is not None:
    passive_physics_diagnostic_result = _write_passive_physics_diagnostic(
        passive_physics_diagnostic, status="evaluation_episode_loop_finished"
    )


def _mean_metric(name: str) -> float:
    values = [result[name] for result in episode_results]
    return _mean([float(value) for value in values if value is not None])


summary = {
    "fall_rate": _mean([float(bool(result["fell"])) for result in episode_results]),
    "mean_run_speed_m_s": _mean_metric("mean_run_speed_m_s"),
    "mean_run_speed_fraction": _mean_metric("mean_run_speed_m_s") / args.speed,
    "run_speed_tracking_rmse_m_s": _mean_metric("run_speed_tracking_rmse_m_s"),
    "run_scheduled_release_rate_by_leg": {
        leg: _mean(
            [
                float(result["run_scheduled_release_rate_by_leg"][leg])
                for result in episode_results
            ]
        )
        for leg in ("front_left", "rear_left", "front_right", "rear_right")
    },
    "run_scheduled_release_quality_by_leg": {
        leg: _mean(
            [
                float(result["run_scheduled_release_quality_by_leg"][leg])
                for result in episode_results
            ]
        )
        for leg in ("front_left", "rear_left", "front_right", "rear_right")
    },
    "mean_run_pair_release_quality": _mean_metric(
        "mean_run_pair_release_quality"
    ),
    "mean_run_pair_release_mismatch": _mean_metric(
        "mean_run_pair_release_mismatch"
    ),
    "run_action_saturation_joint_fraction": _mean_metric(
        "run_action_saturation_joint_fraction"
    ),
    "run_target_limiter_joint_fraction": _mean_metric(
        "run_target_limiter_joint_fraction"
    ),
    "run_target_velocity_ceiling_joint_fraction": _mean_metric(
        "run_target_velocity_ceiling_joint_fraction"
    ),
    "run_joint_tracking_error_rms_deg": _mean_metric(
        "run_joint_tracking_error_rms_deg"
    ),
    "run_joint_tracking_error_p95_deg": _mean_metric(
        "run_joint_tracking_error_p95_deg"
    ),
    "run_joint_tracking_error_max_deg": max(
        float(result["run_joint_tracking_error_max_deg"])
        for result in episode_results
    ),
    "run_joint_tracking_error_rms_deg_by_joint": {
        joint: _mean(
            [
                float(result["run_joint_tracking_error_rms_deg_by_joint"][joint])
                for result in episode_results
            ]
        )
        for joint in episode_results[0]["run_joint_tracking_error_rms_deg_by_joint"]
    },
    "run_joint_tracking_error_p95_deg_by_joint": {
        joint: _mean(
            [
                float(result["run_joint_tracking_error_p95_deg_by_joint"][joint])
                for result in episode_results
            ]
        )
        for joint in episode_results[0]["run_joint_tracking_error_p95_deg_by_joint"]
    },
    "run_action_saturation_fraction_by_joint": {
        joint: _mean(
            [
                float(result["run_action_saturation_fraction_by_joint"][joint])
                for result in episode_results
            ]
        )
        for joint in episode_results[0]["run_action_saturation_fraction_by_joint"]
    },
    "run_target_limiter_fraction_by_joint": {
        joint: _mean(
            [
                float(result["run_target_limiter_fraction_by_joint"][joint])
                for result in episode_results
            ]
        )
        for joint in episode_results[0]["run_target_limiter_fraction_by_joint"]
    },
    "run_target_velocity_ceiling_fraction_by_joint": {
        joint: _mean(
            [
                float(
                    result["run_target_velocity_ceiling_fraction_by_joint"][joint]
                )
                for result in episode_results
            ]
        )
        for joint in episode_results[0][
            "run_target_velocity_ceiling_fraction_by_joint"
        ]
    },
    "mean_absolute_lateral_per_forward": _mean_metric(
        "absolute_lateral_per_forward"
    ),
    "mean_abs_final_heading_error_rad": _mean(
        [abs(float(result["final_heading_error_rad"])) for result in episode_results]
    ),
    "mean_stop_distance_m": _mean_metric("stop_distance_m"),
    "mean_abs_stop_distance_error_m": _mean(
        [abs(float(result["stop_distance_error_m"])) for result in episode_results]
    ),
    "settled_episode_rate": _mean(
        [float(result["settle_time_s"] is not None) for result in episode_results]
    ),
    "mean_settle_time_s": _mean_metric("settle_time_s"),
    "mean_hold_speed_m_s": _mean_metric("mean_hold_speed_m_s"),
    "mean_final_speed_m_s": _mean_metric("final_speed_m_s"),
    "mean_final_stop_position_error_m": _mean_metric(
        "final_stop_position_error_m"
    ),
    "hold_all_feet_contact_fraction": _mean_metric(
        "hold_all_feet_contact_fraction"
    ),
    "hold_stable_fraction": _mean_metric("hold_stable_fraction"),
    "hold_contact_fraction_by_leg": {
        leg: _mean(
            [
                float(result["hold_contact_fraction_by_leg"][leg])
                for result in episode_results
            ]
        )
        for leg in episode_results[0]["hold_contact_fraction_by_leg"]
    },
    "mean_hold_foot_force_n_by_leg": {
        leg: _mean(
            [
                float(result["mean_hold_foot_force_n_by_leg"][leg])
                for result in episode_results
            ]
        )
        for leg in episode_results[0]["mean_hold_foot_force_n_by_leg"]
    },
    "p05_hold_foot_force_n_by_leg": {
        leg: _mean(
            [
                float(result["p05_hold_foot_force_n_by_leg"][leg])
                for result in episode_results
            ]
        )
        for leg in episode_results[0]["p05_hold_foot_force_n_by_leg"]
    },
    "mean_hold_normalized_target_rms": _mean_metric(
        "mean_hold_normalized_target_rms"
    ),
    "mean_abs_hold_normalized_target_by_joint": {
        joint: _mean(
            [
                float(result["mean_abs_hold_normalized_target_by_joint"][joint])
                for result in episode_results
            ]
        )
        for joint in episode_results[0]["mean_abs_hold_normalized_target_by_joint"]
    },
    "rms_joint_acceleration_rad_s2": _mean_metric(
        "rms_joint_acceleration_rad_s2"
    ),
    "rms_body_linear_acceleration_m_s2": _mean_metric(
        "rms_body_linear_acceleration_m_s2"
    ),
    "rms_body_angular_acceleration_rad_s2": _mean_metric(
        "rms_body_angular_acceleration_rad_s2"
    ),
    "mean_abs_action_acceleration_per_step2": _mean_metric(
        "mean_abs_action_acceleration_per_step2"
    ),
    "effort_soft_limit_joint_fraction": _mean_metric(
        "effort_soft_limit_joint_fraction"
    ),
    "mean_applied_effort_fraction_by_phase": {
        phase: _mean(
            [
                float(result["mean_applied_effort_fraction_by_phase"][phase])
                for result in episode_results
            ]
        )
        for phase in ("run", "deceleration", "hold")
    },
    "effort_soft_limit_joint_fraction_by_phase": {
        phase: _mean(
            [
                float(result["effort_soft_limit_joint_fraction_by_phase"][phase])
                for result in episode_results
            ]
        )
        for phase in ("run", "deceleration", "hold")
    },
    "mean_applied_effort_fraction_by_phase_and_joint": {
        phase: {
            joint: _mean(
                [
                    float(
                        result["mean_applied_effort_fraction_by_phase_and_joint"]
                        [phase][joint]
                    )
                    for result in episode_results
                ]
            )
            for joint in episode_results[0][
                "mean_applied_effort_fraction_by_phase_and_joint"
            ][phase]
        }
        for phase in ("run", "deceleration", "hold")
    },
    "effort_soft_limit_fraction_by_phase_and_joint": {
        phase: {
            joint: _mean(
                [
                    float(
                        result["effort_soft_limit_fraction_by_phase_and_joint"]
                        [phase][joint]
                    )
                    for result in episode_results
                ]
            )
            for joint in episode_results[0][
                "effort_soft_limit_fraction_by_phase_and_joint"
            ][phase]
        }
        for phase in ("run", "deceleration", "hold")
    },
    "maximum_applied_effort_fraction": max(
        float(result["maximum_applied_effort_fraction"])
        for result in episode_results
    ),
}
gates = {
    "zero_falls": summary["fall_rate"] == 0.0,
    "run_speed_at_least_65_percent": summary["mean_run_speed_fraction"] >= 0.65,
    "run_speed_at_most_135_percent": summary["mean_run_speed_fraction"] <= 1.35,
    "speed_rmse_within_command_scaled_limit": (
        summary["run_speed_tracking_rmse_m_s"] <= max(0.05, 0.50 * args.speed)
    ),
    "every_leg_releases_in_at_least_70_percent_of_scheduled_swing": (
        min(summary["run_scheduled_release_rate_by_leg"].values()) >= 0.70
    ),
    "paired_release_quality_at_least_0_70": (
        summary["mean_run_pair_release_quality"] >= 0.70
    ),
    "paired_release_mismatch_at_most_0_10": (
        summary["mean_run_pair_release_mismatch"] <= 0.10
    ),
    "settles_in_every_episode": summary["settled_episode_rate"] == 1.0,
    "settles_within_1_5_s": (
        summary["mean_settle_time_s"] is not None
        and summary["mean_settle_time_s"] <= 1.5
    ),
    "final_speed_at_most_0_03_m_s": summary["mean_final_speed_m_s"] <= 0.03,
    "stable_for_at_least_80_percent_of_hold": summary["hold_stable_fraction"] >= 0.80,
    "four_feet_for_at_least_90_percent_of_hold": (
        summary["hold_all_feet_contact_fraction"] >= 0.90
    ),
    "hold_target_rms_at_most_0_30_normalized": (
        summary["mean_hold_normalized_target_rms"] <= 0.30
    ),
    "final_stop_error_at_most_0_08_m": (
        summary["mean_final_stop_position_error_m"] <= 0.08
    ),
    "lateral_ratio_at_most_0_15": (
        summary["mean_absolute_lateral_per_forward"] <= 0.15
    ),
    "heading_error_at_most_8_deg": (
        summary["mean_abs_final_heading_error_rad"] <= math.radians(8.0)
    ),
    "rms_joint_acceleration_at_most_25_rad_s2": (
        summary["rms_joint_acceleration_rad_s2"] <= 25.0
    ),
    "rms_body_linear_acceleration_at_most_2_m_s2": (
        summary["rms_body_linear_acceleration_m_s2"] <= 2.0
    ),
    "rms_body_angular_acceleration_at_most_12_rad_s2": (
        summary["rms_body_angular_acceleration_rad_s2"] <= 12.0
    ),
    "mean_action_acceleration_at_most_0_35_per_step2": (
        summary["mean_abs_action_acceleration_per_step2"] <= 0.35
    ),
    "run_action_saturation_fraction_at_most_0_05": (
        summary["run_action_saturation_joint_fraction"] <= 0.05
    ),
    "run_target_limiter_fraction_at_most_0_10": (
        summary["run_target_limiter_joint_fraction"] <= 0.10
    ),
    "run_target_velocity_ceiling_fraction_at_most_0_15": (
        summary["run_target_velocity_ceiling_joint_fraction"] <= 0.15
    ),
    # V31's failed hardware trace reached 25.4 deg p95 and 33.7 deg maximum
    # target-to-position error.  Reject a simulated candidate before it enters
    # that same delayed physical regime, while retaining modest headroom above
    # V34's first cadence-matched nominal result (8.7 / 13.2 deg).
    "run_joint_tracking_error_p95_at_most_12_deg": (
        summary["run_joint_tracking_error_p95_deg"] <= 12.0
    ),
    "run_joint_tracking_error_max_at_most_25_deg": (
        summary["run_joint_tracking_error_max_deg"] <= 25.0
    ),
    "effort_soft_limit_fraction_at_most_0_10": (
        summary["effort_soft_limit_joint_fraction"] <= 0.10
    ),
}
flat_contact_summary = None
if flat_contact_enabled:
    def _present_extreme(values, reducer):
        present = [float(value) for value in values if value is not None]
        return reducer(present) if present else None

    def _leg_values(phase, leg, metric, statistic):
        return [episode["flat_contact_diagnostics"]["by_phase_and_leg"][phase][leg][metric][statistic]
                for episode in episode_results]

    per_leg = {}
    for leg in foot_leg_names:
        per_leg[leg] = {
            "run_support_tilt_mean_deg": _optional_mean(_leg_values("run", leg, "support_sole_tilt_deg", "mean")),
            "worst_episode_run_support_tilt_p95_deg": _present_extreme(
                _leg_values("run", leg, "support_sole_tilt_deg", "p95"), max),
            "worst_episode_force_confirmed_tilt_p95_deg": _present_extreme(
                _leg_values("run", leg, "force_confirmed_sole_tilt_deg", "p95"), max),
            "worst_episode_core_swing_corner_p05_m": _present_extreme(
                _leg_values("run", leg, "core_swing_lowest_corner_height_m", "p05"), min),
            "minimum_core_swing_corner_height_m": _present_extreme(
                _leg_values("run", leg, "core_swing_lowest_corner_height_m", "min"), min),
            "worst_episode_swing_clearance_deficit_p95_m": _present_extreme(
                _leg_values("run", leg, "swing_clearance_deficit_m", "p95"), max),
            "worst_episode_run_contact_patch_slip_rms_m_s": _present_extreme(
                _leg_values("run", leg, "contact_patch_slip_speed_m_s", "rms"), max),
            "worst_episode_hold_support_tilt_p95_deg": _present_extreme(
                _leg_values("hold", leg, "support_sole_tilt_deg", "p95"), max),
            "core_swing_samples_in_every_episode": all(
                count > 0 for count in _leg_values("run", leg, "core_swing_lowest_corner_height_m", "samples")),
            "force_contact_samples_in_every_episode": all(
                count > 0 for count in _leg_values("run", leg, "contact_patch_slip_speed_m_s", "samples")),
        }
    flat_contact_summary = {
        "qualification": "provisional_simulation_filters_not_hardware_qualification",
        "run_warmup_excluded_s": run_warmup_steps / control_hz,
        "core_swing_minimum_lift_envelope_fraction": .90,
        "terminal_frame_handling": (
            "Auto-reset terminal frames are excluded from geometry and final-position metrics; "
            "fall_phase records the failing step. Final values on a fall are the last nonterminal state."
        ),
        "slip_definition": "World-planar velocity RMS of corners near the lowest edge on a force-confirmed foot; contact-patch approximation, not a measured PhysX contact point.",
        "lag_definition": "Peak centered target/position correlation over 0 to 0.75 s; null without >=1 deg target range, >=0.60 correlation or an interior optimum. Includes control-grid timing uncertainty; not identified servo latency.",
        "by_leg": per_leg,
        "mean_commanded_run_speed_m_s": _optional_mean(
            episode["flat_contact_diagnostics"]["run_commanded_speed_m_s"]["mean"] for episode in episode_results),
        "mean_actual_run_speed_m_s": _optional_mean(
            episode["flat_contact_diagnostics"]["run_actual_forward_speed_m_s"]["mean"] for episode in episode_results),
        "mean_run_packet_held_fraction": _optional_mean(
            episode["flat_contact_diagnostics"]["run_packet_held_fraction"] for episode in episode_results),
        "worst_run_corrected_target_ceiling_fraction_by_joint": {
            joint: _present_extreme(
                (episode["flat_contact_diagnostics"]["run_corrected_target_ceiling_fraction_by_joint"][joint]
                 for episode in episode_results), max)
            for joint in joint_names
        },
        "worst_credible_run_target_to_position_lag_s_by_joint": {
            joint: _present_extreme(
                (episode["flat_contact_diagnostics"]["run_target_to_position_lag_by_joint"].get(joint, {}).get("lag_s")
                 for episode in episode_results), max)
            for joint in joint_names
        },
        "maximum_hold_drift_from_first_hold_sample_m": _present_extreme(
            (episode["flat_contact_diagnostics"]["hold_drift_from_first_hold_sample_m"]["max"]
             for episode in episode_results), max),
        "fall_rate_by_phase": {phase: _mean([float(episode["fall_phase"] == phase) for episode in episode_results])
                               for phase in ("run", "deceleration", "hold")},
    }
    # These added V44 filters reject missing coverage and the toe/drag failure
    # without changing the existing report schema or V43 gates.
    gates.update({
        "v44_every_leg_has_core_swing_and_contact_samples_in_every_episode": all(
            metrics["core_swing_samples_in_every_episode"] and metrics["force_contact_samples_in_every_episode"]
            for metrics in per_leg.values()),
        "v44_worst_support_sole_tilt_p95_at_most_12_deg": all(
            metrics["worst_episode_run_support_tilt_p95_deg"] is not None
            and metrics["worst_episode_run_support_tilt_p95_deg"] <= 12.
            for metrics in per_leg.values()),
        "v44_core_swing_corner_p05_at_least_5_mm": all(
            metrics["worst_episode_core_swing_corner_p05_m"] is not None
            and metrics["worst_episode_core_swing_corner_p05_m"] >= .005
            for metrics in per_leg.values()),
        "v44_contact_patch_slip_rms_within_command_scaled_limit": all(
            metrics["worst_episode_run_contact_patch_slip_rms_m_s"] is not None
            and metrics["worst_episode_run_contact_patch_slip_rms_m_s"] <= max(.015, .5 * args.speed)
            for metrics in per_leg.values()),
    })

crawl_evaluation = None
if args.schedule_aware_crawl:
    crawl_episodes = [episode["flat_contact_diagnostics"]["sequential_crawl"] for episode in episode_results]
    crawl_legs = [metrics for episode in crawl_episodes for metrics in episode["by_leg"].values()]
    pair_gate_names = ("paired_release_quality_at_least_0_70", "paired_release_mismatch_at_most_0_10")
    for name in pair_gate_names:
        gates.pop(name)
    gates.update({
        "crawl_two_complete_core_bouts_per_leg_per_episode_with_at_least_three_samples_each": all(
            metrics["complete_core_bout_count"] >= 2
            and metrics["minimum_samples_per_complete_core_bout"] is not None
            and metrics["minimum_samples_per_complete_core_bout"] >= 3 for metrics in crawl_legs),
        "crawl_no_unexpected_multi_swing_or_nonfinite_force_samples": all(
            episode["steady_run_samples"] > 0 and episode["unexpected_multi_swing_schedule_samples"] == 0
            and episode["nonfinite_foot_force_samples"] == 0 for episode in crawl_episodes),
        "crawl_three_loaded_feet_at_least_95_percent_in_every_episode": all(
            episode["at_least_three_loaded_feet_fraction"] is not None
            and episode["at_least_three_loaded_feet_fraction"] >= .95 for episode in crawl_episodes),
        "crawl_fewer_than_three_loaded_feet_never_longer_than_0_10_s": all(
            episode["maximum_continuous_fewer_than_three_loaded_feet_s"] <= .10 for episode in crawl_episodes),
        "crawl_each_scheduled_support_foot_loaded_at_least_95_percent_per_episode": all(
            metrics["loaded_support_fraction"] is not None and metrics["loaded_support_fraction"] >= .95
            for metrics in crawl_legs),
        "crawl_each_core_swing_released_clear_and_three_supported_at_least_95_percent": all(
            metrics["coordinated_core_fraction"] is not None and metrics["coordinated_core_fraction"] >= .95
            and metrics["worst_complete_core_bout_coordinated_fraction"] is not None
            and metrics["worst_complete_core_bout_coordinated_fraction"] >= .95 for metrics in crawl_legs),
        # Existing cross-episode mean release gate remains; do not let it hide
        # a single failed leg/episode in the new sequential foundation mode.
        "crawl_every_leg_releases_at_least_70_percent_in_every_episode": all(
            min(episode["run_scheduled_release_rate_by_leg"].values()) >= .70 for episode in episode_results),
    })
    crawl_evaluation = {
        **crawl_schedule,
        "not_applicable_gates": {name: "No paired-swing opportunity in validated sequential schedule; replaced by simultaneous loaded-three-support/released-clear-swing gates."
                                 for name in pair_gate_names},
        "unchanged_gates": "All existing fall, speed, straightness, effort, limiter/tracking, sole clearance/tilt/slip and graceful-stop gates remain active.",
        "minimum_complete_core_bouts_per_leg_per_episode": 2,
        "minimum_samples_per_complete_core_bout": 3,
        "support_and_coordinated_core_minimum_fraction": .95,
        "maximum_continuous_under_supported_s": .10,
        "world_path_semantics": "Original world-path displacement/heading metrics retained; no reference-relative subtraction or lateral gate relaxation.",
        "episodes": crawl_episodes,
        "passes_crawl_foundation_gates": all(gates.values()),
        "warning": "Passing this opt-in report establishes only the evaluated commanded-speed crawl foundation. It does not qualify a running goal or authorize hardware deployment. Missing or partial phase coverage fails.",
    }

report = {
    "schema": "drobot.run-stop-evaluation.v2",
    "initialization_version": INITIALIZATION_VERSION,
    "episode_observation_initializations": episode_observation_initializations,
    "initialization_rng_caveat": "V44 v1 intentionally generates one fresh initial observation after all manual reset/phase/command changes. Its existing IMU noise operations consume RNG draws, including draws later masked in nominal mode; later random draws need not match old cached-input evaluations at the same seed. No RNG state is restored or noise path bypassed.",
    "checkpoint": checkpoint,
    **evaluation_source_provenance,
    "zero_policy_actions": args.zero_policy_actions,
    "policy_execution_mode": (
        "UNQUALIFIED_OBSERVATION_DIAGNOSTIC_deterministic_actor_pre_position_fd"
        if args.actor_joint_velocity_fd else "reference_only_zero_residual_actions_not_learned_policy"
        if args.zero_policy_actions else "deterministic_checkpoint_policy"
    ),
    "task": args.task,
    "seed": args.seed,
    "domain_mode": args.domain_mode,
    "fixed_start_phase_zero": args.fixed_start_phase_zero,
    "phase_alignment": {
        "enabled": args.ramp_start_phase_cycles is not None,
        "ramp_start_phase_cycles": args.ramp_start_phase_cycles,
        "phase_tolerance_cycles": args.ramp_start_phase_tolerance_cycles,
        "maximum_wait_s": args.maximum_phase_alignment_wait_s,
    },
    "command": {
        "speed_m_s": args.speed,
        "run_seconds": args.run_seconds,
        "ramp_seconds": args.ramp_seconds,
        "ramp_shape": "cubic_smoothstep",
        "hold_seconds": args.hold_seconds,
    },
    "static_domain_parameters": static_domain_parameters,
    "controller_measurement_contract": controller_measurement_contract,
    "effort_observable": _effort_observable_contract(unwrapped),
    "critic_context": _critic_context_contract(unwrapped),
    "phase_reference_sampling": _phase_reference_sampling_contract(unwrapped),
    "summary": summary,
    "gates": gates,
    "passes_all_gates": all(gates.values()),
    "episodes": episode_results,
    **({"qualification_status": "UNQUALIFIED_EFFORT_OBSERVABLE_ABLATION",
        "passes_all_gates_scope": "unchanged_numeric_gates_not_hardware_qualification",
        "qualifies_hardware": False, "production_export_allowed": False}
       if env_cfg.native_effort_telemetry_enabled else {}),
    **({"qualification_status": "UNQUALIFIED_CRITIC_CONTEXT_ABLATION",
        "passes_all_gates_scope": "unchanged_numeric_gates_not_hardware_qualification",
        "qualifies_hardware": False, "production_export_allowed": False}
       if env_cfg.critic_context_mode != "legacy58" else {}),
    **({"qualification_status": "UNQUALIFIED_REAR_PITCH_LIFT_ABLATION",
        "rear_pitch_lift": unwrapped._rear_pitch_lift_contract,
        "passes_all_gates_scope": "unchanged_numeric_gates_not_policy_or_hardware_qualification",
        "qualifies_running_goal": False, "qualifies_hardware": False,
        "production_export_allowed": False} if rear_pitch_evaluation else {}),
    **({"qualification_status": "UNQUALIFIED_REAR_PITCH_PROJECTION_ABLATION",
        "rear_pitch_projection": unwrapped._rear_pitch_projection_contract,
        "passes_all_gates_scope": "unchanged_numeric_gates_not_policy_or_hardware_qualification",
        "qualifies_running_goal": False, "qualifies_hardware": False,
        "production_export_allowed": False} if rear_projection_evaluation else {}),
    **_sustained_credit_metadata(),
    **({"passive_physics_step_diagnostic": passive_physics_diagnostic_result}
       if passive_physics_diagnostic_result is not None else {}),
    **({"environment_ablation": ablation_provenance} if ablation_provenance is not None else {}),
    **({"flat_contact_summary": flat_contact_summary} if flat_contact_summary is not None else {}),
    **({"sequential_crawl_evaluation": crawl_evaluation,
        "passes_all_gates_scope": "commanded_speed_crawl_foundation_only_not_running",
        "qualifies_running_goal": False} if crawl_evaluation is not None else {}),
    **({"physx_effort_readback": {
        "samples": effort_limit_readbacks,
        "all_samples_confirm_expected_limits": bool(effort_limit_readbacks) and all(
            sample["matches_expected_within_1e_5_nm"] is True for sample in effort_limit_readbacks),
        "semantics": "Read-only live physics-view max-force property versus expected domain values and cached articulation values. Backend DOFs reordered by exact joint names. Failure/unavailable data never substitutes cache for physics. No forces, limits or gate thresholds are changed. Constructor capture precedes vector-wrapper reset; each explicit episode reset and first step is captured. A terminal first step may already have autoreset and is labeled accordingly. Confirms configured limits only, not measured torque or hardware capability.",
    }} if args.physx_effort_readback else {}),
    **({"qualification_status": "UNQUALIFIED_OBSERVATION_DIAGNOSTIC",
        "actor_joint_velocity_fd_diagnostic": actor_velocity_fd.metadata,
        "passes_all_gates_scope": "numeric_diagnostic_gates_only_not_policy_qualification",
        "qualifies_running_goal": False, "qualifies_hardware": False,
        "production_export_allowed": False} if actor_velocity_fd is not None else {}),
}
try:
    if rear_projection_evaluation:
        report["rear_pitch_projection_summary"] = diagnostic_trace.rear_projection_summary()
    if diagnostic_trace is not None:
        report["diagnostic_trace"] = diagnostic_trace.write(report)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if rear_pitch_evaluation:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized + "\n")
        else:
            args.output.write_text(serialized + "\n", encoding="utf-8")
finally:
    env.close()
    simulation_app.close()
