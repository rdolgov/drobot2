"""V44 provenance and export guards; this module does not start Isaac or training."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROFILE = "flat-contact-hardware-matched-run-stop-external-rear-payload"
TASK = "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
EXPERIMENT = "drobot_commanded_run_v44_flat_contact_hardware_matched_stop_external_rear_payload"
METHOD = "flat-sole-corner-clearance.v1"
PHASES = [0.0, 0.5, 0.5, 0.0]
RESIDUAL_SCALES = [0.06] * 4 + [0.20] * 8
VELOCITY_LIMITS = [math.radians(60.0)] * 4 + [math.radians(90.0)] * 8

# These are executable training/deployment settings, not a new model merely
# relabeled from a prior checkpoint. The saved effective training config must
# match before provenance or an ONNX package is produced.
ENV_CONTRACT = {
    "decimation": 2,
    "sim.dt": 1.0 / 120.0,
    "action_mode": "gait_residual",
    "action_history_mode": "applied_target",
    "action_scale_abduction_rad": 0.12,
    "action_scale_hip_rad": 0.30,
    "action_scale_knee_rad": 0.40,
    "residual_action_scale_by_action": RESIDUAL_SCALES,
    "gait_residual_stop_action_scale": 0.0,
    "gait_swing_residual_scale_by_leg": [1.0, 1.0, 1.0, 1.0],
    "gait_swing_residual_ablation_task": None,
    "gait_rear_pitch_lift_m": 0.0,
    "gait_rear_pitch_lift_ablation_task": None,
    "gait_rear_pitch_projection_enabled": False,
    "gait_rear_pitch_projection_ablation_task": None,
    "gait_rear_pitch_descent_fraction": 0.5,
    "penalty_normalized_contact_slip": 0.0,
    "normalized_contact_slip_ablation_task": None,
    "gait_front_stance_tracking_feedback_gain": 0.0,
    "gait_front_stance_tracking_feedback_ablation_task": None,
    "gait_rear_tracking_feedback_gain": 0.0,
    "gait_rear_tracking_feedback_leg_mask": [1.0, 1.0],
    "gait_rear_tracking_feedback_mode": "independent",
    "gait_rear_tracking_feedback_ablation_task": None,
    "decouple_sustained_progress_from_pair_release": False,
    "sustained_progress_decoupling_ablation_task": None,
    "target_velocity_limit_rad_s": math.radians(90.0),
    "target_velocity_limit_rad_s_by_action": VELOCITY_LIMITS,
    "target_packet_step_limit_rad": math.radians(2.0),
    "target_packet_hold_probability": 0.12,
    "target_packet_max_hold_steps": 1,
    "control_delay_step_range": [0, 0],
    "actuator_nominal_stiffness_scale": 1.0,
    "actuator_nominal_damping_scale": 1.0,
    "actuator_sync_drive_gain_telemetry": True,
    "native_effort_telemetry_enabled": False,
    "native_effort_reward_enabled": False,
    "native_effort_ablation_task": None,
    "critic_context_mode": "legacy58",
    "critic_context_ablation_task": None,
    "observation_space": 50,
    "state_space": 58,
    "actuator_target_response_enabled": True,
    "actuator_target_response_nominal_tau_s": 0.030,
    "actuator_target_response_nominal_gain": 0.97,
    "actuator_target_response_tau_s_range": [0.010, 0.060],
    "actuator_target_response_gain_range": [0.90, 1.0],
    "actuator_target_response_tau_s_range_by_action": (
        [[0.010, 0.030]] * 4
        + [[0.020, 0.070], [0.010, 0.040], [0.020, 0.070], [0.010, 0.040]]
        + [[0.020, 0.070], [0.020, 0.060], [0.020, 0.070], [0.020, 0.060]]
    ),
    "penalty_stance_sole_tilt": 12.0,
    "penalty_body_tilt": 8.0,
    "penalty_swing_sole_tilt": 0.0,
    "swing_sole_tilt_ablation_task": None,
    "penalty_swing_corner_clearance": 6.0,
    "penalty_contact_point_slip": 4.0,
    "sole_clearance_target_m": 0.010,
    "sole_clearance_transition_fraction": 0.5,
    "swing_clearance_worst_leg_fraction": 0.0,
    "gait_period_s": 20.0 / 9.0,
    "gait_reference_mode": "phase_table",
    "gait_clock_mode": "speed_scaled",
    "integrate_gait_phase": True,
    "gait_park_at_standstill": True,
    "gait_phase_table_park_contact_override": True,
    "gait_phase_table_lateral_reference_enabled": False,
    "gait_phase_table_forward_reference_enabled": False,
    "normalized_posture_cost_enabled": False,
    "normalized_body_height_cost_enabled": False,
    "sustained_speed_window_s": 3.0,
    "gait_standstill_deadband_m_s": 0.004,
    "gait_speed_min_m_s": 0.03,
    "gait_speed_max_m_s": 0.05,
    "gait_frequency_min_hz": 0.35,
    "gait_frequency_max_hz": 0.45,
    "gait_stride_scale_min": 1.0,
    "gait_duty_factor": 0.55,
    "gait_phase_offsets": PHASES,
    "gait_phase_reference_scale_about_neutral": 1.0,
    "gait_phase_reference_linear_interpolation_enabled": False,
    "gait_phase_reference_interpolation_ablation_task": None,
    "gait_phase_reference_blend_speed_min_m_s": 0.03,
    "gait_phase_reference_blend_speed_max_m_s": 0.05,
    "gait_start_ramp_s": 1.5,
    "gait_contact_transition_fraction": 0.0,
    "gait_stance_fore_aft_m": 0.080,
    "gait_stance_down_m": 0.329341447,
    "gait_stance_center_offset_m": 0.0,
    "gait_forward_body_pitch_rad": 0.0,
    "target_forward_pitch_rad": 0.0,
    "target_base_height_m": 0.375,
    "forward_speed_min_m_s": 0.03,
    "forward_speed_max_m_s": 0.05,
    "command_observation_scale": [10.0, 10.0, 10.0],
    "heading_hold_kp_s": 0.75,
    "heading_hold_max_correction_rad_s": 0.12,
    "run_stop_run_duration_min_s": 10.0,
    "run_stop_run_duration_max_s": 12.0,
    "run_stop_ramp_duration_min_s": 0.55,
    "run_stop_ramp_duration_max_s": 1.0,
    "run_stop_abrupt_fraction": 0.15,
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_equal(name: str, actual: object, expected: object) -> None:
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(f"{name}: expected {expected!r}, received {actual!r}")
        for index, (value, wanted) in enumerate(zip(actual, expected, strict=True)):
            require_equal(f"{name}[{index}]", value, wanted)
    elif isinstance(expected, float):
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isclose(
            actual, expected, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ValueError(f"{name}: expected {expected!r}, received {actual!r}")
    elif actual != expected:
        raise ValueError(f"{name}: expected {expected!r}, received {actual!r}")


def read_config_fields(path: Path, keys: set[str]) -> dict[str, object]:
    """Read selected (optionally dotted) nodes without constructing config objects."""
    import yaml

    def plain(node):
        if isinstance(node, yaml.SequenceNode):
            return [plain(item) for item in node.value]
        if isinstance(node, yaml.ScalarNode):
            if node.tag == "tag:yaml.org,2002:bool":
                return node.value.lower() in ("true", "yes", "on")
            if node.tag == "tag:yaml.org,2002:int":
                return int(node.value)
            if node.tag == "tag:yaml.org,2002:float":
                return float(node.value)
            if node.tag == "tag:yaml.org,2002:null":
                return None
            return node.value
        raise ValueError(f"Expected a scalar or sequence in {path.name}")

    document = yaml.compose(path.read_text(encoding="utf-8"), Loader=yaml.SafeLoader)
    if not isinstance(document, yaml.MappingNode):
        raise ValueError(f"{path.name} must be a mapping")
    result = {}
    for requested in keys:
        node = document
        for component in requested.split("."):
            if not isinstance(node, yaml.MappingNode):
                node = None
                break
            node = next(
                (value for key, value in node.value if key.value == component), None
            )
            if node is None:
                break
        if node is not None:
            result[requested] = plain(node)
    return result


def validate_reference(path: Path, *, speed: float, frequency: float) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require_equal("reference.schema", payload.get("schema"), "drobot.phase-joint-reference.v1")
    require_equal("reference.sample_count", payload.get("sample_count"), 512)
    generation = payload.get("generation", {})
    expected = {
        "method": METHOD,
        "training_profile": PROFILE,
        "duty_factor": 0.55,
        "phase_offsets_fl_rl_fr_rr": PHASES,
        "speed_m_s": speed,
        "frequency_hz": frequency,
        "stance_fore_aft_m": 0.08,
        "sole_corner_clearance_m": 0.010,
        "clearance_envelope": "quintic-symmetric.v1",
        "clearance_transition_fraction": 0.5,
        "target_velocity_limit_deg_s": 90.0,
        "packet_step_limit_deg": 2.0,
    }
    for key, value in expected.items():
        require_equal(f"reference.generation.{key}", generation.get(key), value)
    return {"filename": path.name, "sha256": digest(path), "generation": generation}


def training_snapshot(experiment_root: Path) -> dict[Path, tuple[int, int]]:
    return {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in experiment_root.glob("*/model_*.pt")
        if not path.parent.name.startswith("_")
    }


def validate_saved_training(run_directory: Path) -> dict:
    params = run_directory / "params"
    env_path, agent_path = params / "env.yaml", params / "agent.yaml"
    env = read_config_fields(env_path, set(ENV_CONTRACT) | {
        "gait_phase_reference_path", "gait_phase_reference_low_path"
    })
    for key, expected in ENV_CONTRACT.items():
        require_equal(f"trained env.{key}", env.get(key), expected)
    agent = read_config_fields(agent_path, {"experiment_name"})
    require_equal("trained experiment", agent.get("experiment_name"), EXPERIMENT)
    references = {}
    for label, field, speed, frequency in (
        ("high", "gait_phase_reference_path", 0.05, 0.45),
        ("low", "gait_phase_reference_low_path", 0.03, 0.35),
    ):
        value = env.get(field)
        if not isinstance(value, str):
            raise ValueError(f"Missing trained reference: {field}")
        references[label] = validate_reference(Path(value), speed=speed, frequency=frequency)
    return {
        "environment_contract": ENV_CONTRACT,
        "references": references,
        "params_sha256": {"env.yaml": digest(env_path), "agent.yaml": digest(agent_path)},
    }


def record_completed_training(experiment_root: Path, before: dict) -> None:
    """Bind newly saved weights to effective configs only after successful training."""
    after = training_snapshot(experiment_root)
    for checkpoint, fingerprint in after.items():
        if before.get(checkpoint) == fingerprint:
            continue
        contract = validate_saved_training(checkpoint.parent)
        record = {
            "schema": "drobot.training-provenance.v1",
            "training_profile": PROFILE,
            "training_task": TASK,
            "experiment_name": EXPERIMENT,
            "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": digest(checkpoint),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            **contract,
        }
        checkpoint.with_suffix(".training.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8"
        )


def validate_export_provenance(checkpoint: Path, provenance: Path | None, high: Path, low: Path) -> dict:
    path = provenance or checkpoint.with_suffix(".training.json")
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value in {
        "schema": "drobot.training-provenance.v1",
        "training_profile": PROFILE,
        "training_task": TASK,
        "experiment_name": EXPERIMENT,
        "checkpoint_filename": checkpoint.name,
        "checkpoint_sha256": digest(checkpoint),
    }.items():
        require_equal(f"provenance.{key}", record.get(key), value)
    # Copying an old checkpoint into a V44-named directory is insufficient.
    # Its hash-bound sidecar and the actual saved training configs must agree.
    contract = validate_saved_training(path.parent)
    for key in ("environment_contract", "params_sha256", "references"):
        require_equal(f"provenance.{key}", record.get(key), contract[key])
    for label, reference, speed, frequency in (
        ("high", high, 0.05, 0.45), ("low", low, 0.03, 0.35)
    ):
        exported = validate_reference(reference, speed=speed, frequency=frequency)
        require_equal(f"export reference {label}", exported, record["references"][label])
    return record


def validate_export_arguments(args, residual_scales, target_limits) -> dict:
    """Reject incomplete CLI overrides before loading weights or writing ONNX."""
    expected = {
        "training_profile": PROFILE,
        "training_task": TASK,
        "gait_clock_mode": "speed_scaled",
        "integrated_gait_phase": True,
        "gait_park_at_standstill": True,
        "gait_period_s": 20.0 / 9.0,
        "gait_standstill_deadband_m_s": 0.004,
        "gait_speed_min_m_s": 0.03,
        "gait_speed_max_m_s": 0.05,
        "gait_frequency_min_hz": 0.35,
        "gait_frequency_max_hz": 0.45,
        "gait_stride_scale_min": 1.0,
        "gait_pattern": "diagonal_trot",
        "gait_duty_factor": 0.55,
        "gait_phase_offsets": PHASES,
        "action_mode": "gait_residual",
        "action_history_mode": "applied_target",
        "residual_action_scale": 0.06,
        "gait_residual_stop_action_scale": 0.0,
        "reference_start_ramp_s": 1.5,
        "reference_blend_speed_min_m_s": 0.03,
        "reference_blend_speed_max_m_s": 0.05,
        "heading_hold_enabled": True,
        "heading_hold_kp_s": 0.75,
        "heading_hold_max_correction_rad_s": 0.12,
        "target_velocity_limit_rad_s": math.radians(90.0),
        "max_target_step_deg": 2.0,
        "startup_ramp_rate_deg_s": 45.0,
        "startup_settle_s": 0.5,
        "startup_position_tolerance_deg": 5.0,
        "forward_speed_min_m_s": 0.03,
        "forward_speed_max_m_s": 0.05,
        "recommended_forward_speed_m_s": 0.04,
        "command_observation_scales": [10.0, 10.0, 10.0],
    }
    for key, value in expected.items():
        require_equal(f"export.{key}", getattr(args, key), value)
    require_equal("export.residual_scales", residual_scales, RESIDUAL_SCALES)
    require_equal("export.target_limits", target_limits, VELOCITY_LIMITS)
    if args.reference_phase_table_path is None or args.reference_low_phase_table_path is None:
        raise ValueError("V44 requires both generated flat-contact reference tables")
    output = args.output.resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("V44 export requires a new output path; existing packages are not overwritten")
    if any("v43" in part.lower() for part in output.parts):
        raise ValueError("V44 must not be exported into a V43 package")
    return validate_export_provenance(
        args.checkpoint.resolve(), args.training_provenance,
        args.reference_phase_table_path.resolve(), args.reference_low_phase_table_path.resolve(),
    )
