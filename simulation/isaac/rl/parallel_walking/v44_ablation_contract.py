"""Strict, reusable environment overrides for isolated V44 geometry ablations.

No simulator import, code evaluation, interpolation or arbitrary Hydra key is
allowed. This changes no production defaults and never edits training configs.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .v44_training_contract import TASK
else:
    from v44_training_contract import TASK

PATH_FIELDS = {"gait_phase_reference_path", "gait_phase_reference_low_path"}
NULLABLE_FIELDS = {
    "gait_phase_reference_low_path", "gait_stance_down_by_leg_m",
    "gait_phase_reference_blend_speed_min_m_s", "gait_phase_reference_blend_speed_max_m_s",
}
BOOLEAN_FIELDS = {
    "decouple_sustained_progress_from_pair_release",
    "gait_rear_pitch_projection_enabled",
    "gate_progress_by_scheduled_pair_release",
    "gait_phase_table_lateral_reference_enabled",
    "gait_phase_table_forward_reference_enabled",
    "require_schedule_matched_support",
    "gate_progress_by_schedule_matched_support",
    "use_soft_schedule_matched_progress_gate",
    "use_soft_scheduled_stance_quality",
    "normalized_posture_cost_enabled",
    "normalized_body_height_cost_enabled",
    "gait_phase_reference_linear_interpolation_enabled",
    "native_effort_telemetry_enabled",
    "native_effort_reward_enabled",
}
INTEGER_FIELDS = {"episode_horizon_curriculum_steps"}
ENUM_FIELDS = {"critic_context_mode": {"legacy58", "context90_zero", "context90_causal"},
               "gait_rear_tracking_feedback_mode": {"independent", "fixed_pitch"}}
VECTOR_LENGTHS = {"residual_action_scale_by_action": 12, "gait_phase_offsets": 4,
                  "gait_rear_tracking_feedback_leg_mask": 2,
                  "gait_swing_residual_scale_by_leg": 4,
                  "gait_stance_down_by_leg_m": 4,
                  "rear_payload_combined_mass_scale_range": 2,
                  "rear_payload_combined_com_jitter_m": 3,
                  "robot_mass_scale_range": 2, "actuator_effort_scale_range": 2,
                  "target_velocity_scale_range": 2,
                  "actuator_individual_effort_scale_range": 2,
                  "target_individual_velocity_scale_range": 2}
NUMERIC_FIELDS = {
    "gait_front_stance_tracking_feedback_gain",
    "gait_rear_tracking_feedback_gain",
    "gait_rear_pitch_lift_m",
    "gait_rear_pitch_descent_fraction",
    "residual_action_scale", "gait_frequency_min_hz", "gait_frequency_max_hz",
    "gait_speed_min_m_s", "gait_speed_max_m_s", "gait_period_s", "gait_duty_factor",
    "gait_phase_reference_scale_about_neutral",
    "gait_phase_reference_blend_speed_min_m_s", "gait_phase_reference_blend_speed_max_m_s",
    "gait_start_ramp_s", "gait_stance_down_m", "gait_stance_fore_aft_m",
    "gait_stance_center_offset_m", "gait_forward_body_pitch_rad",
    "target_base_height_m", "target_forward_pitch_rad", "sole_clearance_target_m",
    "sole_clearance_transition_fraction", "gait_stride_scale_min",
    "swing_clearance_worst_leg_fraction",
    "initial_forward_speed_min_m_s", "initial_forward_speed_max_m_s",
    "forward_speed_min_m_s", "forward_speed_max_m_s", "forward_speed_low_band_max_m_s",
    "forward_speed_low_band_fraction", "gait_reference_sigma",
    "minimum_sustained_speed_m_s", "minimum_sustained_speed_fraction",
    "velocity_tracking_sigma_m_s", "rolling_velocity_tracking_sigma_m_s",
    "physical_randomization_nominal_fraction",
    "actuator_nominal_stiffness_scale", "actuator_nominal_damping_scale",
    "dry_robot_mass_scale",
    "active_translation_threshold_m_s", "sustained_speed_window_s",
    "episode_length_s", "initial_training_horizon_s", "final_training_horizon_s",
    "run_stop_run_duration_min_s", "run_stop_run_duration_max_s",
    "schedule_matched_progress_gate_floor", "four_leg_progress_gate_floor",
    "posture_height_deadband_m", "posture_height_normalizer_m",
    "posture_angle_deadband_rad", "posture_angle_normalizer_rad",
}
WEIGHT_FIELDS = {
    "penalty_body_tilt",
    "penalty_normalized_contact_slip",
    "penalty_swing_sole_tilt",
    "reward_gait_reference", "reward_forward_velocity_tracking", "reward_rolling_velocity_tracking",
    "reward_instant_progress", "reward_sustained_progress", "reward_straight_aligned_progress",
    "penalty_stance_sole_tilt", "penalty_swing_corner_clearance", "penalty_contact_point_slip",
    "penalty_action_magnitude", "penalty_abduction_action_magnitude", "penalty_abduction_target_magnitude",
    "penalty_action_rate", "penalty_action_acceleration", "penalty_joint_acceleration",
    "penalty_body_linear_acceleration", "penalty_body_angular_acceleration",
    "penalty_rolling_velocity_error", "penalty_sustained_stall", "penalty_overspeed",
    "penalty_lateral_displacement", "penalty_lateral_corridor",
    "penalty_diagonal_sync", "reward_scheduled_pair_release", "penalty_scheduled_pair_mismatch",
    "reward_three_foot_support", "penalty_missing_scheduled_stance", "penalty_excess_airborne_feet",
    "penalty_normalized_body_height", "penalty_normalized_body_roll", "penalty_normalized_body_pitch",
}
ALLOWED_FIELDS = PATH_FIELDS | set(VECTOR_LENGTHS) | NUMERIC_FIELDS | WEIGHT_FIELDS | BOOLEAN_FIELDS | INTEGER_FIELDS | set(ENUM_FIELDS)


def _normalized_slip_request(values: dict) -> bool:
    if "penalty_normalized_contact_slip" not in values:
        return False
    value = values["penalty_normalized_contact_slip"]
    if type(value) not in (int, float) or value not in (0., .1):
        raise ValueError("Normalized slip study supports only zero or weight0.1")
    if (_swing_tilt_request(values) != 4. or _rear_feedback_request(values) != .25
        or values.get("gait_front_stance_tracking_feedback_gain", 0.) != 0.):
        raise ValueError("Normalized slip comparison requires unchanged34 controller/tilt4")
    return True


def _front_stance_feedback_request(values: dict) -> float:
    gain = values.get("gait_front_stance_tracking_feedback_gain", 0.)
    if type(gain) not in (int, float) or gain not in (0., .25):
        raise ValueError("Front stance study supports only zero or gain0.25")
    if gain and (values.get("penalty_swing_sole_tilt") != 4.
        or _swing_tilt_request(values) != 4. or _rear_feedback_request(values) != .25):
        raise ValueError("Front stance study requires weight4 and both-rear fixed-pitch feedback")
    return float(gain)


def _swing_tilt_request(values: dict) -> float:
    weight = values.get("penalty_swing_sole_tilt", 0.)
    if type(weight) not in (int, float) or weight not in (0., 2., 4.):
        raise ValueError("Swing sole tilt study supports only zero,weight2 or weight4")
    if weight and (values.get("gait_rear_tracking_feedback_gain") not in (.25,.5)
        or values.get("gait_rear_tracking_feedback_mode") != "fixed_pitch"
        or list(values.get("gait_rear_tracking_feedback_leg_mask", [])) != [1.,1.]):
        raise ValueError("Swing sole tilt study requires both-rear fixed-pitch feedback")
    return float(weight)


def _rear_feedback_request(values: dict) -> float:
    gain = values.get("gait_rear_tracking_feedback_gain", 0.0)
    mask = values.get("gait_rear_tracking_feedback_leg_mask", (1., 1.))
    if (not isinstance(mask, (list, tuple)) or len(mask) != 2
        or any(type(v) not in (int, float) for v in mask)
        or tuple(mask) not in ((1., 1.), (0., 1.))
        or (tuple(mask) != (1., 1.) and gain != .25)):
        raise ValueError("Rear feedback mask supports both rear legs or isolated RR at gain0.25")
    mode = values.get("gait_rear_tracking_feedback_mode", "independent")
    if mode not in ("independent", "fixed_pitch") or (mode == "fixed_pitch" and gain not in (.25,.5)):
        raise ValueError("Fixed-pitch feedback requires isolated rear gain0.25 or0.5")
    if type(gain) not in (int, float) or gain not in (0.0, 0.25, 0.5):
        raise ValueError("Rear tracking comparison supports only zero,gain0.25 or0.5")
    if gain == .5 and (mode != "fixed_pitch" or tuple(mask) != (1.,1.)
        or values.get("penalty_swing_sole_tilt") != 4.
        or values.get("gait_front_stance_tracking_feedback_gain",0.) != 0.
        or "penalty_normalized_contact_slip" in values
        or values.get("physical_randomization_nominal_fraction") != 1.):
        raise ValueError("Gain0.5 requires isolated nominal38 both-rear fixed-pitch controller")
    if gain and (values.get("gait_rear_pitch_projection_enabled") is not True
        or values.get("gait_rear_pitch_lift_m") != .005
        or values.get("gait_rear_pitch_descent_fraction", .5) != .5
        or values.get("penalty_action_acceleration", 3.) != 3.
        or values.get("decouple_sustained_progress_from_pair_release", False) is not False):
        raise ValueError("Rear feedback requires original fixed5mm projection and rewards")
    return float(gain)


def _sustained_decoupling_request(values: dict) -> bool:
    flag = values.get("decouple_sustained_progress_from_pair_release", False)
    if type(flag) is not bool:
        raise ValueError("Sustained-progress decoupling must be Boolean")
    if flag:
        if (values.get("gait_rear_pitch_projection_enabled") is not True
            or values.get("gait_rear_pitch_lift_m") != 0.005
            or values.get("gait_rear_pitch_descent_fraction", 0.5) != 0.5
            or values.get("gate_progress_by_scheduled_pair_release") is not True
            or values.get("critic_context_mode", "legacy58") != "legacy58"):
            raise ValueError("Decoupling requires explicit common-F5, five-mm original timing and retained pair gate")
        for name in ("native_effort_telemetry_enabled", "native_effort_reward_enabled"):
            if values.get(name, False) is not False:
                raise ValueError(f"Decoupling cannot combine with {name}")
    return flag


def validate_sustained_progress_config(cfg) -> None:
    """Shared effective-config guard; the environment also checks exact class identity."""
    flag = cfg.decouple_sustained_progress_from_pair_release
    marker = cfg.sustained_progress_decoupling_ablation_task
    if type(flag) is not bool:
        raise ValueError("Sustained-progress decoupling must be Boolean")
    if not flag:
        if marker is not None:
            raise ValueError("Disabled sustained decoupling requires a None marker")
        return
    required = {
        "sustained_progress_decoupling_ablation_task": TASK,
        "gait_rear_pitch_projection_enabled": True, "gait_rear_pitch_lift_m": .005,
        "gait_rear_pitch_descent_fraction": .5,
        "action_mode": "gait_residual", "gait_reference_mode": "phase_table",
        "command_profile": "run_stop", "action_history_mode": "applied_target",
        "action_space": 12, "observation_space": 50, "state_space": 58,
        "critic_context_mode": "legacy58", "native_effort_telemetry_enabled": False,
        "native_effort_reward_enabled": False, "gate_progress_by_scheduled_pair_release": True,
        "gate_progress_by_four_leg_release": True, "gate_progress_by_cycle_four_leg_release": False,
        "gate_positive_progress_by_straight_alignment": True,
        "gate_progress_by_schedule_matched_support": False,
        "track_episode_world_path": True, "sustained_speed_window_s": 3.0,
        "physical_randomization_nominal_fraction": 1.0,
    }
    if type(cfg).__name__ != "DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg":
        raise ValueError("Sustained decoupling requires the isolated V44 task")
    for name, expected in required.items():
        actual = getattr(cfg, name)
        if actual != expected or (type(expected) is bool and type(actual) is not bool):
            raise ValueError(f"Sustained decoupling requires {name}={expected!r}")


def _rear_pitch_request(values: dict) -> bool:
    projection_enabled = values.get("gait_rear_pitch_projection_enabled", False)
    if type(projection_enabled) is not bool:
        raise ValueError("Common-F5 projection flag must be Boolean")
    descent = values.get("gait_rear_pitch_descent_fraction", 0.5)
    if type(descent) not in (int, float) or descent not in (0.3, 0.5):
        raise ValueError("Rear descent fraction accepts only the isolated0.3 or original0.5")
    if descent != 0.5 and (not projection_enabled or values.get("gait_rear_pitch_lift_m") != 0.005):
        raise ValueError("Changed rear descent requires explicit fixed5mm common-F5 projection")
    if projection_enabled and "gait_rear_pitch_lift_m" not in values:
        raise ValueError("Common-F5 projection requires an explicit0/0.005m lift arm")
    if "gait_rear_pitch_lift_m" not in values:
        return False
    value = values["gait_rear_pitch_lift_m"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value not in (0.0, 0.005):
        raise ValueError("Rear pitch lift accepts only the selected0 or0.005m amplitudes")
    if values.get("critic_context_mode", "legacy58") != "legacy58":
        raise ValueError("Rear pitch lift requires the unchanged legacy58 critic")
    for name in ("gait_phase_reference_linear_interpolation_enabled", "gait_phase_table_lateral_reference_enabled",
                 "gait_phase_table_forward_reference_enabled", "normalized_posture_cost_enabled",
                 "normalized_body_height_cost_enabled", "native_effort_reward_enabled"):
        if values.get(name, False) is not False:
            raise ValueError(f"Rear pitch witness cannot combine with {name}")
    if any(v != 1.0 for v in values.get("gait_swing_residual_scale_by_leg", [1.0]*4)):
        raise ValueError("Rear pitch witness requires identity swing residual scales")
    return True


def _context_request_mode(values: dict) -> str:
    mode = values.get("critic_context_mode", "legacy58")
    if not isinstance(mode, str) or mode not in ENUM_FIELDS["critic_context_mode"]:
        raise ValueError("critic_context_mode must be legacy58, context90_zero or context90_causal")
    if mode != "legacy58":
        if values.get("native_effort_telemetry_enabled") is not True or values.get("native_effort_reward_enabled") is not False:
            raise ValueError("Critic90 requires explicit native telemetry true and native reward false")
        for flag in ("gait_phase_table_lateral_reference_enabled", "gait_phase_table_forward_reference_enabled",
                     "gait_phase_reference_linear_interpolation_enabled", "normalized_posture_cost_enabled",
                     "normalized_body_height_cost_enabled"):
            if values.get(flag, False) is not False:
                raise ValueError(f"Critic context v1 cannot combine with {flag}")
    return mode


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate override key {key!r}")
        result[key] = value
    return result


def _finite_number(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def load_v44_overrides(path: Path, task: str) -> tuple[dict, dict]:
    """Read a flat JSON object; resolve reference paths relative to that file."""
    if task != TASK:
        raise ValueError("Environment ablations are supported only for the exact V44 task")
    path = Path(path).resolve(strict=True)
    values = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(values, dict) or not values:
        raise ValueError("Override JSON must be a nonempty flat object of allowed env attributes")
    unknown = set(values) - ALLOWED_FIELDS
    if unknown:
        raise ValueError(f"Unsupported V44 override keys: {sorted(unknown)}")
    if "penalty_body_tilt" in values:
        weight = values["penalty_body_tilt"]
        if (type(weight) not in (int, float) or weight not in (8., 512.)
            or values.get("normalized_posture_cost_enabled") is not False
            or values.get("normalized_body_height_cost_enabled") is not False
            or _swing_tilt_request(values) != 4. or _rear_feedback_request(values) != .25
            or values.get("gait_front_stance_tracking_feedback_gain", 0.) != 0.
            or values.get("penalty_normalized_contact_slip", 0.) != 0.):
            raise ValueError("Body tilt42 requires weight8/512 and unchanged39 feedback/posture rewards")
    effective = {}
    references = {}
    for name, value in values.items():
        if value is None and name in NULLABLE_FIELDS:
            effective[name] = None
            continue
        if name in PATH_FIELDS:
            if not isinstance(value, str) or not value.strip() or "${" in value or any(ord(c) < 32 for c in value):
                raise ValueError(f"{name} must be a plain nonempty JSON reference path without interpolation")
            reference = Path(value)
            reference = (reference if reference.is_absolute() else path.parent / reference).resolve(strict=True)
            if not reference.is_file() or reference.suffix.lower() != ".json":
                raise ValueError(f"{name} must reference an existing JSON file")
            effective[name] = reference.as_posix()
            references[name] = {"path": reference.as_posix(), "sha256": file_sha256(reference)}
        elif name in ENUM_FIELDS:
            if not isinstance(value, str) or value not in ENUM_FIELDS[name]:
                raise ValueError(f"{name} must be one of {sorted(ENUM_FIELDS[name])}")
            effective[name] = value
        elif name in BOOLEAN_FIELDS:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a JSON boolean")
            effective[name] = value
        elif name in INTEGER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive JSON integer")
            effective[name] = value
        elif name in VECTOR_LENGTHS:
            if not isinstance(value, list) or len(value) != VECTOR_LENGTHS[name]:
                raise ValueError(f"{name} must be a list of {VECTOR_LENGTHS[name]} finite numbers")
            effective[name] = [_finite_number(f"{name}[{index}]", item) for index, item in enumerate(value)]
            if name in {"residual_action_scale_by_action", "gait_swing_residual_scale_by_leg"} and any(not 0. <= item <= 1. for item in effective[name]):
                raise ValueError("Residual scales must be within [0, 1]")
            if name == "gait_phase_offsets" and any(not 0. <= item < 1. for item in effective[name]):
                raise ValueError("Gait phase offsets must be within [0, 1)")
            if name == "gait_stance_down_by_leg_m" and any(item <= 0. for item in effective[name]):
                raise ValueError("Leg stance depths must be positive")
            if name.endswith("_scale_range") and not 0. < effective[name][0] <= effective[name][1]:
                raise ValueError(f"{name} must be positive ordered bounds")
            if name in {"actuator_individual_effort_scale_range", "target_individual_velocity_scale_range"} and effective[name][1] > 1.0:
                # These newly exposed qualification-ablation fields may reduce
                # available capacity but must not exceed the nominal envelope.
                # Existing production defaults are unchanged when omitted.
                raise ValueError(f"{name} is a capped qualification override; upper bound cannot exceed 1")
            if name == "rear_payload_combined_com_jitter_m" and any(item < 0. for item in effective[name]):
                raise ValueError("COM jitter magnitudes cannot be negative")
        else:
            effective[name] = _finite_number(name, value)
            if name in WEIGHT_FIELDS and effective[name] < 0.:
                raise ValueError(f"{name} cannot be negative")
    for name in ("gait_period_s", "gait_frequency_min_hz", "gait_frequency_max_hz",
                 "gait_start_ramp_s", "gait_stance_down_m", "target_base_height_m",
                 "sole_clearance_target_m", "gait_reference_sigma", "velocity_tracking_sigma_m_s",
                 "rolling_velocity_tracking_sigma_m_s", "active_translation_threshold_m_s",
                 "sustained_speed_window_s", "episode_length_s", "initial_training_horizon_s",
                 "final_training_horizon_s", "run_stop_run_duration_min_s", "run_stop_run_duration_max_s"):
        if name in effective and effective[name] <= 0.:
            raise ValueError(f"{name} must be positive")
    for name in ("actuator_nominal_stiffness_scale", "actuator_nominal_damping_scale"):
        if name in effective and not 0.25 <= effective[name] <= 8.0:
            raise ValueError(f"{name} must be a bounded hypothesis within [0.25, 8]")
    # Mirror the environment's strict bounds without importing simulator code.
    for name, lower, upper in (
        ("posture_height_deadband_m", 0.0, 0.010),
        ("posture_height_normalizer_m", 0.001, 0.050),
        ("posture_angle_deadband_rad", 0.0, math.radians(5.0)),
        ("posture_angle_normalizer_rad", math.radians(0.5), math.radians(10.0)),
        ("penalty_normalized_body_height", 0.0, 20.0),
        ("penalty_normalized_body_roll", 0.0, 20.0),
        ("penalty_normalized_body_pitch", 0.0, 20.0),
    ):
        if name in effective and not lower <= effective[name] <= upper:
            raise ValueError(f"{name} must be within [{lower}, {upper}]")
    if "dry_robot_mass_scale" in effective and not 0.25 <= effective["dry_robot_mass_scale"] <= 2.0:
        raise ValueError("dry_robot_mass_scale must be within [0.25, 2]; document the measured total and preserved payload")
    if "gait_duty_factor" in effective and not 0. < effective["gait_duty_factor"] < 1.:
        raise ValueError("gait_duty_factor must be within (0, 1)")
    if "sole_clearance_transition_fraction" in effective and not 0. < effective["sole_clearance_transition_fraction"] <= .5:
        raise ValueError("sole_clearance_transition_fraction must be within (0, .5]")
    for name in ("residual_action_scale", "forward_speed_low_band_fraction", "gait_stride_scale_min",
                 "physical_randomization_nominal_fraction", "schedule_matched_progress_gate_floor",
                 "four_leg_progress_gate_floor", "swing_clearance_worst_leg_fraction"):
        if name in effective and not 0. <= effective[name] <= 1.:
            raise ValueError(f"{name} must be within [0, 1]")
    if effective.get("normalized_body_height_cost_enabled", False) and effective.get("normalized_posture_cost_enabled", False):
        raise ValueError("Height-only and combined normalized posture costs are mutually exclusive")
    if effective.get("native_effort_reward_enabled", False) and not effective.get("native_effort_telemetry_enabled", False):
        raise ValueError("Native-effort reward requires explicitly enabled paired telemetry")
    context_mode = _context_request_mode(effective)
    rear_pitch_requested = _rear_pitch_request(effective)
    decoupling_enabled = _sustained_decoupling_request(effective)
    raw_smoothness_changed = effective.get("penalty_action_acceleration", 3.0) != 3.0
    feedback_gain = _rear_feedback_request(effective)
    swing_tilt_changed = _swing_tilt_request(effective) != 0.
    front_gain = _front_stance_feedback_request(effective)
    normalized_slip = _normalized_slip_request(effective)
    provenance = {
        "schema": "drobot.v44-env-ablation.v1", "task": task,
        "override_path": path.as_posix(), "override_sha256": file_sha256(path),
        "supplied_values": values, "effective_values": effective, "reference_files": references,
        "qualification": "isolated_ablation_not_production_or_hardware_qualification",
        "swing_residual_simulation_only": any(value != 1.0 for value in effective.get("gait_swing_residual_scale_by_leg", [1.0] * 4)),
    }
    if normalized_slip:
        provenance["normalized_contact_slip_ablation"] = {
            "weight": effective["penalty_normalized_contact_slip"], "normalizer_m_s": .015,
            "cost_cap": 25., "scope": "Additional dimensionless mean contact-patch speed-squared cost during active run only; original raw slip term retained",
            "controller_physics_limits_and_existing_reward_terms_unchanged": True,
            "production_export_allowed": False,
        }
    if "gait_front_stance_tracking_feedback_gain" in effective:
        provenance["front_stance_feedback_ablation"] = {
            "gain": front_gain, "maximum_correction_deg": 2.,
            "scope": "Front sagittal stance only, equal/opposite requested pair correction, C2 phase/start/command/0.2s-stop gates; before rear feedback/F5",
            "existing_residual_soft_joint_rate_packet_filter_limits_retained": True,
            "actual_or_delivered_sole_pitch_guaranteed": False,
            "production_export_allowed": False,
        }
    if "gait_rear_tracking_feedback_gain" in effective:
        provenance["rear_tracking_feedback_ablation"] = {
            "gain": feedback_gain, "maximum_added_correction_deg": 2.0,
            "leg_mask_RL_RR": list(effective.get("gait_rear_tracking_feedback_leg_mask", (1., 1.))),
            "correction_mode": effective.get("gait_rear_tracking_feedback_mode", "independent"),
            "signal": "Previous delivered target minus current pre-physics measured q",
            "scope": "Rear sagittal joints only, weighted by existing common lift/5mm; before F5 projection and unchanged downstream bounds/rate/filter",
            "front_abduction_stance_and_stop_requested_correction_zero": True,
            "rewards_and_physical_limits_unchanged": not swing_tilt_changed,
            "production_export_allowed": False,
        }
    if "penalty_swing_sole_tilt" in effective:
        provenance["swing_sole_tilt_ablation"] = {
            "weight": effective["penalty_swing_sole_tilt"],
            "signal": "Actual world sole normal: (1-cos tilt)/(1-cos12deg), clipped to25, existing C2 swing envelope and startup weighting",
            "scope": "Mean over scheduled swing feet during run, including premature physical touchdown; zero in stance/ramp/hold",
            "physical_limits_and_existing_reward_terms_unchanged": True,
            "production_export_allowed": False,
        }
    if rear_pitch_requested:
        provenance["rear_descent_timing_ablation"] = {
            "schema": "drobot.v44-rear-descent-timing.v1",
            "descent_fraction": effective.get("gait_rear_pitch_descent_fraction", 0.5),
            "rise_fraction": 0.5,
            "peak_lift_m": effective["gait_rear_pitch_lift_m"],
            "same_fraction_in_projection_and_lift": True,
            "rewards_limits_and_numeric_gates_unchanged": not (decoupling_enabled or raw_smoothness_changed or swing_tilt_changed),
            "production_export_allowed": False,
        }
        provenance["rear_pitch_lift_ablation"] = {
            "schema": "drobot.v44-rear-fixed-pitch-lift.v1",
            "amplitude_m": effective["gait_rear_pitch_lift_m"],
            "preserves": "Original requested distal orientation and abduction; scheduled-stance/startup/park overlay zero",
            "mapping": "Same-branch analytic hip/knee lift before existing joint bounds, delay, packet/rate and actuator response",
            "construction_max_abs_X_excursion_m": 0.0175,
            "construction_max_level_root_tilt_deg": 12.0,
            "fault_policy": "Reject before physics, preserve partial evidence; never substitute fallback targets",
            "actor_observation_count": 50, "critic_observation_count": 58,
            "legacy_rewards_limits_and_numeric_gates_unchanged": not (decoupling_enabled or raw_smoothness_changed or swing_tilt_changed),
            "production_export_allowed": False,
        }
        if effective.get("gait_rear_pitch_projection_enabled", False):
            provenance["rear_pitch_projection_ablation"] = {
                "schema": "drobot.v44-rear-pitch-common-F5-projection.v1",
                "enabled": True, "common_maximum_lift_m": 0.005,
                "common_envelope": "Actual5mm phase/startup/command/contact envelope for BOTH zero and lift arms",
                "composition": "Pi_F(reference+residual), followed by selected0/5mm pitch-preserving O",
                "feasible_float32_proposal": "Exact original target values, no reconstruction round-trip",
                "raw_actor_actions_history_rewards_unchanged": not (decoupling_enabled or raw_smoothness_changed or swing_tilt_changed),
                "fault_policy": "Empty/numerically-invalid domain rejects before physics; no fallback",
                "production_export_allowed": False,
            }
    if raw_smoothness_changed:
        provenance["raw_action_smoothness_ablation"] = {
            "changed_term": "action_acceleration",
            "coefficient": effective["penalty_action_acceleration"],
            "baseline_coefficient": 3.0,
            "signal": "Squared dimensionless sampled-action second difference; not mechanical acceleration",
            "physical_acceleration_terms_and_limits_unchanged_by_this_override": True,
            "production_export_allowed": False,
        }
    if "decouple_sustained_progress_from_pair_release" in effective:
        provenance["sustained_progress_credit_ablation"] = {
            "schema": "drobot.v44-sustained-progress-pair-decoupling.v1",
            "enabled": decoupling_enabled,
            "changed_reward_terms_only": ["sustained_progress"] if decoupling_enabled else [],
            "signed_component_retained": True,
            "actor_history_geometry_limits_and_numeric_gates_unchanged": True,
            "production_export_allowed": False,
        }
    if effective.get("normalized_body_height_cost_enabled", False):
        provenance["body_height_reward_ablation"] = {
            "schema": "drobot.v44-motion-only-normalized-height.v1",
            "simulation_only_not_production_qualification": True,
            "blend": "quintic(clamp((command_speed-deadband)/(run_target_speed-deadband),0,1))",
            "inactive_schedule_fallback": "command translation speed above standstill deadband",
            "settled_or_parked_height": "exact legacy metre-squared height term",
            "tilt": "exact legacy projected-gravity tilt; combined normalized posture rejected",
            "normalization_parameters": "existing posture_height_deadband_m, posture_height_normalizer_m and penalty_normalized_body_height from effective saved config",
        }
    if "gait_phase_reference_linear_interpolation_enabled" in effective:
        interpolation_enabled = effective["gait_phase_reference_linear_interpolation_enabled"]
        provenance["phase_reference_sampling_ablation"] = {
            "schema": "drobot.v44-phase-joint-interpolation.v1",
            "mode": "linear_adjacent" if interpolation_enabled else "floor",
            "nondefault_mode_simulation_only": interpolation_enabled,
            "position_sampling": "periodic adjacent-row lerp in each low/high table, then low-to-high speed blend" if interpolation_enabled else "unchanged floor phase lookup then low-to-high speed blend",
            "contact_sampling": "unchanged floor table; no boolean interpolation",
            "legacy_phase_alias_unchanged": True,
            "production_export_allowed": False,
        }
    if "native_effort_telemetry_enabled" in effective or "native_effort_reward_enabled" in effective:
        provenance["effort_observable_ablation"] = {
            "schema": "drobot.v44-active-effort-observable.v1",
            "native_telemetry_enabled": effective.get("native_effort_telemetry_enabled", False),
            "reward_observable": "native_active_generalized_effort" if effective.get("native_effort_reward_enabled", False) else "retained_clipped_approximate_pd",
            "sampling": "One fresh projected-force getter after final physics substep at existing reward boundary; no extra physics or actor call",
            "native_semantics": "Active generalized joint effort, not isolated drive torque, motor current, electrical power or heating",
            "changed_reward_terms_only": ["effort_rms", "effort_soft_limit", "actuator_torque_speed_envelope", "leg_operating_point_imbalance"] if effective.get("native_effort_reward_enabled", False) else [],
            "legacy_pd_metrics_and_numeric_gates_unchanged": True,
            "production_export_allowed": False,
        }
    if "critic_context_mode" in effective:
        provenance["critic_context_ablation"] = {
            "schema": "drobot.critic-causal-context.v1",
            "mode": context_mode,
            "actor_observation_count": 50,
            "critic_observation_count": 58 if context_mode == "legacy58" else 90,
            "appended_context_count": 0 if context_mode == "legacy58" else 32,
            "zero_control_builds_same_unmasked_context": True,
            "history_timing": "Completed reward history after existing updates; current pre-action command state after existing observation command updates; selected reset slots invalidated",
            "complete_markov_state_claimed": False,
            "reward_controller_and_legacy_phase_semantics_unchanged": True,
            "derived_state_space_and_task_marker": True,
            "production_export_allowed": False,
        }
    return effective, provenance


def apply_v44_overrides(env_cfg, values: dict) -> None:
    """Assign known fields only; reject an incomplete or unsupported config."""
    _context_request_mode(values)
    rear_pitch_requested = _rear_pitch_request(values)
    _sustained_decoupling_request(values)
    for name, value in values.items():
        if name not in ALLOWED_FIELDS or not hasattr(env_cfg, name):
            raise ValueError(f"V44 config does not expose allowed attribute {name!r}")
        existing = getattr(env_cfg, name)
        setattr(env_cfg, name, tuple(value) if isinstance(existing, tuple) and isinstance(value, list) else value)
    if "decouple_sustained_progress_from_pair_release" in values:
        env_cfg.sustained_progress_decoupling_ablation_task = TASK if values["decouple_sustained_progress_from_pair_release"] else None
    validate_sustained_progress_config(env_cfg)
    feedback_gain = _rear_feedback_request(values)
    swing_tilt_weight = _swing_tilt_request(values)
    front_gain = _front_stance_feedback_request(values)
    if _normalized_slip_request(values):
        env_cfg.normalized_contact_slip_ablation_task = TASK
    if "gait_front_stance_tracking_feedback_gain" in values:
        env_cfg.gait_front_stance_tracking_feedback_ablation_task = TASK if front_gain else None
    if "penalty_swing_sole_tilt" in values:
        env_cfg.swing_sole_tilt_ablation_task = TASK if swing_tilt_weight else None
    if "gait_rear_tracking_feedback_gain" in values:
        env_cfg.gait_rear_tracking_feedback_ablation_task = TASK if feedback_gain else None
    if rear_pitch_requested:
        # Both explicit arms are unqualified; only omitted0/None is production.
        env_cfg.gait_rear_pitch_lift_ablation_task = TASK
    if "gait_rear_pitch_projection_enabled" in values:
        env_cfg.gait_rear_pitch_projection_ablation_task = TASK if values["gait_rear_pitch_projection_enabled"] else None
    context_mode = env_cfg.critic_context_mode
    if not isinstance(context_mode, str) or context_mode not in ENUM_FIELDS["critic_context_mode"]:
        raise ValueError("Invalid effective critic context mode")
    context_enabled = context_mode != "legacy58"
    if "critic_context_mode" in values:
        env_cfg.critic_context_ablation_task = TASK if context_enabled else None
        env_cfg.state_space = 90 if context_enabled else 58
    if context_enabled:
        if (type(env_cfg).__name__ != "DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg"
            or env_cfg.critic_context_ablation_task != TASK
            or env_cfg.state_space != 90 or env_cfg.observation_space != 50 or env_cfg.action_space != 12
            or env_cfg.action_mode != "gait_residual" or env_cfg.gait_reference_mode != "phase_table"
            or env_cfg.action_history_mode != "applied_target" or env_cfg.command_profile != "run_stop"
            or env_cfg.native_effort_telemetry_enabled is not True or env_cfg.native_effort_reward_enabled is not False):
            raise ValueError("Critic90 requires strictly marked V44 actor50, PD reward and paired telemetry")
    elif env_cfg.critic_context_ablation_task is not None:
        raise ValueError("Legacy critic requires a None context marker")
    telemetry = env_cfg.native_effort_telemetry_enabled
    native_reward = env_cfg.native_effort_reward_enabled
    if not isinstance(telemetry, bool) or not isinstance(native_reward, bool):
        raise ValueError("Effective native-effort flags must be boolean")
    if native_reward and not telemetry:
        raise ValueError("Native-effort reward requires paired telemetry")
    if "native_effort_telemetry_enabled" in values or "native_effort_reward_enabled" in values:
        env_cfg.native_effort_ablation_task = TASK if telemetry else None
    if telemetry:
        if (type(env_cfg).__name__ != "DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg"
            or env_cfg.native_effort_ablation_task != TASK
            or env_cfg.action_mode != "gait_residual" or env_cfg.gait_reference_mode != "phase_table"
            or env_cfg.command_profile != "run_stop"
            or env_cfg.observation_space != 50 or env_cfg.action_space != 12
            or env_cfg.state_space != (90 if context_enabled else 58)):
            raise ValueError("Native-effort telemetry requires the exact isolated V44 context")
    elif env_cfg.native_effort_ablation_task is not None:
        raise ValueError("Disabled native-effort telemetry requires a None marker")
    interpolation_enabled = env_cfg.gait_phase_reference_linear_interpolation_enabled
    if not isinstance(interpolation_enabled, bool):
        raise ValueError("Effective phase interpolation flag must be boolean")
    if "gait_phase_reference_linear_interpolation_enabled" in values:
        env_cfg.gait_phase_reference_interpolation_ablation_task = TASK if interpolation_enabled else None
    if interpolation_enabled:
        if (type(env_cfg).__name__ != "DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg"
            or env_cfg.gait_phase_reference_interpolation_ablation_task != TASK
            or env_cfg.action_mode != "gait_residual" or env_cfg.gait_reference_mode != "phase_table"
            or env_cfg.command_profile != "run_stop"):
            raise ValueError("Linear phase interpolation requires the exact isolated V44 phase_table/gait_residual/run_stop config")
        if (env_cfg.gait_phase_table_lateral_reference_enabled is not False
            or env_cfg.gait_phase_table_forward_reference_enabled is not False):
            raise ValueError("Linear joint interpolation cannot use floor-sampled body-transfer metadata")
        scale = env_cfg.gait_phase_reference_scale_about_neutral
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale != 1.0:
            raise ValueError("Linear phase interpolation requires scale-about-neutral 1.0")
        if not env_cfg.gait_phase_reference_path or not env_cfg.gait_phase_reference_low_path:
            raise ValueError("Linear phase interpolation requires both high and low references")
    elif env_cfg.gait_phase_reference_interpolation_ablation_task is not None:
        raise ValueError("Disabled phase interpolation requires a None ablation marker")
    if getattr(env_cfg, "normalized_body_height_cost_enabled", False):
        if getattr(env_cfg, "normalized_posture_cost_enabled", False):
            raise ValueError("Effective height-only and combined normalized posture flags cannot both be enabled")
        if (type(env_cfg).__name__ != "DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg"
            or env_cfg.action_mode != "gait_residual" or env_cfg.gait_reference_mode != "phase_table"
            or env_cfg.command_profile != "run_stop"):
            raise ValueError("Height-only normalization requires the exact V44 phase_table/gait_residual/run_stop config")
    if "gait_swing_residual_scale_by_leg" in values:
        env_cfg.gait_swing_residual_ablation_task = (
            TASK if any(value != 1.0 for value in values["gait_swing_residual_scale_by_leg"]) else None
        )
    for lower, upper in (
        ("gait_frequency_min_hz", "gait_frequency_max_hz"),
        ("gait_speed_min_m_s", "gait_speed_max_m_s"),
        ("initial_forward_speed_min_m_s", "initial_forward_speed_max_m_s"),
        ("forward_speed_min_m_s", "forward_speed_max_m_s"),
        ("gait_phase_reference_blend_speed_min_m_s", "gait_phase_reference_blend_speed_max_m_s"),
        ("initial_training_horizon_s", "final_training_horizon_s"),
        ("run_stop_run_duration_min_s", "run_stop_run_duration_max_s"),
    ):
        low, high = getattr(env_cfg, lower), getattr(env_cfg, upper)
        if low is not None and high is not None and not 0. <= low <= high:
            raise ValueError(f"Invalid effective V44 bounds: {lower}, {upper}")


def v44_hydra_overrides(values: dict) -> list[str]:
    """Format loader-validated values without accepting Hydra expressions."""
    if set(values) - ALLOWED_FIELDS:
        raise ValueError("Unsupported V44 Hydra override key")
    context_mode = _context_request_mode(values)
    rear_pitch_requested = _rear_pitch_request(values)
    _sustained_decoupling_request(values)

    def plain(value):
        if isinstance(value, list):
            for item in value:
                plain(item)
        elif isinstance(value, str):
            if "${" in value or any(ord(c) < 32 for c in value):
                raise ValueError("Hydra expressions and control characters are not permitted")
        elif value is not None and not isinstance(value, (bool, int, float)):
            raise ValueError("Only JSON primitives and lists are permitted")

    formatted = []
    if _normalized_slip_request(values):
        formatted.append(f"env.normalized_contact_slip_ablation_task={json.dumps(TASK)}")
    feedback_gain = _rear_feedback_request(values)
    swing_tilt_weight = _swing_tilt_request(values)
    front_gain = _front_stance_feedback_request(values)
    if "gait_front_stance_tracking_feedback_gain" in values:
        formatted.append(f"env.gait_front_stance_tracking_feedback_ablation_task={json.dumps(TASK if front_gain else None)}")
    for name, value in values.items():
        plain(value)
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        if "${" in encoded:
            raise ValueError("Hydra interpolation is not permitted in V44 ablations")
        formatted.append(f"env.{name}={encoded}")
    if "penalty_swing_sole_tilt" in values:
        formatted.append(f"env.swing_sole_tilt_ablation_task={json.dumps(TASK if swing_tilt_weight else None)}")
    if "decouple_sustained_progress_from_pair_release" in values:
        marker = TASK if values["decouple_sustained_progress_from_pair_release"] else None
        formatted.append(f"env.sustained_progress_decoupling_ablation_task={json.dumps(marker)}")
    if "gait_rear_tracking_feedback_gain" in values:
        marker = TASK if feedback_gain else None
        formatted.append(f"env.gait_rear_tracking_feedback_ablation_task={json.dumps(marker)}")
    if rear_pitch_requested:
        formatted.append(f"env.gait_rear_pitch_lift_ablation_task={json.dumps(TASK)}")
    if "gait_rear_pitch_projection_enabled" in values:
        projection_marker = TASK if values["gait_rear_pitch_projection_enabled"] else None
        formatted.append(f"env.gait_rear_pitch_projection_ablation_task={json.dumps(projection_marker)}")
    if "gait_swing_residual_scale_by_leg" in values:
        marker = TASK if any(value != 1.0 for value in values["gait_swing_residual_scale_by_leg"]) else None
        formatted.append(f"env.gait_swing_residual_ablation_task={json.dumps(marker)}")
    if "gait_phase_reference_linear_interpolation_enabled" in values:
        interpolation_enabled = values["gait_phase_reference_linear_interpolation_enabled"]
        if not isinstance(interpolation_enabled, bool):
            raise ValueError("Phase interpolation Hydra value must be boolean")
        marker = TASK if interpolation_enabled else None
        formatted.append(f"env.gait_phase_reference_interpolation_ablation_task={json.dumps(marker)}")
    if "native_effort_telemetry_enabled" in values or "native_effort_reward_enabled" in values:
        telemetry = values.get("native_effort_telemetry_enabled", False)
        native_reward = values.get("native_effort_reward_enabled", False)
        if not isinstance(telemetry, bool) or not isinstance(native_reward, bool) or (native_reward and not telemetry):
            raise ValueError("Native-effort Hydra flags require boolean paired telemetry")
        formatted.append(f"env.native_effort_ablation_task={json.dumps(TASK if telemetry else None)}")
    if "critic_context_mode" in values:
        context_enabled = context_mode != "legacy58"
        formatted.append(f"env.critic_context_ablation_task={json.dumps(TASK if context_enabled else None)}")
        formatted.append(f"env.state_space={90 if context_enabled else 58}")
    return formatted
