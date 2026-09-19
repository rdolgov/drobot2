"""Analyze a recorded V44 diagnostic JSONL without importing Isaac or a policy.

Writes a NEW JSON report only. Missing gait settings require explicit CLI
fallbacks; they are never guessed from current task defaults. This does not
identify motor latency, measure motor torque, or qualify a gait for hardware.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

TRACE_SCHEMA = "drobot.v44-step-diagnostic-trace.v1"
SETTINGS = (
    "gait_start_ramp_s", "sole_clearance_transition_fraction",
    "gait_standstill_deadband_m_s", "sole_clearance_target_m",
)


def finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def vector(value, length: int, name: str, boolean: bool = False) -> list:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if boolean:
        if any(not isinstance(item, bool) for item in value):
            raise ValueError(f"{name} must contain booleans")
        return value
    return [finite(item, name) for item in value]


def distribution(values) -> dict:
    values = list(values)
    keys = ("mean", "rms", "min", "p05", "p50", "p95", "max", "peak_to_peak", "mean_absolute", "p95_absolute")
    if not values:
        return {"samples": 0, **dict.fromkeys(keys)}
    ordered = sorted(values)

    def quantile(sequence, fraction):
        position = fraction * (len(sequence) - 1)
        low = int(position)
        high = min(low + 1, len(sequence) - 1)
        return sequence[low] + (position - low) * (sequence[high] - sequence[low])

    return {
        "samples": len(values), "mean": statistics.fmean(values),
        "rms": math.sqrt(statistics.fmean(value * value for value in values)),
        "min": ordered[0], "p05": quantile(ordered, .05), "p50": quantile(ordered, .50),
        "p95": quantile(ordered, .95), "max": ordered[-1],
        "peak_to_peak": ordered[-1] - ordered[0],
        "mean_absolute": statistics.fmean(abs(value) for value in values),
        "p95_absolute": quantile(sorted(abs(value) for value in values), .95),
    }


def fraction(values) -> float | None:
    values = list(values)
    return statistics.fmean(float(value) for value in values) if values else None


def quaternion_roll_pitch_deg(quaternion: list[float]) -> tuple[float, float]:
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm < 1e-12:
        raise ValueError("Root quaternion has zero norm")
    x, y, z, w = (value / norm for value in quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return math.degrees(roll), math.degrees(pitch)


def load_trace(path: Path) -> tuple[dict, list[dict]]:
    metadata, rows = None, []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Trace line {line_number} must be an object")
            if record.get("record_type") == "metadata" and metadata is None and not rows:
                metadata = record
            elif record.get("record_type") == "step" and metadata is not None:
                rows.append(record)
            else:
                raise ValueError(f"Unexpected record/order at trace line {line_number}")
    if metadata is None or metadata.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"Trace must start with {TRACE_SCHEMA} metadata")
    if metadata.get("sample_count") != len(rows):
        raise ValueError("Trace sample count differs from metadata; file may be incomplete")
    return metadata, rows


def effective_settings(metadata: dict, args) -> tuple[dict, dict]:
    direct = metadata.get("gait_diagnostic_config", {})
    ablation = (metadata.get("environment_ablation") or {}).get("effective_values", {})
    values, provenance = {}, {}
    for name in SETTINGS:
        recorded = direct.get(name, ablation.get(name))
        fallback = getattr(args, name)
        if recorded is None and fallback is None:
            raise ValueError(f"Trace lacks {name}; supply --{name.replace('_', '-')} explicitly")
        if recorded is not None:
            value = finite(recorded, name)
            if fallback is not None and not math.isclose(value, fallback, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"CLI {name} conflicts with the recorded value")
            source = "trace.gait_diagnostic_config" if name in direct else "trace.environment_ablation.effective_values"
        else:
            value, source = finite(fallback, name), "explicit_cli_fallback"
        values[name] = value
        provenance[name] = {"value": value, "source": source}
    if values["gait_start_ramp_s"] <= 0.0 or values["sole_clearance_target_m"] <= 0.0:
        raise ValueError("Ramp time and clearance target must be positive")
    if not 0.0 < values["sole_clearance_transition_fraction"] <= .5:
        raise ValueError("Clearance transition fraction must be in (0, .5]")
    if values["gait_standstill_deadband_m_s"] < 0.0:
        raise ValueError("Standstill deadband cannot be negative")
    return values, provenance


def prepare_rows(metadata: dict, rows: list[dict], settings: dict) -> None:
    joints, legs = metadata["joint_names"], metadata["foot_names"]
    for label, names in (("joint_names", joints), ("foot_names", legs)):
        if not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError(f"Invalid/duplicate {label}")
    offsets = vector(metadata["gait_phase_offsets_cycles"], len(legs), "phase offsets")
    if any(not 0.0 <= offset < 1.0 for offset in offsets):
        raise ValueError("Phase offsets must be in [0, 1)")
    duty = finite(metadata["gait_duty_factor"], "gait_duty_factor")
    dt = finite(metadata["step_dt_s"], "step_dt_s")
    if not 0.0 < duty < 1.0 or dt <= 0.0:
        raise ValueError("Invalid duty factor or step interval")
    previous_step = {}
    for row in rows:
        episode, step = row["episode_index"], row["step_index"]
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0 or isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("Episode and step indices must be nonnegative integers")
        if step <= previous_step.get(episode, -1):
            raise ValueError("Duplicate/out-of-order step in an episode")
        previous_step[episode] = step
        if row["stage"] not in ("run", "ramp", "hold"):
            raise ValueError("Unknown trace stage")
        if not isinstance(row["included_in_steady_run_metrics"], bool) or not isinstance(row["packet_held"], bool):
            raise ValueError("Steady-run and held-packet flags must be boolean")
        for name in ("episode_sim_time_s", "interval_start_sim_time_s", "packet_elapsed_budget_s", "applied_gait_phase_cycles"):
            finite(row[name], name)
        for name in ("requested_target_rad", "delivered_target_rad", "joint_position_rad", "joint_velocity_rad_s", "applied_torque_nm", "effort_limit_nm"):
            row[name] = vector(row[name], len(joints), name)
        if any(limit <= 0.0 for limit in row["effort_limit_nm"]):
            raise ValueError("Effort limits must be positive")
        status = row["response_filtered_target_status"]
        if status == "available":
            row["_filter"] = vector(row["response_filtered_target_rad"], len(joints), "response filtered target")
        elif status == "filter_disabled":
            row["_filter"] = row["delivered_target_rad"]
        elif status == "filter_target_attribute_unavailable":
            row["_filter"] = None
        else:
            raise ValueError(f"Unknown filter target status {status!r}")
        for name in ("root_position_world_m", "root_linear_velocity_world_m_s", "root_angular_velocity_world_rad_s", "command_next_observation_vx_vy_yawrate", "command_applied_vx_vy_yawrate"):
            row[name] = vector(row[name], 3, name)
        quaternion = vector(row["root_quaternion_world_xyzw"], 4, "root quaternion")
        row["_roll_deg"], row["_pitch_deg"] = quaternion_roll_pitch_deg(quaternion)
        row["foot_force_norm_n"] = vector(row["foot_force_norm_n"], len(legs), "foot forces")
        for name in ("foot_contact", "scheduled_contact", "scheduled_release"):
            row[name] = vector(row[name], len(legs), name, boolean=True)
        if row["scheduled_release"] != [not scheduled and not contact for scheduled, contact in zip(row["scheduled_contact"], row["foot_contact"])]:
            raise ValueError("Recorded scheduled release conflicts with contact flags")
        feet = row["feet"]
        if feet is not None:
            if feet["names"] != legs:
                raise ValueError("Foot geometry ordering differs from metadata")
            for name in ("minimum_corner_height_above_terrain_origin_m", "sole_tilt_from_level_rad"):
                feet[name] = vector(feet[name], len(legs), name)
        row["_core_eligible"] = (
            row["stage"] == "run" and row["included_in_steady_run_metrics"]
            and row["interval_start_sim_time_s"] >= settings["gait_start_ramp_s"]
            and row["command_next_observation_vx_vy_yawrate"][0] > settings["gait_standstill_deadband_m_s"]
        )
        row["_envelope"] = []
        row["_leg_phase"] = []
        transition = settings["sole_clearance_transition_fraction"]
        for offset in offsets:
            phase = (row["applied_gait_phase_cycles"] + offset) % 1.0
            u = min(1.0, max(0.0, (phase - duty) / (1.0 - duty)))
            enter, leave = min(1.0, u / transition), min(1.0, (1.0 - u) / transition)
            envelope = enter**3 * (10.0 - 15.0 * enter + 6.0 * enter**2) * leave**3 * (10.0 - 15.0 * leave + 6.0 * leave**2)
            row["_leg_phase"].append(phase)
            row["_envelope"].append(envelope)


def joint_report(rows: list[dict], index: int) -> dict:
    def angles(name):
        return distribution(math.degrees(row[name][index]) for row in rows)

    def error(first, second):
        return distribution(
            math.degrees(row[first][index] - row[second][index])
            for row in rows if row[first] is not None and row[second] is not None
        )

    return {
        "requested_target_deg": angles("requested_target_rad"),
        "delivered_target_deg": angles("delivered_target_rad"),
        "filtered_target_deg": distribution(math.degrees(row["_filter"][index]) for row in rows if row["_filter"] is not None),
        "actual_position_deg": angles("joint_position_rad"),
        "actual_velocity_deg_s": angles("joint_velocity_rad_s"),
        "requested_minus_delivered_deg": error("requested_target_rad", "delivered_target_rad"),
        "delivered_minus_filtered_deg": error("delivered_target_rad", "_filter"),
        "filtered_minus_actual_deg": error("_filter", "joint_position_rad"),
        "delivered_minus_actual_deg": error("delivered_target_rad", "joint_position_rad"),
        "estimated_torque_nm": distribution(row["applied_torque_nm"][index] for row in rows),
        "estimated_abs_effort_fraction": distribution(abs(row["applied_torque_nm"][index]) / row["effort_limit_nm"][index] for row in rows),
        "estimated_hard_cap_gt_0p98_fraction": fraction(abs(row["applied_torque_nm"][index]) / row["effort_limit_nm"][index] > .98 for row in rows),
    }


def root_report(rows: list[dict]) -> dict:
    return {
        "world_z_m": distribution(row["root_position_world_m"][2] for row in rows),
        "roll_deg": distribution(row["_roll_deg"] for row in rows),
        "pitch_deg": distribution(row["_pitch_deg"] for row in rows),
        "world_vertical_speed_m_s": distribution(row["root_linear_velocity_world_m_s"][2] for row in rows),
    }


def leg_report(rows: list[dict], index: int, joint_indices: dict, settings: dict) -> dict:
    geometry = [row for row in rows if row["feet"] is not None]
    stable_swing = [row for row in geometry if row["_core_eligible"] and not row["scheduled_contact"][index]]
    return {
        "samples": len(rows), "root": root_report(rows),
        "physical_contact_fraction": fraction(row["foot_contact"][index] for row in rows),
        "no_force_contact_fraction": fraction(not row["foot_contact"][index] for row in rows),
        "scheduled_contact_fraction": fraction(row["scheduled_contact"][index] for row in rows),
        "scheduled_release_fraction": fraction(row["scheduled_release"][index] for row in rows),
        "foot_force_norm_n": distribution(row["foot_force_norm_n"][index] for row in rows),
        "corner_height_above_terrain_origin_m": distribution(row["feet"]["minimum_corner_height_above_terrain_origin_m"][index] for row in geometry),
        "sole_tilt_deg": distribution(math.degrees(row["feet"]["sole_tilt_from_level_rad"][index]) for row in geometry),
        "lift_envelope_fraction": distribution(row["_envelope"][index] for row in rows),
        "post_startup_swing_clearance_deficit_m": distribution(max(0.0, settings["sole_clearance_target_m"] * row["_envelope"][index] - row["feet"]["minimum_corner_height_above_terrain_origin_m"][index]) for row in stable_swing),
        "by_joint": {name: joint_report(rows, joint_index) for name, joint_index in joint_indices.items()},
    }


def analyze(metadata: dict, rows: list[dict], settings: dict) -> dict:
    joints, legs = metadata["joint_names"], metadata["foot_names"]
    ownership = {leg: {name: index for index, name in enumerate(joints) if name.startswith(leg + "_")} for leg in legs}
    if sum(len(indices) for indices in ownership.values()) != len(joints) or any(not indices for indices in ownership.values()):
        raise ValueError("Every joint must map uniquely to a recorded leg-name prefix")
    groups = {stage: [row for row in rows if row["stage"] == stage] for stage in ("run", "ramp", "hold")}
    groups["steady_run"] = [row for row in rows if row["stage"] == "run" and row["included_in_steady_run_metrics"]]
    # Select each episode's tail independently. A fall-truncated hold is not
    # comparable to a completed stationary tail and must not enter this group.
    completed_episodes = {
        outcome["episode"] for outcome in metadata.get("episode_outcomes", [])
        if outcome.get("fell") is False
    }
    tail_intervals = []
    hold_tail = []
    for episode in sorted(completed_episodes):
        hold_rows = [row for row in groups["hold"] if row["episode_index"] == episode]
        if not hold_rows:
            continue
        end = max(row["episode_sim_time_s"] for row in hold_rows)
        start = end - .5
        selected = [row for row in hold_rows if row["interval_start_sim_time_s"] >= start - 1e-9]
        hold_tail.extend(selected)
        groups[f"hold_final_0p5s_episode_{episode}"] = selected
        tail_intervals.append({
            "episode_index": episode,
            "requested_interval_start_s": start, "interval_end_s": end,
            "sample_count": len(selected),
            "sampled_simulated_duration_s": len(selected) * metadata["step_dt_s"],
            "first_included_interval_start_s": min((row["interval_start_sim_time_s"] for row in selected), default=None),
            "completed_without_recorded_fall": True,
        })
    groups["hold_final_0p5s"] = hold_tail
    result = {}
    for group_name, selected in groups.items():
        by_leg = {}
        for index, leg in enumerate(legs):
            subsets = {
                "all": selected,
                "scheduled_support": [row for row in selected if row["scheduled_contact"][index]],
                "scheduled_swing": [row for row in selected if not row["scheduled_contact"][index]],
                "support_union_actual_or_scheduled": [row for row in selected if row["scheduled_contact"][index] or row["foot_contact"][index]],
                "core_swing": [row for row in selected if not row["scheduled_contact"][index] and row["_core_eligible"] and row["_envelope"][index] >= .90],
            }
            by_leg[leg] = {scope: leg_report(subset, index, ownership[leg], settings) for scope, subset in subsets.items()}
        result[group_name] = {
            "samples": len(selected), "episode_count": len({row["episode_index"] for row in selected}),
            "sampled_simulated_duration_s": len(selected) * metadata["step_dt_s"],
            "packet_held_fraction": fraction(row["packet_held"] for row in selected),
            "packet_elapsed_budget_s": distribution(row["packet_elapsed_budget_s"] for row in selected),
            "root": root_report(selected),
            "by_joint": {name: joint_report(selected, index) for index, name in enumerate(joints)},
            "by_leg_and_scope": by_leg,
        }
        if group_name == "hold_final_0p5s":
            result[group_name]["per_episode_selected_intervals"] = tail_intervals
        elif group_name.startswith("hold_final_0p5s_episode_"):
            episode = int(group_name.removeprefix("hold_final_0p5s_episode_"))
            result[group_name]["per_episode_selected_intervals"] = [interval for interval in tail_intervals if interval["episode_index"] == episode]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Recorded _DiagnosticTrace JSONL")
    parser.add_argument("--output", type=Path, required=True, help="New analysis JSON; never overwrite")
    for name in SETTINGS:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None, help="Explicit fallback only when missing from trace metadata")
    args = parser.parse_args()
    trace, output = args.trace.resolve(strict=True), args.output.resolve()
    if output == trace or output.exists():
        parser.error("--output must be a new file distinct from the input trace")
    try:
        metadata, rows = load_trace(trace)
        settings, provenance = effective_settings(metadata, args)
        prepare_rows(metadata, rows, settings)
        groups = analyze(metadata, rows, settings)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        parser.error(f"Cannot analyze trace: {exc}")
    with trace.open("rb") as stream:
        trace_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "schema": "drobot.v44-step-diagnostic-analysis.v1",
        "qualification": "offline_diagnostic_only_not_hardware_qualification",
        "input_trace": {"path": str(trace), "sha256": trace_sha256, "trace_id": metadata.get("trace_id"), "samples": len(rows)},
        "source_metadata": metadata,
        "effective_gait_settings": provenance,
        "definitions": {
            "stages": "run/ramp/hold are recorded labels; steady_run uses the recorded inclusion flag.",
            "hold_final_0p5s": "Last 0.5 simulated seconds of each episode's hold, selected independently from interval start times. Only metadata-confirmed no-fall episodes enter; explicit per-episode sample counts/durations are reported. This selects a tail, not proof of unloaded/static joint calibration.",
            "support": "scheduled_support uses recorded scheduled_contact; support_union also includes force-confirmed contact.",
            "swing": "scheduled_swing uses recorded !scheduled_contact, including transitions; recorded masks are authoritative for stop/park behavior.",
            "core_swing": "Scheduled swing, recorded steady-run inclusion, interval start >= explicit gait_start_ramp_s, next-observation vx > explicit standstill deadband, and quintic envelope >=0.90. No early-startup or stop-stage core success is inferred.",
            "envelope": "Recorded applied phase + recorded leg offset modulo1; u=clamp((phase-duty)/(1-duty),0,1); quintic(clamp(u/transition))*quintic(clamp((1-u)/transition)). Same analytical formula as evaluator; floating-point threshold boundaries may differ at machine precision.",
            "errors": "Signed first-minus-second angular differences at each post-physics trace snapshot, reported in degrees; absolute summaries are also included.",
            "excursion": "Distribution peak_to_peak is max-minus-min over selected samples, not a per-cycle amplitude or gain estimate.",
            "filter_disabled": "Explicit filter_disabled treats delivered target as filter output; unavailable filter telemetry remains missing, never substituted with actual position.",
            "hard_cap": "Fraction of selected snapshots with abs(estimated applied_torque)/recorded effort_limit >0.98. This is not the lower rated-torque soft-limit exceedance metric.",
            "root": "World-frame root Z, not inferred ground-relative height. Normalized xyzw quaternion gives XYZ roll/pitch; positive pitch is nose-down for body +X forward.",
            "empty": "Empty subsets use null metrics, never false zero success. Groups pool snapshots across episodes; inspect episode outcomes in source metadata.",
        },
        "limitations": [
            "Implicit-actuator applied_torque is an estimate, not measured motor torque or resolved PhysX drive torque.",
            "Hard-cap occupancy and tracking error alone cannot identify servo stall, stiffness, damping, contact equilibrium or payload error.",
            "Delivered/filter/state values are interval-end snapshots, not simultaneous command issue times or substep averages.",
            "No cross-correlation lag estimate is made; snapshot errors do not identify communication latency or a unique response time constant.",
            "World corner geometry is recorded actual state, not a counterfactual reconstruction under perfect target tracking.",
            "Root motion, loaded joint deflection and gait timing remain coupled; use controlled matched trials before changing physical parameters.",
            "Contact is the recorded net-force norm threshold, not necessarily vertical load or physical toe/heel contact location.",
            "Terminal auto-reset frames are absent; sampled duration and final samples do not prove a trial completed without falling.",
        ],
        "groups": groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Wrote offline diagnostic analysis: {output} ({len(rows)} recorded steps; no simulator or policy executed)")


if __name__ == "__main__":
    main()
