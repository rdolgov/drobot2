"""Simulation-only consistency replay of completed real host target writes.

Example (run with the configured Isaac Lab Python, not the onboard runtime):
  python replay_recorded_targets_v44.py --trial-dir <extracted-trial> \
    --gain-hypothesis-json <two-gain-scales.json> \
    --trace-jsonl <new-trace.jsonl> --output-json <new-report.json>

The gain JSON must contain exactly actuator_nominal_stiffness_scale and
actuator_nominal_damping_scale, both finite numbers in [0.25, 8]. All other
physical settings are the V44 nominal plant. No model inference, hardware API,
controller limiter, packet hold or synthetic stop processing is executed.
This free-standing ground replay is NOT hardware calibration or qualification.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import sys
from uuid import uuid4


GAIN_FIELDS = {"actuator_nominal_stiffness_scale", "actuator_nominal_damping_scale"}
V43_TASK = "Drobot-Commanded-Run-Foot-Warp-Trot-Residual-Stop-External-Rear-Payload"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def vector(value, size, label):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} numbers")
    return [finite(item, f"{label}[{index}]") for index, item in enumerate(value)]


def serializable(value):
    """Config snapshot / nonfinite failure reporting, not code evaluation."""
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}"
    return str(value)


def load_inputs(trial, gain_path):
    gains = json.loads(gain_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    if not isinstance(gains, dict) or set(gains) != GAIN_FIELDS:
        raise ValueError(f"Gain JSON must contain exactly {sorted(GAIN_FIELDS)}")
    gains = {key: finite(value, key) for key, value in gains.items()}
    if any(not 0.25 <= value <= 8.0 for value in gains.values()):
        raise ValueError("Gain scales must be in [0.25, 8]")
    metadata = json.loads((trial / "metadata.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    samples = [json.loads(line) for line in (trial / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    events = [json.loads(line) for line in (trial / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if metadata.get("schema") != "drobot.rl-trial.v1" or metadata.get("status") != "complete":
        raise ValueError("Replay requires a completed drobot.rl-trial.v1 recording")
    if metadata.get("dropped_samples", 0) or metadata.get("dropped_events", 0):
        raise ValueError("A dropped-data recording cannot define this complete-target replay")
    if not samples:
        raise ValueError("Recording has no samples")
    names, ids = metadata["robot"]["joint_order"], metadata["robot"]["servo_ids"]
    if len(names) != 12 or len(set(names)) != 12 or len(ids) != 12 or len(set(ids)) != 12:
        raise ValueError("Recording needs 12 unique ordered joints and servo IDs")
    sensor_times = [finite(row["sensor_time_s"]["joints"], "joint sensor timestamp") for row in samples]
    if any(b < a for a, b in zip(sensor_times, sensor_times[1:])):
        raise ValueError("Recorded joint timestamps must be nondecreasing")
    first = samples[0]
    initial_q = vector(first["joints"]["position_rad"], 12, "first encoder position")
    recorded_dq = first["joints"].get("velocity_rad_s")
    initial_dq = vector(recorded_dq, 12, "first encoder velocity") if recorded_dq is not None else [0.0] * 12
    held = vector(metadata["trial"]["initial_joint_position_rad"], 12, "initial held target")
    initial_held = held.copy()
    id_index = {str(servo_id): index for index, servo_id in enumerate(ids)}
    writes = []
    for event in events:
        if event.get("type") != "motor_output":
            continue
        timestamp = finite(event["output_completed_s"], "host-write completion")
        if writes and timestamp <= writes[-1]["host_monotonic_s"]:
            raise ValueError("Host-write completion timestamps must strictly increase")
        changes = event["commanded_target_deg_by_servo_id"]
        if not isinstance(changes, dict) or not changes or set(changes) - set(id_index):
            raise ValueError("Motor output has invalid/missing joint IDs")
        for servo_id, degrees in changes.items():
            held[id_index[servo_id]] = math.radians(finite(degrees, "completed target degrees"))
        writes.append({"host_monotonic_s": timestamp, "targets_rad": held.copy(),
                       "output_sequence": event["sequence"],
                       "policy_target_sequence": event["policy_target_sequence"]})
    if not writes or writes[0]["host_monotonic_s"] < sensor_times[0] or writes[-1]["host_monotonic_s"] > sensor_times[-1]:
        raise ValueError("Recorded encoder interval must cover all completed writes")
    duration = sensor_times[-1] - sensor_times[0]
    if not 0.0 < duration <= 120.0:
        raise ValueError("Replay supports a bounded recording of at most 120 seconds")
    return metadata, gains, writes, initial_q, initial_dq, initial_held, sensor_times[0], sensor_times[-1], recorded_dq is not None


def response_summary(rows, joint_names, frequency_hz, real_end_elapsed_s):
    """Pure post-run waveform calculations; no environment or policy calls."""
    import numpy as np

    if not rows:
        return {"status": "no_samples"}
    selected = {}
    for row in rows:
        selected[row["last_target_selection_sim_time_s"]] = row["delivered_target_rad"]
    selected_times = np.array(sorted(selected))
    selected_targets = np.array([selected[time] for time in selected_times], dtype=float)
    sample_times = np.array([row["replay_sim_time_s"] for row in rows])
    times = np.array([row["recorded_elapsed_s"] for row in rows])
    q = np.array([row["joint_position_rad"] for row in rows], dtype=float)
    # Reconstruct right-continuous actual 60 Hz selections at encoder time, not
    # the preceding-substep target stored alongside each post-physics snapshot.
    target = selected_targets[np.maximum(0, np.searchsorted(selected_times, sample_times, side="right") - 1)]
    target, q = np.rad2deg(target), np.rad2deg(q)
    finite_rows = np.isfinite(target).all(axis=1) & np.isfinite(q).all(axis=1)
    requested_run = (times >= 2.0) & (times <= 20.0) & finite_rows
    failure_times = [row["recorded_elapsed_s"] for row in rows if not row["finite_state"]
                     or row["diagnostic_low_height"] or row["diagnostic_excessive_tilt"]]
    first_failure = min(failure_times) if failure_times else None
    run = requested_run & (times < first_failure if first_failure is not None else True)
    run_span = float(np.ptp(times[run])) if run.any() else 0.0
    sample_interval = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    result = {"status": "available", "run_elapsed_window_s": [2.0, 20.0],
              "run_sample_count": int(run.sum()), "fundamental_frequency_hz": frequency_hz,
              "actual_prefall_fit_elapsed_window_s": [float(times[run][0]), float(times[run][-1])] if run.any() else None,
              "requested_run_window_complete": bool(requested_run.any() and times[requested_run][0] <= 2.0 + sample_interval
                                                     and times[requested_run][-1] >= 20.0 - sample_interval),
              "requested_run_window_entirely_prefall": first_failure is None or first_failure > 20.0,
              "first_low_height_tilt_or_nonfinite_elapsed_s": first_failure,
              "prefall_fit_period_count": run_span * frequency_hz,
              "joint_names": joint_names,
              "method": "Offset + linear drift + three harmonics in requested recorded elapsed 2–20 s, STRICTLY BEFORE the first low-height, excessive-tilt or nonfinite flag. At least four periods are required. Input reconstructed from actual 60 Hz selections as right-continuous ZOH at q timestamp. Phase-equivalent lag is modulo a period, not identified firmware delay. The diagnostic flags do not prove real contact/support equivalence."}
    if run.sum() >= 20 and frequency_hz > 0 and run_span * frequency_hz >= 4.0:
        t = times[run]
        omega = 2 * np.pi * frequency_hz
        columns = [np.ones_like(t), t - t.mean()]
        for harmonic in range(1, 4):
            columns.extend([np.cos(harmonic * omega * t), np.sin(harmonic * omega * t)])
        design = np.column_stack(columns)
        def fit(signal):
            coefficients = np.linalg.lstsq(design, signal, rcond=None)[0]
            variance = np.sum((signal - signal.mean()) ** 2)
            return complex(coefficients[2], -coefficients[3]), (
                float(1 - np.sum((signal - design @ coefficients) ** 2) / variance) if variance > 1e-12 else None)
        per_joint = []
        for j, name in enumerate(joint_names):
            u_phasor, u_r2 = fit(target[run, j])
            q_phasor, q_r2 = fit(q[run, j])
            ratio = q_phasor / u_phasor if abs(u_phasor) > 1e-12 else None
            per_joint.append({"joint": name,
                              "target_fundamental_amplitude_deg": float(abs(u_phasor)),
                              "encoder_fundamental_amplitude_deg": float(abs(q_phasor)),
                              "fundamental_amplitude_ratio": float(abs(ratio)) if ratio is not None else None,
                              "fundamental_phase_lag_deg": float(-np.angle(ratio) * 180 / np.pi) if ratio is not None else None,
                              "fundamental_phase_equivalent_lag_s": float(-np.angle(ratio) / omega) if ratio is not None else None,
                              "target_harmonic_fit_r_squared": u_r2, "encoder_harmonic_fit_r_squared": q_r2,
                              "aligned_target_minus_q_signed_mean_deg": float(np.mean(target[run, j] - q[run, j])),
                              "aligned_target_minus_q_abs_p95_deg": float(np.percentile(np.abs(target[run, j] - q[run, j]), 95)),
                              "low_excitation_warning": bool(abs(u_phasor) < 1.0)})
        result["per_joint"] = per_joint
    else:
        result["harmonic_fit_unavailable_reason"] = "Fewer than four recorded-frequency periods or 20 samples before the first failure flag in requested 2–20 s window. No post-fall plant fit substituted."
    tail = (times >= real_end_elapsed_s - 0.3) & (times <= real_end_elapsed_s) & finite_rows
    result["last_recorded_0p3s_neutral_window"] = {
        "sample_count": int(tail.sum()), "requested_elapsed_window_s": [real_end_elapsed_s - 0.3, real_end_elapsed_s],
        "window_complete": bool(tail.any() and times[tail][0] <= real_end_elapsed_s - 0.3 + sample_interval
                                and times[tail][-1] >= real_end_elapsed_s - sample_interval),
        "after_any_low_height_tilt_or_nonfinite_flag": first_failure is not None and first_failure <= real_end_elapsed_s,
        "interpretation": "Descriptive target tracking only, not static actuator calibration. If a prior fall flag exists, support/load are especially incomparable with the real near-neutral hold."}
    if tail.any():
        error = target[tail] - q[tail]
        result["last_recorded_0p3s_neutral_window"].update({
            "actual_elapsed_window_s": [float(times[tail][0]), float(times[tail][-1])],
            "pooled_abs_error_p95_deg": float(np.percentile(np.abs(error), 95)),
            "pooled_abs_error_max_deg": float(np.max(np.abs(error))),
            "per_joint_target_range_deg": np.ptp(target[tail], axis=0).tolist(),
            "per_joint_final_target_deg": target[tail][-1].tolist(),
            "per_joint_mean_position_deg": np.mean(q[tail], axis=0).tolist(),
            "per_joint_position_range_deg": np.ptp(q[tail], axis=0).tolist(),
            "per_joint_signed_mean_error_deg": np.mean(error, axis=0).tolist(),
            "per_joint_abs_error_p95_deg": np.percentile(np.abs(error), 95, axis=0).tolist(),
            "note": "Comparable coordinate/time window only; simulated support/load need not match real near-neutral tail."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--gain-hypothesis-json", type=Path, required=True)
    parser.add_argument("--trace-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.trial_dir = args.trial_dir.resolve(strict=True)
    args.gain_hypothesis_json = args.gain_hypothesis_json.resolve(strict=True)
    args.trace_jsonl, args.output_json = args.trace_jsonl.resolve(), args.output_json.resolve()
    if args.trace_jsonl == args.output_json or args.trace_jsonl.exists() or args.output_json.exists():
        parser.error("Trace/report must be distinct new paths; existing files are never overwritten")
    try:
        metadata, gains, writes, initial_q, initial_dq, initial_held, real_start, real_end, has_recorded_dq = load_inputs(args.trial_dir, args.gain_hypothesis_json)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    source_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(source_dir.parent))
    from v44_training_contract import TASK
    source_provenance = {name: sha256(source_dir / name) for name in (
        "replay_recorded_targets_v44.py", "commanded_walking_env.py", "commanded_walking_env_cfg.py")}
    trace_id = str(uuid4())
    report = {"schema": "drobot.v44-recorded-target-replay.v1", "trace_id": trace_id,
              "qualification": "DIAGNOSTIC_ONLY_NOT_HARDWARE_CALIBRATION_OR_POLICY_QUALIFICATION",
              "status": "not_started", "task": TASK, "seed": args.seed,
              "trial_id": metadata["recording_id"], "recorded_policy": metadata["policy"],
              "recorded_robot": metadata["robot"], "recorded_payload": metadata.get("payload"),
              "input_sha256": {name: sha256(args.trial_dir / name) for name in ("metadata.json", "samples.jsonl", "events.jsonl")},
              "gain_hypothesis": {"path": str(args.gain_hypothesis_json), "sha256": sha256(args.gain_hypothesis_json), "values": gains},
              "source_sha256": source_provenance,
              "real_start_joint_sensor_monotonic_s": real_start, "real_end_joint_sensor_monotonic_s": real_end,
              "initial_joint_velocity_source": "first_recorded_encoder_frame" if has_recorded_dq else "unavailable_assumed_zero",
              "requested_replay_duration_s": real_end - real_start,
              "trace_path": str(args.trace_jsonl), "report_path": str(args.output_json)}
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True, device=args.device)
    simulation_app, env, rows = launcher.app, None, []
    try:
        import torch
        import parallel_walking  # noqa: F401 -- task registration, no policy loading
        from isaaclab_tasks.utils import load_cfg_from_registry
        from parallel_walking.commanded_walking_env import DrobotCommandedWalkingEnv, _rotate_vectors_xyzw

        cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
        current_v43 = load_cfg_from_registry(V43_TASK, "env_cfg_entry_point")
        plant_prefixes = ("actuator_", "shoe_", "rear_payload_", "dry_robot_")
        plant_names = ("robot", "terrain", "sim", "contact_sensor", "robot_mass_scale_range")
        def plant_snapshot(config):
            return {key: serializable(value) for key, value in config.to_dict().items()
                    if key in plant_names or key.startswith(plant_prefixes)}
        raw_v44, raw_v43 = plant_snapshot(cfg), plant_snapshot(current_v43)
        report["current_source_v43_to_v44_plant_differences"] = {
            key: {"current_v43": raw_v43.get(key), "current_v44": raw_v44.get(key)}
            for key in sorted(set(raw_v43) | set(raw_v44)) if raw_v43.get(key) != raw_v44.get(key)}
        nominal_overrides = {
            "physical_randomization_nominal_fraction": 1.0,
            "rear_payload_combined_mass_scale_range": (1.0, 1.0),
            "rear_payload_combined_com_jitter_m": (0.0, 0.0, 0.0),
            "robot_mass_scale_range": (1.0, 1.0), "actuator_effort_scale_range": (1.0, 1.0),
            "target_velocity_scale_range": (1.0, 1.0), "disable_time_limit": True,
            "reset_joint_position_noise_rad": 0.0, "reset_xy_jitter_m": 0.0,
            "reset_roll_pitch_noise_rad": 0.0, "reset_yaw_noise_rad": 0.0,
            "randomize_gait_start_phase_quarters": False, "seed": args.seed,
        }
        for key, value in {**nominal_overrides, **gains}.items():
            setattr(cfg, key, value)
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        if not math.isclose(cfg.sim.dt, 1.0 / 120.0) or cfg.decimation != 2:
            raise ValueError("Replay contract requires unchanged 120 Hz physics / 60 Hz target update")
        report["effective_plant_config"] = plant_snapshot(cfg)
        report["forced_nominal_values"] = serializable(nominal_overrides)
        usd = Path(cfg.robot.spawn.usd_path).resolve(strict=True)
        report["robot_usd"] = {"path": str(usd), "sha256": sha256(usd)}
        env = DrobotCommandedWalkingEnv(cfg=cfg)
        env.reset(seed=args.seed)
        joint_names = list(env._robot.joint_names)
        recorded_names = metadata["robot"]["joint_order"]
        if set(joint_names) != set(recorded_names) or len(joint_names) != 12:
            raise ValueError("Simulated and recorded joint names must match exactly")
        reorder = [recorded_names.index(name) for name in joint_names]
        def ordered_tensor(values):
            return torch.tensor([[values[index] for index in reorder]], device=env.device, dtype=torch.float32)
        q0, dq0, held0 = ordered_tensor(initial_q), ordered_tensor(initial_dq), ordered_tensor(initial_held)
        env_ids = torch.tensor([0], device=env.device, dtype=torch.int32)
        env._robot.write_joint_position_to_sim_index(position=q0, env_ids=env_ids)
        env._robot.write_joint_velocity_to_sim_index(velocity=dq0, env_ids=env_ids)
        env._processed_actions.copy_(held0)
        env._desired_targets.copy_(held0)
        env._previous_targets.copy_(held0)
        env._actuator_response_targets.copy_(q0)
        env._robot.set_joint_position_target_index(target=q0)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.0)
        physics_dt, target_dt = float(cfg.sim.dt), float(env.step_dt)
        count = math.ceil((real_end - real_start) / physics_dt)
        write_times = [item["host_monotonic_s"] for item in writes]
        selected_index, selected_at = -1, 0.0
        source_first_policy = json.loads((args.trial_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
        elapsed_origin = float(source_first_policy["monotonic_time_s"] - source_first_policy["elapsed_s"])

        def values(tensor):
            return serializable(tensor.detach().cpu().tolist())

        def snapshot(physics_index, sim_time, initial=False):
            data = env._robot.data
            sole_ids = env._sole_body_ids
            sole_quat = data.body_link_quat_w.torch[0, sole_ids]
            corner_offsets = _rotate_vectors_xyzw(sole_quat.unsqueeze(1), env._sole_local_corners.unsqueeze(0))
            corners = data.body_link_pos_w.torch[0, sole_ids].unsqueeze(1) + corner_offsets
            body_masses = data.body_mass.torch[0]
            coms = data.body_com_pos_w.torch[0]
            whole_com = (body_masses.unsqueeze(-1) * coms).sum(dim=0) / body_masses.sum()
            root = data.root_pos_w.torch[0]
            foot_forces = env._foot_forces()[0]
            finite_fields = {
                "joint_position": data.joint_pos.torch, "joint_velocity": data.joint_vel.torch,
                "root_position": root, "root_quaternion": data.root_quat_w.torch,
                "root_linear_velocity": data.root_lin_vel_w.torch, "root_angular_velocity": data.root_ang_vel_w.torch,
                "body_mass": body_masses, "body_com": coms, "whole_com": whole_com,
                "applied_torque": data.applied_torque.torch, "foot_forces": foot_forces,
                "base_contact_force": env._base_contact_force(), "sole_corners": corners,
                "delivered_target": env._processed_actions, "response_target": env._actuator_response_targets,
            }
            nonfinite_fields = [name for name, tensor in finite_fields.items() if not bool(torch.isfinite(tensor).all())]
            finite_state = not nonfinite_fields
            real_time = real_start + sim_time
            elapsed = real_time - elapsed_origin
            stage = ("initial" if initial else "recorded_run" if elapsed < metadata["trial"]["duration_s"]
                     else "recorded_return" if real_time <= write_times[-1] else "recorded_constant_target_tail")
            active_write = writes[selected_index] if selected_index >= 0 else None
            return {
                "record_type": "sample", "physics_step_index": physics_index, "replay_sim_time_s": sim_time,
                "interval_start_sim_time_s": max(0.0, sim_time - physics_dt),
                "equivalent_real_monotonic_s": real_time, "recorded_elapsed_s": elapsed, "stage": stage,
                "last_target_selection_sim_time_s": selected_at,
                "selected_source_write_index": selected_index,
                "selected_source_output_sequence": active_write["output_sequence"] if active_write else None,
                "selected_source_write_monotonic_s": active_write["host_monotonic_s"] if active_write else None,
                "delivered_target_rad": values(env._processed_actions[0]),
                "response_filtered_target_rad": values(env._actuator_response_targets[0]) if env._actuator_target_response_enabled else None,
                "joint_position_rad": values(data.joint_pos.torch[0]), "joint_velocity_rad_s": values(data.joint_vel.torch[0]),
                "applied_torque_nm": values(data.applied_torque.torch[0]),
                "root_position_world_m": values(root), "root_quaternion_world_xyzw": values(data.root_quat_w.torch[0]),
                "root_linear_velocity_world_m_s": values(data.root_lin_vel_w.torch[0]),
                "root_angular_velocity_world_rad_s": values(data.root_ang_vel_w.torch[0]),
                "whole_robot_com_world_m": values(whole_com), "body_com_positions_world_m": values(coms),
                "foot_force_norm_n": values(foot_forces),
                "foot_contact": values(foot_forces > cfg.scheduled_release_force_threshold_n),
                "sole_corners_world_m": values(corners),
                "minimum_corner_height_above_terrain_origin_m": values(corners[..., 2].amin(dim=1) - env._terrain.env_origins[0, 2]),
                "base_contact_force_norm_n": float(env._base_contact_force()[0].item()),
                "finite_state": finite_state,
                "nonfinite_fields": nonfinite_fields,
                "diagnostic_low_height": bool(root[2] - env._terrain.env_origins[0, 2] < cfg.minimum_base_height_m),
                "diagnostic_excessive_tilt": bool(-data.projected_gravity_b.torch[0, 2] < cfg.minimum_upright_cosine),
            }

        report.update({"status": "running", "joint_names": joint_names, "foot_names": list(env._foot_leg_names),
                       "body_names": list(env._robot.body_names), "physics_dt_s": physics_dt, "target_update_dt_s": target_dt,
                       "initial_joint_position_rad": values(q0[0]), "initial_joint_velocity_rad_s": values(dq0[0]),
                       "initial_assumed_held_target_rad": values(held0[0]),
                       "body_masses_kg": values(env._robot.data.body_mass.torch[0]),
                       "actual_total_mass_kg": float(env._robot.data.body_mass.torch[0].sum().item()),
                       "effort_limit_nm": values(env._actuator_effort_limit_by_joint[0]),
                       "drive_stiffness_nm_rad": values(env._actuator_drive_stiffness_by_joint[0]),
                       "drive_damping_nm_s_rad": values(env._actuator_drive_damping_by_joint[0]),
                       "response_tau_s": values(env._actuator_response_tau_s[0]), "response_gain": values(env._actuator_response_gain[0]),
                       "response_filter_enabled": bool(env._actuator_target_response_enabled),
                       "foot_static_friction": values(env._foot_static_friction[0]),
                       "foot_dynamic_friction": values(env._foot_dynamic_friction[0]),
                       "contact_force_threshold_n": cfg.scheduled_release_force_threshold_n,
                       "recorded_elapsed_origin_monotonic_s": elapsed_origin})
        initial_snapshot = snapshot(-1, 0.0, initial=True)
        rows.append(initial_snapshot)
        report["initialized_world_condition"] = {
            key: initial_snapshot[key] for key in (
                "root_position_world_m", "root_quaternion_world_xyzw", "root_linear_velocity_world_m_s",
                "root_angular_velocity_world_rad_s", "whole_robot_com_world_m", "body_com_positions_world_m",
                "sole_corners_world_m", "minimum_corner_height_above_terrain_origin_m", "foot_force_norm_n",
                "base_contact_force_norm_n", "finite_state", "nonfinite_fields", "diagnostic_low_height",
                "diagnostic_excessive_tilt")}
        report["initialized_world_condition"]["contact_note"] = (
            "Before the first advancing physics step; initial sensor forces can be stale or zero. "
            "Root/world COM/contact are authored simulation initial conditions, not reconstructed real-world measurements.")
        if not initial_snapshot["finite_state"]:
            raise ValueError("Initial simulated state contains nonfinite fields")
        selected_counts = [0] * len(writes)
        with torch.inference_mode():
            for index in range(count):
                if not simulation_app.is_running():
                    report["status"] = "interrupted"
                    break
                interval_start = index * physics_dt
                if index % cfg.decimation == 0:
                    selected_at = interval_start
                    selected_index = bisect.bisect_right(write_times, real_start + interval_start) - 1
                    target = held0 if selected_index < 0 else ordered_tensor(writes[selected_index]["targets_rad"])
                    env._processed_actions.copy_(target)
                    env._desired_targets.copy_(target)
                    if selected_index >= 0:
                        selected_counts[selected_index] += 1
                # This is the installed DirectRLEnv physics-substep ordering,
                # deliberately omitting policy/host preprocessing and resets.
                env._sim_step_counter += 1
                env._apply_action()
                env.scene.write_data_to_sim()
                env.sim.step(render=False)
                env.scene.update(dt=physics_dt)
                row = snapshot(index, (index + 1) * physics_dt)
                rows.append(row)
                if not row["finite_state"]:
                    report["status"] = "nonfinite_state_aborted"
                    break
            else:
                report["status"] = "completed"
        report["replayed_physics_steps"] = len(rows) - 1
        report["completed_replay_duration_s"] = rows[-1]["replay_sim_time_s"]
        report["source_writes_never_selected_by_60hz_resampling"] = [index for index, selected in enumerate(selected_counts) if not selected]
        report["source_write_count"] = len(writes)
        report["nominal_target_selection_quantization_bound_s"] = target_dt
        report["first_diagnostic_low_height_or_tilt_elapsed_s"] = next(
            (row["recorded_elapsed_s"] for row in rows if row["diagnostic_low_height"] or row["diagnostic_excessive_tilt"]), None)
        report["response_summary"] = response_summary(rows, joint_names,
            finite(metadata["trial"]["gait_clock"]["frequency_hz"], "recorded gait frequency"), real_end - elapsed_origin)
    except Exception as exc:
        report["status"], report["error"] = "error", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["semantics"] = {
            "qualification": "Open-loop target consistency study only, not a gait-policy evaluation, hardware calibration or deployable model.",
            "time": "Simulated t=0 is the first distinct real joint encoder timestamp. Retain actual recorded write times; latest completed write is ZOH-selected only at 60 Hz, then physics/filter runs at 120 Hz. Resampling adds less than one target-update period and can omit brief targets; omissions are explicitly listed. The last step may exceed the recorded endpoint by less than 1/120 s.",
            "bypassed": "No policy inference, gait-reference generator, host target limiter, packet loss/hold, artificial control delay, heading controller, reward, RL termination/reset or synthetic stop interpolation. Actual recorded neutral-return writes are replayed unchanged.",
            "preserved": "V44 normal _apply_action response filter, actual joint limits/drive/effort clamp, gravity, body mass/inertia and shoe/ground contact. No root fixation, external support or replay-driven world-pose constraint.",
            "initialization": "Initial q from first recorded joint frame; dq from that frame when present, otherwise explicit zero fallback. Initial assumed held target comes from metadata's initial_joint_position_rad. Root uses authored V44 nominal pose and zero velocity; its real height, support/contact and external handling are unknown. Filter state starts at measured q, an explicit unmeasured assumption. No unrecorded settling interval is inserted.",
            "ordering": "All joint vectors follow joint_names, reordered from the recording by exact names. Body vectors follow body_names and foot vectors follow foot_names. Quaternion order is xyzw.",
            "angles": "Radians in model coordinates. Recorded target degrees precede rounding to motor encoder ticks; no raw-byte capture/servo acknowledgement exists, so exact wire target uncertainty is up to about 0.044 degree.",
            "torque": "applied_torque_nm is Isaac Lab's synchronized implicit-PD approximation, not independently measured PhysX joint torque or hardware current. Actual effort cap and drive gains are included in metadata.",
            "contacts": "Force norm threshold, not video-verified contact/toe pressure. COM is simulated mass-weighted body COM. Initial/low-height/tilt flags are diagnostic; the free replay continues after a fall and never auto-resets, unless nonfinite state occurs.",
            "plant_history": "V44 inherits current V43 asset/body configuration but adds target-response hypotheses and gain-telemetry synchronization. The report compares current-source configs only; it does not prove equality to the historical V43 training asset or the real assembled robot. Real contact, firmware profile/PID, load/support and compliant parts remain unknown.",
            "output": "Snapshots are buffered; JSON output occurs after stepping. Not a host/controller timing benchmark. Subsequent offline fitting should compare both selected targets and original event timestamps, and distinguish pre-fall intervals.",
        }
        try:
            for path in (args.trace_jsonl, args.output_json):
                path.parent.mkdir(parents=True, exist_ok=True)
            with args.trace_jsonl.open("x", encoding="utf-8") as stream:
                header = {"record_type": "metadata", **report, "sample_count": len(rows)}
                stream.write(json.dumps(serializable(header), allow_nan=False) + "\n")
                for row in rows:
                    stream.write(json.dumps(serializable(row), allow_nan=False, separators=(",", ":")) + "\n")
            report["trace_sha256"] = sha256(args.trace_jsonl)
            with args.output_json.open("x", encoding="utf-8") as stream:
                json.dump(serializable(report), stream, indent=2, allow_nan=False)
                stream.write("\n")
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                simulation_app.close()


if __name__ == "__main__":
    main()
