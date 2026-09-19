"""Opt-in, passive per-physics-step diagnostics for a single V44 PhysX env.

This module does not import or launch Isaac. ``PassivePhysxStepDiagnostic`` is
installed explicitly by an existing evaluator; its wrapper calls the original
``sim.step`` exactly once. It never sets a target, force, limit, or physics
property. Host copies add overhead: this is NOT a timing benchmark.

See passive_physx_step_diagnostic.md for the integration boundary and caveats.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import statistics
from uuid import uuid4


SCHEMA = "drobot.passive-physx-step-diagnostic.v1"
V44_TASK = "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"


def _stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not values:
        return {"samples": 0, "mean": None, "rms": None, "min": None, "max": None}
    return {"samples": len(values), "mean": statistics.mean(values),
            "rms": math.sqrt(statistics.mean(v * v for v in values)),
            "min": min(values), "max": max(values)}


def _vector(snapshot, name, count):
    value = snapshot.get(name) if snapshot else None
    if not isinstance(value, list) or len(value) != count:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        return None
    return value


def summarize_physics_steps(metadata, records):
    """Offline arithmetic only; finite differences are intervals, not exact qdot.

    Each complete pre/post pair spans ONE physics step. Consecutive post/post
    differences are accepted only for adjacent recorded wrapper calls in the
    same episode. Missing channels remain unavailable rather than become zero.
    """
    names = metadata["joint_names"]
    count, dt = len(names), float(metadata["physics_dt_s"])
    groups = {}
    for record in records:
        key = (record["episode_index"], record["stage"])
        groups.setdefault(key, []).append(record)
    result = {"schema": SCHEMA + ".summary", "trace_id": metadata["trace_id"],
              "interpretation": "Diagnostic differences, not pass/fail gates or identified motor torque. Post-physics cache precedes scene.update. Interval derivatives need not equal instantaneous solver velocity.",
              "groups": []}
    for (episode, stage), rows in groups.items():
        joint_values = [{key: [] for key in (
            "pre_raw_q_rad", "post_raw_q_rad", "pre_raw_qd_rad_s", "post_raw_qd_rad_s",
            "pre_to_post_velocity_rad_s", "consecutive_post_velocity_rad_s",
            "post_velocity_minus_interval_velocity_rad_s", "trapezoidal_velocity_minus_interval_velocity_rad_s",
            "pre_cache_minus_raw_q_rad", "post_cache_minus_raw_q_rad",
            "pre_cache_minus_raw_qd_rad_s", "post_cache_minus_raw_qd_rad_s",
            "pre_approximate_torque_minus_raw_state_pd_nm", "post_approximate_torque_minus_raw_state_pd_nm",
            "pre_projected_force_minus_approximate_torque_nm", "post_projected_force_minus_approximate_torque_nm",
        )} for _ in names]
        substeps, previous = {}, None
        for row in rows:
            pre, post = row.get("pre"), row.get("post")
            pq = _vector(pre, "backend_joint_position_rad", count)
            qq = _vector(post, "backend_joint_position_rad", count)
            pv = _vector(pre, "backend_joint_velocity_rad_s", count)
            qv = _vector(post, "backend_joint_velocity_rad_s", count)
            for label, snapshot in (("pre", pre), ("post", post)):
                if not snapshot:
                    continue
                raw_q = _vector(snapshot, "backend_joint_position_rad", count)
                raw_v = _vector(snapshot, "backend_joint_velocity_rad_s", count)
                cache_q = _vector(snapshot, "cached_joint_position_rad", count)
                cache_v = _vector(snapshot, "cached_joint_velocity_rad_s", count)
                torque = _vector(snapshot, "approximate_applied_torque_nm", count)
                projected = _vector(snapshot, "backend_projected_joint_force_nm", count)
                pd = _vector(snapshot, "pd_from_backend_state_nm", count)
                for i in range(count):
                    dest = joint_values[i]
                    if raw_q is not None:
                        dest[label + "_raw_q_rad"].append(raw_q[i])
                    if raw_v is not None:
                        dest[label + "_raw_qd_rad_s"].append(raw_v[i])
                    for suffix, first, second in (
                        ("cache_minus_raw_q_rad", cache_q, raw_q),
                        ("cache_minus_raw_qd_rad_s", cache_v, raw_v),
                        ("approximate_torque_minus_raw_state_pd_nm", torque, pd),
                        ("projected_force_minus_approximate_torque_nm", projected, torque),
                    ):
                        if first is not None and second is not None:
                            dest[label + "_" + suffix].append(first[i] - second[i])
            if pq is not None and qq is not None:
                for i in range(count):
                    interval = (qq[i] - pq[i]) / dt
                    joint_values[i]["pre_to_post_velocity_rad_s"].append(interval)
                    if qv is not None:
                        joint_values[i]["post_velocity_minus_interval_velocity_rad_s"].append(qv[i] - interval)
                    if pv is not None and qv is not None:
                        joint_values[i]["trapezoidal_velocity_minus_interval_velocity_rad_s"].append((pv[i] + qv[i]) / 2 - interval)
            if qq is not None and qv is not None:
                buckets = substeps.setdefault(row["physics_substep_in_control"], [[] for _ in names])
                for i in range(count):
                    buckets[i].append((qq[i], qv[i]))
            if previous is not None and row["wrapper_call_index"] == previous["wrapper_call_index"] + 1:
                last = _vector(previous.get("post"), "backend_joint_position_rad", count)
                if last is not None and qq is not None:
                    for i in range(count):
                        joint_values[i]["consecutive_post_velocity_rad_s"].append((qq[i] - last[i]) / dt)
            previous = row
        result["groups"].append({
            "episode_index": episode, "stage": stage, "recorded_physics_steps": len(rows),
            "failed_physics_steps": sum(row.get("physics_step_error") is not None for row in rows),
            "snapshot_error_count": sum(len((row.get(side) or {}).get("errors", {})) for row in rows for side in ("pre", "post")),
            "by_joint": {name: {key: _stats(values) for key, values in joint_values[i].items()} for i, name in enumerate(names)},
            "by_substep_and_joint": {str(substep): {name: {
                "post_q_rad": _stats(pair[0] for pair in buckets[i]),
                "post_qd_rad_s": _stats(pair[1] for pair in buckets[i]),
            } for i, name in enumerate(names)} for substep, buckets in substeps.items()},
        })
    return result


class PassivePhysxStepDiagnostic:
    """Caller-selected capture windows; bounded memory; no disk I/O while stepping."""

    def __init__(self, env, *, task, provenance, max_physics_steps=1000):
        if task != V44_TASK or int(env.num_envs) != 1:
            raise ValueError("This diagnostic is restricted to the single-environment V44 evaluation")
        if getattr(env, "_physics_handles_decimation", True):
            raise ValueError("Backend-managed decimation cannot be labeled individual physics steps")
        if isinstance(max_physics_steps, bool) or not isinstance(max_physics_steps, int) or not 1 <= max_physics_steps <= 12000:
            raise ValueError("max_physics_steps must be an integer from 1 to 12000")
        self.env, self.robot, self.sim = env, env._robot, env.sim
        if not type(self.robot).__module__.startswith("isaaclab_physx."):
            raise ValueError("Expected the installed isaaclab_physx articulation backend")
        self.names = list(self.robot.joint_names)
        backend_names = list(self.robot.backend_joint_names)
        if len(self.names) != 12 or len(set(self.names)) != 12 or set(self.names) != set(backend_names) or len(backend_names) != 12:
            raise ValueError("Expected exactly 12 uniquely named V44 rotational DOFs in each order")
        self.order = [backend_names.index(name) for name in self.names]
        self.dt = float(env.physics_dt)
        if not math.isfinite(self.dt) or not math.isclose(self.dt, 1 / 120, rel_tol=0, abs_tol=1e-9):
            raise ValueError("This diagnostic requires the existing 120 Hz physics configuration")
        self.limit, self.records = max_physics_steps, []
        self.context = None
        self.wrapper_calls = self.eligible_calls = self.omitted_at_limit = 0
        self.lifecycle_errors = []
        self.installed = False
        self.metadata = {
            "record_type": "metadata", "schema": SCHEMA, "trace_id": str(uuid4()),
            "task": task, "provenance": json.loads(json.dumps(provenance, allow_nan=False)),
            "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "joint_names": self.names, "backend_joint_names": backend_names,
            "backend_indices_in_user_order": self.order, "physics_dt_s": self.dt,
            "control_dt_s": float(env.step_dt), "decimation": int(env.cfg.decimation),
            "max_physics_steps": self.limit,
            "semantics": {
                "pre": "After _apply_action and scene.write_data_to_sim; immediately before the original sim.step call.",
                "post": "Immediately after original sim.step returns; BEFORE DirectRLEnv scene.update advances caches. Cached/approximate torque values may deliberately still represent pre-step state.",
                "copy_order": "Copy Lab cached fields first, then direct backend getters. Each tensor is copied to host before the next getter; no live tensor references are buffered.",
                "ordering_units": "All joint vectors follow joint_names: rotational q in rad, qdot in rad/s, effort in Nm, K in Nm/rad, B in Nm*s/rad.",
                "approximate_torque": "Lab implicit-actuator PD estimate, not measured PhysX drive torque. Raw-state PD reuses copied target/K/B/limit buffers; no parameters are altered.",
                "projected_joint_force": "get_dof_projected_joint_forces projects incoming joint force into the motion axis. An independent backend quantity, not a calibrated motor-current reading or guaranteed isolated drive torque.",
                "contact_forces": "No contact force arrays are captured here. Existing net_forces_w data is NORMAL contact force only, omitting friction. Projected joint-force differences cannot locate or identify frictional self-stress without separate friction/contact-point evidence.",
                "timing": "Host synchronization overhead is intentionally added. This is not a wall-clock/control performance measurement. Finite-difference velocity is an interval quantity, not an exact instantaneous-velocity identity.",
            },
        }

    def _snapshot(self):
        import warp as wp

        row = {"errors": {}, "articulation_cache_timestamp_s": float(self.robot.data._sim_timestamp)}
        data = self.robot.data

        def copy(name, getter, *, backend=False):
            try:
                tensor = getter()
                tensor = wp.to_torch(tensor) if backend else tensor.torch
                values = tensor.detach().cpu().clone().tolist()
                if len(values) != 1 or len(values[0]) != 12:
                    raise ValueError("Expected a [1,12] joint tensor")
                values = values[0]
                if backend:
                    values = [values[index] for index in self.order]
                if any(not math.isfinite(value) for value in values):
                    raise ValueError("Nonfinite joint data")
                row[name] = values
            except Exception as exc:
                row[name] = None
                row["errors"][name] = f"{type(exc).__name__}: {exc}"[:500]

        # Never let a raw getter refresh an aliased backend buffer before these copies.
        for name, attribute in (
            ("cached_joint_position_rad", "joint_pos"),
            ("cached_joint_velocity_rad_s", "joint_vel"),
            ("position_target_rad", "joint_pos_target"),
            ("velocity_target_rad_s", "joint_vel_target"),
            ("feedforward_effort_nm", "joint_effort_target"),
            ("approximate_applied_torque_nm", "applied_torque"),
            ("approximate_computed_torque_nm", "computed_torque"),
            ("cached_stiffness_nm_rad", "joint_stiffness"),
            ("cached_damping_nm_s_rad", "joint_damping"),
            ("cached_effort_limit_nm", "joint_effort_limits"),
        ):
            copy(name, lambda attribute=attribute: getattr(data, attribute))
        for name, method in (
            ("backend_joint_position_rad", "get_dof_positions"),
            ("backend_joint_velocity_rad_s", "get_dof_velocities"),
            ("backend_projected_joint_force_nm", "get_dof_projected_joint_forces"),
        ):
            copy(name, lambda method=method: getattr(self.robot.root_view, method)(), backend=True)
        fields = [row.get(key) for key in (
            "position_target_rad", "velocity_target_rad_s", "feedforward_effort_nm",
            "cached_stiffness_nm_rad", "cached_damping_nm_s_rad", "cached_effort_limit_nm",
            "backend_joint_position_rad", "backend_joint_velocity_rad_s",
        )]
        row["pd_from_backend_state_nm"] = None
        if all(value is not None for value in fields):
            target, target_v, feedforward, k, b, limit, q, v = fields
            pd = [k[i] * (target[i] - q[i]) + b[i] * (target_v[i] - v[i]) + feedforward[i] for i in range(12)]
            if any(not math.isfinite(value) for value in pd) or any(value < 0 for value in limit):
                row["errors"]["pd_from_backend_state_nm"] = "Nonfinite PD arithmetic or negative effort limit"
            else:
                row["pd_from_backend_state_nm"] = [max(-limit[i], min(limit[i], pd[i])) for i in range(12)]
        return row

    def _safe_snapshot(self):
        try:
            return self._snapshot()
        except Exception as exc:
            return {"errors": {"snapshot": f"{type(exc).__name__}: {exc}"[:500]}}

    def __enter__(self):
        if self.installed or getattr(self.sim.step, "_drobot_passive_physx_step", False):
            raise RuntimeError("A passive physics wrapper is already installed")
        self.original = self.sim.step
        self.had_instance_step = "step" in vars(self.sim)
        self.original_instance_step = vars(self.sim).get("step")

        @wraps(self.original)
        def wrapped(*args, **kwargs):
            self.wrapper_calls += 1
            if self.context is None:
                return self.original(*args, **kwargs)
            context = dict(self.context)
            self.context["physics_substep_in_control"] += 1
            if not context.pop("capture"):
                return self.original(*args, **kwargs)
            self.eligible_calls += 1
            if len(self.records) >= self.limit:
                self.omitted_at_limit += 1
                return self.original(*args, **kwargs)
            context.update(record_type="physics_step", wrapper_call_index=self.wrapper_calls,
                           env_sim_step_counter_before_call=int(self.env._sim_step_counter),
                           nominal_episode_interval_start_s=context["control_step_index"] * float(self.env.step_dt) + context["physics_substep_in_control"] * self.dt,
                           physics_step_error=None, pre=self._safe_snapshot())
            try:
                result = self.original(*args, **kwargs)
            except BaseException as exc:
                context["physics_step_error"] = f"{type(exc).__name__}: {exc}"[:500]
                context["post"] = None
                self.records.append(context)
                raise
            context["post"] = self._safe_snapshot()
            self.records.append(context)
            return result

        wrapped._drobot_passive_physx_step = True
        self.wrapped = wrapped
        self.sim.step = wrapped
        self.installed = True
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.context = None
        if self.sim.step is not self.wrapped:
            self.lifecycle_errors.append("sim.step changed while diagnostic was installed; refusing to overwrite another owner")
        elif self.had_instance_step:
            self.sim.step = self.original_instance_step
        else:
            delattr(self.sim, "step")
        self.installed = False
        return False

    @contextmanager
    def control_step(self, *, episode_index, control_step_index, stage, capture=True):
        if not self.installed or self.context is not None:
            raise RuntimeError("control_step requires an installed recorder with no active context")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (episode_index, control_step_index)):
            raise ValueError("Episode/control indices must be nonnegative integers")
        if stage not in {"run", "ramp", "hold"} or not isinstance(capture, bool):
            raise ValueError("Expected run/ramp/hold stage and boolean capture")
        self.context = {"episode_index": episode_index, "control_step_index": control_step_index,
                        "stage": stage, "capture": capture, "physics_substep_in_control": 0}
        try:
            yield
        finally:
            self.context = None

    def write_jsonl(self, path):
        """Exclusive output, after unhooking; buffered partial data survives failures."""
        if self.installed:
            raise RuntimeError("Uninstall the recorder before writing diagnostic output")
        metadata = dict(self.metadata)
        metadata.update(recorded_physics_steps=len(self.records), eligible_physics_steps=self.eligible_calls,
                        omitted_at_limit=self.omitted_at_limit, capture_truncated=self.omitted_at_limit > 0,
                        lifecycle_errors=list(self.lifecycle_errors))
        metadata["offline_summary"] = summarize_physics_steps(metadata, self.records)
        with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
            for record in (metadata, *self.records):
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        return metadata
