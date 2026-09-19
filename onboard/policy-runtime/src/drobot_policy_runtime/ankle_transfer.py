"""Experimental V2.3 transfer actor and its 82-input, 16-output contract.

Port of ankle_v23_transfer/env.py and ankle_v21/env.py. This separate contract
never feeds ankle weights through the older 50-input walking runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .runtime import PolicyCommand, PolicyStepSample

SCHEMA = "drobot.ankle-v23-transfer.numpy.v1"
CHECKPOINT_SHA256 = "d0b1e4e31fe39919dcedefd2e816dc48b34aeadfd366ccbaf9b9d4c32f7d069d"
LEGS = ("front_left", "rear_left", "front_right", "rear_right")
ACTION_NAMES = tuple(f"{leg}_{kind}" for kind in
                     ("hip_abduction", "hip_flexion", "knee", "ankle_pitch") for leg in LEGS)
SERVO_IDS = (1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12, 13, 15, 14, 16)
UPPER = np.repeat(np.deg2rad([25., 60., 90., 35.]) * .95, 4).astype(np.float32)
LOWER = -UPPER
SCALES = np.repeat([.18, .70, .90, .60], 4).astype(np.float32)
RESIDUAL = np.repeat([.14, .24, .30, .20], 4).astype(np.float32)
RATES = np.repeat(np.deg2rad([60., 100., 100., 100.]), 4).astype(np.float32)
FRONT = np.array([1., -1., 1., -1.], dtype=np.float32)
SIDE = np.array([1., 1., -1., -1.], dtype=np.float32)
PHASE_BIAS = np.array([0., .5, .75, .25], dtype=np.float32)
SHIFT = np.array([[-.012, -.018], [.026, -.018], [-.012, .018], [.026, .018]], dtype=np.float32)
PREVIOUS = (3, 2, 0, 1)
LENGTH = .159896689
KNEE = -math.acos((.070 ** 2 + .275 ** 2 - 2 * LENGTH ** 2) / (2 * LENGTH ** 2))
HIP = math.atan2(.070, .275) - math.atan2(math.sin(KNEE), 1 + math.cos(KNEE))
NEUTRAL = np.concatenate((np.zeros(4), FRONT * HIP, FRONT * KNEE,
                          FRONT * (-HIP - KNEE))).astype(np.float32)
OBSERVATION_ORDER = [
    "command_xyz_times_10[3]", "clock_sin_cos[2]", "imu_angular_velocity_body[3]",
    "projected_gravity_body[3]", "imu_linear_acceleration_body_over_9_81[3]",
    "joint_position_minus_neutral[16]", "joint_velocity_times_0_25[16]",
    "previous_applied_target_normalized[16]", "filtered_action[16]", "expected_stance[4]",
]


@dataclass(frozen=True)
class AnkleTargetConfig:
    neutral_joint_position_rad: tuple = tuple(float(x) for x in NEUTRAL)
    action_scale_rad: tuple = tuple(float(x) for x in SCALES)
    target_velocity_limit_rad_s: float = math.radians(100)
    target_velocity_limit_rad_s_by_action: tuple = tuple(float(x) for x in RATES)
    max_target_step_rad: float = math.radians(2)
    startup_ramp_rate_deg_s: float = 20.
    startup_settle_s: float = 1.
    startup_position_tolerance_deg: float = 5.


def load_metadata(path: Path) -> dict:
    payload = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (payload.get("schema") != SCHEMA or payload.get("action_names") != list(ACTION_NAMES)
            or payload.get("servo_ids") != list(SERVO_IDS)
            or payload.get("observation_order") != OBSERVATION_ORDER
            or payload.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
            or payload.get("weights_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise ValueError("Ankle policy identity or 82/16 contract mismatch")
    return payload


class AnkleTransferPolicy:
    """CPU NumPy evaluation of the frozen RSL-RL ELU/Beta actor."""

    def __init__(self, path: Path):
        self.metadata = load_metadata(path)
        with np.load(path, allow_pickle=False) as archive:
            self.mean = archive["obs_normalizer._mean"].reshape(82).copy()
            self.std = archive["obs_normalizer._std"].reshape(82).copy()
            self.layers = [(archive[f"mlp.{i}.weight"].copy(), archive[f"mlp.{i}.bias"].copy())
                           for i in (0, 2, 4, 6)]
        for (weight, bias), (rows, cols) in zip(self.layers, ((256, 82), (256, 256), (128, 256), (32, 128)), strict=True):
            if weight.shape != (rows, cols) or bias.shape != (rows,):
                raise ValueError("Ankle actor layer dimensions do not match")
        arrays = [self.mean, self.std, *(a for layer in self.layers for a in layer)]
        if any(not np.all(np.isfinite(a)) for a in arrays) or np.any(self.std < 0):
            raise ValueError("Ankle actor weights contain invalid values")

    def infer(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (82,) or not np.all(np.isfinite(x)):
            raise ValueError("Ankle policy requires 82 finite observations")
        x = (x - self.mean) / (self.std + .01)
        for index, (weight, bias) in enumerate(self.layers):
            x = weight @ x + bias
            if index < 3:
                x = np.maximum(x, 0) + np.expm1(np.minimum(x, 0))
        alpha, beta = np.logaddexp(0., x.reshape(2, 16)) + 1.
        action = 2. * alpha / (alpha + beta) - 1.
        if not np.all(np.isfinite(action)):
            raise RuntimeError("Ankle actor produced invalid actions")
        return action.astype(np.float32)


def frequency(speed):
    sweep = .08 + .03 * np.clip((speed - .015) / .006, 0, 1)
    return float(max(speed, .015) * .82 / sweep)


class AnkleTransferLoop:
    """Proprioceptive transfer reference, action filter and bounded motor targets."""

    def __init__(self, policy, imu_source, joint_source, motor_sink, *, command,
                 control_hz, initial_target_rad, joint_target_config, step_observer=None):
        if control_hz != 60. or not .015 <= command.forward_m_s <= .021:
            raise ValueError("Ankle policy requires 60 Hz and 0.015–0.021 m/s")
        self.policy, self.imu_source, self.joint_source = policy, imu_source, joint_source
        self.motor_sink, self.observer = motor_sink, step_observer
        self.speed = float(command.forward_m_s)
        self.initial = np.asarray(initial_target_rad, dtype=np.float32).copy()
        if self.initial.shape != (16,) or not np.all(np.isfinite(self.initial)):
            raise ValueError("Ankle runtime requires 16 measured start positions")

    def run(self, *, duration_s, stop_event, normal_stop_ramp_s, normal_stop_hold_s):
        if normal_stop_ramp_s != 1. or normal_stop_hold_s != 1.:
            raise ValueError("Ankle stopping contract requires one-second ramp and hold")
        dt, cadence = 1. / 60., frequency(self.speed)
        filtered, expected = np.zeros(16, dtype=np.float32), np.ones(4, dtype=np.float32)
        target = self.initial.copy()
        start = time.monotonic()
        deadline, last_step, misses, sequence = start, start - dt, 0, 0
        while not stop_event.is_set():
            now = time.monotonic()
            age = now - start
            if age >= duration_s + normal_stop_ramp_s + normal_stop_hold_s:
                return
            elapsed = now - last_step
            if elapsed > .12:
                raise RuntimeError("RL control loop missed its 120 ms deadline (ankle)")
            ramp = float(np.clip((age - .8) / 1.5, 0, 1) * np.clip(duration_s + 1. - age, 0, 1))
            speed = self.speed * ramp
            phase = 2. * math.pi * age * cadence
            imu, joints = self.imu_source.read(), self.joint_source.read()
            q = np.asarray(joints.position_rad, dtype=np.float32)
            qd = np.asarray(joints.velocity_rad_s, dtype=np.float32)
            gravity = np.asarray(imu.projected_gravity_body, dtype=np.float32)
            if q.shape != (16,) or qd.shape != (16,):
                raise RuntimeError("Ankle feedback must contain 16 joints")
            if np.any(q < LOWER - math.radians(5)) or np.any(q > UPPER + math.radians(5)):
                raise RuntimeError("Measured pose is outside the ankle policy envelope")
            applied = None
            reader = getattr(self.motor_sink, "read_applied_target", None)
            if reader:
                applied = reader()
            base = target.copy() if applied is None else np.asarray(applied.joint_target_rad, dtype=np.float32)
            if base.shape != (16,) or not np.all(np.isfinite(base)):
                raise RuntimeError("Invalid completed ankle motor target")
            if applied is not None and now - applied.monotonic_time_s > .12:
                raise RuntimeError("Completed ankle motor target is stale")
            observation = np.concatenate((
                [speed * 10., 0., 0., math.sin(phase), math.cos(phase)],
                imu.angular_velocity_body_rad_s, gravity,
                imu.linear_acceleration_body_m_s2 / 9.81, q - NEUTRAL, qd * .25,
                np.clip((base - NEUTRAL) / SCALES, -1, 1), filtered, expected,
            )).astype(np.float32)
            inference_start = time.monotonic()
            action = self.policy.infer(observation)
            inference_end = time.monotonic()
            # Training uses a fixed 1/60 s filter and target-rate step.
            filtered += dt / (.10 + dt) * (np.clip(action, -1, 1) - filtered)
            leg_phase = (age * cadence + PHASE_BIAS) % 1.
            selected = int(np.argmax(leg_phase))
            transfer = float(np.clip((leg_phase[selected] - .75) / .07, 0, 1))
            blend = transfer ** 3 * (10 + transfer * (-15 + 6 * transfer))
            shift = (SHIFT[PREVIOUS[selected]] * (1 - blend) + SHIFT[selected] * blend)
            shift = shift * ramp * (speed > .005)
            stance = leg_phase < .82
            u = np.clip((leg_phase - .82) / .18, 0, 1)
            smooth = u ** 3 * (10 + u * (-15 + 6 * u))
            sweep = np.where(stance, .5 - leg_phase / .82, -.5 + smooth)
            lift = .018 * np.sin(math.pi * u) ** 2 * (~stance) * ramp * (speed > .005)
            forward = FRONT * .070 + sweep * (speed * .82 / cadence) - shift[0]
            down = .275 - lift
            cosine = np.clip((forward ** 2 + down ** 2 - 2 * LENGTH ** 2) / (2 * LENGTH ** 2), -.999, .999)
            knee = -FRONT * np.arccos(cosine)
            hip = np.arctan2(forward, down) - np.arctan2(np.sin(knee), 1 + np.cos(knee))
            # R^T * (0,0,-1) has X = 2*(qw*qy-qz*qx), the training pitch sine.
            pitch = math.asin(float(np.clip(gravity[0], -1, 1)))
            reference = np.concatenate((-SIDE * math.atan2(float(shift[1]), .363908389 + .05),
                                        hip, knee, pitch - q[4:8] - q[8:12]))
            expected = (stance | (speed < .005)).astype(np.float32)
            requested = np.clip(reference + filtered * RESIDUAL, LOWER, UPPER)
            step_limit = np.minimum(RATES * min(dt, elapsed), math.radians(2))
            target = base + np.clip(requested - base, -step_limit, step_limit)
            if stop_event.is_set():
                return
            sink_start = time.monotonic()
            self.motor_sink.write(action, target, sink_start)
            sink_end = time.monotonic()
            if self.observer:
                self.observer(PolicyStepSample(
                    sequence=sequence, elapsed_s=age, monotonic_time_s=now,
                    command=PolicyCommand(forward_m_s=speed), phase_sin=math.sin(phase),
                    phase_cos=math.cos(phase), imu=imu, joints=joints, observation=observation,
                    action=action, requested_target_rad=requested, rate_limited_target_rad=target.copy(),
                    target_elapsed_s=elapsed, observation_duration_s=inference_start - now,
                    inference_duration_s=inference_end - inference_start,
                    motor_sink_duration_s=sink_end - sink_start, gait_frequency_hz=cadence,
                    heading_hold_enabled=False, heading_reference_yaw_rad=0.,
                    heading_current_relative_rad=0., heading_desired_relative_rad=0.,
                    heading_error_rad=0., heading_correction_rad_s=0., effective_yaw_rad_s=0.,
                    missed_deadlines_total=misses, applied_motor_target=applied,
                    target_limit_base_rad=base.copy(),
                ))
            sequence += 1
            last_step = now
            deadline += dt
            if deadline < sink_end:
                misses += 1
                deadline = sink_end
            stop_event.wait(max(0., deadline - time.monotonic()))
