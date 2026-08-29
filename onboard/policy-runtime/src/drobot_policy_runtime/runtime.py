"""Shared 60 Hz policy loop; ROS can reuse this class later."""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contract import (
    ACTION_NAMES,
    GRAVITY_M_S2,
    JOINT_LOWER_RAD,
    JOINT_UPPER_RAD,
    SERVO_ID_BY_ACTION_NAME,
    SERVO_VELOCITY_LIMIT_RAD_S,
    GaitClockConfig,
    JointTargetConfig,
    normalized_action_to_joint_target,
)
from .policy import OnnxWalkingPolicy
from .sources import ImuSample, ImuSource, JointStateSample, JointStateSource


@dataclass(frozen=True)
class PolicyCommand:
    forward_m_s: float = 0.15
    lateral_m_s: float = 0.0
    yaw_rad_s: float = 0.0


@dataclass(frozen=True)
class PolicyStepSample:
    """All synchronized inputs and outputs from one successful policy step."""

    sequence: int
    elapsed_s: float
    monotonic_time_s: float
    command: PolicyCommand
    phase_sin: float
    phase_cos: float
    imu: ImuSample
    joints: JointStateSample
    observation: np.ndarray
    action: np.ndarray
    requested_target_rad: np.ndarray
    rate_limited_target_rad: np.ndarray
    target_elapsed_s: float
    gait_frequency_hz: float
    missed_deadlines_total: int


class MotorSink(Protocol):
    def write(
        self,
        action: np.ndarray,
        joint_target_rad: np.ndarray,
        monotonic_time_s: float,
    ) -> None: ...


class PrintMotorSink:
    """Print semantic motor targets; this class never opens a servo bus."""

    def __init__(self, print_hz: float = 5.0) -> None:
        self._period_s = 1.0 / print_hz
        self._next_print_s = 0.0

    def write(
        self,
        action: np.ndarray,
        joint_target_rad: np.ndarray,
        monotonic_time_s: float,
    ) -> None:
        if monotonic_time_s < self._next_print_s:
            return
        self._next_print_s = monotonic_time_s + self._period_s
        motors = [
            {
                "servo_id": SERVO_ID_BY_ACTION_NAME[name],
                "joint": name,
                "target_deg": round(math.degrees(float(joint_target_rad[index])), 3),
                "normalized_action": round(float(action[index]), 5),
            }
            for index, name in enumerate(ACTION_NAMES)
        ]
        print(
            json.dumps(
                {
                    "mode": "PRINT_ONLY_NO_SERVO_WRITES",
                    "monotonic_time_s": round(monotonic_time_s, 6),
                    "motors": motors,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )


class WalkingPolicyLoop:
    def __init__(
        self,
        policy: OnnxWalkingPolicy,
        imu_source: ImuSource,
        joint_source: JointStateSource,
        motor_sink: MotorSink,
        *,
        command: PolicyCommand = PolicyCommand(),
        control_hz: float = 60.0,
        initial_target_rad: np.ndarray | None = None,
        gait_clock: GaitClockConfig | None = None,
        joint_target_config: JointTargetConfig | None = None,
        step_observer: Callable[[PolicyStepSample], None] | None = None,
    ) -> None:
        if not 10.0 <= control_hz <= 200.0:
            raise ValueError("control_hz must be in [10, 200]")
        self.policy = policy
        self.imu_source = imu_source
        self.joint_source = joint_source
        self.motor_sink = motor_sink
        self.command = command
        self.control_hz = control_hz
        self.gait_clock = gait_clock or getattr(
            policy,
            "gait_clock_config",
            GaitClockConfig(),
        )
        self.joint_target_config = joint_target_config or getattr(
            policy,
            "joint_target_config",
            JointTargetConfig(),
        )
        self.step_observer = step_observer
        self.step_count = 0
        self.missed_deadlines = 0
        self._phase_cycles = 0.0
        self._last_phase_time_s: float | None = None
        self._last_target_time_s: float | None = None
        self.previous_action = np.zeros(12, dtype=np.float32)
        initial_target = (
            self.joint_target_config.neutral_array
            if initial_target_rad is None
            else np.asarray(initial_target_rad, dtype=np.float32)
        )
        if initial_target.shape != (12,) or not np.all(np.isfinite(initial_target)):
            raise ValueError("initial_target_rad must contain 12 finite values")
        self.previous_target_rad = initial_target.copy()

    def _reference_target(self, elapsed_s: float) -> np.ndarray | None:
        table = self.joint_target_config.reference_array
        if table is None:
            return None
        table_index = math.floor(self._phase_cycles * len(table)) % len(table)
        desired = table[table_index]
        ramp_duration_s = self.joint_target_config.reference_start_ramp_s
        ramp = (
            1.0
            if ramp_duration_s <= 0.0
            else float(np.clip(elapsed_s / ramp_duration_s, 0.0, 1.0))
        )
        stride_scale = self.gait_clock.stride_scale(
            self.command.forward_m_s,
            self.command.lateral_m_s,
        )
        motion_scale = ramp * stride_scale
        neutral = self.joint_target_config.neutral_array
        return neutral + (desired - neutral) * motion_scale

    def _observation(
        self,
        phase_time_s: float,
    ) -> tuple[np.ndarray, ImuSample, JointStateSample, float, float, float]:
        imu = self.imu_source.read()
        joints = self.joint_source.read()
        gait_frequency_hz = self.gait_clock.frequency_hz(
            self.command.forward_m_s,
            self.command.lateral_m_s,
        )
        if self._last_phase_time_s is not None:
            phase_elapsed_s = max(0.0, phase_time_s - self._last_phase_time_s)
            self._phase_cycles = (
                self._phase_cycles + gait_frequency_hz * phase_elapsed_s
            ) % 1.0
        self._last_phase_time_s = phase_time_s
        phase_angle = 2.0 * math.pi * self._phase_cycles
        phase_sin = math.sin(phase_angle)
        phase_cos = math.cos(phase_angle)
        observation = np.concatenate(
            (
                np.asarray(
                    (
                        self.command.forward_m_s,
                        self.command.lateral_m_s,
                        self.command.yaw_rad_s,
                        phase_sin,
                        phase_cos,
                    ),
                    dtype=np.float32,
                ),
                np.asarray(imu.angular_velocity_body_rad_s, dtype=np.float32),
                np.asarray(imu.projected_gravity_body, dtype=np.float32),
                np.asarray(imu.linear_acceleration_body_m_s2, dtype=np.float32)
                / GRAVITY_M_S2,
                np.asarray(joints.position_rad, dtype=np.float32)
                - self.joint_target_config.neutral_array,
                np.asarray(joints.velocity_rad_s, dtype=np.float32)
                / SERVO_VELOCITY_LIMIT_RAD_S,
                self.previous_action,
            )
        ).astype(np.float32)
        return (
            observation,
            imu,
            joints,
            phase_sin,
            phase_cos,
            gait_frequency_hz,
        )

    def step(self, start_s: float, now_s: float) -> None:
        (
            observation,
            imu,
            joints,
            phase_sin,
            phase_cos,
            gait_frequency_hz,
        ) = self._observation(
            now_s,
        )
        action = self.policy.infer(observation)
        output_time_s = time.monotonic()
        elapsed_s = output_time_s - start_s
        target_elapsed_s = (
            1.0 / self.control_hz
            if self._last_target_time_s is None
            else max(0.0, output_time_s - self._last_target_time_s)
        )
        reference_target = self._reference_target(max(0.0, now_s - start_s))
        base_target = (
            self.joint_target_config.neutral_array
            if reference_target is None
            else reference_target
        )
        action_scale = (
            self.joint_target_config.residual_action_scale
            if self.joint_target_config.action_mode == "gait_residual"
            else 1.0
        )
        requested_target = np.clip(
            base_target
            + self.joint_target_config.action_scale_array
            * action_scale
            * np.clip(action, -1.0, 1.0),
            JOINT_LOWER_RAD,
            JOINT_UPPER_RAD,
        ).astype(np.float32)
        target = normalized_action_to_joint_target(
            action,
            self.previous_target_rad,
            target_elapsed_s,
            self.joint_target_config,
            reference_target,
        )
        self.motor_sink.write(action, target, output_time_s)
        if self.step_observer is not None:
            try:
                self.step_observer(
                    PolicyStepSample(
                        sequence=self.step_count,
                        elapsed_s=elapsed_s,
                        monotonic_time_s=output_time_s,
                        command=self.command,
                        phase_sin=phase_sin,
                        phase_cos=phase_cos,
                        imu=imu,
                        joints=joints,
                        observation=observation,
                        action=action,
                        requested_target_rad=requested_target,
                        rate_limited_target_rad=target,
                        target_elapsed_s=target_elapsed_s,
                        gait_frequency_hz=gait_frequency_hz,
                        missed_deadlines_total=self.missed_deadlines,
                    )
                )
            except Exception:
                # Recording/telemetry observers must never disturb motor control.
                pass
        self.step_count += 1
        self.previous_action = action
        self.previous_target_rad = target
        self._last_target_time_s = output_time_s

    def run(
        self,
        duration_s: float = 0.0,
        stop_event: threading.Event | None = None,
    ) -> None:
        if duration_s < 0.0:
            raise ValueError("duration_s cannot be negative")
        start_s = time.monotonic()
        deadline_s = start_s
        period_s = 1.0 / self.control_hz
        while (
            (duration_s == 0.0 or time.monotonic() - start_s < duration_s)
            and (stop_event is None or not stop_event.is_set())
        ):
            now_s = time.monotonic()
            self.step(start_s, now_s)
            deadline_s += period_s
            completed_s = time.monotonic()
            if deadline_s <= completed_s:
                skipped = math.floor((completed_s - deadline_s) / period_s) + 1
                self.missed_deadlines += skipped
                deadline_s += skipped * period_s
            wait_s = max(0.0, deadline_s - time.monotonic())
            if stop_event is None:
                time.sleep(wait_s)
            elif stop_event.wait(wait_s):
                break
