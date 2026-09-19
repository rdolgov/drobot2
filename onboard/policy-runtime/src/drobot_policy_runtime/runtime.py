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
    HeadingHoldConfig,
    JointTargetConfig,
    RunStopConfig,
    normalized_action_to_joint_target,
)
from .policy import (
    ACTION_HISTORY_APPLIED_TARGET,
    ACTION_HISTORY_REQUESTED,
    OnnxWalkingPolicy,
)
from .sources import ImuSample, ImuSource, JointStateSample, JointStateSource


@dataclass(frozen=True)
class PolicyCommand:
    forward_m_s: float = 0.15
    lateral_m_s: float = 0.0
    yaw_rad_s: float = 0.0


@dataclass(frozen=True)
class AppliedMotorTarget:
    """Last host-completed bus write, not a motor/encoder acknowledgement."""

    joint_target_rad: np.ndarray
    monotonic_time_s: float
    sequence: int


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
    observation_duration_s: float
    inference_duration_s: float
    motor_sink_duration_s: float
    gait_frequency_hz: float
    heading_hold_enabled: bool
    heading_reference_yaw_rad: float
    heading_current_relative_rad: float
    heading_desired_relative_rad: float
    heading_error_rad: float
    heading_correction_rad_s: float
    effective_yaw_rad_s: float
    missed_deadlines_total: int
    applied_motor_target: AppliedMotorTarget | None = None
    target_limit_base_rad: np.ndarray | None = None


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
        heading_hold: HeadingHoldConfig | None = None,
        joint_target_config: JointTargetConfig | None = None,
        run_stop_config: RunStopConfig | None = None,
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
        self.heading_hold = heading_hold or getattr(
            policy,
            "heading_hold_config",
            HeadingHoldConfig(),
        )
        self.joint_target_config = joint_target_config or getattr(
            policy,
            "joint_target_config",
            JointTargetConfig(),
        )
        self.run_stop_config = run_stop_config or getattr(
            policy,
            "run_stop_config",
            RunStopConfig(),
        )
        self.experimental_mapper = getattr(policy, "experimental_mapper", None)
        if self.experimental_mapper is not None:
            if getattr(motor_sink, "offline_only", False) is not True:
                raise ValueError("Study35 is restricted to an explicitly offline sink")
            if any(x is not None for x in (gait_clock, heading_hold, joint_target_config, run_stop_config)) or control_hz != 60.0:
                raise ValueError("Study35 does not permit runtime contract overrides")
        self.last_experimental_mapping = None
        self._mapping_stop_start_s: float | None = None
        self._last_phase_frequency_hz: float | None = None
        self.action_history_mode = getattr(
            policy,
            "action_history_mode",
            ACTION_HISTORY_REQUESTED,
        )
        self.command_observation_scale = np.asarray(
            getattr(
                policy,
                "command_observation_scale",
                (1.0, 1.0, 1.0),
            ),
            dtype=np.float32,
        )
        if self.command_observation_scale.shape != (3,) or not np.all(
            np.isfinite(self.command_observation_scale)
        ):
            raise ValueError("command observation scale must contain three finite values")
        if np.any(self.command_observation_scale <= 0.0):
            raise ValueError("command observation scales must be positive")
        if self.action_history_mode not in (
            ACTION_HISTORY_REQUESTED,
            ACTION_HISTORY_APPLIED_TARGET,
        ):
            raise ValueError(
                f"Unsupported policy action history: {self.action_history_mode!r}"
            )
        if (
            self.action_history_mode == ACTION_HISTORY_APPLIED_TARGET
            and self.joint_target_config.action_mode
            not in ("direct", "gait_residual", "integrated_target_velocity")
        ):
            raise ValueError(
                "applied-target action history requires direct, gait-residual, "
                "or integrated-target-velocity actions"
            )
        self.step_observer = step_observer
        self.step_count = 0
        self.missed_deadlines = 0
        self._phase_cycles = 0.0
        self._last_phase_time_s: float | None = None
        self._last_target_time_s: float | None = None
        self._heading_reference_yaw_rad: float | None = None
        self._heading_desired_relative_rad = 0.0
        self.previous_action = np.zeros(12, dtype=np.float32)
        initial_target = (
            self.joint_target_config.neutral_array
            if initial_target_rad is None
            else np.asarray(initial_target_rad, dtype=np.float32)
        )
        if initial_target.shape != (12,) or not np.all(np.isfinite(initial_target)):
            raise ValueError("initial_target_rad must contain 12 finite values")
        if self.experimental_mapper is not None and not np.array_equal(initial_target,self.joint_target_config.neutral_array):
            raise ValueError("Study35 offline start requires calibrated center targets")
        self.previous_target_rad = initial_target.copy()
        self._acknowledged_target_rad = initial_target.copy()
        self._applied_motor_target: AppliedMotorTarget | None = None

    def _action_history_target(self) -> np.ndarray:
        reader = getattr(self.motor_sink, "read_applied_target", None)
        if not callable(reader):
            # Existing synchronous/print sinks retain their established contract.
            return self.previous_target_rad
        acknowledged = reader()
        if acknowledged is None:
            # An asynchronous sink must not promote queued targets to applied
            # history simply because its output worker is delayed.
            return self._acknowledged_target_rad
        target = np.asarray(acknowledged.joint_target_rad, dtype=np.float32)
        timestamp = float(acknowledged.monotonic_time_s)
        sequence = int(acknowledged.sequence)
        if target.shape != (12,) or not np.all(np.isfinite(target)):
            raise RuntimeError("Applied motor target must contain 12 finite joint values")
        if not math.isfinite(timestamp) or sequence < 0:
            raise RuntimeError("Applied motor target acknowledgement is invalid")
        previous = self._applied_motor_target
        if previous is not None and (
            sequence < previous.sequence or timestamp < previous.monotonic_time_s
        ):
            raise RuntimeError("Applied motor target acknowledgement moved backwards")
        self._acknowledged_target_rad = target.copy()
        self._applied_motor_target = AppliedMotorTarget(target.copy(), timestamp, sequence)
        return self._acknowledged_target_rad

    def _reference_target(self, elapsed_s: float) -> np.ndarray | None:
        table = self.joint_target_config.reference_array
        if table is None:
            return None
        table_index = math.floor(self._phase_cycles * len(table)) % len(table)
        desired = table[table_index]
        low_table = self.joint_target_config.reference_low_array
        if low_table is not None:
            blend = self.joint_target_config.reference_blend_fraction(
                self.command.forward_m_s,
                self.command.lateral_m_s,
            )
            desired = low_table[table_index] + blend * (
                desired - low_table[table_index]
            )
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
    ) -> tuple[
        np.ndarray,
        ImuSample,
        JointStateSample,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:
        imu = self.imu_source.read()
        joints = self.joint_source.read()
        gait_frequency_hz = self.gait_clock.frequency_hz(
            self.command.forward_m_s,
            self.command.lateral_m_s,
        )
        phase_elapsed_s = 0.0
        if self._last_phase_time_s is not None:
            phase_elapsed_s = max(0.0, phase_time_s - self._last_phase_time_s)
            if self.experimental_mapper is not None:
                # The previous command drove the preceding interval. Preserve
                # the simulator float32 clock at nominal dt; use real elapsed
                # time when intervals vary, never catch-up policy invocations.
                self._phase_cycles = float(np.remainder(
                    np.float32(self._phase_cycles) + np.float32(self._last_phase_frequency_hz)*np.float32(phase_elapsed_s),
                    np.float32(1.)))
            else:
                self._phase_cycles = (
                    self._phase_cycles + gait_frequency_hz * phase_elapsed_s
                ) % 1.0
        self._last_phase_frequency_hz = gait_frequency_hz
        self._last_phase_time_s = phase_time_s
        if self._heading_reference_yaw_rad is None:
            self._heading_reference_yaw_rad = float(imu.heading_yaw_rad)
        else:
            self._heading_desired_relative_rad = math.atan2(
                math.sin(
                    self._heading_desired_relative_rad
                    + self.command.yaw_rad_s * phase_elapsed_s
                ),
                math.cos(
                    self._heading_desired_relative_rad
                    + self.command.yaw_rad_s * phase_elapsed_s
                ),
            )
        heading_current_relative_rad = math.atan2(
            math.sin(float(imu.heading_yaw_rad) - self._heading_reference_yaw_rad),
            math.cos(float(imu.heading_yaw_rad) - self._heading_reference_yaw_rad),
        )
        heading_error_rad = math.atan2(
            math.sin(
                heading_current_relative_rad - self._heading_desired_relative_rad
            ),
            math.cos(
                heading_current_relative_rad - self._heading_desired_relative_rad
            ),
        )
        effective_yaw_rad_s, heading_correction_rad_s = (
            self.heading_hold.effective_yaw_rad_s(
                self.command.yaw_rad_s,
                heading_error_rad,
            )
        )
        phase_angle = 2.0 * math.pi * self._phase_cycles
        phase_sin = math.sin(phase_angle)
        phase_cos = math.cos(phase_angle)
        delivered_target = self._action_history_target()
        action_history = self.previous_action
        if self.action_history_mode == ACTION_HISTORY_APPLIED_TARGET:
            action_history = np.clip(
                (
                    delivered_target
                    - self.joint_target_config.neutral_array
                )
                / self.joint_target_config.action_scale_array,
                -1.0,
                1.0,
            ).astype(np.float32)
        observation = np.concatenate(
            (
                np.asarray(
                    (
                        self.command.forward_m_s,
                        self.command.lateral_m_s,
                        effective_yaw_rad_s,
                    ),
                    dtype=np.float32,
                )
                * self.command_observation_scale,
                np.asarray((phase_sin, phase_cos), dtype=np.float32),
                np.asarray(imu.angular_velocity_body_rad_s, dtype=np.float32),
                np.asarray(imu.projected_gravity_body, dtype=np.float32),
                np.asarray(imu.linear_acceleration_body_m_s2, dtype=np.float32)
                / GRAVITY_M_S2,
                np.asarray(joints.position_rad, dtype=np.float32)
                - self.joint_target_config.neutral_array,
                np.asarray(joints.velocity_rad_s, dtype=np.float32)
                / SERVO_VELOCITY_LIMIT_RAD_S,
                action_history,
            )
        ).astype(np.float32)
        if self.experimental_mapper is not None:
            observation = np.clip(observation, -20., 20.)
        return (
            observation,
            imu,
            joints,
            phase_sin,
            phase_cos,
            gait_frequency_hz,
            self._heading_reference_yaw_rad,
            heading_current_relative_rad,
            self._heading_desired_relative_rad,
            heading_error_rad,
            heading_correction_rad_s,
            effective_yaw_rad_s,
        )

    def step(self, start_s: float, now_s: float) -> None:
        observation_started_s = time.monotonic()
        (
            observation,
            imu,
            joints,
            phase_sin,
            phase_cos,
            gait_frequency_hz,
            heading_reference_yaw_rad,
            heading_current_relative_rad,
            heading_desired_relative_rad,
            heading_error_rad,
            heading_correction_rad_s,
            effective_yaw_rad_s,
        ) = self._observation(
            now_s,
        )
        observation_completed_s = time.monotonic()
        action = self.policy.infer(observation)
        output_time_s = time.monotonic()
        elapsed_s = output_time_s - start_s
        target_elapsed_s = (
            1.0 / self.control_hz
            if self._last_target_time_s is None
            else max(0.0, output_time_s - self._last_target_time_s)
        )
        previous_target = self.previous_target_rad
        if callable(getattr(self.motor_sink, "read_applied_target", None)):
            previous_target = self._acknowledged_target_rad
            if self._applied_motor_target is not None:
                # Do not subtract the usual queue delay a second time on every
                # producer tick. This is a request budget, not a prediction of
                # the next write time: the bus guard enforces actual elapsed
                # time and packet limits independently.
                target_elapsed_s = max(
                    target_elapsed_s,
                    max(0.0, output_time_s - self._applied_motor_target.monotonic_time_s),
                )
        if self.experimental_mapper is not None:
            self.last_experimental_mapping = self.experimental_mapper.map(
                action, phase_cycles=self._phase_cycles, age_s=max(0., now_s-start_s),
                command=(self.command.forward_m_s, self.command.lateral_m_s, self.command.yaw_rad_s),
                previous_delivered=previous_target, measured_q=joints.position_rad,
                elapsed_s=target_elapsed_s,
                stop_elapsed_s=(None if self._mapping_stop_start_s is None else max(0., now_s-self._mapping_stop_start_s)),
            )
            requested_target = self.last_experimental_mapping["requested"]
            target = self.last_experimental_mapping["delivered"]
        else:
            reference_target = self._reference_target(max(0.0, now_s - start_s))
            base_target = (
                self.joint_target_config.neutral_array
                if reference_target is None
                else reference_target
            )
            motion_requested = (
                math.hypot(
                    self.command.forward_m_s,
                    self.command.lateral_m_s,
                )
                > self.gait_clock.standstill_deadband_m_s
                or abs(self.command.yaw_rad_s) > 1.0e-4
            )
            target_action = np.clip(action, -1.0, 1.0)
            if (
                self.joint_target_config.action_mode == "gait_residual"
                and not motion_requested
            ):
                target_action = (
                    target_action
                    * self.joint_target_config.gait_residual_stop_action_scale
                )
            residual_scale = (
                self.joint_target_config.residual_action_scale_array
                if self.joint_target_config.action_mode == "gait_residual"
                else 1.0
            )
            if self.joint_target_config.action_mode == "integrated_target_velocity":
                maximum_delta = np.minimum(
                    self.joint_target_config.target_velocity_limit_array
                    * target_elapsed_s,
                    self.joint_target_config.max_target_step_rad,
                )
                if motion_requested:
                    requested_target = np.clip(
                        previous_target
                        + maximum_delta * target_action,
                        JOINT_LOWER_RAD,
                        JOINT_UPPER_RAD,
                    ).astype(np.float32)
                elif (
                    self.joint_target_config.integrated_stop_target_residual_scale
                    > 0.0
                ):
                    # At zero command the actor selects a small posture residual
                    # around calibrated center. Slew toward that reachable target
                    # with the same elapsed-time and packet-step limits used while
                    # moving, so stopping cannot introduce a target jump.
                    stop_target = np.clip(
                        self.joint_target_config.neutral_array
                        + self.joint_target_config.action_scale_array
                        * self.joint_target_config.integrated_stop_target_residual_scale
                        * target_action,
                        JOINT_LOWER_RAD,
                        JOINT_UPPER_RAD,
                    )
                    requested_target = np.clip(
                        previous_target
                        + np.clip(
                            stop_target - previous_target,
                            -maximum_delta,
                            maximum_delta,
                        ),
                        JOINT_LOWER_RAD,
                        JOINT_UPPER_RAD,
                    ).astype(np.float32)
                elif np.all(maximum_delta > 0.0):
                    # A velocity policy can otherwise output zero and freeze an
                    # accumulated non-neutral target.  The zero-command contract
                    # always homes through the same bounded packet increments.  A
                    # model-declared residual retains limited authority to damp the
                    # body and finish load transfer without defeating neutral homing.
                    homing_action = np.clip(
                        (
                            self.joint_target_config.neutral_array
                            - previous_target
                        )
                        / maximum_delta,
                        -1.0,
                        1.0,
                    )
                    effective_action = np.clip(
                        homing_action
                        + self.joint_target_config.integrated_stop_action_scale
                        * target_action,
                        -1.0,
                        1.0,
                    )
                    requested_target = np.clip(
                        previous_target + maximum_delta * effective_action,
                        JOINT_LOWER_RAD,
                        JOINT_UPPER_RAD,
                    ).astype(np.float32)
                else:
                    requested_target = previous_target.copy()
            else:
                requested_target = np.clip(
                    base_target
                    + self.joint_target_config.action_scale_array
                    * residual_scale
                    * target_action,
                    JOINT_LOWER_RAD,
                    JOINT_UPPER_RAD,
                ).astype(np.float32)
            target = (
                requested_target
                if self.joint_target_config.action_mode
                == "integrated_target_velocity"
                else normalized_action_to_joint_target(
                    target_action,
                    previous_target,
                    target_elapsed_s,
                    self.joint_target_config,
                    reference_target,
                )
            )
        motor_sink_started_s = time.monotonic()
        self.motor_sink.write(action, target, output_time_s)
        motor_sink_completed_s = time.monotonic()
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
                        observation_duration_s=max(
                            0.0, observation_completed_s - observation_started_s
                        ),
                        inference_duration_s=max(
                            0.0, output_time_s - observation_completed_s
                        ),
                        motor_sink_duration_s=max(
                            0.0, motor_sink_completed_s - motor_sink_started_s
                        ),
                        gait_frequency_hz=gait_frequency_hz,
                        heading_hold_enabled=self.heading_hold.enabled,
                        heading_reference_yaw_rad=heading_reference_yaw_rad,
                        heading_current_relative_rad=heading_current_relative_rad,
                        heading_desired_relative_rad=heading_desired_relative_rad,
                        heading_error_rad=heading_error_rad,
                        heading_correction_rad_s=heading_correction_rad_s,
                        effective_yaw_rad_s=effective_yaw_rad_s,
                        missed_deadlines_total=self.missed_deadlines,
                        applied_motor_target=self._applied_motor_target,
                        target_limit_base_rad=previous_target.copy(),
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
        *,
        normal_stop_ramp_s: float = 0.0,
        normal_stop_hold_s: float = 0.0,
    ) -> None:
        values = (duration_s, normal_stop_ramp_s, normal_stop_hold_s)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("policy-loop durations must be finite")
        if duration_s < 0.0:
            raise ValueError("duration_s cannot be negative")
        if normal_stop_ramp_s < 0.0:
            raise ValueError("normal_stop_ramp_s cannot be negative")
        if normal_stop_hold_s < 0.0:
            raise ValueError("normal_stop_hold_s cannot be negative")
        if duration_s == 0.0 and (
            normal_stop_ramp_s > 0.0 or normal_stop_hold_s > 0.0
        ):
            raise ValueError("an indefinite run cannot schedule a normal stop")
        start_s = time.monotonic()
        deadline_s = start_s
        period_s = 1.0 / self.control_hz
        run_command = self.command
        align_stop_phase = (
            duration_s > 0.0
            and normal_stop_ramp_s > 0.0
            and self.run_stop_config.phase_alignment_enabled
        )
        alignment_pending = align_stop_phase
        ramp_start_elapsed_s = duration_s
        stop_end_elapsed_s: float | None = (
            None
            if alignment_pending
            else duration_s + normal_stop_ramp_s + normal_stop_hold_s
        )
        if self.experimental_mapper is not None and align_stop_phase:
            # This candidate fixed the stop tick at episode reset, using the
            # nominal phase at the requested duration, then rounded its wait
            # once to a control tick. Do not align from the previous live tick.
            ramp_start_elapsed_s = self.experimental_mapper.stop_start_age_s(
                duration_s, run_command.forward_m_s)
            stop_end_elapsed_s = ramp_start_elapsed_s + normal_stop_ramp_s + normal_stop_hold_s
            alignment_pending = False
        while stop_event is None or not stop_event.is_set():
            now_s = time.monotonic()
            elapsed_s = max(0.0, now_s - start_s)
            if (
                alignment_pending
                and elapsed_s >= duration_s
            ):
                gait_frequency_hz = self.gait_clock.frequency_hz(
                    run_command.forward_m_s,
                    run_command.lateral_m_s,
                )
                alignment_wait_s = self.run_stop_config.alignment_delay_s(
                    self._phase_cycles,
                    gait_frequency_hz,
                )
                ramp_start_elapsed_s = elapsed_s + alignment_wait_s
                stop_end_elapsed_s = (
                    ramp_start_elapsed_s
                    + normal_stop_ramp_s
                    + normal_stop_hold_s
                )
                alignment_pending = False
            if (
                duration_s > 0.0
                and not alignment_pending
                and elapsed_s + (1.e-9 if self.experimental_mapper is not None else 0.) >= ramp_start_elapsed_s
            ):
                ramp_elapsed_s = elapsed_s - ramp_start_elapsed_s
                if self.experimental_mapper is not None:
                    # Simulator commands advance before the next observation:
                    # the first stop sample represents one elapsed interval.
                    ramp_elapsed_s += period_s
                    self._mapping_stop_start_s = start_s + ramp_start_elapsed_s - period_s
                command_scale = (
                    0.0
                    if normal_stop_ramp_s <= 0.0
                    else self.run_stop_config.ramp_command_scale(
                        ramp_elapsed_s / normal_stop_ramp_s
                    )
                )
                self.command = PolicyCommand(
                    forward_m_s=run_command.forward_m_s * command_scale,
                    lateral_m_s=run_command.lateral_m_s * command_scale,
                    yaw_rad_s=run_command.yaw_rad_s * command_scale,
                )
            if (
                duration_s > 0.0
                and stop_end_elapsed_s is not None
                and elapsed_s + (1.e-9 if self.experimental_mapper is not None else 0.) >= stop_end_elapsed_s
            ):
                break
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
        if duration_s > 0.0 and (stop_event is None or not stop_event.is_set()):
            self.command = PolicyCommand(
                forward_m_s=0.0,
                lateral_m_s=0.0,
                yaw_rad_s=0.0,
            )
