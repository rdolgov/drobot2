"""Shared 60 Hz policy loop; ROS can reuse this class later."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .contract import (
    ACTION_NAMES,
    DEFAULT_JOINT_POSITION_RAD,
    GAIT_PERIOD_S,
    GRAVITY_M_S2,
    SERVO_ID_BY_ACTION_NAME,
    SERVO_VELOCITY_LIMIT_RAD_S,
    normalized_action_to_joint_target,
)
from .policy import OnnxWalkingPolicy
from .sources import ImuSource, JointStateSource


@dataclass(frozen=True)
class PolicyCommand:
    forward_m_s: float = 0.15
    lateral_m_s: float = 0.0
    yaw_rad_s: float = 0.0


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
    ) -> None:
        if not 10.0 <= control_hz <= 200.0:
            raise ValueError("control_hz must be in [10, 200]")
        self.policy = policy
        self.imu_source = imu_source
        self.joint_source = joint_source
        self.motor_sink = motor_sink
        self.command = command
        self.control_hz = control_hz
        self.previous_action = np.zeros(12, dtype=np.float32)
        self.previous_target_rad = DEFAULT_JOINT_POSITION_RAD.copy()

    def _observation(self, elapsed_s: float) -> np.ndarray:
        imu = self.imu_source.read()
        joints = self.joint_source.read()
        phase_angle = 2.0 * math.pi * ((elapsed_s / GAIT_PERIOD_S) % 1.0)
        return np.concatenate(
            (
                np.asarray(
                    (
                        self.command.forward_m_s,
                        self.command.lateral_m_s,
                        self.command.yaw_rad_s,
                        math.sin(phase_angle),
                        math.cos(phase_angle),
                    ),
                    dtype=np.float32,
                ),
                np.asarray(imu.angular_velocity_body_rad_s, dtype=np.float32),
                np.asarray(imu.projected_gravity_body, dtype=np.float32),
                np.asarray(imu.linear_acceleration_body_m_s2, dtype=np.float32)
                / GRAVITY_M_S2,
                np.asarray(joints.position_rad, dtype=np.float32)
                - DEFAULT_JOINT_POSITION_RAD,
                np.asarray(joints.velocity_rad_s, dtype=np.float32)
                / SERVO_VELOCITY_LIMIT_RAD_S,
                self.previous_action,
            )
        ).astype(np.float32)

    def step(self, start_s: float, now_s: float) -> None:
        action = self.policy.infer(self._observation(now_s - start_s))
        target = normalized_action_to_joint_target(
            action, self.previous_target_rad, self.control_hz
        )
        self.motor_sink.write(action, target, now_s)
        self.previous_action = action
        self.previous_target_rad = target

    def run(self, duration_s: float = 0.0) -> None:
        if duration_s < 0.0:
            raise ValueError("duration_s cannot be negative")
        start_s = time.monotonic()
        deadline_s = start_s
        period_s = 1.0 / self.control_hz
        while duration_s == 0.0 or time.monotonic() - start_s < duration_s:
            now_s = time.monotonic()
            self.step(start_s, now_s)
            deadline_s += period_s
            time.sleep(max(0.0, deadline_s - time.monotonic()))

