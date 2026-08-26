"""The exact deployable observation, action, and joint-target contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


OBSERVATION_SIZE = 50
ACTION_SIZE = 12
SERVO_VELOCITY_LIMIT_RAD_S = 4.5836625
GRAVITY_M_S2 = 9.81
GAIT_PERIOD_S = 0.8
MAX_TARGET_STEP_RAD = math.radians(5.0)


@dataclass(frozen=True)
class GaitClockConfig:
    """Model-declared gait-clock behavior shared by training and deployment."""

    mode: str = "fixed"
    fixed_period_s: float = GAIT_PERIOD_S
    standstill_deadband_m_s: float = 0.0
    speed_min_m_s: float = 0.0
    speed_max_m_s: float = 0.10
    frequency_min_hz: float = 1.0 / GAIT_PERIOD_S
    frequency_max_hz: float = 1.0 / GAIT_PERIOD_S

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "speed_scaled"}:
            raise ValueError("gait clock mode must be 'fixed' or 'speed_scaled'")
        values = (
            self.fixed_period_s,
            self.standstill_deadband_m_s,
            self.speed_min_m_s,
            self.speed_max_m_s,
            self.frequency_min_hz,
            self.frequency_max_hz,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gait clock values must be finite")
        if self.fixed_period_s <= 0.0:
            raise ValueError("fixed gait period must be positive")
        if self.standstill_deadband_m_s < 0.0:
            raise ValueError("gait standstill deadband cannot be negative")
        if self.speed_min_m_s < self.standstill_deadband_m_s:
            raise ValueError("minimum gait speed cannot be below the deadband")
        if self.speed_max_m_s <= self.speed_min_m_s:
            raise ValueError("maximum gait speed must exceed minimum gait speed")
        if self.frequency_min_hz <= 0.0:
            raise ValueError("minimum gait frequency must be positive")
        if self.frequency_max_hz < self.frequency_min_hz:
            raise ValueError(
                "maximum gait frequency cannot be below minimum gait frequency"
            )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any] | None) -> GaitClockConfig:
        """Load v2 cadence metadata or preserve the v1 fixed-period contract."""

        payload = dict(metadata or {})
        gait = payload.get("gait_clock")
        if not isinstance(gait, Mapping):
            return cls(
                fixed_period_s=float(
                    payload.get("gait_period_s", GAIT_PERIOD_S)
                )
            )
        mode = str(gait.get("mode", "fixed"))
        fixed_period_s = float(
            gait.get("period_s", payload.get("gait_period_s", GAIT_PERIOD_S))
        )
        if mode == "fixed":
            return cls(fixed_period_s=fixed_period_s)
        return cls(
            mode=mode,
            fixed_period_s=fixed_period_s,
            standstill_deadband_m_s=float(gait.get("standstill_deadband_m_s", 0.0)),
            speed_min_m_s=float(gait["speed_min_m_s"]),
            speed_max_m_s=float(gait["speed_max_m_s"]),
            frequency_min_hz=float(gait["frequency_min_hz"]),
            frequency_max_hz=float(gait["frequency_max_hz"]),
        )

    def frequency_hz(
        self,
        forward_m_s: float,
        lateral_m_s: float = 0.0,
    ) -> float:
        if self.mode == "fixed":
            return 1.0 / self.fixed_period_s
        command_speed = math.hypot(float(forward_m_s), float(lateral_m_s))
        if command_speed <= self.standstill_deadband_m_s:
            return 0.0
        fraction = np.clip(
            (command_speed - self.speed_min_m_s)
            / (self.speed_max_m_s - self.speed_min_m_s),
            0.0,
            1.0,
        )
        return float(
            self.frequency_min_hz
            + fraction * (self.frequency_max_hz - self.frequency_min_hz)
        )

ACTION_NAMES = (
    "front_left_hip_abduction",
    "rear_left_hip_abduction",
    "front_right_hip_abduction",
    "rear_right_hip_abduction",
    "front_left_hip_flexion",
    "rear_left_hip_flexion",
    "front_right_hip_flexion",
    "rear_right_hip_flexion",
    "front_left_knee",
    "rear_left_knee",
    "front_right_knee",
    "rear_right_knee",
)

STANCE_ANGLE_RAD = 0.5239596454
DEFAULT_JOINT_POSITION_RAD = np.asarray(
    (
        0.0,
        0.0,
        0.0,
        0.0,
        STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        STANCE_ANGLE_RAD,
        -STANCE_ANGLE_RAD,
        STANCE_ANGLE_RAD,
    ),
    dtype=np.float32,
)
ACTION_SCALE_RAD = np.asarray(
    (0.12, 0.12, 0.12, 0.12, 0.30, 0.30, 0.30, 0.30, 0.40, 0.40, 0.40, 0.40),
    dtype=np.float32,
)
JOINT_LOWER_RAD = np.asarray(
    (-0.436,) * 4 + (-1.047,) * 4 + (-1.571,) * 4,
    dtype=np.float32,
)
JOINT_UPPER_RAD = -JOINT_LOWER_RAD

# Physical manifest order is FL, FR, RL, RR, while Isaac's tensor order is
# FL, RL, FR, RR inside each joint type.
SERVO_ID_BY_ACTION_NAME = {
    "front_left_hip_abduction": 1,
    "front_left_hip_flexion": 2,
    "front_left_knee": 3,
    "front_right_hip_abduction": 4,
    "front_right_hip_flexion": 5,
    "front_right_knee": 6,
    "rear_left_hip_abduction": 7,
    "rear_left_hip_flexion": 8,
    "rear_left_knee": 9,
    "rear_right_hip_abduction": 10,
    "rear_right_hip_flexion": 11,
    "rear_right_knee": 12,
}


def normalized_action_to_joint_target(
    action: np.ndarray,
    previous_target_rad: np.ndarray,
    elapsed_s: float,
) -> np.ndarray:
    """Convert an action using real elapsed time, never invocation count."""

    step_duration_s = float(elapsed_s)
    if not math.isfinite(step_duration_s) or step_duration_s < 0.0:
        raise ValueError("elapsed_s must be finite and non-negative")
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    desired = np.clip(
        DEFAULT_JOINT_POSITION_RAD + ACTION_SCALE_RAD * action,
        JOINT_LOWER_RAD,
        JOINT_UPPER_RAD,
    )
    # The wall-time velocity bound prevents compressed scheduler iterations from
    # advancing the target too quickly. The per-packet cap also avoids a large
    # jump after a delayed cycle and matches the guarded hardware sink.
    max_delta = min(
        SERVO_VELOCITY_LIMIT_RAD_S * step_duration_s,
        MAX_TARGET_STEP_RAD,
    )
    return np.clip(
        desired,
        previous_target_rad - max_delta,
        previous_target_rad + max_delta,
    ).astype(np.float32)
