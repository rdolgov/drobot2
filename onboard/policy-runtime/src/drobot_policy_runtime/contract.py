"""The exact deployable observation, action, and joint-target contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

OBSERVATION_SIZE = 50
ACTION_SIZE = 12
SERVO_VELOCITY_LIMIT_RAD_S = 4.5836625
GRAVITY_M_S2 = 9.81
GAIT_PERIOD_S = 0.8
# Match the conservative 120 deg/s crawl ceiling at 60 Hz. The hardware sink
# independently clamps finite commands to the same bound.
MAX_TARGET_STEP_RAD = math.radians(2.0)


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
    stride_scale_min: float = 1.0

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
            self.stride_scale_min,
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
        if not 0.0 <= self.stride_scale_min <= 1.0:
            raise ValueError("minimum gait stride scale must be in [0, 1]")

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
            stride_scale_min=float(gait.get("stride_scale_min", 1.0)),
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

    def stride_scale(
        self,
        forward_m_s: float,
        lateral_m_s: float = 0.0,
    ) -> float:
        """Return the deployable command-dependent reference amplitude."""

        if self.mode == "fixed":
            return 1.0
        command_speed = math.hypot(float(forward_m_s), float(lateral_m_s))
        if command_speed <= self.standstill_deadband_m_s:
            return 0.0
        fraction = np.clip(
            (command_speed - self.speed_min_m_s)
            / (self.speed_max_m_s - self.speed_min_m_s),
            0.0,
            1.0,
        )
        return float(self.stride_scale_min + fraction * (1.0 - self.stride_scale_min))


@dataclass(frozen=True)
class HeadingHoldConfig:
    """Model-declared relative-yaw feedback through the existing yaw command."""

    enabled: bool = False
    mode: str = "disabled"
    kp_s: float = 0.0
    max_correction_rad_s: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "relative_yaw_feedback"}:
            raise ValueError(f"Unsupported heading-hold mode: {self.mode}")
        if not math.isfinite(self.kp_s) or self.kp_s < 0.0:
            raise ValueError("Heading-hold kp must be finite and non-negative")
        if (
            not math.isfinite(self.max_correction_rad_s)
            or self.max_correction_rad_s < 0.0
        ):
            raise ValueError(
                "Heading-hold maximum correction must be finite and non-negative"
            )
        if self.enabled:
            if self.mode != "relative_yaw_feedback":
                raise ValueError(
                    "Enabled heading hold requires relative_yaw_feedback mode"
                )
            if self.kp_s <= 0.0 or self.max_correction_rad_s <= 0.0:
                raise ValueError(
                    "Enabled heading hold requires positive kp and correction limit"
                )

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> HeadingHoldConfig:
        payload = dict(metadata or {})
        heading = payload.get("heading_hold")
        if not isinstance(heading, Mapping) or not bool(heading.get("enabled", False)):
            return cls()
        return cls(
            enabled=True,
            mode=str(heading.get("mode", "relative_yaw_feedback")),
            kp_s=float(heading["kp_s"]),
            max_correction_rad_s=float(heading["max_correction_rad_s"]),
        )

    def effective_yaw_rad_s(
        self,
        requested_yaw_rad_s: float,
        heading_error_rad: float,
    ) -> tuple[float, float]:
        """Return effective yaw input and the bounded feedback correction."""

        requested = float(requested_yaw_rad_s)
        error = float(heading_error_rad)
        if not math.isfinite(requested) or not math.isfinite(error):
            raise ValueError("Heading-hold inputs must be finite")
        if not self.enabled:
            return requested, 0.0
        correction = max(
            -self.max_correction_rad_s,
            min(self.max_correction_rad_s, self.kp_s * error),
        )
        return requested - correction, correction


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


@dataclass(frozen=True)
class JointTargetConfig:
    """Model-declared neutral pose and deployable target dynamics."""

    neutral_joint_position_rad: tuple[float, ...] = tuple(
        float(value) for value in DEFAULT_JOINT_POSITION_RAD
    )
    action_scale_rad: tuple[float, ...] = tuple(
        float(value) for value in ACTION_SCALE_RAD
    )
    target_velocity_limit_rad_s: float = SERVO_VELOCITY_LIMIT_RAD_S
    max_target_step_rad: float = MAX_TARGET_STEP_RAD
    startup_ramp_rate_deg_s: float = 45.0
    startup_settle_s: float = 0.5
    startup_position_tolerance_deg: float = 5.0
    action_mode: str = "direct"
    residual_action_scale: float = 1.0
    residual_action_scale_by_action: tuple[float, ...] | None = None
    reference_start_ramp_s: float = 0.0
    reference_joint_position_rad: tuple[tuple[float, ...], ...] = ()

    def __post_init__(self) -> None:
        if len(self.neutral_joint_position_rad) != ACTION_SIZE:
            raise ValueError("neutral joint position must contain 12 values")
        if len(self.action_scale_rad) != ACTION_SIZE:
            raise ValueError("action scale must contain 12 values")
        if (
            self.residual_action_scale_by_action is not None
            and len(self.residual_action_scale_by_action) != ACTION_SIZE
        ):
            raise ValueError("per-action residual scale must contain 12 values")
        values = (
            *self.neutral_joint_position_rad,
            *self.action_scale_rad,
            self.target_velocity_limit_rad_s,
            self.max_target_step_rad,
            self.startup_ramp_rate_deg_s,
            self.startup_settle_s,
            self.startup_position_tolerance_deg,
            self.residual_action_scale,
            *(self.residual_action_scale_by_action or ()),
            self.reference_start_ramp_s,
            *(value for row in self.reference_joint_position_rad for value in row),
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("joint target contract values must be finite")
        neutral = np.asarray(self.neutral_joint_position_rad, dtype=np.float32)
        scale = np.asarray(self.action_scale_rad, dtype=np.float32)
        if np.any(neutral < JOINT_LOWER_RAD) or np.any(neutral > JOINT_UPPER_RAD):
            raise ValueError("neutral joint position exceeds the policy envelope")
        if np.any(scale <= 0.0):
            raise ValueError("action scales must be positive")
        if self.target_velocity_limit_rad_s <= 0.0:
            raise ValueError("target velocity limit must be positive")
        if self.max_target_step_rad <= 0.0:
            raise ValueError("maximum target step must be positive")
        if not 1.0 <= self.startup_ramp_rate_deg_s <= 180.0:
            raise ValueError("startup ramp rate must be in [1, 180] deg/s")
        if not 0.0 <= self.startup_settle_s <= 10.0:
            raise ValueError("startup settle time must be in [0, 10] seconds")
        if not 0.1 <= self.startup_position_tolerance_deg <= 15.0:
            raise ValueError("startup position tolerance must be in [0.1, 15] deg")
        if self.action_mode not in {"direct", "gait_residual"}:
            raise ValueError("action mode must be 'direct' or 'gait_residual'")
        if not 0.0 < self.residual_action_scale <= 1.0:
            raise ValueError("residual action scale must be in (0, 1]")
        if self.residual_action_scale_by_action is not None and any(
            not 0.0 < value <= 1.0
            for value in self.residual_action_scale_by_action
        ):
            raise ValueError("per-action residual scales must be in (0, 1]")
        if not 0.0 <= self.reference_start_ramp_s <= 10.0:
            raise ValueError("reference start ramp must be in [0, 10] seconds")
        if any(len(row) != ACTION_SIZE for row in self.reference_joint_position_rad):
            raise ValueError("each gait reference sample must contain 12 values")
        if self.action_mode == "gait_residual" and not self.reference_joint_position_rad:
            raise ValueError("gait-residual policies require a joint reference table")

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, Any] | None,
    ) -> JointTargetConfig:
        payload = dict(metadata or {})
        target = payload.get("joint_target_contract")
        if not isinstance(target, Mapping):
            target = {}
        startup = payload.get("startup")
        if not isinstance(startup, Mapping):
            startup = {}
        action_contract = payload.get("action_contract")
        if not isinstance(action_contract, Mapping):
            action_contract = {}
        residual_scales = action_contract.get("residual_scale_by_action")
        if residual_scales is not None and not isinstance(
            residual_scales,
            (list, tuple),
        ):
            raise TypeError("per-action residual scale must be an array")
        gait_reference = payload.get("gait_reference")
        if not isinstance(gait_reference, Mapping):
            gait_reference = {}
        reference_rows = gait_reference.get("joint_position_rad", ())
        if not isinstance(reference_rows, (list, tuple)):
            raise TypeError("gait reference joint positions must be an array")
        return cls(
            neutral_joint_position_rad=tuple(
                float(value)
                for value in target.get(
                    "neutral_joint_position_rad",
                    DEFAULT_JOINT_POSITION_RAD,
                )
            ),
            action_scale_rad=tuple(
                float(value)
                for value in target.get("action_scale_rad", ACTION_SCALE_RAD)
            ),
            target_velocity_limit_rad_s=float(
                target.get(
                    "target_velocity_limit_rad_s",
                    SERVO_VELOCITY_LIMIT_RAD_S,
                )
            ),
            max_target_step_rad=float(
                target.get("max_target_step_rad", MAX_TARGET_STEP_RAD)
            ),
            startup_ramp_rate_deg_s=float(
                startup.get("ramp_rate_deg_s", 45.0)
            ),
            startup_settle_s=float(startup.get("settle_s", 0.5)),
            startup_position_tolerance_deg=float(
                startup.get("position_tolerance_deg", 5.0)
            ),
            action_mode=str(action_contract.get("mode", "direct")),
            residual_action_scale=float(action_contract.get("residual_scale", 1.0)),
            residual_action_scale_by_action=(
                None
                if residual_scales is None
                else tuple(float(value) for value in residual_scales)
            ),
            reference_start_ramp_s=float(gait_reference.get("start_ramp_s", 0.0)),
            reference_joint_position_rad=tuple(
                tuple(float(value) for value in row) for row in reference_rows
            ),
        )

    @property
    def neutral_array(self) -> np.ndarray:
        return np.asarray(self.neutral_joint_position_rad, dtype=np.float32)

    @property
    def action_scale_array(self) -> np.ndarray:
        return np.asarray(self.action_scale_rad, dtype=np.float32)

    @property
    def residual_action_scale_array(self) -> np.ndarray:
        if self.residual_action_scale_by_action is None:
            return np.full(
                ACTION_SIZE,
                self.residual_action_scale,
                dtype=np.float32,
            )
        return np.asarray(self.residual_action_scale_by_action, dtype=np.float32)

    @property
    def reference_array(self) -> np.ndarray | None:
        if not self.reference_joint_position_rad:
            return None
        return np.asarray(self.reference_joint_position_rad, dtype=np.float32)

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
    config: JointTargetConfig | None = None,
    reference_target_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Convert an action using real elapsed time, never invocation count."""

    step_duration_s = float(elapsed_s)
    if not math.isfinite(step_duration_s) or step_duration_s < 0.0:
        raise ValueError("elapsed_s must be finite and non-negative")
    target_config = config or JointTargetConfig()
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if target_config.action_mode == "gait_residual":
        if reference_target_rad is None:
            raise ValueError("gait-residual action requires a reference target")
        base_target = np.asarray(reference_target_rad, dtype=np.float32)
        if base_target.shape != (ACTION_SIZE,) or not np.all(np.isfinite(base_target)):
            raise ValueError("reference target must contain 12 finite values")
        action = target_config.residual_action_scale_array * action
    else:
        base_target = target_config.neutral_array
    desired = np.clip(
        base_target + target_config.action_scale_array * action,
        JOINT_LOWER_RAD,
        JOINT_UPPER_RAD,
    )
    # The wall-time velocity bound prevents compressed scheduler iterations from
    # advancing the target too quickly. The per-packet cap also avoids a large
    # jump after a delayed cycle and matches the guarded hardware sink.
    max_delta = min(
        target_config.target_velocity_limit_rad_s * step_duration_s,
        target_config.max_target_step_rad,
    )
    return np.clip(
        desired,
        previous_target_rad - max_delta,
        previous_target_rad + max_delta,
    ).astype(np.float32)
