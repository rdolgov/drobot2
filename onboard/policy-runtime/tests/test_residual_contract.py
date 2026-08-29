from __future__ import annotations

import numpy as np
from drobot_policy_runtime.contract import (
    GaitClockConfig,
    JointTargetConfig,
    normalized_action_to_joint_target,
)


def test_speed_scaled_clock_preserves_low_speed_stride() -> None:
    clock = GaitClockConfig.from_metadata(
        {
            "gait_clock": {
                "mode": "speed_scaled",
                "speed_min_m_s": 0.003,
                "speed_max_m_s": 0.015,
                "frequency_min_hz": 0.06,
                "frequency_max_hz": 0.30,
                "stride_scale_min": 1.0,
            }
        }
    )

    assert clock.frequency_hz(0.003) == 0.06
    assert clock.frequency_hz(0.015) == 0.30
    assert clock.stride_scale(0.003) == 1.0


def test_residual_target_is_applied_around_reference() -> None:
    neutral = tuple(0.0 for _ in range(12))
    reference = tuple(0.1 for _ in range(12))
    config = JointTargetConfig.from_metadata(
        {
            "joint_target_contract": {
                "neutral_joint_position_rad": neutral,
                "action_scale_rad": tuple(0.2 for _ in range(12)),
                "target_velocity_limit_rad_s": 100.0,
                "max_target_step_rad": 1.0,
            },
            "action_contract": {
                "mode": "gait_residual",
                "residual_scale": 0.25,
            },
            "gait_reference": {
                "start_ramp_s": 1.5,
                "joint_position_rad": [reference],
            },
        }
    )

    target = normalized_action_to_joint_target(
        np.ones(12, dtype=np.float32),
        np.asarray(reference, dtype=np.float32),
        elapsed_s=1.0,
        config=config,
        reference_target_rad=np.asarray(reference, dtype=np.float32),
    )

    np.testing.assert_allclose(target, 0.15, atol=1e-6)
