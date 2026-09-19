from __future__ import annotations

import time

import numpy as np
import pytest
from drobot_policy_runtime.contract import (
    GaitClockConfig,
    JointTargetConfig,
    RunStopConfig,
    normalized_action_to_joint_target,
)
from drobot_policy_runtime.policy import ACTION_HISTORY_APPLIED_TARGET
from drobot_policy_runtime.runtime import PolicyCommand, WalkingPolicyLoop
from drobot_policy_runtime.sources import LevelImuSource, NeutralJointStateSource


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


def test_integrated_target_uses_per_action_velocity_limits() -> None:
    config = JointTargetConfig.from_metadata(
        {
            "joint_target_contract": {
                "target_velocity_limit_rad_s": 1.0,
                "target_velocity_limit_rad_s_by_action": [0.5] * 4
                + [1.0] * 8,
                "max_target_step_rad": 1.0,
            },
            "action_contract": {
                "mode": "integrated_target_velocity",
            },
        }
    )

    target = normalized_action_to_joint_target(
        np.ones(12, dtype=np.float32),
        np.zeros(12, dtype=np.float32),
        elapsed_s=0.1,
        config=config,
    )

    np.testing.assert_allclose(target[:4], 0.05, atol=1e-6)
    np.testing.assert_allclose(target[4:], 0.10, atol=1e-6)


def _dual_reference_config() -> JointTargetConfig:
    return JointTargetConfig.from_metadata(
        {
            "joint_target_contract": {
                "neutral_joint_position_rad": [0.0] * 12,
            },
            "action_contract": {
                "mode": "gait_residual",
                "residual_scale": 0.04,
                "residual_scale_by_action": [0.04] * 4 + [0.12] * 8,
            },
            "gait_reference": {
                "start_ramp_s": 0.0,
                "joint_position_rad": [[0.20] * 12, [0.40] * 12],
                "low_speed_joint_position_rad": [[0.10] * 12, [0.30] * 12],
                "blend_speed_min_m_s": 0.10,
                "blend_speed_max_m_s": 0.14,
            },
        }
    )


def test_dual_reference_contract_blends_by_command_speed() -> None:
    config = _dual_reference_config()

    assert config.reference_blend_fraction(0.08) == 0.0
    assert config.reference_blend_fraction(0.10) == 0.0
    assert config.reference_blend_fraction(0.12) == pytest.approx(0.5)
    assert config.reference_blend_fraction(0.14) == 1.0
    assert config.reference_blend_fraction(0.18) == 1.0


def test_runtime_uses_the_same_dual_table_interpolation() -> None:
    class _Policy:
        action_history_mode = ACTION_HISTORY_APPLIED_TARGET

    class _MotorSink:
        def write(self, action, joint_target_rad, monotonic_time_s) -> None:
            del action, joint_target_rad, monotonic_time_s

    config = _dual_reference_config()
    loop = WalkingPolicyLoop(
        _Policy(),
        object(),
        object(),
        _MotorSink(),
        command=PolicyCommand(forward_m_s=0.12),
        joint_target_config=config,
    )
    loop._phase_cycles = 0.0

    np.testing.assert_allclose(loop._reference_target(1.0), 0.15, atol=1e-6)
    loop._phase_cycles = 0.5
    np.testing.assert_allclose(loop._reference_target(1.0), 0.35, atol=1e-6)


def test_dual_reference_contract_rejects_mismatched_table_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        JointTargetConfig.from_metadata(
            {
                "action_contract": {"mode": "gait_residual"},
                "gait_reference": {
                    "joint_position_rad": [[0.0] * 12, [0.0] * 12],
                    "low_speed_joint_position_rad": [[0.0] * 12],
                    "blend_speed_min_m_s": 0.10,
                    "blend_speed_max_m_s": 0.14,
                },
            }
        )


def test_v43_phase_aligned_smooth_stop_contract() -> None:
    config = RunStopConfig.from_metadata(
        {
            "run_stop_contract": {
                "recommended_stop_ramp_s": 0.90,
                "recommended_stopped_hold_s": 2.0,
                "ramp_shape": "cubic_smoothstep",
                "phase_alignment": {
                    "enabled": True,
                    "ramp_start_phase_cycles": 0.05,
                    "phase_tolerance_cycles": 0.03,
                    "maximum_wait_s": 0.50,
                },
            }
        }
    )

    assert config.ramp_command_scale(0.0) == 1.0
    assert config.ramp_command_scale(0.5) == pytest.approx(0.5)
    assert config.ramp_command_scale(1.0) == 0.0
    assert config.alignment_delay_s(0.05, 1.60) == 0.0
    assert config.alignment_delay_s(0.80, 1.60) == pytest.approx(0.15625)
    assert config.alignment_delay_s(0.30, 1.15) == 0.50


def test_zero_command_suppresses_v43_gait_residual() -> None:
    class _Policy:
        action_history_mode = ACTION_HISTORY_APPLIED_TARGET

        def infer(self, observation: np.ndarray) -> np.ndarray:
            del observation
            return np.ones(12, dtype=np.float32)

    class _MotorSink:
        def __init__(self) -> None:
            self.target = None

        def write(self, action, joint_target_rad, monotonic_time_s) -> None:
            del action, monotonic_time_s
            self.target = joint_target_rad.copy()

    config = JointTargetConfig.from_metadata(
        {
            "action_contract": {
                "mode": "gait_residual",
                "residual_scale": 0.04,
                "zero_command_residual_scale": 0.0,
            },
            "gait_reference": {
                "joint_position_rad": [[0.20] * 12],
            },
        }
    )
    sink = _MotorSink()
    loop = WalkingPolicyLoop(
        _Policy(),
        LevelImuSource(),
        NeutralJointStateSource(),
        sink,
        command=PolicyCommand(forward_m_s=0.0),
        joint_target_config=config,
        gait_clock=GaitClockConfig.from_metadata(
            {
                "gait_clock": {
                    "mode": "speed_scaled",
                    "integrated_phase": True,
                    "park_at_standstill": True,
                    "standstill_deadband_m_s": 0.004,
                    "speed_min_m_s": 0.10,
                    "speed_max_m_s": 0.14,
                    "frequency_min_hz": 1.15,
                    "frequency_max_hz": 1.60,
                    "stride_scale_min": 1.0,
                }
            }
        ),
    )
    started = time.monotonic()
    loop.step(started, started)

    np.testing.assert_allclose(sink.target, config.neutral_array, atol=1e-6)
