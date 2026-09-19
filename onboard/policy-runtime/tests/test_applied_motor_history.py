from __future__ import annotations

import numpy as np
import pytest

from drobot_policy_runtime.contract import JointTargetConfig
from drobot_policy_runtime.policy import ACTION_HISTORY_APPLIED_TARGET
from drobot_policy_runtime.runtime import AppliedMotorTarget, WalkingPolicyLoop
from drobot_policy_runtime.sources import LevelImuSource, NeutralJointStateSource


class Policy:
    action_history_mode = ACTION_HISTORY_APPLIED_TARGET


class AsyncSink:
    acknowledgement = None

    def read_applied_target(self):
        return self.acknowledgement

    def write(self, action, joint_target_rad, monotonic_time_s):
        # Enqueueing a target alone must not claim it has reached the motor bus.
        self.queued = joint_target_rad.copy()


def loop_with(sink):
    config = JointTargetConfig(neutral_joint_position_rad=(0.0,) * 12)
    return WalkingPolicyLoop(
        Policy(), LevelImuSource(), NeutralJointStateSource(), sink,
        joint_target_config=config, initial_target_rad=np.zeros(12, dtype=np.float32),
    )


def test_applied_history_uses_completed_bus_target_not_latest_queue():
    sink = AsyncSink()
    loop = loop_with(sink)
    loop.previous_target_rad[:] = 0.10  # Runtime has queued a newer target.
    sink.acknowledgement = AppliedMotorTarget(np.full(12, 0.04), 100.0, 4)
    observation = loop._observation(101.0)[0]
    np.testing.assert_allclose(
        observation[-12:], 0.04 / loop.joint_target_config.action_scale_array
    )
    assert loop._applied_motor_target.sequence == 4


def test_unacknowledged_output_preserves_armed_initial_history():
    sink = AsyncSink()
    loop = loop_with(sink)
    loop.previous_target_rad[:] = 0.10
    np.testing.assert_allclose(loop._observation(101.0)[0][-12:], 0.0)


def test_unchanged_acknowledgement_is_valid_but_regression_is_rejected():
    sink = AsyncSink()
    loop = loop_with(sink)
    sink.acknowledgement = AppliedMotorTarget(np.full(12, 0.04), 100.0, 4)
    loop._observation(101.0)
    loop.previous_target_rad[:] = 0.10
    loop._observation(102.0)
    assert loop._applied_motor_target.sequence == 4
    sink.acknowledgement = AppliedMotorTarget(np.full(12, 0.03), 99.0, 3)
    with pytest.raises(RuntimeError, match="moved backwards"):
        loop._observation(103.0)


def test_bad_acknowledgement_cannot_enter_actor_history():
    sink = AsyncSink()
    loop = loop_with(sink)
    sink.acknowledgement = AppliedMotorTarget(np.full(12, np.nan), 100.0, 1)
    with pytest.raises(RuntimeError, match="12 finite"):
        loop._observation(101.0)


def test_legacy_sink_retains_synchronous_target_history():
    class SynchronousSink:
        def write(self, action, joint_target_rad, monotonic_time_s):
            pass

    loop = loop_with(SynchronousSink())
    loop.previous_target_rad[:] = 0.04
    np.testing.assert_allclose(
        loop._observation(101.0)[0][-12:],
        0.04 / loop.joint_target_config.action_scale_array,
    )


def test_held_motor_output_does_not_accumulate_multiple_queued_steps(monkeypatch):
    from drobot_policy_runtime import runtime

    class MovingPolicy(Policy):
        def infer(self, observation):
            return np.ones(12, dtype=np.float32)

    now = [100.0]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    sink = AsyncSink()
    loop = WalkingPolicyLoop(
        MovingPolicy(), LevelImuSource(), NeutralJointStateSource(), sink,
        joint_target_config=JointTargetConfig(neutral_joint_position_rad=(0.0,) * 12),
        initial_target_rad=np.zeros(12, dtype=np.float32),
    )
    sink.acknowledgement = AppliedMotorTarget(np.zeros(12), 99.95, 0)
    loop.step(99.0, now[0])
    first = sink.queued.copy()
    now[0] += 0.02
    loop.step(99.0, now[0])
    np.testing.assert_allclose(sink.queued, first)
    sink.acknowledgement = AppliedMotorTarget(first.copy(), 100.02, 1)
    now[0] += 0.05
    loop.step(99.0, now[0])
    np.testing.assert_allclose(sink.queued - first, first, atol=1e-6)


def test_producer_budget_does_not_subtract_queue_delay_twice(monkeypatch):
    from drobot_policy_runtime import runtime

    class MovingPolicy(Policy):
        def infer(self, observation):
            return np.ones(12, dtype=np.float32)

    now = [100.02]
    monkeypatch.setattr(runtime.time, "monotonic", lambda: now[0])
    sink = AsyncSink()
    captured = []
    config = JointTargetConfig(
        neutral_joint_position_rad=(0.0,) * 12,
        target_velocity_limit_rad_s=1.0,
        max_target_step_rad=0.2,
    )
    loop = WalkingPolicyLoop(
        MovingPolicy(), LevelImuSource(), NeutralJointStateSource(), sink,
        joint_target_config=config, initial_target_rad=np.zeros(12, dtype=np.float32),
        step_observer=captured.append,
    )
    sink.acknowledgement = AppliedMotorTarget(np.zeros(12), 100.0, 0)
    loop.step(99.0, now[0])
    sink.acknowledgement = AppliedMotorTarget(sink.queued.copy(), 100.035, 1)
    now[0] = 100.04
    loop.step(99.0, now[0])
    # Actual worker remains authoritative; producer uses20ms production budget,
    # not just5ms age since a queued write happened to finish.
    assert captured[-1].target_elapsed_s == pytest.approx(0.02)
    np.testing.assert_allclose(sink.queued, 0.04, atol=1e-6)
