"""Offline regression cases for ankle producer timing; no model weights used."""
from types import SimpleNamespace

import numpy as np
import pytest

from drobot_policy_runtime import ankle_transfer as ankle
from drobot_policy_runtime.runtime import AppliedMotorTarget, PolicyCommand


def run_steps(monkeypatch, *, acknowledgement_age=0.0, count=3, slow_first_step=0.0):
    now = [100.0]
    monkeypatch.setattr(ankle.time, "monotonic", lambda: now[0])
    samples = []

    class Stop:
        def is_set(self):
            return len(samples) >= count

        def wait(self, seconds):
            now[0] += seconds
            return self.is_set()

    class Policy:
        def infer(self, observation):
            if not samples:
                now[0] += slow_first_step
            return np.ones(16, dtype=np.float32)

    class Sink:
        def read_applied_target(self):
            # Deliberately reuse the same completed write while the producer
            # runs ahead. Publishing does not constitute a bus completion.
            return AppliedMotorTarget(ankle.NEUTRAL.copy(), 100.0 - acknowledgement_age, 1)

        def write(self, action, target, monotonic_time_s):
            pass

    imu = SimpleNamespace(
        read=lambda: SimpleNamespace(
            angular_velocity_body_rad_s=np.zeros(3),
            projected_gravity_body=np.array([0., 0., -1.]),
            linear_acceleration_body_m_s2=np.array([0., 0., 9.81]),
        ),
    )
    joints = SimpleNamespace(
        read=lambda: SimpleNamespace(
            position_rad=ankle.NEUTRAL.copy(), velocity_rad_s=np.zeros(16),
        ),
    )
    loop = ankle.AnkleTransferLoop(
        Policy(), imu, joints, Sink(), command=PolicyCommand(forward_m_s=.018),
        control_hz=60., initial_target_rad=ankle.NEUTRAL.copy(),
        joint_target_config=ankle.AnkleTargetConfig(), step_observer=samples.append,
    )
    loop.run(duration_s=1., stop_event=Stop(), normal_stop_ramp_s=1., normal_stop_hold_s=1.)
    return samples


def test_reused_completed_target_allows_elapsed_progress_without_exceeding_packet_cap(monkeypatch):
    samples = run_steps(monkeypatch)
    assert samples[2].target_elapsed_s == pytest.approx(2 / 60)
    # Hip-abduction residual is far enough from neutral to expose the former
    # one-degree ceiling, while the physical packet cap still stays at two.
    delta = np.degrees(samples[2].rate_limited_target_rad[0] - ankle.NEUTRAL[0])
    assert delta == pytest.approx(2., abs=1e-5)
    for sample in samples:
        assert sample.applied_motor_target.sequence == 1
        np.testing.assert_allclose(sample.observation[46:62], 0., atol=1e-6)
        assert np.max(np.abs(sample.rate_limited_target_rad - ankle.NEUTRAL)) <= np.radians(2) + 1e-6


def test_old_completed_target_still_caps_every_joint_at_two_degrees(monkeypatch):
    samples = run_steps(monkeypatch, acknowledgement_age=.08, count=1)
    assert samples[0].target_elapsed_s == pytest.approx(.08)
    assert np.max(np.abs(samples[0].rate_limited_target_rad - ankle.NEUTRAL)) <= np.radians(2) + 1e-6


def test_stale_bus_history_still_stops_the_policy(monkeypatch):
    with pytest.raises(RuntimeError, match="Completed ankle motor target is stale"):
        run_steps(monkeypatch, acknowledgement_age=.121, count=1)


def test_overrun_skips_expired_slots_without_an_immediate_catch_up_step(monkeypatch):
    samples = run_steps(monkeypatch, count=2, slow_first_step=.04)
    assert samples[1].elapsed_s == pytest.approx(.05)
    assert samples[1].missed_deadlines_total == 2
