"""Offline target-handoff checks using the demo bus, never physical servos."""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drobot_hardware_test_apps import four_leg_control
from drobot_hardware_test_apps.four_leg_control import (
    FourLegDemoBus, FourLegSession, load_dashboard_config,
)
from drobot_policy_runtime.ankle_transfer import AnkleTargetConfig, SERVO_IDS


@pytest.fixture
def session():
    now = [100.]
    manifest = Path(__file__).resolve().parents[3] / "robot-runtime/four-leg.toml"
    dashboard = load_dashboard_config(manifest)
    result = FourLegSession(dashboard, FourLegDemoBus(dashboard), clock=lambda: now[0], persist_calibration=False)
    result.start(start_worker=False)
    for profile in result.profiles:
        result.arm_leg(profile.number, safety_ack=True)
    by_id = {
        motor.servo_id: (profile, motor, result.controllers[profile.number])
        for profile in result.profiles for motor in profile.config.motors
    }
    result.rl_motor_order = tuple(by_id[sid] for sid in SERVO_IDS)
    result.rl_controller = SimpleNamespace(
        active=True, joint_target_config=AnkleTargetConfig(), record_event=lambda *args: None,
    )
    initial = {
        (profile.number, motor.name): controller.targets_deg[motor.name]
        for profile, motor, controller in result.rl_motor_order
    }
    result._prime_rl_output_locked(initial)
    base = result._read_rl_applied_target().joint_target_rad.copy()
    try:
        yield result, now, base
    finally:
        result.rl_controller = None
        result.close()


def test_latest_target_wakes_worker_and_consumes_notification_with_slot(session):
    robot, now, base = session
    robot._apply_rl_targets(np.zeros(16), base + math.radians(.5), now[0])
    robot._apply_rl_targets(np.zeros(16), base + math.radians(1.), now[0])
    assert robot.rl_target_ready.is_set()
    assert robot.rl_replaced_pending_target_count == 1
    now[0] += 1 / 60
    robot.advance_once(elapsed_s=1 / 60)
    assert not robot.rl_target_ready.is_set()
    assert robot.rl_pending_target_deg is None
    assert robot.rl_applied_target_sequence == 2
    # A publication after consumption must keep its own notification.
    robot._apply_rl_targets(np.zeros(16), base + math.radians(1.5), now[0])
    assert robot.rl_target_ready.is_set()


def test_woken_outputs_keep_actual_packet_and_velocity_limits(session):
    robot, now, base = session
    robot._apply_rl_targets(np.zeros(16), base + math.radians(2.), now[0])
    now[0] += .05
    robot.advance_once(elapsed_s=.05)
    first = robot._read_rl_applied_target()
    np.testing.assert_allclose(first.joint_target_rad - base, math.radians(2), atol=1e-6)
    robot._apply_rl_targets(np.zeros(16), first.joint_target_rad + math.radians(2), now[0])
    now[0] += .005
    robot.advance_once(elapsed_s=.005)
    second = robot._read_rl_applied_target()
    rates = np.array(AnkleTargetConfig().target_velocity_limit_rad_s_by_action)
    np.testing.assert_allclose(second.joint_target_rad - first.joint_target_rad, rates * .005, atol=1e-6)


def test_stop_clears_queued_target_and_worker_notification(session):
    robot, now, base = session
    robot._apply_rl_targets(np.zeros(16), base + math.radians(1.), now[0])
    robot._stop_rl_output_locked()
    assert not robot.rl_target_ready.is_set()
    assert robot.rl_pending_target_deg is None
    with pytest.raises(RuntimeError, match="motor output is inactive"):
        robot._apply_rl_targets(np.zeros(16), base, now[0])


def test_worker_handles_fresh_target_before_its_periodic_deadline(session, monkeypatch):
    robot, now, base = session
    events = []
    robot.rl_controller.record_event = lambda kind, payload: events.append((kind, payload))
    monkeypatch.setattr(four_leg_control.time, "monotonic", lambda: now[0])
    robot._apply_rl_targets(np.zeros(16), base + math.radians(1.), now[0])
    advance = robot.advance_once
    starts = []

    def once(*, elapsed_s):
        starts.append(now[0])
        now[0] += .02
        advance(elapsed_s=elapsed_s)
        robot.stop_event.set()
        robot.rl_target_ready.set()

    monkeypatch.setattr(robot, "advance_once", once)
    robot._worker_loop()
    assert starts == [100.]  # The initial periodic deadline was still future.
    cycles = [payload for kind, payload in events if kind == "motor_worker_cycle"]
    assert len(cycles) == 1
    assert cycles[0]["wake_reason"] == "policy_target"
    assert cycles[0]["output_packets"] == 1
