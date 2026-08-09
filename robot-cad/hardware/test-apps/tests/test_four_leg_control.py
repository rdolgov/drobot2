from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from drobot_leg_testbed.model import load_calibration, save_calibration

from drobot_hardware_test_apps.four_leg_control import (
    FourLegDemoBus,
    FourLegSession,
    load_dashboard_config,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self) -> None:
        self.now = 20.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _session() -> tuple[FourLegSession, FourLegDemoBus, FakeClock]:
    dashboard = load_dashboard_config(ROOT / "config" / "four-leg.toml")
    bus = FourLegDemoBus(dashboard)
    clock = FakeClock()
    session = FourLegSession(
        dashboard,
        bus,
        clock=clock,
        persist_calibration=False,
    )
    session.start(start_worker=False)
    return session, bus, clock


def test_manifest_loads_verified_ids_and_directions() -> None:
    dashboard = load_dashboard_config(ROOT / "config" / "four-leg.toml")

    ids = [
        motor.servo_id for profile in dashboard.legs for motor in profile.config.motors
    ]
    directions = [
        tuple(motor.direction for motor in profile.config.motors)
        for profile in dashboard.legs
    ]

    assert ids == list(range(1, 13))
    assert directions == [(-1, 1, 1), (1, 1, 1), (-1, -1, -1), (1, 1, 1)]
    assert dashboard.bus.torque_limit == 300
    assert dashboard.bus.speed == 350


def test_start_requires_all_motors_and_disarms_every_id() -> None:
    session, bus, _clock = _session()
    try:
        state = session.snapshot()

        assert state["summary"]["online_count"] == 12
        assert state["summary"]["armed_count"] == 0
        assert state["summary"]["health"] == "nominal"
        assert bus.torque == set()
    finally:
        session.close()


def test_joint_arm_requires_ack_and_large_target_is_ramped() -> None:
    session, _bus, _clock = _session()
    try:
        with pytest.raises(ValueError, match="support"):
            session.arm(2, 1, safety_ack=False)

        session.arm(2, 1, safety_ack=True)
        session.set_target(2, 1, 20.0)
        session.advance_once()
        motor = session.snapshot()["legs"][1]["motors"][0]

        assert motor["desired_deg"] == 20.0
        assert motor["commanded_deg"] == pytest.approx(1.5)
        assert motor["armed"] is True
    finally:
        session.close()


def test_same_joint_names_on_different_legs_do_not_collide() -> None:
    session, _bus, _clock = _session()
    try:
        session.arm(2, "hip_flexion", safety_ack=True)
        session.arm(3, "hip_flexion", safety_ack=True)
        session.set_target(2, "hip_flexion", 15.0)
        session.set_target(3, "hip_flexion", -15.0)
        session.advance_once()
        state = session.snapshot()

        assert state["legs"][1]["motors"][1]["commanded_deg"] == pytest.approx(1.5)
        assert state["legs"][2]["motors"][1]["commanded_deg"] == pytest.approx(-1.5)
    finally:
        session.close()


def test_leg_arm_zero_and_disarm_are_scoped() -> None:
    session, _bus, _clock = _session()
    try:
        session.arm_leg(1, safety_ack=True)
        session.arm(2, 1, safety_ack=True)
        session.set_target(1, 1, 15.0)
        session.zero_armed_leg(1)
        session.disarm_leg(1)
        state = session.snapshot()

        assert state["legs"][0]["armed_count"] == 0
        assert state["legs"][1]["armed_count"] == 1
        assert state["summary"]["armed_count"] == 1
    finally:
        session.close()


def test_center_all_requires_confirmation_and_targets_calibrated_zero() -> None:
    session, bus, _clock = _session()
    try:
        with pytest.raises(ValueError, match="support"):
            session.center_all(safety_ack=False, confirmation="CENTER ALL 12")
        with pytest.raises(ValueError, match="confirmation"):
            session.center_all(safety_ack=True, confirmation="")

        session.center_all(safety_ack=True, confirmation="CENTER ALL 12")
        state = session.snapshot()

        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))
        assert all(
            motor["desired_deg"] == 0.0
            for leg in state["legs"]
            for motor in leg["motors"]
        )
        assert "calibrated zero" in state["last_event"].lower()
    finally:
        session.close()


def test_capture_zero_all_requires_disarm_and_saves_backups(tmp_path: Path) -> None:
    source = load_dashboard_config(ROOT / "config" / "four-leg.toml")
    profiles = []
    for profile in source.legs:
        calibration_path = tmp_path / profile.calibration_path.name
        save_calibration(profile.calibration, calibration_path)
        profiles.append(replace(profile, calibration_path=calibration_path))
    dashboard = replace(source, legs=tuple(profiles))
    bus = FourLegDemoBus(dashboard)
    session = FourLegSession(dashboard, bus, persist_calibration=True)
    session.start(start_worker=False)
    try:
        with pytest.raises(ValueError, match="support"):
            session.capture_zero_all(
                safety_ack=False,
                confirmation="CAPTURE ZERO ALL",
            )
        session.arm(1, 1, safety_ack=True)
        with pytest.raises(RuntimeError, match="Disarm all"):
            session.capture_zero_all(
                safety_ack=True,
                confirmation="CAPTURE ZERO ALL",
            )
        session.disarm_all()

        for servo_id in range(1, 13):
            bus.positions[servo_id] += servo_id
        expected = dict(bus.positions)
        session.capture_zero_all(
            safety_ack=True,
            confirmation="CAPTURE ZERO ALL",
        )

        state = session.snapshot()
        assert state["summary"]["armed_count"] == 0
        assert all(
            motor["measured_deg"] == pytest.approx(0.0)
            for leg in state["legs"]
            for motor in leg["motors"]
        )
        for profile in session.profiles:
            saved = load_calibration(profile.calibration_path, profile.config)
            assert {motor.servo_id: motor.center_tick for motor in saved.motors} == {
                motor.servo_id: expected[motor.servo_id] for motor in saved.motors
            }
        backups = list((tmp_path / "backups").glob("*/*.json"))
        assert len(backups) == 4
    finally:
        session.close()


def test_lost_browser_heartbeat_disarms_all_twelve() -> None:
    session, bus, clock = _session()
    try:
        session.arm_leg(4, safety_ack=True)
        clock.advance(4.0)
        session.advance_once()
        state = session.snapshot()

        assert state["any_armed"] is False
        assert bus.torque == set()
        assert "heartbeat lost" in state["last_event"].lower()
    finally:
        session.close()


def test_monitoring_thresholds_create_visible_warnings() -> None:
    session, bus, _clock = _session()
    try:
        bus.voltage_v[12] = 10.4
        bus.temperature_c[8] = 60
        bus.current_ma[4] = 3100.0
        state = session.snapshot()

        assert state["summary"]["health"] == "warning"
        assert any("Low servo voltage" in item for item in state["summary"]["warnings"])
        assert any("temperature" in item for item in state["summary"]["warnings"])
        assert any("Leg 2" in item for item in state["summary"]["warnings"])
    finally:
        session.close()
