from __future__ import annotations

import importlib
import math
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from drobot_leg_testbed.model import load_calibration, save_calibration

from drobot_hardware_test_apps.crawl_gait import quasistatic_crawl_degrees
from drobot_hardware_test_apps.four_leg_control import (
    FourLegDemoBus,
    FourLegHTTPServer,
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
    assert [profile.corner for profile in dashboard.legs] == [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ]
    assert dashboard.server.ramp_rate_deg_s == 45.0
    assert dashboard.crawl.period_s == 20.0
    assert dashboard.crawl.cycles == 4
    assert dashboard.crawl.stride_m == pytest.approx(0.025)
    assert dashboard.crawl.lift_m == pytest.approx(0.013)
    assert dashboard.crawl.stance_down_m == pytest.approx(0.305)
    assert dashboard.crawl.stance_fore_aft_m == pytest.approx(0.035)
    assert dashboard.crawl.weight_shift_forward_m == pytest.approx(0.020)


def test_dashboard_crawl_matches_isaac_reference_and_motor_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = load_dashboard_config(ROOT / "config" / "four-leg.toml")
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    runtime = importlib.import_module("simulation.isaac._quadruped_runtime")
    config = dashboard.crawl

    for sample in range(81):
        gait_time_s = config.period_s * sample / 80
        dashboard_pose, _state = quasistatic_crawl_degrees(
            gait_time_s,
            period_s=config.period_s,
            stride_m=config.stride_m,
            lift_m=config.lift_m,
            weight_shift_forward_m=config.weight_shift_forward_m,
            weight_shift_lateral_m=config.weight_shift_lateral_m,
            down_m=config.stance_down_m,
            fore_aft_m=config.stance_fore_aft_m,
            abduction_deg=config.abduction_deg,
        )
        isaac_pose, _isaac_state = runtime.quasistatic_crawl_by_name(
            gait_time_s,
            period_s=config.period_s,
            stride_m=config.stride_m,
            lift_m=config.lift_m,
            weight_shift_forward_m=config.weight_shift_forward_m,
            weight_shift_lateral_m=config.weight_shift_lateral_m,
            down_m=config.stance_down_m,
            fore_aft_m=config.stance_fore_aft_m,
            abduction_deg=config.abduction_deg,
        )
        for profile in dashboard.legs:
            for motor in profile.config.motors:
                degrees = dashboard_pose[(profile.corner, motor.name)]
                assert degrees == pytest.approx(
                    math.degrees(isaac_pose[f"{profile.corner}_{motor.name}"])
                )
                assert motor.min_deg <= degrees <= motor.max_deg


def test_start_requires_all_motors_and_disarms_every_id() -> None:
    session, bus, _clock = _session()
    try:
        state = session.snapshot()

        assert state["summary"]["online_count"] == 12
        assert state["summary"]["armed_count"] == 0
        assert state["summary"]["health"] == "nominal"
        assert state["runtime"] == {"mode": "demo", "port": None}
        assert bus.torque == set()
    finally:
        session.close()


def test_dashboard_port_is_exclusive() -> None:
    first = FourLegHTTPServer(("127.0.0.1", 0), None, "first")
    try:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            assert (
                first.socket.getsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_EXCLUSIVEADDRUSE,
                )
                == 1
            )
        with pytest.raises(OSError):
            FourLegHTTPServer(
                ("127.0.0.1", first.server_port),
                None,
                "second",
            )
    finally:
        first.server_close()


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
        assert motor["commanded_deg"] == pytest.approx(2.25)
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

        assert state["legs"][1]["motors"][1]["commanded_deg"] == pytest.approx(2.25)
        assert state["legs"][2]["motors"][1]["commanded_deg"] == pytest.approx(-2.25)
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


def test_crawl_requires_guarded_disarmed_start_and_manual_motion_is_locked() -> None:
    session, bus, _clock = _session()
    try:
        with pytest.raises(ValueError, match="corner map"):
            session.start_crawl_forward(
                safety_ack=False,
                confirmation="WALK FORWARD",
            )
        with pytest.raises(ValueError, match="confirmation"):
            session.start_crawl_forward(safety_ack=True, confirmation="")

        session.arm(1, 1, safety_ack=True)
        with pytest.raises(RuntimeError, match="Disarm all 12"):
            session.start_crawl_forward(
                safety_ack=True,
                confirmation="WALK FORWARD",
            )
        session.disarm_all()

        session.start_crawl_forward(
            safety_ack=True,
            confirmation="WALK FORWARD",
        )
        state = session.snapshot()

        assert state["crawl"]["active"] is True
        assert state["crawl"]["stage"] == "preparing"
        assert state["crawl"]["corner_map"] == {
            "1": "front_left",
            "2": "front_right",
            "3": "rear_left",
            "4": "rear_right",
        }
        assert state["crawl"]["duration_s"] == 80.0
        assert state["crawl"]["stride_mm"] == pytest.approx(25.0)
        assert state["crawl"]["lift_mm"] == pytest.approx(13.0)
        assert state["crawl"]["stance_down_mm"] == pytest.approx(305.0)
        assert state["crawl"]["stance_fore_aft_mm"] == pytest.approx(35.0)
        assert state["crawl"]["weight_shift_forward_mm"] == pytest.approx(20.0)
        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))
        with pytest.raises(RuntimeError, match="Stop the active crawl"):
            session.set_target(1, 1, 5.0)

        session.stop_crawl()
        state = session.snapshot()
        assert state["crawl"]["active"] is False
        assert state["summary"]["armed_count"] == 0
        assert bus.torque == set()
    finally:
        session.close()


def test_crawl_preflight_rejects_off_center_pose_and_telemetry_warning() -> None:
    session, bus, _clock = _session()
    try:
        bus.positions[1] += 500
        with pytest.raises(RuntimeError, match="Center the robot"):
            session.start_crawl_forward(
                safety_ack=True,
                confirmation="WALK FORWARD",
            )

        bus.positions[1] -= 500
        bus.voltage_v[12] = 10.4
        with pytest.raises(RuntimeError, match="telemetry warnings"):
            session.start_crawl_forward(
                safety_ack=True,
                confirmation="WALK FORWARD",
            )
    finally:
        session.close()


def test_crawl_completes_configured_cycles_and_holds_stance() -> None:
    session, bus, clock = _session()
    try:
        session.start_crawl_forward(
            safety_ack=True,
            confirmation="WALK FORWARD",
        )
        for tick in range(2000):
            if tick % 10 == 0:
                session.heartbeat()
            clock.advance(0.05)
            session.advance_once()
            if session.crawl_stage == "complete":
                break

        state = session.snapshot()
        assert tick < 1999
        assert state["crawl"]["stage"] == "complete"
        assert state["crawl"]["phase"] == "holding_stance"
        assert state["crawl"]["progress"] == 1.0
        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))
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
