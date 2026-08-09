from __future__ import annotations

import importlib
import math
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from drobot_leg_testbed.model import degrees_to_raw, load_calibration, save_calibration

from drobot_hardware_test_apps.crawl_gait import (
    COORDINATED_PUSH_SWING_ORDER,
    coordinated_push_crawl_degrees,
    coordinated_push_stance_degrees,
    outward_bent_crawl_stance_degrees,
)
from drobot_hardware_test_apps.four_leg_control import (
    FourLegDemoBus,
    FourLegHTTPServer,
    FourLegSession,
    load_dashboard_config,
)

HARDWARE_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[4]
MANIFEST = HARDWARE_ROOT / "robot-runtime" / "four-leg.toml"


class FakeClock:
    def __init__(self) -> None:
        self.now = 20.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _session() -> tuple[FourLegSession, FourLegDemoBus, FakeClock]:
    dashboard = load_dashboard_config(MANIFEST)
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
    dashboard = load_dashboard_config(MANIFEST)

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
    assert dashboard.crawl.cycles == 1
    assert dashboard.crawl.stride_m == pytest.approx(0.035)
    assert dashboard.crawl.lift_m == pytest.approx(0.016)
    assert dashboard.crawl.stance_down_m == pytest.approx(0.272960722)
    assert dashboard.crawl.stance_fore_aft_m == pytest.approx(0.113064033)
    assert dashboard.crawl.abduction_deg == pytest.approx(15.0)
    assert dashboard.crawl.weight_shift_forward_m == pytest.approx(0.016)
    assert dashboard.crawl.weight_shift_lateral_m == pytest.approx(0.012)


def test_dashboard_crawl_matches_isaac_reference_and_motor_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = load_dashboard_config(MANIFEST)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    runtime = importlib.import_module("simulation.isaac._quadruped_runtime")
    config = dashboard.crawl

    for sample in range(81):
        gait_time_s = config.period_s * sample / 80
        dashboard_pose, _state = coordinated_push_crawl_degrees(
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
        isaac_pose, _isaac_state = runtime.coordinated_push_crawl_by_name(
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


def test_coordinated_push_is_periodic_and_moves_every_planted_foot() -> None:
    config = load_dashboard_config(MANIFEST).crawl
    expected_order = ("front_left", "rear_right", "front_right", "rear_left")
    assert COORDINATED_PUSH_SWING_ORDER == expected_order

    start_pose, _start_state = coordinated_push_crawl_degrees(
        0.0,
        period_s=config.period_s,
        stride_m=config.stride_m,
        lift_m=config.lift_m,
        weight_shift_forward_m=config.weight_shift_forward_m,
        weight_shift_lateral_m=config.weight_shift_lateral_m,
        down_m=config.stance_down_m,
        fore_aft_m=config.stance_fore_aft_m,
        abduction_deg=config.abduction_deg,
    )
    end_pose, _end_state = coordinated_push_crawl_degrees(
        config.period_s,
        period_s=config.period_s,
        stride_m=config.stride_m,
        lift_m=config.lift_m,
        weight_shift_forward_m=config.weight_shift_forward_m,
        weight_shift_lateral_m=config.weight_shift_lateral_m,
        down_m=config.stance_down_m,
        fore_aft_m=config.stance_fore_aft_m,
        abduction_deg=config.abduction_deg,
    )
    assert end_pose == pytest.approx(start_pose)

    for step_index, expected_swing in enumerate(expected_order):
        step_start = step_index / len(expected_order)
        pose, state = coordinated_push_crawl_degrees(
            config.period_s * (step_start + 0.25 * 0.40),
            period_s=config.period_s,
            stride_m=config.stride_m,
            lift_m=config.lift_m,
            weight_shift_forward_m=config.weight_shift_forward_m,
            weight_shift_lateral_m=config.weight_shift_lateral_m,
            down_m=config.stance_down_m,
            fore_aft_m=config.stance_fore_aft_m,
            abduction_deg=config.abduction_deg,
        )
        assert state["swing_corner"] == expected_swing
        assert state["phase"] == "swing_push"
        assert pose[("front_left", "knee")] < 0.0
        assert pose[("front_right", "knee")] < 0.0
        assert pose[("rear_left", "knee")] > 0.0
        assert pose[("rear_right", "knee")] > 0.0

    _pose, push_start = coordinated_push_crawl_degrees(
        config.period_s * 0.25 * 0.220001,
        period_s=config.period_s,
        stride_m=config.stride_m,
        lift_m=config.lift_m,
        weight_shift_forward_m=config.weight_shift_forward_m,
        weight_shift_lateral_m=config.weight_shift_lateral_m,
        down_m=config.stance_down_m,
        fore_aft_m=config.stance_fore_aft_m,
        abduction_deg=config.abduction_deg,
    )
    _pose, push_end = coordinated_push_crawl_degrees(
        config.period_s * 0.25 * 0.579999,
        period_s=config.period_s,
        stride_m=config.stride_m,
        lift_m=config.lift_m,
        weight_shift_forward_m=config.weight_shift_forward_m,
        weight_shift_lateral_m=config.weight_shift_lateral_m,
        down_m=config.stance_down_m,
        fore_aft_m=config.stance_fore_aft_m,
        abduction_deg=config.abduction_deg,
    )
    assert push_start["phase"] == "swing_push"
    assert push_end["phase"] == "swing_push"
    deltas = {
        corner: push_end["foot_offsets_m"][corner]
        - push_start["foot_offsets_m"][corner]
        for corner in expected_order
    }
    assert deltas["front_left"] == pytest.approx(config.stride_m, abs=1e-9)
    assert deltas["rear_right"] == pytest.approx(-config.stride_m / 2.0, abs=1e-9)
    assert deltas["front_right"] == pytest.approx(-config.stride_m / 4.0, abs=1e-9)
    assert deltas["rear_left"] == pytest.approx(-config.stride_m / 4.0, abs=1e-9)


def test_uniform_ready_stance_is_same_direction_and_45_degree_bend() -> None:
    pose = coordinated_push_stance_degrees(
        down_m=0.295447,
        abduction_deg=0.0,
    )
    for joint in ("hip_abduction", "hip_flexion", "knee"):
        values = [pose[(corner, joint)] for corner in COORDINATED_PUSH_SWING_ORDER]
        assert values == pytest.approx([values[0]] * 4)
    assert pose[("front_left", "hip_flexion")] == pytest.approx(-22.5, abs=0.01)
    assert pose[("front_left", "knee")] == pytest.approx(45.0, abs=0.01)


def test_walking_stance_opens_front_and_rear_legs_away_from_body() -> None:
    config = load_dashboard_config(MANIFEST).crawl
    pose = outward_bent_crawl_stance_degrees(
        down_m=config.stance_down_m,
        fore_aft_m=config.stance_fore_aft_m,
        abduction_deg=config.abduction_deg,
    )

    assert pose[("front_left", "hip_abduction")] == pytest.approx(
        -config.abduction_deg
    )
    assert pose[("rear_left", "hip_abduction")] == pytest.approx(
        -config.abduction_deg
    )
    assert pose[("front_right", "hip_abduction")] == pytest.approx(
        config.abduction_deg
    )
    assert pose[("rear_right", "hip_abduction")] == pytest.approx(
        config.abduction_deg
    )
    assert pose[("front_left", "knee")] < 0.0
    assert pose[("front_right", "knee")] < 0.0
    assert pose[("rear_left", "knee")] > 0.0
    assert pose[("rear_right", "knee")] > 0.0
    assert pose[("front_left", "hip_flexion")] > 0.0
    assert pose[("rear_left", "hip_flexion")] < 0.0
    assert pose[("front_left", "hip_flexion")] == pytest.approx(45.0, abs=0.01)
    assert pose[("front_left", "knee")] == pytest.approx(-45.0, abs=0.01)
    assert pose[("rear_left", "hip_flexion")] == pytest.approx(-45.0, abs=0.01)
    assert pose[("rear_left", "knee")] == pytest.approx(45.0, abs=0.01)


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


def test_walk_stance_arms_missing_motors_and_targets_mirrored_angles() -> None:
    session, bus, _clock = _session()
    try:
        with pytest.raises(ValueError, match="support"):
            session.set_crawl_stance(
                safety_ack=False,
                confirmation="SET WIDE WALK STANCE",
            )
        session.arm(1, 1, safety_ack=True)
        session.set_crawl_stance(
            safety_ack=True,
            confirmation="SET WIDE WALK STANCE",
        )
        state = session.snapshot()

        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))
        desired_by_leg = []
        for leg in state["legs"]:
            desired = {motor["name"]: motor["desired_deg"] for motor in leg["motors"]}
            expected_abduction = -15.0 if leg["corner"].endswith("_left") else 15.0
            assert desired["hip_abduction"] == pytest.approx(expected_abduction)
            expected_knee = 45.0 if leg["corner"].startswith("rear_") else -45.0
            assert desired["knee"] == pytest.approx(expected_knee)
            desired_by_leg.append(desired)
        assert desired_by_leg[0]["hip_flexion"] > 0.0
        assert desired_by_leg[1]["hip_flexion"] > 0.0
        assert desired_by_leg[2]["hip_flexion"] < 0.0
        assert desired_by_leg[3]["hip_flexion"] < 0.0
        leg_1 = session.profiles[0]
        leg_1_knee = leg_1.config.motor("knee")
        assert degrees_to_raw(
            -45.0,
            leg_1_knee,
            leg_1.calibration.motor(leg_1_knee),
        ) == 493
    finally:
        session.close()


def test_crawl_requires_guarded_disarmed_start_and_manual_motion_is_locked() -> None:
    session, bus, _clock = _session()
    try:
        with pytest.raises(ValueError, match="supported"):
            session.start_crawl_forward(
                safety_ack=False,
                confirmation="TEST COORDINATED MOTION",
            )
        with pytest.raises(ValueError, match="confirmation"):
            session.start_crawl_forward(safety_ack=True, confirmation="")

        session.arm(1, 1, safety_ack=True)
        with pytest.raises(RuntimeError, match="Disarm all 12"):
            session.start_crawl_forward(
                safety_ack=True,
                confirmation="TEST COORDINATED MOTION",
            )
        session.disarm_all()

        session.start_crawl_forward(
            safety_ack=True,
            confirmation="TEST COORDINATED MOTION",
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
        assert state["crawl"]["duration_s"] == 20.0
        assert state["crawl"]["pattern"] == "coordinated_support_push"
        assert state["crawl"]["supported_test_only"] is True
        assert state["crawl"]["stride_mm"] == pytest.approx(35.0)
        assert state["crawl"]["lift_mm"] == pytest.approx(16.0)
        assert state["crawl"]["stance_down_mm"] == pytest.approx(272.960722)
        assert state["crawl"]["stance_fore_aft_mm"] == pytest.approx(113.064033)
        assert state["crawl"]["abduction_deg"] == pytest.approx(15.0)
        assert state["crawl"]["weight_shift_forward_mm"] == pytest.approx(16.0)
        assert state["crawl"]["weight_shift_lateral_mm"] == pytest.approx(12.0)
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
                confirmation="TEST COORDINATED MOTION",
            )

        bus.positions[1] -= 500
        bus.voltage_v[12] = 10.4
        with pytest.raises(RuntimeError, match="telemetry warnings"):
            session.start_crawl_forward(
                safety_ack=True,
                confirmation="TEST COORDINATED MOTION",
            )
    finally:
        session.close()


def test_crawl_completes_configured_cycles_and_holds_stance() -> None:
    session, bus, clock = _session()
    try:
        session.start_crawl_forward(
            safety_ack=True,
            confirmation="TEST COORDINATED MOTION",
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
        assert state["crawl"]["phase"] == "holding_wide_mirrored_stance"
        assert state["crawl"]["progress"] == 1.0
        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))
    finally:
        session.close()


def test_capture_zero_all_requires_disarm_and_saves_backups(tmp_path: Path) -> None:
    source = load_dashboard_config(MANIFEST)
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
