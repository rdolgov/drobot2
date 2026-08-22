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
    DIAGONAL_PAIR_SWING_ORDER,
    DISTRIBUTED_PUSH_SWING_ORDER,
    coordinated_push_crawl_degrees,
    coordinated_push_stance_degrees,
    diagonal_pair_gait_degrees,
    distributed_push_crawl_degrees,
    hardware_joint_sequence_degrees,
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
    assert dashboard.bus.torque_limit == 900
    assert dashboard.bus.speed == 3400
    assert dashboard.bus.acceleration == 254
    assert dashboard.bus.max_command_step_deg == pytest.approx(15.0)
    assert [profile.corner for profile in dashboard.legs] == [
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    ]
    assert dashboard.server.ramp_rate_deg_s == 270.0
    assert dashboard.server.heartbeat_timeout_s == 20.0
    assert dashboard.monitoring.voltage_warning_low_v == pytest.approx(11.0)
    assert dashboard.monitoring.voltage_spread_warning_v == pytest.approx(0.3)
    assert dashboard.monitoring.voltage_sag_warning_v == pytest.approx(0.6)
    assert dashboard.monitoring.temperature_warning_c == 55
    assert dashboard.monitoring.leg_current_warning_ma == pytest.approx(2500.0)
    assert dashboard.monitoring.motor_stall_current_warning_ma == pytest.approx(
        1200.0
    )
    assert dashboard.monitoring.stall_tracking_error_deg == pytest.approx(8.0)
    assert dashboard.monitoring.stall_speed_raw_max == 20
    assert dashboard.monitoring.battery_series_cells == 3
    assert dashboard.crawl.period_s == pytest.approx(4.0)
    assert dashboard.crawl.stride_m == pytest.approx(0.096)
    assert dashboard.crawl.lift_m == pytest.approx(0.035)
    assert dashboard.crawl.support_extension_m == pytest.approx(0.0)
    assert dashboard.crawl.stance_down_m == pytest.approx(0.329341447)
    assert dashboard.crawl.stance_fore_aft_m == pytest.approx(0.080)
    assert dashboard.crawl.abduction_deg == pytest.approx(0.0)
    assert dashboard.crawl.weight_shift_forward_m == pytest.approx(0.006)
    assert dashboard.crawl.weight_shift_lateral_m == pytest.approx(0.0)


def test_dashboard_crawl_matches_isaac_reference_and_motor_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dashboard = load_dashboard_config(MANIFEST)
    monkeypatch.syspath_prepend(str(REPO_ROOT))
    runtime = importlib.import_module("simulation.isaac._quadruped_runtime")
    config = dashboard.crawl

    for sample in range(81):
        gait_time_s = config.period_s * sample / 80
        dashboard_pose, _state = distributed_push_crawl_degrees(
            gait_time_s,
            period_s=config.period_s,
            stride_m=config.stride_m,
            lift_m=config.lift_m,
            support_extension_m=config.support_extension_m,
            weight_shift_forward_m=config.weight_shift_forward_m,
            weight_shift_lateral_m=config.weight_shift_lateral_m,
            down_m=config.stance_down_m,
            fore_aft_m=config.stance_fore_aft_m,
            abduction_deg=config.abduction_deg,
        )
        isaac_pose, _isaac_state = runtime.distributed_push_crawl_by_name(
            gait_time_s,
            period_s=config.period_s,
            stride_m=config.stride_m,
            lift_m=config.lift_m,
            support_extension_m=config.support_extension_m,
            weight_shift_forward_m=config.weight_shift_forward_m,
            weight_shift_lateral_m=config.weight_shift_lateral_m,
            down_m=config.stance_down_m,
            fore_aft_m=config.stance_fore_aft_m,
            abduction_deg=config.abduction_deg,
        )
        assert _state["flat_sole_nominal_support"] is True
        assert _state["support_extension_holds_contact_x"] is False
        assert _state["flat_sole_support"] is True
        assert _state["all_planted_soles_flat"] is True
        for corner, sole_pitch_deg in _state["sole_pitch_deg"].items():
            swing_is_airborne = (
                corner == _state["swing_corner"]
                and _state["phase"] in (
                    "lift",
                    "swing",
                    "lower",
                )
            )
            if not swing_is_airborne:
                assert sole_pitch_deg == pytest.approx(0.0, abs=1e-9)
        for profile in dashboard.legs:
            for motor in profile.config.motors:
                degrees = dashboard_pose[(profile.corner, motor.name)]
                assert degrees == pytest.approx(
                    math.degrees(isaac_pose[f"{profile.corner}_{motor.name}"])
                )
                assert motor.min_deg <= degrees <= motor.max_deg

    assert DISTRIBUTED_PUSH_SWING_ORDER == (
        "rear_right",
        "front_right",
        "rear_left",
        "front_left",
    )


def test_diagonal_pair_gait_is_periodic_flat_and_inside_motor_limits() -> None:
    dashboard = load_dashboard_config(MANIFEST)
    config = dashboard.crawl
    poses: list[dict[tuple[str, str], float]] = []
    seen_pairs: list[tuple[str, ...]] = []

    for sample in range(161):
        pose, state = diagonal_pair_gait_degrees(
            config.period_s * sample / 160,
            period_s=config.period_s,
            stride_m=config.stride_m,
            lift_m=config.lift_m,
            support_extension_m=config.support_extension_m,
            down_m=config.stance_down_m,
            fore_aft_m=config.stance_fore_aft_m,
            abduction_deg=config.abduction_deg,
        )
        poses.append(pose)
        swing_pair = tuple(state["swing_pair"])
        if swing_pair not in seen_pairs:
            seen_pairs.append(swing_pair)
        for corner in state["expected_support_corners"]:
            assert state["sole_pitch_deg"][corner] == pytest.approx(0.0, abs=1e-9)
        for profile in dashboard.legs:
            for motor in profile.config.motors:
                degrees = pose[(profile.corner, motor.name)]
                assert motor.min_deg <= degrees <= motor.max_deg

    assert DIAGONAL_PAIR_SWING_ORDER == (
        ("front_left", "rear_right"),
        ("front_right", "rear_left"),
    )
    assert seen_pairs == list(DIAGONAL_PAIR_SWING_ORDER)
    assert poses[-1] == pytest.approx(poses[0])


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


def test_hardware_joint_sequence_runs_requested_targets_twice() -> None:
    config = load_dashboard_config(MANIFEST).crawl

    def pose_at(seconds: float) -> dict[tuple[str, str], float]:
        pose, _state = hardware_joint_sequence_degrees(
            seconds,
            period_s=config.period_s,
            down_m=config.stance_down_m,
            fore_aft_m=config.stance_fore_aft_m,
            abduction_deg=config.abduction_deg,
        )
        return pose

    transition_s = config.period_s / 8.0
    stance = pose_at(0.0)
    first_leg_1 = pose_at(transition_s)
    first_leg_2 = pose_at(2.0 * transition_s)
    first_leg_4 = pose_at(3.0 * transition_s)
    second_stance = pose_at(4.0 * transition_s)
    second_leg_1 = pose_at(5.0 * transition_s)
    second_leg_2 = pose_at(6.0 * transition_s)
    second_leg_4 = pose_at(7.0 * transition_s)
    final_stance = pose_at(config.period_s)

    assert first_leg_1[("front_left", "hip_flexion")] == pytest.approx(90.0)
    assert first_leg_2[("front_right", "hip_flexion")] == pytest.approx(70.0)
    assert first_leg_4[("rear_right", "knee")] == pytest.approx(20.0)
    assert second_stance == pytest.approx(stance)
    assert second_leg_1 == pytest.approx(first_leg_1)
    assert second_leg_2 == pytest.approx(first_leg_2)
    assert second_leg_4 == pytest.approx(first_leg_4)
    assert final_stance == pytest.approx(stance)


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


def test_hardware_ui_can_start_disconnected_with_motion_disabled() -> None:
    dashboard = load_dashboard_config(MANIFEST)
    bus = FourLegDemoBus(dashboard)
    session = FourLegSession(dashboard, bus, persist_calibration=False)
    original_require_motor = bus.require_motor

    def missing_motor(motor):
        if motor.servo_id == 7:
            raise RuntimeError("No motor answered at ID 7")
        return original_require_motor(motor)

    bus.require_motor = missing_motor
    session.start(start_worker=False, allow_disconnected=True)
    try:
        assert bus.torque == set()
        assert session.fault == (
            "Servo bus unavailable; automatic reconnect failed: "
            "No motor answered at ID 7"
        )
        assert session.last_event == (
            "Startup fault; waiting for servo bus reconnect"
        )
    finally:
        session.close()


def test_telemetry_fault_reopens_adapter_and_returns_disarmed() -> None:
    session, bus, _clock = _session()
    try:
        session.arm(1, 1, safety_ack=True)
        assert bus.torque == {1}

        original_status = bus.status
        failed = False
        reopen_count = 0
        bus.port_name = "/dev/ttyACM0"

        def fail_once(motor):
            nonlocal failed
            if not failed:
                failed = True
                raise RuntimeError("USB adapter disconnected")
            return original_status(motor)

        def reopen() -> None:
            nonlocal reopen_count
            reopen_count += 1
            bus.port_name = "/dev/ttyACM1"

        bus.status = fail_once
        bus.reopen = reopen

        state = session.snapshot()

        assert reopen_count == 1
        assert state["summary"]["online_count"] == 12
        assert state["summary"]["armed_count"] == 0
        assert state["runtime"]["port"] == "/dev/ttyACM1"
        assert state["fault"] is None
        assert state["last_event"] == (
            "Servo bus reconnected; all 12 motors online and disarmed"
        )
        assert bus.torque == set()
    finally:
        session.close()


def test_missing_adapter_recovery_returns_one_stable_fault() -> None:
    session, bus, _clock = _session()
    try:
        def unavailable(_motor):
            raise RuntimeError("Serial bus is not open")

        def cannot_reopen() -> None:
            raise RuntimeError("No likely USB serial adapter found")

        bus.status = unavailable
        bus.reopen = cannot_reopen

        expected = (
            "Servo bus unavailable; automatic reconnect failed: "
            "No likely USB serial adapter found"
        )
        with pytest.raises(RuntimeError, match="Servo bus unavailable") as first:
            session.snapshot()
        with pytest.raises(RuntimeError, match="Servo bus unavailable") as second:
            session.snapshot()

        assert str(first.value) == expected
        assert str(second.value) == expected
        assert session.fault == expected
        assert session.last_event == (
            "Telemetry fault; waiting for servo bus reconnect"
        )
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
                confirmation="SET GAIT START STANCE",
            )
        session.arm(1, 1, safety_ack=True)
        session.set_crawl_stance(
            safety_ack=True,
            confirmation="SET GAIT START STANCE",
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
                confirmation="TEST DISTRIBUTED CRAWL",
            )
        with pytest.raises(ValueError, match="confirmation"):
            session.start_crawl_forward(safety_ack=True, confirmation="")

        session.arm(1, 1, safety_ack=True)
        with pytest.raises(RuntimeError, match="Disarm all 12"):
            session.start_crawl_forward(
                safety_ack=True,
                confirmation="TEST DISTRIBUTED CRAWL",
            )
        session.disarm_all()

        session.start_crawl_forward(
            safety_ack=True,
            confirmation="TEST DISTRIBUTED CRAWL",
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
        assert state["crawl"]["run_until_stopped"] is True
        assert state["crawl"]["duration_s"] is None
        assert state["crawl"]["completed_cycles"] == 0
        assert state["crawl"]["pattern"] == "rectangular_flat_support_crawl_v8"
        assert state["crawl"]["supported_test_only"] is True
        assert state["crawl"]["stride_mm"] == pytest.approx(96.0)
        assert state["crawl"]["lift_mm"] == pytest.approx(35.0)
        assert state["crawl"]["support_extension_mm"] == pytest.approx(0.0)
        assert state["crawl"]["stance_down_mm"] == pytest.approx(329.341447)
        assert state["crawl"]["stance_fore_aft_mm"] == pytest.approx(80.0)
        assert state["crawl"]["abduction_deg"] == pytest.approx(0.0)
        assert state["crawl"]["weight_shift_forward_mm"] == pytest.approx(6.0)
        assert state["crawl"]["weight_shift_lateral_mm"] == pytest.approx(0.0)
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


def test_crawl_accepts_off_center_pose_and_allows_telemetry_warning() -> None:
    session, bus, _clock = _session()
    try:
        bus.positions[1] += 500
        bus.voltage_v[12] = 10.4
        session.start_crawl_forward(
            safety_ack=True,
            confirmation="TEST DISTRIBUTED CRAWL",
        )
        state = session.snapshot()
        assert state["crawl"]["active"] is True
        assert state["summary"]["health"] == "warning"
        assert any("Low servo voltage" in item for item in state["summary"]["warnings"])
    finally:
        session.close()


def test_diagonal_pair_mode_has_separate_start_route_and_state() -> None:
    session, bus, _clock = _session()
    try:
        with pytest.raises(ValueError, match="confirmation"):
            session.start_diagonal_pair_forward(
                safety_ack=True,
                confirmation="",
            )

        session.start_diagonal_pair_forward(
            safety_ack=True,
            confirmation="TEST DIAGONAL PAIR GAIT",
        )
        state = session.snapshot()

        assert state["crawl"]["active"] is True
        assert state["crawl"]["mode"] == "diagonal_pair"
        assert state["crawl"]["pattern"] == "diagonal_pair_flat_support_gait_v1"
        assert state["crawl"]["airborne_leg_count"] == 2
        assert state["crawl"]["planted_support_leg_count"] == 2
        assert state["crawl"]["run_until_stopped"] is True
        assert state["crawl"]["duration_s"] is None
        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))

        session.stop_crawl()
        assert session.snapshot()["summary"]["armed_count"] == 0
    finally:
        session.close()


def test_crawl_repeats_until_explicit_stop() -> None:
    session, bus, clock = _session()
    try:
        session.start_crawl_forward(
            safety_ack=True,
            confirmation="TEST DISTRIBUTED CRAWL",
        )
        for tick in range(2000):
            if tick % 10 == 0:
                session.heartbeat()
            clock.advance(0.05)
            session.advance_once()
            if session.snapshot()["crawl"]["completed_cycles"] >= 2:
                break

        state = session.snapshot()
        assert tick < 1999
        assert state["crawl"]["active"] is True
        assert state["crawl"]["stage"] == "walking"
        assert state["crawl"]["completed_cycles"] >= 2
        assert 0.0 <= state["crawl"]["progress"] < 1.0
        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))

        session.stop_crawl()
        state = session.snapshot()
        assert state["crawl"]["active"] is False
        assert state["summary"]["armed_count"] == 0
        assert bus.torque == set()
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


def test_lost_browser_heartbeat_is_warning_only_and_gait_continues() -> None:
    session, bus, clock = _session()
    try:
        session.start_crawl_forward(
            safety_ack=True,
            confirmation="TEST DISTRIBUTED CRAWL",
        )
        clock.advance(19.9)
        session.advance_once()
        assert session.snapshot()["crawl"]["active"] is True

        clock.advance(0.2)
        session.advance_once()
        state = session.snapshot()

        assert state["any_armed"] is True
        assert state["crawl"]["active"] is True
        assert state["summary"]["armed_count"] == 12
        assert bus.torque == set(range(1, 13))
        assert state["browser_heartbeat"]["recent"] is False
        assert state["browser_heartbeat"]["warning_only"] is True
        assert state["browser_heartbeat"]["controls_motion"] is False
        assert any(
            "warning only" in warning.lower()
            for warning in state["summary"]["warnings"]
        )

        session.heartbeat("test-browser")
        state = session.snapshot()
        assert state["browser_heartbeat"]["recent"] is True
        assert state["browser_heartbeat"]["source"] == "test-browser"
        assert state["browser_heartbeat"]["received_count"] == 2
        assert state["crawl"]["active"] is True
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


def test_live_telemetry_warning_does_not_interrupt_armed_motors() -> None:
    session, bus, _clock = _session()
    try:
        session.arm_leg(2, safety_ack=True)
        bus.current_ma[4] = 2600.0

        state = session.snapshot()

        assert state["any_armed"] is True
        assert state["summary"]["armed_count"] == 3
        assert bus.torque == {4, 5, 6}
        assert state["summary"]["health"] == "warning"
        assert state["fault"] is None
    finally:
        session.close()


def test_power_analytics_detect_sag_and_possible_stall() -> None:
    session, bus, clock = _session()
    try:
        idle = session.snapshot()
        assert idle["power"]["idle_reference_voltage_v"] == pytest.approx(12.2)
        assert idle["power"]["battery_charge"]["status"] == "good"
        assert idle["power"]["battery_charge"][
            "average_cell_voltage_v"
        ] == pytest.approx(12.2 / 3.0)

        session.arm(1, 1, safety_ack=True)
        stalled_position = bus.positions[1]
        session.set_target(1, 1, 15.0)
        session.advance_once()
        bus.positions[1] = stalled_position
        bus.voltage_v[1] = 11.5
        bus.current_ma[1] = 1300.0
        clock.advance(1.0)
        state = session.snapshot()

        assert state["power"]["instantaneous_w"] == pytest.approx(14.95)
        assert state["power"]["voltage_sag_v"] == pytest.approx(0.7)
        assert state["power"]["possible_stall_ids"] == [1]
        assert state["legs"][0]["power_w"] == pytest.approx(14.95)
        assert state["legs"][0]["motors"][0]["possible_stall"] is True
        assert state["power"]["energy_wh"] > 0.0
        assert any(
            "voltage sag" in item.lower()
            for item in state["summary"]["warnings"]
        )
        assert any("stall" in item.lower() for item in state["summary"]["warnings"])

        with pytest.raises(RuntimeError, match="Disarm all motors"):
            session.reset_power_analytics()
    finally:
        session.close()
