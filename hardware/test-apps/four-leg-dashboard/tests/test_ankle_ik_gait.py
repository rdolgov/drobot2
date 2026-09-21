"""Focused contracts for the new pure IK controller, without hardware access."""
import math
import pytest

from drobot_hardware_test_apps.ankle_ik_gait import (
    AnkleIKConfig, AnkleIKClock, LINK_M, ankle_leg_ik, ankle_ik_gait_degrees,
)


def test_ankle_pivot_ik_matches_two_link_forward_geometry():
    for leg in ("front_left", "front_right", "rear_left", "rear_right"):
        for x, z in ((.08, .295), (-.08, .295), (.02 if leg.startswith("front_") else -.02, .27)):
            hip, knee, ankle = map(math.radians, ankle_leg_ik(leg, x, z))
            assert LINK_M*(math.sin(hip)+math.sin(hip+knee)) == pytest.approx(x)
            assert LINK_M*(math.cos(hip)+math.cos(hip+knee)) == pytest.approx(z)
            assert hip+knee+ankle == pytest.approx(0.)


@pytest.mark.parametrize("mode", ["distributed", "diagonal_pair"])
def test_periodic_trajectory_preserves_limits_and_phase_continuity(mode):
    cfg = AnkleIKConfig()
    first, _ = ankle_ik_gait_degrees(0., cfg, mode)
    last, _ = ankle_ik_gait_degrees(cfg.period_s, cfg, mode)
    assert first == pytest.approx(last)
    previous = first
    for tick in range(1, 3841):
        current, state = ankle_ik_gait_degrees(tick*cfg.period_s/3840, cfg, mode)
        assert len(current) == 16
        assert max(abs(v) for k, v in current.items() if k[1] == "ankle_pitch") <= 33.25
        assert max(abs(current[k]-previous[k]) for k in current) < .5
        assert len(state["expected_support_corners"]) >= (2 if mode == "diagonal_pair" else 3)
        previous = current


def test_gate_cannot_advance_from_sent_targets_without_encoder_tracking():
    clock = AnkleIKClock(AnkleIKConfig(), "distributed")
    clock.time_s = .36*4.
    clock.gate = "lift"
    target, _ = ankle_ik_gait_degrees(clock.time_s, clock.config, clock.mode)
    measured = dict(target)
    measured[("rear_right", "knee")] += 8.
    zero = {key: 0. for key in target}
    before = clock.time_s
    with pytest.raises(RuntimeError, match="tracking timeout"):
        for _ in range(130):
            clock.tick(1./60., measured, zero, target)
            assert clock.time_s == before


def test_stop_completes_current_plant_gate_then_holds():
    clock = AnkleIKClock(AnkleIKConfig(), "distributed")
    clock.time_s = .86*4.
    clock.gate = "firm_plant"
    clock.stop_requested = True
    target, _ = ankle_ik_gait_degrees(clock.time_s, clock.config, clock.mode)
    before = clock.time_s
    for _ in range(12):
        clock.tick(1./60., target, {k: 0. for k in target}, target)
    assert clock.stopped
    assert clock.time_s == before


def test_gate_rejects_missing_or_nonfinite_feedback():
    clock = AnkleIKClock(AnkleIKConfig(), "distributed")
    target, _ = ankle_ik_gait_degrees(0.)
    with pytest.raises(ValueError, match="sixteen"):
        clock.tick(1./60., {}, {}, target)
    bad = dict(target)
    bad[("front_left", "knee")] = math.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        clock.tick(1./60., bad, {k: 0. for k in target}, target)


def test_dashboard_sends_all_sixteen_joints_and_stops_after_plant():
    from pathlib import Path
    from drobot_hardware_test_apps.four_leg_control import FourLegDemoBus, FourLegSession, load_dashboard_config
    root = Path(__file__).resolve().parents[4]
    dashboard = load_dashboard_config(root / "hardware/robot-runtime/four-leg.toml")
    bus = FourLegDemoBus(dashboard)
    now = [100.]
    session = FourLegSession(dashboard, bus, clock=lambda: now[0], persist_calibration=False)
    session.start(start_worker=False)
    try:
        session.start_crawl_forward(safety_ack=True, confirmation="TEST DISTRIBUTED CRAWL")
        for _ in range(450):
            now[0] += 1./60.
            session.heartbeat()
            session.advance_once(elapsed_s=1./60.)
        assert session.crawl_stage == "walking"
        assert set(bus.torque) == set(range(1, 17))
        assert all("ankle_pitch" in controller.targets_deg for controller in session.controllers.values())
        session.stop_crawl()
        for _ in range(300):
            now[0] += 1./60.
            session.heartbeat()
            session.advance_once(elapsed_s=1./60.)
            if session.crawl_stage == "holding":
                break
        assert session.crawl_stage == "holding"
        assert session.ik_clock.stopped
        assert session.snapshot()["crawl"]["ankle_ik"]
    finally:
        session.close()
