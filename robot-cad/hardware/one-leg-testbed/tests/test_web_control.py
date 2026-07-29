from __future__ import annotations

from pathlib import Path

import pytest

from drobot_leg_testbed.model import calibration_from_centers, load_config
from drobot_leg_testbed.web_control import ControlSession, DemoBus

ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self) -> None:
        self.now = 10.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _session() -> tuple[ControlSession, FakeClock]:
    config = load_config(ROOT / "leg.example.toml")
    calibration = calibration_from_centers(
        config,
        {motor.name: 2048 for motor in config.motors},
    )
    clock = FakeClock()
    session = ControlSession(
        config,
        calibration,
        DemoBus(config, calibration),
        clock=clock,
    )
    session.start(start_worker=False)
    return session, clock


def test_web_arm_requires_safety_acknowledgement() -> None:
    session, _clock = _session()
    try:
        with pytest.raises(ValueError, match="fixture"):
            session.arm(1, safety_ack=False)

        session.arm(1, safety_ack=True)

        assert session.snapshot()["motors"][0]["armed"] is True
    finally:
        session.close()


def test_large_web_destination_is_ramped_in_small_steps() -> None:
    session, _clock = _session()
    try:
        session.arm(1, safety_ack=True)
        session.set_target(1, 20.0)

        session.advance_once()
        first = session.snapshot()["motors"][0]

        assert first["desired_deg"] == 20.0
        assert first["commanded_deg"] == pytest.approx(1.5)

        for _index in range(20):
            session.advance_once()
        final = session.snapshot()["motors"][0]

        assert final["commanded_deg"] == 20.0
        assert final["measured_deg"] == pytest.approx(20.0, abs=0.05)
    finally:
        session.close()


def test_lost_browser_heartbeat_disarms_all_motors() -> None:
    session, clock = _session()
    try:
        session.arm(1, safety_ack=True)
        clock.advance(4.0)

        session.advance_once()
        state = session.snapshot()

        assert state["any_armed"] is False
        assert state["motors"][0]["torque_enabled"] is False
        assert "heartbeat lost" in state["last_event"].lower()
    finally:
        session.close()
