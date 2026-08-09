from __future__ import annotations

from pathlib import Path

from drobot_leg_testbed.model import load_config
from drobot_leg_testbed.transport import (
    REG_TORQUE_ENABLE,
    SET_MIDDLE_COMMAND,
    STSBus,
)

ROOT = Path(__file__).resolve().parents[1]


def test_set_middle_position_disarms_unlocks_verifies_and_relocks(
    monkeypatch,
) -> None:
    motor = load_config(ROOT / "leg.example.toml").motor(1)
    bus = STSBus("demo", 1_000_000)
    reads = iter((3691, 2048))
    calls: list[tuple] = []

    monkeypatch.setattr(
        bus,
        "require_motor",
        lambda selected: calls.append(("ping", selected.servo_id)),
    )
    monkeypatch.setattr(bus, "read_position", lambda servo_id: next(reads))
    monkeypatch.setattr(
        bus,
        "disable_torque",
        lambda servo_id: calls.append(("disable", servo_id)),
    )
    monkeypatch.setattr(
        bus,
        "unlock_config",
        lambda servo_id: calls.append(("unlock", servo_id)),
    )
    monkeypatch.setattr(
        bus,
        "lock_config",
        lambda servo_id: calls.append(("lock", servo_id)),
    )
    monkeypatch.setattr(
        bus,
        "_write1",
        lambda servo_id, address, value: calls.append(
            ("write1", servo_id, address, value)
        ),
    )
    monkeypatch.setattr(
        "drobot_leg_testbed.transport.time.sleep",
        lambda _seconds: None,
    )

    before, after = bus.set_middle_position(motor)

    assert (before, after) == (3691, 2048)
    assert calls == [
        ("ping", 1),
        ("disable", 1),
        ("unlock", 1),
        ("write1", 1, REG_TORQUE_ENABLE, SET_MIDDLE_COMMAND),
        ("lock", 1),
    ]


def test_write_position_sign_encodes_negative_extended_target(monkeypatch) -> None:
    class Packet:
        def scs_toscs(self, value: int, sign_bit: int) -> int:
            calls.append(("encode", value, sign_bit))
            return (-value) | (1 << sign_bit) if value < 0 else value

        def WritePosEx(
            self,
            servo_id: int,
            position: int,
            speed: int,
            acceleration: int,
        ) -> tuple[int, int]:
            calls.append(
                ("write", servo_id, position, speed, acceleration)
            )
            return 0, 0

    config = load_config(ROOT / "leg.example.toml")
    calls: list[tuple] = []
    bus = STSBus("demo", config.bus.baudrate)
    bus.packet = Packet()
    bus._comm_success = 0
    monkeypatch.setattr(bus, "_write2", lambda *_args: None)

    bus.write_position(3, -19, config.bus)

    assert calls == [
        ("encode", -19, 15),
        ("write", 3, 0x8013, config.bus.speed, config.bus.acceleration),
    ]
