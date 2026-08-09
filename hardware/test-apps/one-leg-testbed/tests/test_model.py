from __future__ import annotations

import json
from pathlib import Path

import pytest

from drobot_leg_testbed.model import (
    calibration_from_centers,
    degrees_to_raw,
    load_calibration,
    load_config,
    raw_to_degrees,
    save_calibration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_has_three_unique_numbered_motors() -> None:
    config = load_config(ROOT / "leg.example.toml")

    assert [motor.number for motor in config.motors] == [1, 2, 3]
    assert [motor.servo_id for motor in config.motors] == [1, 2, 3]
    assert [motor.name for motor in config.motors] == [
        "hip_abduction",
        "hip_flexion",
        "knee",
    ]
    assert config.bus.baudrate == 1_000_000
    assert config.bus.torque_limit == 300


def test_angle_conversion_round_trips_and_obeys_direction() -> None:
    config = load_config(ROOT / "leg.example.toml")
    calibration = calibration_from_centers(
        config,
        {motor.name: 2048 for motor in config.motors},
    )
    motor = config.motor(1)
    motor_calibration = calibration.motor(motor)

    raw = degrees_to_raw(10.0, motor, motor_calibration)

    assert raw > 2048
    assert raw_to_degrees(raw, motor, motor_calibration) == pytest.approx(
        10.0,
        abs=0.05,
    )


def test_angle_conversion_rejects_encoder_boundary_crossing() -> None:
    config = load_config(ROOT / "leg.example.toml")
    calibration = calibration_from_centers(
        config,
        {
            "hip_abduction": 4090,
            "hip_flexion": 2048,
            "knee": 2048,
        },
    )
    motor = config.motor("hip_abduction")
    motor_calibration = calibration.motor(motor)

    with pytest.raises(ValueError, match="encoder"):
        degrees_to_raw(1.0, motor, motor_calibration)


def test_angle_conversion_rejects_configured_limit() -> None:
    config = load_config(ROOT / "leg.example.toml")
    calibration = calibration_from_centers(
        config,
        {motor.name: 2048 for motor in config.motors},
    )
    motor = config.motor("hip_abduction")

    with pytest.raises(ValueError, match="outside"):
        degrees_to_raw(31.0, motor, calibration.motor(motor))


def test_calibration_round_trip(tmp_path: Path) -> None:
    config = load_config(ROOT / "leg.example.toml")
    calibration = calibration_from_centers(
        config,
        {
            "hip_abduction": 2001,
            "hip_flexion": 2048,
            "knee": 2100,
        },
    )
    path = tmp_path / "calibration.json"

    save_calibration(calibration, path)
    loaded = load_calibration(path, config)

    assert loaded == calibration
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    source = (ROOT / "leg.example.toml").read_text(encoding="utf-8")
    duplicate = source.replace("id = 2", "id = 1")
    path = tmp_path / "bad.toml"
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="IDs must be unique"):
        load_config(path)
