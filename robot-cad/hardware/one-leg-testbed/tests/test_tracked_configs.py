from pathlib import Path

import pytest

from drobot_leg_testbed.model import load_calibration, load_config

CONFIG_DIR = Path(__file__).parents[1] / "config"
EXPECTED = {
    1: ((1, 2, 3), (1, -1, -1), (2048, 2048, 2048)),
    2: ((4, 5, 6), (1, 1, 1), (2048, 2047, 2047)),
    3: ((7, 8, 9), (1, 1, 1), (2048, 2048, 2047)),
    4: ((10, 11, 12), (1, 1, 1), (2048, 2047, 2047)),
}


@pytest.mark.parametrize("leg_number", EXPECTED)
def test_tracked_leg_profile_matches_verified_hardware(leg_number: int) -> None:
    config = load_config(CONFIG_DIR / f"leg-{leg_number}.toml")
    calibration = load_calibration(
        CONFIG_DIR / f"calibration-leg-{leg_number}.json",
        config,
    )
    expected_ids, expected_directions, expected_centers = EXPECTED[leg_number]

    assert tuple(motor.servo_id for motor in config.motors) == expected_ids
    assert tuple(motor.direction for motor in config.motors) == expected_directions
    assert tuple(motor.center_tick for motor in calibration.motors) == expected_centers
    assert config.bus.baudrate == 1_000_000
    assert config.bus.torque_limit == 300
    assert config.bus.speed == 350
    assert config.bus.acceleration == 10
    assert config.bus.max_command_step_deg == 5.0
