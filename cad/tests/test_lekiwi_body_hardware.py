from functools import lru_cache

import pytest
from build123d import Location, Vector

from drobot_cad.parts import (
    adafruit_bno085,
    lekiwi_12v_battery_reference,
    quadruped_body,
    quadruped_electronics_tray,
    waveshare_bus_servo_adapter_a,
)


@lru_cache(maxsize=1)
def generated_battery():
    return lekiwi_12v_battery_reference.gen_step()


@lru_cache(maxsize=1)
def generated_adapter():
    return waveshare_bus_servo_adapter_a.gen_step()


@lru_cache(maxsize=1)
def generated_tray():
    return quadruped_electronics_tray.gen_step()


def test_lekiwi_battery_proxy_matches_measured_installed_envelope() -> None:
    battery = generated_battery()

    assert battery.is_valid
    assert len(battery.solids()) == 1
    assert tuple(battery.bounding_box().size) == pytest.approx(
        lekiwi_12v_battery_reference.INSTALLED_ENVELOPE_XYZ_MM
    )


def test_battery_has_clearance_inside_existing_body_bay() -> None:
    battery_bounds = generated_battery().moved(
        Location((0.0, 0.0, quadruped_body.BODY_FLOOR_MM))
    ).bounding_box()

    assert battery_bounds.min.X == pytest.approx(-35.0)
    assert battery_bounds.max.X == pytest.approx(35.0)
    assert battery_bounds.min.Y == pytest.approx(-33.0)
    assert battery_bounds.max.Y == pytest.approx(33.0)
    assert battery_bounds.max.Z == pytest.approx(44.0)
    assert (
        quadruped_body.BATTERY_CLEAR_WIDTH_Y_MM - battery_bounds.size.Y
    ) / 2.0 == pytest.approx(1.0)
    assert (
        quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM - battery_bounds.max.Z
    ) == pytest.approx(12.0)


def test_exact_waveshare_adapter_uses_official_board_envelope() -> None:
    adapter = generated_adapter()
    bounds = adapter.bounding_box()

    assert len(adapter.solids()) == 117
    assert tuple(bounds.size) == pytest.approx(
        waveshare_bus_servo_adapter_a.DETAIL_BOUNDS_SIZE_XYZ_MM,
        abs=1e-6,
    )
    assert bounds.min.Z == pytest.approx(
        waveshare_bus_servo_adapter_a.DETAIL_MIN_Z_MM,
        abs=1e-6,
    )
    assert bounds.max.Z == pytest.approx(
        waveshare_bus_servo_adapter_a.DETAIL_MAX_Z_MM,
        abs=1e-6,
    )


def test_controller_fit_proxy_matches_exact_reference_bounds() -> None:
    exact_bounds = generated_adapter().bounding_box()
    proxy_bounds = waveshare_bus_servo_adapter_a.make_fit_proxy().bounding_box()

    assert tuple(proxy_bounds.min) == pytest.approx(tuple(exact_bounds.min))
    assert tuple(proxy_bounds.max) == pytest.approx(tuple(exact_bounds.max))


def test_servo_adapter_standoff_pattern_matches_official_mount() -> None:
    center_x, center_y = quadruped_electronics_tray.SERVO_ADAPTER_CENTER_XY_MM
    local_centers = tuple(
        (x_mm - center_x, y_mm - center_y)
        for x_mm, y_mm in (
            quadruped_electronics_tray.SERVO_ADAPTER_MOUNT_CENTERS_XY_MM
        )
    )

    for actual, expected in zip(
        local_centers,
        waveshare_bus_servo_adapter_a.MOUNT_HOLE_CENTERS_XY_MM,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    assert waveshare_bus_servo_adapter_a.MOUNT_HOLE_SPACING_XY_MM == pytest.approx(
        (37.0, 28.0)
    )


def test_servo_adapter_mount_holes_are_open_in_tray_standoffs() -> None:
    tray = generated_tray()
    test_z = (
        quadruped_electronics_tray.SERVO_ADAPTER_BOARD_DATUM_Z_MM / 2.0
    )

    for x_mm, y_mm in (
        quadruped_electronics_tray.SERVO_ADAPTER_MOUNT_CENTERS_XY_MM
    ):
        assert not tray.is_inside(Vector(x_mm, y_mm, test_z))
        assert tray.is_inside(
            Vector(
                x_mm
                + (
                    quadruped_electronics_tray
                    .SERVO_ADAPTER_STANDOFF_OUTER_DIAMETER_MM
                    / 2.0
                    - 0.3
                ),
                y_mm,
                quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM + 1.0,
            )
        )


def test_installed_adapter_clears_tray_lid_and_body_imu() -> None:
    adapter_bounds = waveshare_bus_servo_adapter_a.make_fit_proxy().moved(
        Location(
            (
                *quadruped_electronics_tray.SERVO_ADAPTER_CENTER_XY_MM,
                (
                    quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM
                    + quadruped_electronics_tray.SERVO_ADAPTER_BOARD_DATUM_Z_MM
                ),
            )
        )
    ).bounding_box()
    tray_top_z_mm = (
        quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM
        + quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM
    )
    lid_locator_bottom_z_mm = quadruped_body.BODY_BASE_HEIGHT_Z_MM - 3.0
    imu_bounds = adafruit_bno085.gen_step().moved(
        Location(
            (
                0.0,
                0.0,
                (
                    quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM
                    + quadruped_electronics_tray.IMU_BOARD_BOTTOM_Z_MM
                ),
            )
        )
    ).bounding_box()

    assert adapter_bounds.min.Z - tray_top_z_mm == pytest.approx(0.4, abs=0.01)
    assert lid_locator_bottom_z_mm - adapter_bounds.max.Z > 18.0
    assert imu_bounds.min.X - adapter_bounds.max.X > 14.0
    assert adapter_bounds.min.X > -quadruped_body.ELECTRONICS_TRAY_LENGTH_X_MM / 2.0
    assert adapter_bounds.max.Y < quadruped_body.ELECTRONICS_TRAY_WIDTH_Y_MM / 2.0
