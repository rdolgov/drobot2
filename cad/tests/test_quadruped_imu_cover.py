from functools import lru_cache

import pytest
from build123d import Location, Vector

from drobot_cad.parts import (
    adafruit_bno085,
    quadruped_body,
    quadruped_electronics_tray,
    quadruped_imu_cover,
)


@lru_cache(maxsize=1)
def generated_cover():
    return quadruped_imu_cover.gen_step()


def test_cover_is_one_valid_printable_solid_with_expected_envelope() -> None:
    cover = generated_cover()
    bounds = cover.bounding_box()

    assert cover.is_valid
    assert len(cover.solids()) == 1
    assert tuple(bounds.size) == pytest.approx(
        (
            quadruped_imu_cover.COVER_LENGTH_X_MM,
            quadruped_imu_cover.COVER_WIDTH_Y_MM,
            quadruped_imu_cover.COVER_TOTAL_HEIGHT_MM,
        )
    )
    assert bounds.min.Z == pytest.approx(0.0)


def test_cover_m2_holes_match_exact_board_and_tray_axes_and_remain_open() -> None:
    cover = generated_cover()

    assert quadruped_imu_cover.MOUNT_CENTERS_XY_MM == (
        adafruit_bno085.MOUNT_HOLE_CENTERS_SENSOR_XY_MM
    )
    assert quadruped_imu_cover.MOUNT_CENTERS_XY_MM == (
        quadruped_electronics_tray.IMU_MOUNT_CENTERS_XY_MM
    )
    for x_mm, y_mm in quadruped_imu_cover.MOUNT_CENTERS_XY_MM:
        assert not cover.is_inside(
            Vector(
                x_mm,
                y_mm,
                quadruped_imu_cover.COVER_TOTAL_HEIGHT_MM / 2.0,
            )
        )
        assert cover.is_inside(
            Vector(
                x_mm + quadruped_imu_cover.SPACER_OUTER_DIAMETER_MM / 2.0 - 0.2,
                y_mm,
                1.0,
            )
        )


def test_cover_roof_protects_sensor_while_all_sides_stay_open() -> None:
    cover = generated_cover()

    assert cover.is_inside(
        Vector(
            0.0,
            0.0,
            (
                quadruped_imu_cover.COVER_ROOF_UNDERSIDE_Z_MM
                + quadruped_imu_cover.COVER_ROOF_THICKNESS_MM / 2.0
            ),
        )
    )
    assert not cover.is_inside(
        Vector(
            0.0,
            0.0,
            quadruped_imu_cover.COVER_ROOF_UNDERSIDE_Z_MM / 2.0,
        )
    )
    modeled_component_height_above_pcb_mm = (
        adafruit_bno085.BOARD_SIZE_XYZ_MM[2]
        - adafruit_bno085.PCB_THICKNESS_MM
    )
    assert (
        quadruped_imu_cover.COVER_ROOF_UNDERSIDE_Z_MM
        - modeled_component_height_above_pcb_mm
    ) == pytest.approx(3.0)


def test_installed_cover_clears_exact_board_and_body_lid() -> None:
    board = adafruit_bno085.gen_step()
    cover_on_board = generated_cover().moved(
        Location((0.0, 0.0, adafruit_bno085.PCB_THICKNESS_MM))
    )
    intersection = board.intersect(cover_on_board)
    assert intersection is None or intersection.volume == pytest.approx(
        0.0, abs=1e-6
    )

    installed_cover_top_z_mm = (
        quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM
        + quadruped_electronics_tray.IMU_BOARD_BOTTOM_Z_MM
        + adafruit_bno085.PCB_THICKNESS_MM
        + quadruped_imu_cover.COVER_TOTAL_HEIGHT_MM
    )
    lid_locator_bottom_z_mm = quadruped_body.BODY_BASE_HEIGHT_Z_MM - 3.0
    assert lid_locator_bottom_z_mm - installed_cover_top_z_mm > 20.0
