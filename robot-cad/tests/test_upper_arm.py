from functools import lru_cache

import pytest
from build123d import Vector

from robot_cad.parts import upper_arm


@lru_cache(maxsize=1)
def migrated_upper_arm():
    return upper_arm.gen_step()


def test_upper_arm_is_one_valid_solid() -> None:
    part = migrated_upper_arm()

    assert part.is_valid
    assert len(part.solids()) == 1


def test_upper_arm_bounds_match_migrated_geometry() -> None:
    bounds = migrated_upper_arm().bounding_box()

    assert tuple(bounds.min) == pytest.approx((-74.2, -3.614304, -31.7), abs=0.05)
    assert tuple(bounds.max) == pytest.approx((17.553, 27.614304, 31.7), abs=0.05)
    assert tuple(bounds.size) == pytest.approx((91.753, 31.228608, 63.4), abs=0.05)


def test_upper_arm_volume_matches_migrated_geometry() -> None:
    assert float(migrated_upper_arm().volume) == pytest.approx(
        84392.91099500656,
        rel=1e-5,
    )


def test_middle_half_circle_is_enlarged_without_cutting_motor_bay() -> None:
    part = migrated_upper_arm()
    center_z = upper_arm.MIDDLE_OPENING_CENTER_Z_MM
    negative_tip_x = (
        upper_arm.MIDDLE_OPENING_LEFT_CAP_CENTER_X_MM
        - upper_arm.MIDDLE_OPENING_CAP_RADIUS_X_MM
    )

    assert part.is_inside(Vector(negative_tip_x - 1.0, 12.0, center_z))
    assert not part.is_inside(
        Vector(upper_arm.MIDDLE_OPENING_LEFT_CAP_CENTER_X_MM, 12.0, center_z)
    )
    assert not part.is_inside(
        Vector(upper_arm.MIDDLE_OPENING_RIGHT_CAP_CENTER_X_MM, 12.0, center_z)
    )
    assert part.is_inside(Vector(-70.0, -2.0, center_z))
