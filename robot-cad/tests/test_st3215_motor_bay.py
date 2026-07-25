from functools import lru_cache

import pytest

from robot_cad.parts.st3215_motor_bay import gen_step


@lru_cache(maxsize=1)
def migrated_motor_bay():
    return gen_step()


def test_motor_bay_is_one_valid_solid() -> None:
    part = migrated_motor_bay()

    assert part.is_valid
    assert len(part.solids()) == 1


def test_motor_bay_bounds_match_migrated_geometry() -> None:
    bounds = migrated_motor_bay().bounding_box()

    assert tuple(bounds.min) == pytest.approx((-16.0, -15.6117, -22.025), abs=1e-6)
    assert tuple(bounds.max) == pytest.approx((0.0, 15.6117, 22.025), abs=1e-6)
    assert tuple(bounds.size) == pytest.approx((16.0, 31.2234, 44.05), abs=1e-6)


def test_motor_bay_volume_matches_migrated_geometry() -> None:
    assert float(migrated_motor_bay().volume) == pytest.approx(
        9745.45116260892,
        rel=1e-5,
    )
