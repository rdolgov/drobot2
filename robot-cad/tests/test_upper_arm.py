from functools import lru_cache

import pytest

from robot_cad.parts.upper_arm import gen_step


@lru_cache(maxsize=1)
def migrated_upper_arm():
    return gen_step()


def test_upper_arm_is_one_valid_solid() -> None:
    part = migrated_upper_arm()

    assert part.is_valid
    assert len(part.solids()) == 1


def test_upper_arm_bounds_match_migrated_geometry() -> None:
    bounds = migrated_upper_arm().bounding_box()

    assert tuple(bounds.min) == pytest.approx((-44.2, -3.645729, -35.6), abs=0.05)
    assert tuple(bounds.max) == pytest.approx((77.054028, 27.645729, 31.7), abs=0.05)
    assert tuple(bounds.size) == pytest.approx((121.254028, 31.291458, 67.3), abs=0.05)


def test_upper_arm_volume_matches_migrated_geometry() -> None:
    assert float(migrated_upper_arm().volume) == pytest.approx(
        109097.15784699321,
        rel=1e-5,
    )
