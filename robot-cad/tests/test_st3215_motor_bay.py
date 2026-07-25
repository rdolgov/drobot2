from functools import lru_cache

import pytest
from build123d import GeomType

from robot_cad.parts import st3215_motor_bay


@lru_cache(maxsize=1)
def migrated_motor_bay():
    return st3215_motor_bay.gen_step()


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
        9282.994037878747,
        rel=1e-5,
    )


def test_motor_bay_has_eight_diamond_vents_on_each_side_wall() -> None:
    part = migrated_motor_bay()
    outer_y = (
        st3215_motor_bay.ST3215_CATALOG_WIDTH_Z_MM
        + 2.0 * st3215_motor_bay.SOCKET_CLEARANCE_Y_PER_SIDE_MM
        + 2.0 * st3215_motor_bay.SOCKET_WALL_MM
    )
    outer_z = (
        st3215_motor_bay.ST3215_CATALOG_HEIGHT_Y_MM
        + st3215_motor_bay.SOCKET_CLEARANCE_Z_TOTAL_MM
        + 2.0 * st3215_motor_bay.SOCKET_WALL_MM
    )
    side_faces = [
        face
        for face in part.faces()
        if face.geom_type == GeomType.PLANE
        and abs(face.normal_at().Y) > 0.99
        and abs(abs(face.center().Y) - outer_y / 2.0) < 1.0e-4
    ]
    vent_count = (
        len(st3215_motor_bay.VENT_DIAMOND_COLUMNS_X_MM)
        * len(st3215_motor_bay.VENT_DIAMOND_ROWS_Z_MM)
    )
    vent_area = (
        st3215_motor_bay.VENT_DIAMOND_WIDTH_X_MM
        * st3215_motor_bay.VENT_DIAMOND_HEIGHT_Z_MM
        / 2.0
    )
    expected_side_area = (
        st3215_motor_bay.SOCKET_LENGTH_X_MM * outer_z
        - vent_count * vent_area
    )

    assert len(side_faces) == 2
    for face in side_faces:
        assert len(face.wires()) == vent_count + 1
        assert float(face.area) == pytest.approx(expected_side_area, abs=1e-6)
