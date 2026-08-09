from functools import lru_cache

import pytest
from build123d import Vector

from drobot_cad.parts import st3215_hip_body_mount


@lru_cache(maxsize=1)
def generated_mount():
    return st3215_hip_body_mount.gen_step()


@lru_cache(maxsize=1)
def generated_plate():
    return st3215_hip_body_mount.make_mounting_plate()


def test_body_mount_is_one_valid_connected_solid() -> None:
    mount = generated_mount()

    assert mount.is_valid
    assert len(mount.solids()) == 1


def test_mounting_plate_is_larger_than_the_connected_fork_edge() -> None:
    assert (
        st3215_hip_body_mount.MOUNTING_PLATE_WIDTH_Y_MM
        > st3215_hip_body_mount.ROOT_EXTENSION_WIDTH_Y_MM
    )
    assert (
        st3215_hip_body_mount.MOUNTING_PLATE_HEIGHT_Z_MM
        > 2.0 * st3215_hip_body_mount.ROOT_EXTENSION_RADIUS_Z_MM
    )


def test_body_mount_has_expected_overall_bounds() -> None:
    bounds = generated_mount().bounding_box()

    assert bounds.min.X == pytest.approx(-32.0, abs=0.05)
    assert bounds.max.X == pytest.approx(65.084989, abs=0.05)
    assert bounds.size.Y == pytest.approx(76.0, abs=0.05)
    assert bounds.size.Z == pytest.approx(76.0, abs=0.05)


def test_four_m4_holes_pass_through_the_plate() -> None:
    plate = generated_plate()
    hole_radius = (
        st3215_hip_body_mount.BODY_BOLT_CLEARANCE_DIAMETER_MM / 2.0
    )

    for y_mm, z_mm in st3215_hip_body_mount.BODY_BOLT_HOLE_CENTERS_YZ_MM:
        assert not plate.is_inside(
            Vector(
                st3215_hip_body_mount.MOUNTING_PLATE_CENTER_X_MM,
                y_mm,
                z_mm,
            )
        )
        assert plate.is_inside(
            Vector(
                st3215_hip_body_mount.MOUNTING_PLATE_CENTER_X_MM,
                y_mm + hole_radius + 0.5,
                z_mm,
            )
        )


def test_plate_has_positive_overlap_with_the_reused_fork() -> None:
    fork = st3215_hip_body_mount.placed_motor_fork()
    plate = generated_plate()
    overlap = fork & plate

    assert overlap is not None
    assert float(overlap.volume) > 1900.0


def test_reused_motor_fork_is_fully_preserved() -> None:
    fork = st3215_hip_body_mount.placed_motor_fork()
    missing_fork = fork - generated_mount()

    assert not missing_fork.solids()
    assert float(missing_fork.volume) == pytest.approx(0.0, abs=1.0e-5)

