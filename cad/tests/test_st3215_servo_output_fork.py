from functools import lru_cache

import pytest
from build123d import Vector

from drobot_cad.parts import st3215_servo_output_fork, upper_arm


@lru_cache(maxsize=1)
def generated_fork():
    return st3215_servo_output_fork.gen_step()


@lru_cache(maxsize=1)
def generated_core():
    return st3215_servo_output_fork.gen_core_step()


def test_servo_output_fork_is_one_valid_solid() -> None:
    fork = generated_fork()

    assert fork.is_valid
    assert len(fork.solids()) == 1


def test_full_edge_extension_reaches_thirty_millimeters_into_negative_x() -> None:
    bounds = generated_fork().bounding_box()

    assert bounds.min.X == pytest.approx(-30.0, abs=0.05)
    assert bounds.max.X == pytest.approx(65.084989, abs=0.05)


def test_root_extension_covers_the_complete_legacy_split_edge() -> None:
    bounds = st3215_servo_output_fork.make_root_extension().bounding_box()

    assert tuple(bounds.min) == pytest.approx((-30.0, -12.0, -31.7), abs=0.05)
    assert tuple(bounds.max) == pytest.approx((0.0, 12.0, 31.7), abs=0.05)
    assert generated_fork().is_inside(Vector(-0.1, 0.0, 31.0))
    assert generated_fork().is_inside(Vector(-0.1, 0.0, -31.0))


def test_attachment_plane_is_perpendicular_to_x() -> None:
    assert st3215_servo_output_fork.cut_plane_normal_global() == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert st3215_servo_output_fork.cut_plane_alignment_angle_deg() == pytest.approx(
        0.0
    )


def test_output_axis_is_preserved_in_local_coordinates() -> None:
    point = st3215_servo_output_fork.output_axis_location_local().position
    direction = st3215_servo_output_fork.output_axis_direction_local()

    assert tuple(point) == pytest.approx((53.084989, 0.0, 0.0), abs=1.0e-5)
    assert tuple(direction) == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-6)


def test_centerline_root_ligament_is_at_least_three_millimeters() -> None:
    fork = generated_fork()

    assert fork.is_inside(Vector(3.0, 0.0, 0.0))


def test_extension_is_additive_and_preserves_the_original_fork_core() -> None:
    core = generated_core()
    missing_core = core - generated_fork()

    assert core.is_valid
    assert len(core.solids()) == 1
    assert float(core.volume) == pytest.approx(48717.65301998467, rel=1.0e-5)
    assert not missing_core.solids()
    assert float(missing_core.volume) == pytest.approx(0.0, abs=1.0e-5)


def test_extended_fork_volume_matches_the_generated_baseline() -> None:
    assert float(generated_fork().volume) == pytest.approx(
        84203.18991346893,
        rel=1.0e-5,
    )


def test_reference_fork_side_is_integral_to_the_full_upper_arm() -> None:
    arm = upper_arm.gen_step()
    reference_fork = st3215_servo_output_fork.retain_fork_side(
        st3215_servo_output_fork._load_reference_body()
    )
    uncovered_fork = reference_fork - arm

    assert not uncovered_fork.solids()
    assert float(uncovered_fork.volume) == pytest.approx(0.0, abs=1.0e-5)
