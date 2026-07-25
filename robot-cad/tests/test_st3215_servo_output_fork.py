from functools import lru_cache

import pytest

from robot_cad.parts import st3215_servo_output_fork, upper_arm


@lru_cache(maxsize=1)
def generated_fork():
    return st3215_servo_output_fork.gen_step()


def test_servo_output_fork_is_one_valid_solid() -> None:
    fork = generated_fork()

    assert fork.is_valid
    assert len(fork.solids()) == 1


def test_servo_output_fork_starts_at_local_attachment_plane() -> None:
    bounds = generated_fork().bounding_box()

    assert bounds.min.X == pytest.approx(0.0, abs=0.05)


def test_output_axis_is_preserved_in_local_coordinates() -> None:
    point = st3215_servo_output_fork.output_axis_location_local().position
    direction = st3215_servo_output_fork.output_axis_direction_local()

    assert tuple(point) == pytest.approx((50.182161, 0.0, -4.516395), abs=1.0e-5)
    assert tuple(direction) == pytest.approx(
        (0.089638, 0.0, 0.995974),
        abs=1.0e-6,
    )


def test_placed_fork_reconstructs_the_pre_split_upper_arm() -> None:
    arm = upper_arm.gen_step()
    placed_fork = generated_fork().moved(
        st3215_servo_output_fork.placement_in_upper_arm_coordinates()
    )
    reconstructed = arm.fuse(placed_fork)
    bounds = reconstructed.bounding_box()

    assert reconstructed.is_valid
    assert len(reconstructed.solids()) == 1
    assert float(reconstructed.volume) == pytest.approx(
        129142.6023015041,
        rel=1.0e-5,
    )
    assert tuple(bounds.min) == pytest.approx((-74.2, -3.614304, -35.6), abs=0.05)
    assert tuple(bounds.max) == pytest.approx((77.084989, 27.614304, 31.7), abs=0.05)
