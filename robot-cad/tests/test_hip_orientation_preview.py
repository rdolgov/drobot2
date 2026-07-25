from functools import lru_cache

import pytest
from build123d import Vertex

from robot_cad.assembly import hip_orientation_preview


def mapped_direction(location, local_direction):
    origin = Vertex(0.0, 0.0, 0.0).moved(location).center()
    endpoint = Vertex(*local_direction).moved(location).center()
    return tuple(endpoint - origin)


@lru_cache(maxsize=1)
def positioned_components():
    return (
        hip_orientation_preview.placed_fork(),
        hip_orientation_preview.placed_motor_bay(),
    )


@lru_cache(maxsize=1)
def positioned_servo():
    return hip_orientation_preview.placed_installed_servo()


def test_fork_longitudinal_direction_faces_down() -> None:
    direction = mapped_direction(
        hip_orientation_preview.fork_face_down_location(),
        (1.0, 0.0, 0.0),
    )

    assert direction == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-9)


def test_motor_bay_has_all_three_requested_ninety_degree_rotations() -> None:
    location = hip_orientation_preview.motor_bay_facing_left_location()
    insertion_direction = mapped_direction(location, (-1.0, 0.0, 0.0))
    diamond_face_direction = mapped_direction(location, (0.0, 1.0, 0.0))
    cross_direction = mapped_direction(location, (0.0, 0.0, 1.0))

    assert insertion_direction == pytest.approx((-1.0, 0.0, 0.0), abs=1.0e-9)
    assert diamond_face_direction == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-9)
    assert cross_direction == pytest.approx((0.0, -1.0, 0.0), abs=1.0e-9)


def test_fork_and_motor_bay_are_perpendicular() -> None:
    fork_direction = mapped_direction(
        hip_orientation_preview.fork_face_down_location(),
        (1.0, 0.0, 0.0),
    )
    bay_insertion_direction = mapped_direction(
        hip_orientation_preview.motor_bay_facing_left_location(),
        (-1.0, 0.0, 0.0),
    )
    dot_product = sum(
        a * b
        for a, b in zip(
            fork_direction,
            bay_insertion_direction,
            strict=True,
        )
    )

    assert dot_product == pytest.approx(0.0, abs=1.0e-9)


def test_bay_is_seated_on_top_and_centered_over_the_oval_tip() -> None:
    fork, bay = positioned_components()
    fork_bounds = fork.bounding_box()
    bay_bounds = bay.bounding_box()

    assert fork_bounds.max.Z == pytest.approx(
        hip_orientation_preview.FORK_OVAL_TOP_WORLD_Z_MM,
        abs=0.05,
    )
    assert bay_bounds.min.Z == pytest.approx(
        hip_orientation_preview.FORK_OVAL_TOP_WORLD_Z_MM,
        abs=0.05,
    )
    assert bay_bounds.center().X == pytest.approx(
        hip_orientation_preview.FORK_OVAL_CENTER_WORLD_X_MM,
        abs=1.0e-6,
    )
    assert fork_bounds.center().Y == pytest.approx(0.0, abs=1.0e-6)
    assert bay_bounds.center().Y == pytest.approx(0.0, abs=1.0e-6)


def test_orientation_preview_preserves_two_valid_components() -> None:
    fork, bay = positioned_components()
    overlap = fork & bay

    assert fork.is_valid
    assert bay.is_valid
    assert len(fork.solids()) == 1
    assert len(bay.solids()) == 1
    assert fork.distance_to(bay) == pytest.approx(0.0, abs=1.0e-6)
    assert overlap is None or float(overlap.volume) == pytest.approx(0.0, abs=1.0e-6)


def test_installed_st3215_clears_the_fork() -> None:
    fork, _ = positioned_components()
    servo = positioned_servo()
    collision_volume = 0.0

    for servo_solid in servo.solids():
        overlap = fork & servo_solid
        if overlap is not None:
            collision_volume += float(overlap.volume)

    assert collision_volume == pytest.approx(0.0, abs=1.0e-6)
