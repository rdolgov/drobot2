from functools import lru_cache

import pytest
from build123d import Vertex

from robot_cad.parts import st3215_hip


def mapped_direction(location, local_direction):
    origin = Vertex(0.0, 0.0, 0.0).moved(location).center()
    endpoint = Vertex(*local_direction).moved(location).center()
    return tuple(endpoint - origin)


@lru_cache(maxsize=1)
def positioned_components():
    return (
        st3215_hip.placed_fork(),
        st3215_hip.placed_motor_bay(printable=True),
    )


@lru_cache(maxsize=1)
def positioned_servo():
    return st3215_hip.placed_installed_servo(printable=True)


@lru_cache(maxsize=1)
def generated_hip():
    return st3215_hip.gen_step()


def test_approved_orientation_is_preserved() -> None:
    fork_direction = mapped_direction(
        st3215_hip.fork_face_down_location(),
        (1.0, 0.0, 0.0),
    )
    bay_location = st3215_hip.motor_bay_fused_location()
    bay_facing_direction = mapped_direction(bay_location, (-1.0, 0.0, 0.0))
    bay_rolled_local_y = mapped_direction(bay_location, (0.0, 1.0, 0.0))

    assert fork_direction == pytest.approx((0.0, 0.0, -1.0), abs=1.0e-9)
    assert bay_facing_direction == pytest.approx((-1.0, 0.0, 0.0), abs=1.0e-9)
    assert bay_rolled_local_y == pytest.approx((0.0, 0.0, 1.0), abs=1.0e-9)
    assert sum(
        a * b
        for a, b in zip(fork_direction, bay_facing_direction, strict=True)
    ) == pytest.approx(0.0, abs=1.0e-9)


def test_hidden_seating_overlap_and_left_alignment() -> None:
    fork, bay = positioned_components()
    fork_bounds = fork.bounding_box()
    bay_bounds = bay.bounding_box()
    overlap = fork & bay

    assert fork_bounds.max.Z == pytest.approx(0.0, abs=0.05)
    assert bay_bounds.min.Z == pytest.approx(
        -st3215_hip.HIP_JOIN_OVERLAP_MM,
        abs=0.05,
    )
    assert bay_bounds.min.X == pytest.approx(
        st3215_hip.FORK_ROOT_LEFT_EDGE_WORLD_X_MM,
        abs=0.05,
    )
    assert overlap is not None
    assert float(overlap.volume) > 1.0


def test_printable_hip_is_one_valid_solid() -> None:
    hip = generated_hip()
    bounds = hip.bounding_box()

    assert hip.is_valid
    assert len(hip.solids()) == 1
    assert bounds.size.X == pytest.approx(67.3, abs=0.05)
    assert bounds.size.Y == pytest.approx(44.05, abs=0.05)
    assert bounds.size.Z == pytest.approx(96.1084, abs=0.05)


def test_installed_st3215_still_clears_the_fork() -> None:
    fork, _ = positioned_components()
    servo = positioned_servo()
    collision_volume = 0.0

    for servo_solid in servo.solids():
        overlap = fork & servo_solid
        if overlap is not None:
            collision_volume += float(overlap.volume)

    assert collision_volume == pytest.approx(0.0, abs=1.0e-6)
