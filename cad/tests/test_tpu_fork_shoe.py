from functools import lru_cache

import pytest
from build123d import Location, Vector

from drobot_cad.assembly import tpu_fork_shoe_fit_preview
from drobot_cad.parts import tpu_fork_shoe, upper_arm


@lru_cache(maxsize=1)
def generated_shoe():
    return tpu_fork_shoe.gen_step()


def test_shoe_is_one_valid_printable_solid_with_expected_envelope() -> None:
    shoe = generated_shoe()
    bounds = shoe.bounding_box()

    assert shoe.is_valid
    assert len(shoe.solids()) == 1
    assert tuple(bounds.min) == pytest.approx((-11.5, -24.0, -24.0), abs=0.05)
    assert tuple(bounds.max) == pytest.approx((72.5, 24.0, 24.0), abs=0.05)


def test_four_rod_clearances_reuse_the_existing_fork_pattern() -> None:
    shoe = generated_shoe()
    assert len(tpu_fork_shoe.FORK_ROD_HOLE_CENTERS_XY_MM) == 4
    assert (
        2.0 * tpu_fork_shoe.FORK_ROD_PATTERN_OFFSET_MM
    ) == pytest.approx(9.899494)

    for x_mm, y_mm in tpu_fork_shoe.FORK_ROD_HOLE_CENTERS_XY_MM:
        assert not shoe.is_inside(Vector(x_mm, y_mm, 0.0))
        assert shoe.is_inside(
            Vector(x_mm + tpu_fork_shoe.ROD_PAD_RADIUS_MM - 0.3, y_mm, 0.0)
        )


def test_hub_and_local_pads_preserve_fork_clearance() -> None:
    assert (
        tpu_fork_shoe.FORK_INNER_BOSS_FACE_Z_MM
        - tpu_fork_shoe.HUB_HALF_WIDTH_Z_MM
    ) == pytest.approx(tpu_fork_shoe.FORK_CLEARANCE_PER_SIDE_MM)
    assert (
        tpu_fork_shoe.FORK_LOCAL_ROD_FACE_Z_MM
        - tpu_fork_shoe.ROD_PAD_FACE_Z_MM
    ) == pytest.approx(tpu_fork_shoe.FORK_CLEARANCE_PER_SIDE_MM)


def test_oval_rocker_has_no_flat_contact_pad_and_keeps_shell_wall() -> None:
    shoe = generated_shoe()
    assert tpu_fork_shoe.CONTACT_OUTER_AXIAL_RADIUS_X_MM == pytest.approx(30.0)
    assert tpu_fork_shoe.CONTACT_OUTER_RADIAL_RADIUS_MM == pytest.approx(24.0)
    assert tpu_fork_shoe.CONTACT_INNER_AXIAL_RADIUS_X_MM == pytest.approx(26.0)
    assert tpu_fork_shoe.CONTACT_INNER_RADIAL_RADIUS_MM == pytest.approx(20.0)
    assert tpu_fork_shoe.SHELL_WALL_MM == pytest.approx(4.0)
    assert shoe.is_inside(
        Vector(
            tpu_fork_shoe.CONTACT_CENTER_X_MM
            + tpu_fork_shoe.CONTACT_OUTER_AXIAL_RADIUS_X_MM
            - 0.5,
            0.0,
            0.0,
        )
    )
    assert not shoe.is_inside(
        Vector(tpu_fork_shoe.CONTACT_CENTER_X_MM, 15.0, 0.0)
    )
    nose_x_mm = (
        tpu_fork_shoe.CONTACT_CENTER_X_MM
        + tpu_fork_shoe.CONTACT_OUTER_AXIAL_RADIUS_X_MM
    )
    assert all(
        not (
            face.geom_type.name == "PLANE"
            and face.bounding_box().min.X == pytest.approx(nose_x_mm, abs=0.05)
            and face.bounding_box().max.X == pytest.approx(nose_x_mm, abs=0.05)
        )
        for face in shoe.faces()
    )


def test_installed_shoe_clears_the_existing_lower_leg() -> None:
    installed = generated_shoe().moved(
        Location(upper_arm.DISTAL_FORK_AXIS_MM)
    )
    overlap = upper_arm.gen_step().intersect(installed)
    overlap_volume = (
        0.0 if overlap is None else sum(float(solid.volume) for solid in overlap)
    )
    assert overlap_volume == pytest.approx(0.0, abs=1.0e-5)


def test_fit_preview_has_lower_leg_shoe_and_two_rod_envelopes() -> None:
    preview = tpu_fork_shoe_fit_preview.gen_step()
    assert tuple(child.label for child in preview.children) == (
        "existing_distal_lower_leg",
        "printable_tpu_fork_shoe",
        "m3x75_threaded_rod_envelope_1",
        "m3x75_threaded_rod_envelope_2",
    )
