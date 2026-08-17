from functools import lru_cache

import pytest
from build123d import Location, Vector

from drobot_cad.assembly import rectangular_fork_shoe_fit_preview
from drobot_cad.parts import rectangular_fork_shoe, upper_arm


@lru_cache(maxsize=1)
def generated_shoe():
    return rectangular_fork_shoe.make_rectangular_fork_shoe()


def test_shoe_is_one_valid_solid_with_expected_source_envelope() -> None:
    shoe = generated_shoe()
    bounds = shoe.bounding_box()

    assert shoe.is_valid
    assert len(shoe.solids()) == 1
    assert tuple(bounds.min) == pytest.approx((-11.5, -50.0, -30.0), abs=0.05)
    assert tuple(bounds.max) == pytest.approx((30.0, 50.0, 30.0), abs=0.05)


def test_sole_is_long_fore_aft_and_has_a_flat_contact_face() -> None:
    assert rectangular_fork_shoe.SOLE_LENGTH_FORE_AFT_MM == pytest.approx(100.0)
    assert rectangular_fork_shoe.SOLE_WIDTH_LATERAL_MM == pytest.approx(60.0)
    assert rectangular_fork_shoe.SOLE_FACE_X_MM == pytest.approx(30.0)

    shoe = generated_shoe()
    assert shoe.is_inside(Vector(29.5, 0.0, 0.0))
    assert any(
        face.geom_type.name == "PLANE"
        and face.bounding_box().min.X
        == pytest.approx(rectangular_fork_shoe.SOLE_FACE_X_MM, abs=0.05)
        and face.bounding_box().max.X
        == pytest.approx(rectangular_fork_shoe.SOLE_FACE_X_MM, abs=0.05)
        for face in shoe.faces()
    )


def test_four_rods_reuse_the_existing_fork_pattern() -> None:
    shoe = generated_shoe()
    assert len(rectangular_fork_shoe.RECOMMENDED_ROD_HOLE_CENTERS_XY_MM) == 4
    assert (
        2.0 * rectangular_fork_shoe.FORK_ROD_PATTERN_OFFSET_MM
    ) == pytest.approx(9.899494)

    for x_mm, y_mm in rectangular_fork_shoe.FORK_ROD_HOLE_CENTERS_XY_MM:
        assert not shoe.is_inside(Vector(x_mm, y_mm, 0.0))
        assert shoe.is_inside(
            Vector(
                x_mm + rectangular_fork_shoe.ROD_PAD_RADIUS_MM - 0.3,
                y_mm,
                0.0,
            )
        )


def test_fastener_access_envelope_clears_the_sole_and_reinforcement() -> None:
    assert (
        rectangular_fork_shoe.MIN_HOLE_CENTER_TO_SOLE_BACK_X_MM
    ) == pytest.approx(19.050253)
    assert (
        rectangular_fork_shoe.MIN_HOLE_CENTER_TO_REINFORCEMENT_X_MM
    ) == pytest.approx(15.050253)
    assert (
        rectangular_fork_shoe.MIN_TOOL_ENVELOPE_TO_REINFORCEMENT_X_MM
    ) == pytest.approx(10.550253)


def test_hub_preserves_fork_clearance() -> None:
    assert (
        rectangular_fork_shoe.FORK_INNER_BOSS_FACE_Z_MM
        - rectangular_fork_shoe.HUB_HALF_WIDTH_Z_MM
    ) == pytest.approx(rectangular_fork_shoe.FORK_CLEARANCE_PER_SIDE_MM)
    assert (
        rectangular_fork_shoe.FORK_LOCAL_ROD_FACE_Z_MM
        - rectangular_fork_shoe.ROD_PAD_FACE_Z_MM
    ) == pytest.approx(rectangular_fork_shoe.FORK_CLEARANCE_PER_SIDE_MM)


def test_installed_shoe_clears_the_existing_lower_leg() -> None:
    installed = generated_shoe().moved(Location(upper_arm.DISTAL_FORK_AXIS_MM))
    overlap = upper_arm.gen_step().intersect(installed)
    overlap_volume = (
        0.0 if overlap is None else sum(float(solid.volume) for solid in overlap)
    )
    assert overlap_volume == pytest.approx(0.0, abs=1.0e-5)


def test_fit_preview_includes_four_rods_and_bilateral_driver_access() -> None:
    preview = rectangular_fork_shoe_fit_preview.make_fit_preview()
    labels = tuple(child.label for child in preview.children)

    assert labels[:2] == (
        "existing_distal_lower_leg",
        "printable_rectangular_fork_shoe",
    )
    assert sum(label.startswith("m3x75_threaded_rod_envelope_") for label in labels) == 4
    assert sum(label.startswith("m3_driver_access_negative_z_") for label in labels) == 4
    assert sum(label.startswith("m3_driver_access_positive_z_") for label in labels) == 4
