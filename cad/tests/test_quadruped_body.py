from functools import lru_cache

import pytest
from build123d import Align, Box, Location, Vector

from drobot_cad.parts import (
    adafruit_bno085,
    quadruped_body,
    quadruped_body_lid,
    quadruped_electronics_tray,
)


@lru_cache(maxsize=1)
def generated_base():
    return quadruped_body.gen_step()


@lru_cache(maxsize=1)
def generated_lid():
    return quadruped_body_lid.gen_step()


@lru_cache(maxsize=1)
def generated_tray():
    return quadruped_electronics_tray.gen_step()


@pytest.mark.parametrize(
    "shape_factory",
    [generated_base, generated_lid, generated_tray],
)
def test_each_printable_body_piece_is_one_valid_solid(shape_factory) -> None:
    shape = shape_factory()

    assert shape.is_valid
    assert len(shape.solids()) == 1


def test_body_base_has_expected_x2d_safe_bounds() -> None:
    bounds = generated_base().bounding_box()

    assert tuple(bounds.size) == pytest.approx((220.0, 170.0, 96.0))
    assert tuple(bounds.min) == pytest.approx((-110.0, -85.0, 0.0))
    assert quadruped_body.x2d_print_footprint_with_brim() == pytest.approx(
        (230.0, 180.0, 96.0)
    )
    assert (
        quadruped_body.x2d_print_footprint_with_brim()[0]
        <= quadruped_body.X2D_DUAL_NOZZLE_SHARED_VOLUME_MM[0]
    )
    assert (
        quadruped_body.x2d_print_footprint_with_brim()[1]
        <= quadruped_body.X2D_DUAL_NOZZLE_SHARED_VOLUME_MM[1]
    )


def test_all_sixteen_hip_holes_pass_through_reinforced_side_walls() -> None:
    base = generated_base()

    for x_mm in quadruped_body.HIP_MOUNT_HOLE_X_MM:
        for z_mm in quadruped_body.HIP_MOUNT_HOLE_Z_MM:
            for side in (-1.0, 1.0):
                wall_y = side * (
                    quadruped_body.BODY_WIDTH_Y_MM / 2.0
                    - quadruped_body.BODY_WALL_MM / 2.0
                )
                assert not base.is_inside(Vector(x_mm, wall_y, z_mm))
                assert base.is_inside(Vector(x_mm + 4.0, wall_y, z_mm))


def test_battery_clear_envelope_is_open_inside_the_retaining_rail() -> None:
    base = generated_base()
    battery_envelope = Box(
        quadruped_body.BATTERY_CLEAR_LENGTH_X_MM,
        quadruped_body.BATTERY_CLEAR_WIDTH_Y_MM,
        quadruped_body.BATTERY_CLEAR_HEIGHT_Z_MM,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).moved(Location((0.0, 0.0, quadruped_body.BODY_FLOOR_MM)))
    intersection = base.intersect(battery_envelope)

    assert intersection is None or intersection.volume == pytest.approx(
        0.0, abs=1e-6
    )


def test_lid_locator_has_declared_per_side_clearance() -> None:
    cavity_length = (
        quadruped_body.BODY_LENGTH_X_MM - 2.0 * quadruped_body.BODY_WALL_MM
    )
    cavity_width = (
        quadruped_body.BODY_WIDTH_Y_MM - 2.0 * quadruped_body.BODY_WALL_MM
    )
    lip = quadruped_body_lid.make_locator_lip().bounding_box()

    assert cavity_length - lip.size.X == pytest.approx(
        2.0 * quadruped_body_lid.LID_LOCATOR_CLEARANCE_PER_SIDE_MM
    )
    assert cavity_width - lip.size.Y == pytest.approx(
        2.0 * quadruped_body_lid.LID_LOCATOR_CLEARANCE_PER_SIDE_MM
    )


def test_seated_lid_has_no_volume_intersection_with_body_base() -> None:
    base = generated_base()
    seated_lid = generated_lid().moved(
        Location((0.0, 0.0, quadruped_body.BODY_BASE_HEIGHT_Z_MM))
    )
    intersection = base.intersect(seated_lid)

    assert intersection is None or intersection.volume == pytest.approx(
        0.0, abs=1e-6
    )


def test_tray_and_lid_mount_holes_are_open() -> None:
    tray = generated_tray()
    lid = generated_lid()

    for x_mm, y_mm in quadruped_body.TRAY_MOUNT_CENTERS_XY_MM:
        assert not tray.is_inside(Vector(x_mm, y_mm, 1.5))
    for x_mm, y_mm in quadruped_body.LID_BOSS_CENTERS_XY_MM:
        assert not lid.is_inside(Vector(x_mm, y_mm, 2.0))


def test_exact_bno085_mount_is_centred_open_and_lid_clear() -> None:
    tray = generated_tray()

    for actual, expected in zip(
        quadruped_electronics_tray.IMU_MOUNT_CENTERS_XY_MM,
        adafruit_bno085.MOUNT_HOLE_CENTERS_SENSOR_XY_MM,
        strict=True,
    ):
        assert actual == pytest.approx(expected)
    for x_mm, y_mm in quadruped_electronics_tray.IMU_MOUNT_CENTERS_XY_MM:
        assert not tray.is_inside(
            Vector(
                x_mm,
                y_mm,
                quadruped_electronics_tray.IMU_BOARD_BOTTOM_Z_MM / 2.0,
            )
        )
        assert tray.is_inside(
            Vector(
                x_mm + quadruped_electronics_tray.IMU_STANDOFF_OUTER_DIAMETER_MM
                / 2.0
                - 0.3,
                y_mm,
                quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM + 1.0,
            )
        )

    board_top_z_mm = (
        quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM
        + quadruped_electronics_tray.IMU_BOARD_BOTTOM_Z_MM
        + adafruit_bno085.BOARD_SIZE_XYZ_MM[2]
    )
    lid_locator_bottom_z_mm = quadruped_body.BODY_BASE_HEIGHT_Z_MM - 3.0
    assert lid_locator_bottom_z_mm - board_top_z_mm > 25.0


def test_symmetric_side_service_ports_are_open_between_hip_fields() -> None:
    base = generated_base()
    wall_center_offset = (
        quadruped_body.BODY_WIDTH_Y_MM / 2.0
        - quadruped_body.BODY_WALL_MM / 2.0
    )

    for side in (-1.0, 1.0):
        wall_y = side * wall_center_offset
        assert quadruped_body.SIDE_CABLE_PORT_LENGTH_X_MM == pytest.approx(32.0)
        assert quadruped_body.SIDE_CABLE_PORT_HEIGHT_Z_MM == pytest.approx(20.0)
        assert not base.is_inside(
            Vector(
                quadruped_body.SIDE_CABLE_PORT_CENTER_X_MM,
                wall_y,
                quadruped_body.SIDE_CABLE_PORT_CENTER_Z_MM,
            )
        )
        assert not base.is_inside(
            Vector(15.0, wall_y, quadruped_body.SIDE_CABLE_PORT_CENTER_Z_MM)
        )
        assert base.is_inside(
            Vector(
                quadruped_body.SIDE_CABLE_PORT_LENGTH_X_MM / 2.0 + 3.0,
                wall_y,
                quadruped_body.SIDE_CABLE_PORT_CENTER_Z_MM,
            )
        )
        assert base.is_inside(
            Vector(
                quadruped_body.SIDE_CABLE_PORT_CENTER_X_MM,
                wall_y,
                quadruped_body.SIDE_CABLE_PORT_CENTER_Z_MM
                + quadruped_body.SIDE_CABLE_PORT_HEIGHT_Z_MM / 2.0
                + 2.0,
            )
        )


def test_body_walls_have_dense_10_mm_pitch_m3_mounting_grids() -> None:
    base = generated_base()
    front_rear_centers = (
        quadruped_body.front_rear_mounting_grid_centers_yz_mm()
    )
    side_centers = quadruped_body.side_mounting_grid_centers_xz_mm()
    wall_x = (
        quadruped_body.BODY_LENGTH_X_MM / 2.0
        - quadruped_body.BODY_WALL_MM / 2.0
    )
    wall_y = (
        quadruped_body.BODY_WIDTH_Y_MM / 2.0
        - quadruped_body.BODY_WALL_MM / 2.0
    )

    assert len(front_rear_centers) == 83
    assert len(side_centers) == 12
    assert quadruped_body.BODY_WALL_MOUNTING_GRID_PITCH_MM == pytest.approx(10.0)

    for side in (-1.0, 1.0):
        for y_mm, z_mm in front_rear_centers:
            assert not base.is_inside(Vector(side * wall_x, y_mm, z_mm))
        for x_mm, z_mm in side_centers:
            assert not base.is_inside(Vector(x_mm, side * wall_y, z_mm))

    # The former fixed-height tray posts must be absent from the open interior.
    former_post_centers = (
        (-80.0, -55.0),
        (-80.0, 55.0),
        (80.0, -55.0),
        (80.0, 55.0),
    )
    for x_mm, y_mm in former_post_centers:
        assert not base.is_inside(Vector(x_mm, y_mm, 30.0))


def test_dense_floor_grid_is_open_centered_and_tray_compatible() -> None:
    base = generated_base()
    test_z = quadruped_body.BODY_FLOOR_MM / 2.0
    grid_centers = quadruped_body.floor_mounting_grid_centers_xy_mm()

    assert len(grid_centers) == 255
    assert quadruped_body.FLOOR_MOUNTING_GRID_PITCH_MM == pytest.approx(10.0)
    assert set(quadruped_body.TRAY_MOUNT_CENTERS_XY_MM).issubset(grid_centers)

    rail_opening_half_x = (
        quadruped_body.BATTERY_CLEAR_LENGTH_X_MM
        + 2.0 * quadruped_body.BATTERY_RAIL_CLEARANCE_PER_SIDE_MM
    ) / 2.0
    rail_opening_half_y = (
        quadruped_body.BATTERY_CLEAR_WIDTH_Y_MM
        + 2.0 * quadruped_body.BATTERY_RAIL_CLEARANCE_PER_SIDE_MM
    ) / 2.0
    rail_outer_half_x = (
        rail_opening_half_x + quadruped_body.BATTERY_RAIL_THICKNESS_MM
    )
    rail_outer_half_y = (
        rail_opening_half_y + quadruped_body.BATTERY_RAIL_THICKNESS_MM
    )
    protected_offset = (
        quadruped_body.FLOOR_MOUNTING_GRID_M3_CLEARANCE_DIAMETER_MM / 2.0
        + quadruped_body.FLOOR_MOUNTING_GRID_MIN_WEB_MM
    )

    for x_mm, y_mm in grid_centers:
        assert not base.is_inside(Vector(x_mm, y_mm, test_z))
        assert (
            (
                abs(x_mm) < rail_opening_half_x - protected_offset
                and abs(y_mm) < rail_opening_half_y - protected_offset
            )
            or (
                abs(x_mm) > rail_outer_half_x + protected_offset
                or abs(y_mm) > rail_outer_half_y + protected_offset
            )
        )

    # Center mounting points are open while the raised retaining rail survives.
    assert not base.is_inside(Vector(0.0, 0.0, test_z))
    rail_test_z = (
        quadruped_body.BODY_FLOOR_MM
        + quadruped_body.BATTERY_RAIL_HEIGHT_Z_MM / 2.0
    )
    assert base.is_inside(Vector(0.0, rail_opening_half_y + 1.5, rail_test_z))
    assert base.is_inside(Vector(rail_opening_half_x + 1.5, 0.0, rail_test_z))

    for x_mm in quadruped_body.BASE_UTILITY_M2_X_MM:
        for y_mm in quadruped_body.BASE_UTILITY_Y_MM:
            assert not base.is_inside(Vector(x_mm, y_mm, test_z))
            assert abs(y_mm) > (
                quadruped_body.BATTERY_CLEAR_WIDTH_Y_MM / 2.0
                + quadruped_body.BATTERY_RAIL_CLEARANCE_PER_SIDE_MM
                + quadruped_body.BATTERY_RAIL_THICKNESS_MM
            )
    for x_mm in quadruped_body.BASE_UTILITY_M3_X_MM:
        for y_mm in quadruped_body.BASE_UTILITY_Y_MM:
            assert not base.is_inside(Vector(x_mm, y_mm, test_z))


def test_lid_leg_ports_and_utility_mounting_holes_are_open() -> None:
    lid = generated_lid()
    test_z = quadruped_body.BODY_LID_THICKNESS_Z_MM / 2.0

    for x_mm in quadruped_body_lid.LID_CABLE_PORT_X_MM:
        for y_mm in quadruped_body_lid.LID_CABLE_PORT_Y_MM:
            assert not lid.is_inside(Vector(x_mm, y_mm, test_z))
    for x_mm in quadruped_body_lid.LID_UTILITY_M2_X_MM:
        for y_mm in quadruped_body_lid.LID_UTILITY_Y_MM:
            assert not lid.is_inside(Vector(x_mm, y_mm, test_z))
    for x_mm, y_mm in (
        quadruped_body_lid.LID_REAR_UTILITY_M3_CENTERS_XY_MM
        + quadruped_body_lid.LEKIWI_CAMERA_MOUNT_HOLE_CENTERS_XY_MM
    ):
        assert not lid.is_inside(Vector(x_mm, y_mm, test_z))


def test_lid_has_dense_10_mm_pitch_m3_mounting_grid_with_keepouts() -> None:
    lid = generated_lid()
    test_z = quadruped_body.BODY_LID_THICKNESS_Z_MM / 2.0
    grid_centers = quadruped_body_lid.lid_mounting_grid_centers_xy_mm()

    assert len(grid_centers) == 215
    assert min(x_mm for x_mm, _ in grid_centers) == pytest.approx(-90.0)
    assert max(x_mm for x_mm, _ in grid_centers) == pytest.approx(90.0)
    assert min(y_mm for _, y_mm in grid_centers) == pytest.approx(-60.0)
    assert max(y_mm for _, y_mm in grid_centers) == pytest.approx(60.0)
    assert quadruped_body_lid.LID_MOUNTING_GRID_PITCH_MM == pytest.approx(10.0)

    for x_mm, y_mm in grid_centers:
        assert not lid.is_inside(Vector(x_mm, y_mm, test_z))

    # The grid preserves usable material immediately beside representative
    # ventilation and camera-cable keep-outs.
    assert lid.is_inside(Vector(-76.5, -42.0, test_z))
    assert lid.is_inside(Vector(76.5, 0.0, test_z))


def test_lid_lekiwi_camera_interface_is_open_and_on_20_mm_pitch() -> None:
    lid = generated_lid()
    test_z = quadruped_body.BODY_LID_THICKNESS_Z_MM / 2.0
    camera_holes = quadruped_body_lid.LEKIWI_CAMERA_MOUNT_HOLE_CENTERS_XY_MM

    assert camera_holes == (
        (quadruped_body_lid.LEKIWI_CAMERA_MOUNT_CENTER_X_MM, -20.0),
        (quadruped_body_lid.LEKIWI_CAMERA_MOUNT_CENTER_X_MM, 0.0),
        (quadruped_body_lid.LEKIWI_CAMERA_MOUNT_CENTER_X_MM, 20.0),
    )
    assert camera_holes[1][1] - camera_holes[0][1] == pytest.approx(
        quadruped_body_lid.LEKIWI_CAMERA_MOUNT_HOLE_PITCH_Y_MM
    )
    assert camera_holes[2][1] - camera_holes[1][1] == pytest.approx(
        quadruped_body_lid.LEKIWI_CAMERA_MOUNT_HOLE_PITCH_Y_MM
    )
    for x_mm, y_mm in camera_holes:
        assert not lid.is_inside(Vector(x_mm, y_mm, test_z))

    assert not lid.is_inside(
        Vector(
            quadruped_body_lid.LEKIWI_CAMERA_CABLE_PORT_CENTER_X_MM,
            quadruped_body_lid.LEKIWI_CAMERA_CABLE_PORT_CENTER_Y_MM,
            test_z,
        )
    )
    assert lid.is_inside(
        Vector(
            quadruped_body_lid.LEKIWI_CAMERA_CABLE_PORT_CENTER_X_MM
            - quadruped_body_lid.LEKIWI_CAMERA_CABLE_PORT_LENGTH_X_MM / 2.0
            - 2.0,
            quadruped_body_lid.LEKIWI_CAMERA_CABLE_PORT_CENTER_Y_MM,
            test_z,
        )
    )
