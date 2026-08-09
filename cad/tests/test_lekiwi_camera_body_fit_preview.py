import pytest
from build123d import Vector

from drobot_cad.assembly import lekiwi_camera_body_fit_preview as preview
from drobot_cad.parts import quadruped_body, quadruped_body_lid


def test_reference_proxies_are_valid_connected_solids() -> None:
    mount = preview.make_mount_reference_proxy()
    camera = preview.make_camera_reference_proxy()

    assert mount.is_valid
    assert camera.is_valid
    assert len(mount.solids()) == 1
    assert len(camera.solids()) == 1


def test_mount_reference_uses_the_lid_camera_row() -> None:
    mount = preview.make_mount_reference_proxy()
    test_z = preview.LEKIWI_MOUNT_FOOT_THICKNESS_Z_MM / 2.0

    for _, y_mm in quadruped_body_lid.LEKIWI_CAMERA_MOUNT_HOLE_CENTERS_XY_MM:
        assert not mount.is_inside(Vector(0.0, y_mm, test_z))

    assert preview.MOUNT_WORLD_X_MM == pytest.approx(
        quadruped_body_lid.LEKIWI_CAMERA_MOUNT_CENTER_X_MM
    )
    assert preview.MOUNT_WORLD_Z_MM == pytest.approx(
        quadruped_body.BODY_LID_THICKNESS_Z_MM
    )


def test_mounted_reference_stays_inside_lid_footprint() -> None:
    mounted = preview.make_mount_reference_proxy().moved(
        preview.mount_world_location()
    )
    bounds = mounted.bounding_box()

    assert bounds.min.X >= -quadruped_body.BODY_LENGTH_X_MM / 2.0
    assert bounds.max.X <= quadruped_body.BODY_LENGTH_X_MM / 2.0
    assert bounds.min.Y >= -quadruped_body.BODY_WIDTH_Y_MM / 2.0
    assert bounds.max.Y <= quadruped_body.BODY_WIDTH_Y_MM / 2.0
