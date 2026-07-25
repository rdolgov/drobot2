from pathlib import Path

import pytest
import yaml

from robot_cad.parts import st3215_motor_bay, upper_arm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "filename",
    [
        "assembly.yaml",
        "mechanical-interfaces.yaml",
        "st3215-motor-bay.yaml",
        "upper-arm.yaml",
    ],
)
def test_yaml_specs_are_valid_versioned_documents(filename: str) -> None:
    document = yaml.safe_load((PROJECT_ROOT / "specs" / filename).read_text())

    assert isinstance(document, dict)
    assert document["schema_version"] == 1


def test_motor_bay_yaml_matches_fit_critical_parameters() -> None:
    document = yaml.safe_load((PROJECT_ROOT / "specs" / "st3215-motor-bay.yaml").read_text())
    socket = document["socket"]
    ventilation = document["ventilation"]

    assert socket["clearance_y_per_side_mm"] == pytest.approx(
        st3215_motor_bay.SOCKET_CLEARANCE_Y_PER_SIDE_MM
    )
    assert socket["clearance_z_total_mm"] == pytest.approx(
        st3215_motor_bay.SOCKET_CLEARANCE_Z_TOTAL_MM
    )
    assert socket["wall_mm"] == pytest.approx(st3215_motor_bay.SOCKET_WALL_MM)
    assert socket["length_x_mm"] == pytest.approx(st3215_motor_bay.SOCKET_LENGTH_X_MM)
    assert socket["stop_thickness_mm"] == pytest.approx(st3215_motor_bay.SOCKET_STOP_THICKNESS_MM)
    assert socket["outer_side_perimeter_fillet_radius_mm"] == pytest.approx(
        st3215_motor_bay.OUTER_SIDE_PERIMETER_FILLET_RADIUS_MM
    )
    assert ventilation["side_walls"] == "both"
    assert ventilation["pattern"] == "diamond"
    assert ventilation["columns_x_mm"] == pytest.approx(st3215_motor_bay.VENT_DIAMOND_COLUMNS_X_MM)
    assert ventilation["rows_z_mm"] == pytest.approx(st3215_motor_bay.VENT_DIAMOND_ROWS_Z_MM)
    assert ventilation["diamond_width_x_mm"] == pytest.approx(
        st3215_motor_bay.VENT_DIAMOND_WIDTH_X_MM
    )
    assert ventilation["diamond_height_z_mm"] == pytest.approx(
        st3215_motor_bay.VENT_DIAMOND_HEIGHT_Z_MM
    )
    assert ventilation["wall_overtravel_mm"] == pytest.approx(
        st3215_motor_bay.VENT_WALL_OVERTRAVEL_MM
    )


def test_upper_arm_yaml_matches_migrated_feature_parameters() -> None:
    document = yaml.safe_load((PROJECT_ROOT / "specs" / "upper-arm.yaml").read_text())
    features = document["features"]
    transition = document["transition"]
    opening = document["enlarged_middle_opening"]

    assert features["mount_panel_cut_plane_x_mm"] == pytest.approx(
        upper_arm.MOUNT_PANEL_CUT_PLANE_X_MM
    )
    assert features["motor_bay_center_y_mm"] == pytest.approx(
        upper_arm.ST3215_MOTOR_BAY_CENTER_Y_MM
    )
    assert features["motor_bay_center_z_mm"] == pytest.approx(
        upper_arm.ST3215_MOTOR_BAY_CENTER_Z_MM
    )
    assert features["motor_bay_join_overlap_mm"] == pytest.approx(
        upper_arm.ST3215_MOTOR_BAY_JOIN_OVERLAP_MM
    )
    assert features["negative_x_extension_mm"] == pytest.approx(
        upper_arm.UPPER_ARM_NEGATIVE_X_EXTENSION_MM
    )
    assert transition["length_mm"] == pytest.approx(upper_arm.SMOOTH_TRANSITION_LENGTH_MM)
    assert opening["enabled"] is upper_arm.ENLARGE_MIDDLE_HALF_CIRCLE
    assert opening["left_cap_center_x_mm"] == pytest.approx(
        upper_arm.MIDDLE_OPENING_LEFT_CAP_CENTER_X_MM
    )
    assert opening["right_cap_center_x_mm"] == pytest.approx(
        upper_arm.MIDDLE_OPENING_RIGHT_CAP_CENTER_X_MM
    )
    assert opening["height_z_mm"] == pytest.approx(
        upper_arm.MIDDLE_OPENING_HEIGHT_Z_MM
    )
    assert opening["cap_radius_x_mm"] == pytest.approx(
        upper_arm.MIDDLE_OPENING_CAP_RADIUS_X_MM
    )
