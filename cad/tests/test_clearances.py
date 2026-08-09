from drobot_cad.parameters import DEFAULT_MANUFACTURING


def test_default_clearances_are_not_below_project_minimums() -> None:
    assert DEFAULT_MANUFACTURING.minimum_structural_wall_mm >= 3.0
    assert DEFAULT_MANUFACTURING.minimum_boss_wall_mm >= 2.5
    assert DEFAULT_MANUFACTURING.servo_body_clearance_mm >= 0.25
    assert DEFAULT_MANUFACTURING.moving_component_clearance_mm >= 1.0
    assert DEFAULT_MANUFACTURING.cable_envelope_clearance_mm >= 2.0
