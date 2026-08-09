"""Shared mechanical parameters.

Values are expressed in millimeters unless their names state otherwise.
Part-specific dimensions belong in the corresponding part specification.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ManufacturingParameters:
    process: str = "fdm"
    minimum_structural_wall_mm: float = 3.0
    minimum_boss_wall_mm: float = 2.5
    servo_body_clearance_mm: float = 0.25
    moving_component_clearance_mm: float = 1.00
    cable_envelope_clearance_mm: float = 2.00
    m3_normal_clearance_diameter_mm: float = 3.4


DEFAULT_MANUFACTURING = ManufacturingParameters()
