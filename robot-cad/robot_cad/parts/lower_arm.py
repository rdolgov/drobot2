"""Lower-arm design contract placeholder.

Add geometry only after its joint spacing, hardware interfaces, and acceptance
tests are captured under ``specs/``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class LowerArmSpec:
    joint_spacing_mm: float
    width_mm: float
    height_mm: float
