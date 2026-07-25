"""Servo-bracket design contract placeholder.

The final model should be driven by an authoritative vendor servo STEP model,
not a guessed hardware envelope.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ServoBracketSpec:
    servo_part_number: str
    body_clearance_mm: float = 0.30
    fastener: str = "M3"
