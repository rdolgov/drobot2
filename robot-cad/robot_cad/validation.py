"""Reusable deterministic checks for CAD geometry and interface metadata."""

from collections.abc import Iterable
from math import isclose
from typing import Protocol

from robot_cad.interfaces import InterfaceFrame


class CadPart(Protocol):
    @property
    def is_valid(self) -> bool: ...

    def solids(self) -> Iterable[object]: ...


def require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive; received {value}")


def require_at_least(name: str, value: float, minimum: float) -> None:
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; received {value}")


def validate_interface_frame(frame: InterfaceFrame, *, tolerance: float = 1e-9) -> None:
    if not frame.name.startswith("frame_"):
        raise ValueError(f"Interface name must start with 'frame_': {frame.name}")
    if not isclose(frame.axis_magnitude, 1.0, abs_tol=tolerance):
        raise ValueError(f"{frame.name} axis must be a unit vector: {frame.axis}")


def validate_part(part: CadPart, *, expected_solid_count: int = 1) -> None:
    if not part.is_valid:
        raise ValueError("CAD geometry is invalid")
    solid_count = len(list(part.solids()))
    if solid_count != expected_solid_count:
        raise ValueError(
            f"Expected {expected_solid_count} connected solid(s), found {solid_count}"
        )
