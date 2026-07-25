"""Assembly metadata scaffold shared with future CAD and URDF generators."""

from dataclasses import dataclass

from robot_cad.interfaces import InterfaceFrame


@dataclass(frozen=True)
class RevoluteConnection:
    name: str
    parent_component: str
    child_component: str
    parent_frame: InterfaceFrame
    child_frame: InterfaceFrame
    minimum_deg: float
    maximum_deg: float


@dataclass(frozen=True)
class RobotArmAssemblySpec:
    root_component: str = "base"
