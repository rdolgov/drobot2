"""Named mechanical frames shared by parts, assemblies, and future URDF data."""

from dataclasses import dataclass
from math import dist, sqrt

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class InterfaceFrame:
    name: str
    xyz_mm: Vector3
    axis: Vector3
    role: str

    @property
    def axis_magnitude(self) -> float:
        return sqrt(sum(component * component for component in self.axis))

    def distance_to(self, other: "InterfaceFrame") -> float:
        return dist(self.xyz_mm, other.xyz_mm)


ST3215_MOTOR_BAY_INTERFACES = {
    "frame_attachment_datum": InterfaceFrame(
        name="frame_attachment_datum",
        xyz_mm=(0.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        role="flat YZ attachment face at X=0",
    ),
    "frame_servo_install": InterfaceFrame(
        name="frame_servo_install",
        xyz_mm=(-11.1117, 0.0, 9.425),
        axis=(1.0, 0.0, 0.0),
        role="ST3215 catalog origin after the +90 degree X installation rotation",
    ),
}

UPPER_ARM_INTERFACES = {
    "frame_st3215_bay_attachment": InterfaceFrame(
        name="frame_st3215_bay_attachment",
        xyz_mm=(-58.2, 12.0, -1.95),
        axis=(1.0, 0.0, 0.0),
        role="placed ST3215 motor-bay X=0 datum",
    ),
    "frame_st3215_servo_install": InterfaceFrame(
        name="frame_st3215_servo_install",
        xyz_mm=(-69.3117, 12.0, 7.475),
        axis=(1.0, 0.0, 0.0),
        role="installed ST3215 catalog origin in upper-arm coordinates",
    ),
}
