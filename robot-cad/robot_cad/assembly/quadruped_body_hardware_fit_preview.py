"""Body-only hardware fit review at the full quadruped's exact placements."""

from __future__ import annotations

from pathlib import Path

from build123d import Color, Location

from robot_cad.assembly import quadruped_robot
from robot_cad.parts import (
    adafruit_bno085,
    lekiwi_12v_battery_reference,
    quadruped_body,
    quadruped_body_lid,
    quadruped_electronics_tray,
    waveshare_bus_servo_adapter_a,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COMPONENT_ORDER = (
    "body_base",
    "body_battery",
    "electronics_tray",
    "body_servo_bus_adapter",
    "body_imu",
    "body_lid",
    "lekiwi_camera_assembly",
)


def gen_step():
    """Return body hardware without the four heavy detailed leg modules."""
    from cadpy.assembly import AssemblyHelper

    asm = AssemblyHelper("quadruped_body_hardware_fit_preview")
    asm.add(
        quadruped_body.gen_step(),
        "body_base",
        color=Color(0.16, 0.23, 0.32),
    )
    asm.add(
        lekiwi_12v_battery_reference.gen_step().moved(
            quadruped_robot.battery_location()
        ),
        "body_battery",
        color=Color(0.12, 0.14, 0.16),
    )
    asm.add(
        quadruped_electronics_tray.gen_step().moved(
            Location((0.0, 0.0, quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM))
        ),
        "electronics_tray",
        color=Color(0.95, 0.70, 0.18),
    )
    asm.add(
        waveshare_bus_servo_adapter_a.make_fit_proxy().moved(
            quadruped_robot.servo_bus_adapter_location()
        ),
        "body_servo_bus_adapter",
        color=Color(0.12, 0.48, 0.24),
    )
    asm.add(
        adafruit_bno085.gen_step().moved(
            Location(
                (
                    0.0,
                    0.0,
                    (
                        quadruped_body.ELECTRONICS_TRAY_BOTTOM_Z_MM
                        + quadruped_electronics_tray.IMU_BOARD_BOTTOM_Z_MM
                    ),
                )
            )
        ),
        "body_imu",
        color=Color(0.12, 0.30, 0.62),
    )
    asm.add(
        quadruped_body_lid.gen_step().moved(
            Location((0.0, 0.0, quadruped_body.BODY_BASE_HEIGHT_Z_MM))
        ),
        "body_lid",
        color=Color(0.38, 0.48, 0.60),
    )
    asm.add_module(
        "lekiwi_camera_assembly",
        quadruped_robot._camera_reference_children(),
    )
    return asm.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT
        / "exports"
        / "step"
        / "quadruped_body_hardware_fit_preview.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
