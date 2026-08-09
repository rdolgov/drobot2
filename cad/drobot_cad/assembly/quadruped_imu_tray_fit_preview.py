"""Fit preview for the centrally mounted exact Adafruit BNO085.

This is review geometry, not an additional printable part. The gold tray and
light cover are manufacturing geometry; the blue board is the exact vendor
STEP positioned at its installed height.
"""

from __future__ import annotations

from pathlib import Path

from build123d import Color, Location

from drobot_cad.parts import (
    adafruit_bno085,
    quadruped_electronics_tray,
    quadruped_imu_cover,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def gen_step():
    """Return the tray, exact BNO085, and protective cover."""
    from cadpy.assembly import AssemblyHelper

    asm = AssemblyHelper("quadruped_imu_tray_fit_preview")
    asm.add(
        quadruped_electronics_tray.gen_step(),
        "printable_electronics_tray",
        color=Color(0.95, 0.70, 0.18),
    )
    asm.add(
        adafruit_bno085.gen_step().moved(
            Location(
                (
                    0.0,
                    0.0,
                    quadruped_electronics_tray.IMU_BOARD_BOTTOM_Z_MM,
                )
            )
        ),
        "exact_adafruit_bno085_stemma_qt",
        color=Color(0.12, 0.30, 0.62),
    )
    asm.add(
        quadruped_imu_cover.gen_step().moved(
            Location(
                (
                    0.0,
                    0.0,
                    (
                        quadruped_electronics_tray.IMU_BOARD_BOTTOM_Z_MM
                        + adafruit_bno085.PCB_THICKNESS_MM
                    ),
                )
            )
        ),
        "printable_imu_cover",
        color=Color(0.72, 0.78, 0.84),
    )
    return asm.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT
        / "exports"
        / "step"
        / "quadruped_imu_tray_fit_preview.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
