"""Distal lower-leg fit preview for the rigid low-profile shoe."""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Color, Cylinder, Location

from drobot_cad.parts import rigid_fork_shoe, upper_arm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROD_ENVELOPE_DIAMETER_MM = 3.0
ROD_ENVELOPE_LENGTH_MM = 75.0


def shoe_location() -> Location:
    """Place shoe-local origin on the lower leg's distal fork axis."""
    return Location(upper_arm.DISTAL_FORK_AXIS_MM)


def make_rod_envelope(x_mm: float, y_mm: float):
    """Return one M3 x 75 mm threaded-rod fit envelope."""
    return Cylinder(
        ROD_ENVELOPE_DIAMETER_MM / 2.0,
        ROD_ENVELOPE_LENGTH_MM,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(shoe_location() * Location((x_mm, y_mm, 0.0)))


def gen_step():
    """Return lower leg, installed rigid shoe, and two rod envelopes."""
    from cadpy.assembly import AssemblyHelper

    asm = AssemblyHelper("rigid_fork_shoe_fit_preview")
    asm.add(
        upper_arm.gen_step(),
        "existing_distal_lower_leg",
        color=Color(0.22, 0.55, 0.82),
    )
    asm.add(
        rigid_fork_shoe.gen_step().moved(shoe_location()),
        "printable_rigid_fork_shoe",
        color=Color(0.84, 0.48, 0.12),
    )
    for index, (x_mm, y_mm) in enumerate(
        rigid_fork_shoe.RECOMMENDED_ROD_HOLE_CENTERS_XY_MM,
        start=1,
    ):
        asm.add(
            make_rod_envelope(x_mm, y_mm),
            f"m3x75_threaded_rod_envelope_{index}",
            color=Color(0.72, 0.74, 0.78),
        )
    return asm.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT / "exports" / "step" / "rigid_fork_shoe_fit_preview.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
