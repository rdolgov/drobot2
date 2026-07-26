"""Exact catalog ST3215 geometry exported for URDF/Isaac visual use.

The immutable step.parts model remains under ``vendor/``.  This generator
creates a derived STEP/STL visual asset under ``exports/`` without modifying
or simplifying the vendor geometry.  Physics continues to use the audited
55 g mass and box collision/inertia model because the catalog STEP contains
one invalid solid.
"""

from __future__ import annotations

from pathlib import Path

from build123d import import_step

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ST3215_VENDOR_STEP = (
    PROJECT_ROOT
    / "vendor"
    / "servos"
    / "waveshare_feetech_st3215_servo.step"
)


def gen_step():
    """Return the unmodified exact Waveshare/Feetech ST3215 catalog model."""
    servo = import_step(ST3215_VENDOR_STEP)
    servo.label = "waveshare_feetech_st3215_servo_visual"
    return servo


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "st3215_servo_visual.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
