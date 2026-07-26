"""Removable ventilated electronics tray above the quadruped battery bay."""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Box, Cylinder, Location, Shape

from robot_cad.parts import quadruped_body

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAY_CORNER_RADIUS_MM = 8.0
TRAY_M3_CLEARANCE_DIAMETER_MM = 3.4
TRAY_VENT_SLOT_LENGTH_X_MM = 28.0
TRAY_VENT_SLOT_WIDTH_Y_MM = 4.0
TRAY_VENT_X_MM = (-52.0, 0.0, 52.0)
TRAY_VENT_Y_MM = (-34.0, 0.0, 34.0)


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def make_tray_blank() -> Shape:
    """Return the rounded tray plate before holes and ventilation."""
    return quadruped_body._rounded_prism(
        quadruped_body.ELECTRONICS_TRAY_LENGTH_X_MM,
        quadruped_body.ELECTRONICS_TRAY_WIDTH_Y_MM,
        quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM,
        minimum_z_mm=0.0,
        corner_radius_mm=TRAY_CORNER_RADIUS_MM,
    )


def make_mount_hole_tools() -> tuple[Shape, ...]:
    """Return four M3 clearance cutters aligned to the body standoffs."""
    cutter_height = (
        quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Cylinder(
            TRAY_M3_CLEARANCE_DIAMETER_MM / 2.0,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    -quadruped_body.BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for x_mm, y_mm in quadruped_body.TRAY_STANDOFF_CENTERS_XY_MM
    )


def make_vent_tools() -> tuple[Shape, ...]:
    """Return a nine-slot tray ventilation and wire-routing pattern."""
    cutter_height = (
        quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Box(
            TRAY_VENT_SLOT_LENGTH_X_MM,
            TRAY_VENT_SLOT_WIDTH_Y_MM,
            cutter_height,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    -quadruped_body.BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for x_mm in TRAY_VENT_X_MM
        for y_mm in TRAY_VENT_Y_MM
    )


def make_electronics_tray() -> Shape:
    """Return the complete removable electronics tray."""
    finished = make_tray_blank() - (
        make_mount_hole_tools() + make_vent_tools()
    )
    return _one_valid_solid(finished, "quadruped_electronics_tray")


def gen_step() -> Shape:
    """Return the STEP-ready electronics tray."""
    return make_electronics_tray()


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT / "exports" / "step" / "quadruped_electronics_tray.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
