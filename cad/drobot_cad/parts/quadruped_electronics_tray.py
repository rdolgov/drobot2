"""Removable ventilated electronics tray above the quadruped battery bay."""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Box, Cylinder, Location, Shape

from drobot_cad.parts import (
    adafruit_bno085,
    quadruped_body,
    waveshare_bus_servo_adapter_a,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAY_CORNER_RADIUS_MM = 8.0
TRAY_M3_CLEARANCE_DIAMETER_MM = 3.4
TRAY_VENT_SLOT_LENGTH_X_MM = 28.0
TRAY_VENT_SLOT_WIDTH_Y_MM = 4.0
TRAY_VENT_X_MM = (-52.0, 0.0, 52.0)
TRAY_VENT_Y_MM = (-34.0, 0.0, 34.0)

# Rigid, body-axis-aligned IMU interface.  M2 nylon hardware avoids putting
# steel fasteners beside the BNO085 magnetometer.  The sensing package, rather
# than the asymmetric PCB outline, is centred on the robot body origin.
IMU_M2_CLEARANCE_DIAMETER_MM = 2.4
IMU_STANDOFF_OUTER_DIAMETER_MM = 6.0
IMU_STANDOFF_HEIGHT_MM = 4.0
IMU_STANDOFF_FUSION_OVERLAP_MM = 0.5
IMU_BOARD_BOTTOM_Z_MM = (
    quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM + IMU_STANDOFF_HEIGHT_MM
)
IMU_MOUNT_CENTERS_XY_MM = (
    adafruit_bno085.MOUNT_HOLE_CENTERS_SENSOR_XY_MM
)

# Rear-left controller position leaves the body-centred IMU clear while
# keeping the USB, power, and servo connectors inside the serviceable tray
# area.  The exact controller's underside components stop 0.4 mm above the
# tray when its PCB datum is seated on these 4 mm standoffs.
SERVO_ADAPTER_CENTER_XY_MM = (-48.0, 32.0)
SERVO_ADAPTER_M2_CLEARANCE_DIAMETER_MM = 2.4
SERVO_ADAPTER_STANDOFF_OUTER_DIAMETER_MM = 6.0
SERVO_ADAPTER_STANDOFF_HEIGHT_MM = 4.0
SERVO_ADAPTER_STANDOFF_FUSION_OVERLAP_MM = 0.5
SERVO_ADAPTER_BOARD_DATUM_Z_MM = (
    quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM
    + SERVO_ADAPTER_STANDOFF_HEIGHT_MM
)
SERVO_ADAPTER_MOUNT_CENTERS_XY_MM = tuple(
    (
        SERVO_ADAPTER_CENTER_XY_MM[0] + local_x_mm,
        SERVO_ADAPTER_CENTER_XY_MM[1] + local_y_mm,
    )
    for local_x_mm, local_y_mm in (
        waveshare_bus_servo_adapter_a.MOUNT_HOLE_CENTERS_XY_MM
    )
)


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
        for x_mm, y_mm in quadruped_body.TRAY_MOUNT_CENTERS_XY_MM
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


def make_imu_standoffs() -> tuple[Shape, ...]:
    """Return four tray-connected standoffs for the exact BNO085 pattern."""
    return tuple(
        Cylinder(
            IMU_STANDOFF_OUTER_DIAMETER_MM / 2.0,
            IMU_STANDOFF_HEIGHT_MM + IMU_STANDOFF_FUSION_OVERLAP_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM
                    - IMU_STANDOFF_FUSION_OVERLAP_MM,
                )
            )
        )
        for x_mm, y_mm in IMU_MOUNT_CENTERS_XY_MM
    )


def make_imu_mount_hole_tools() -> tuple[Shape, ...]:
    """Return four M2 clearance cutters through tray and IMU standoffs."""
    cutter_height = (
        IMU_BOARD_BOTTOM_Z_MM + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Cylinder(
            IMU_M2_CLEARANCE_DIAMETER_MM / 2.0,
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
        for x_mm, y_mm in IMU_MOUNT_CENTERS_XY_MM
    )


def make_servo_adapter_standoffs() -> tuple[Shape, ...]:
    """Return tray-connected standoffs for the exact Waveshare board."""
    return tuple(
        Cylinder(
            SERVO_ADAPTER_STANDOFF_OUTER_DIAMETER_MM / 2.0,
            (
                SERVO_ADAPTER_STANDOFF_HEIGHT_MM
                + SERVO_ADAPTER_STANDOFF_FUSION_OVERLAP_MM
            ),
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    x_mm,
                    y_mm,
                    (
                        quadruped_body.ELECTRONICS_TRAY_THICKNESS_Z_MM
                        - SERVO_ADAPTER_STANDOFF_FUSION_OVERLAP_MM
                    ),
                )
            )
        )
        for x_mm, y_mm in SERVO_ADAPTER_MOUNT_CENTERS_XY_MM
    )


def make_servo_adapter_mount_hole_tools() -> tuple[Shape, ...]:
    """Return four M2 clearance cutters through the controller standoffs."""
    cutter_height = (
        SERVO_ADAPTER_BOARD_DATUM_Z_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM
    )
    return tuple(
        Cylinder(
            SERVO_ADAPTER_M2_CLEARANCE_DIAMETER_MM / 2.0,
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
        for x_mm, y_mm in SERVO_ADAPTER_MOUNT_CENTERS_XY_MM
    )


def make_electronics_tray() -> Shape:
    """Return the complete removable electronics tray."""
    reinforced_blank = make_tray_blank().fuse(
        *make_imu_standoffs(),
        *make_servo_adapter_standoffs(),
    )
    finished = reinforced_blank - (
        make_mount_hole_tools()
        + make_vent_tools()
        + make_imu_mount_hole_tools()
        + make_servo_adapter_mount_hole_tools()
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
