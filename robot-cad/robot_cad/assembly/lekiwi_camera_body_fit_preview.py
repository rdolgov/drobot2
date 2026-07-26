"""Positioning preview for the upstream LeKiwi camera mount on the body lid.

The green mount and purple camera are documented fit/reference envelopes.
Manufacture the immutable upstream ``vendor/references/lekiwi`` STL rather than
this preview geometry.

Coordinate convention:
    - origin: quadruped lid footprint center at its seating datum
    - +X: robot front
    - +Y: robot left
    - +Z: up
"""

from __future__ import annotations

from pathlib import Path

from build123d import Align, Axis, Box, Color, Cylinder, Location, Shape

from robot_cad.parts import quadruped_body, quadruped_body_lid

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Measured immutable LeKiwi base_camera_mount.stl envelope and interface.
LEKIWI_MOUNT_MIN_X_MM = -5.0
LEKIWI_MOUNT_DEPTH_X_MM = 16.661073
LEKIWI_MOUNT_WIDTH_Y_MM = 48.0
LEKIWI_MOUNT_HEIGHT_Z_MM = 45.184589
LEKIWI_MOUNT_FOOT_THICKNESS_Z_MM = 3.0
LEKIWI_MOUNT_PANEL_THICKNESS_X_MM = 3.0
LEKIWI_MOUNT_HOLE_RADIUS_MM = (
    quadruped_body.UTILITY_M3_CLEARANCE_DIAMETER_MM / 2.0
)
LEKIWI_MOUNT_CABLE_CUTOUT_YZ_MM = (10.0, 8.0)

# Measured upstream Camera-Model-v3-1.stl envelope.
CAMERA_REFERENCE_DEPTH_X_MM = 21.5
CAMERA_REFERENCE_WIDTH_Y_MM = 38.0
CAMERA_REFERENCE_HEIGHT_Z_MM = 38.0
CAMERA_REFERENCE_BOTTOM_Z_MM = 4.0
CAMERA_LENS_RADIUS_MM = 6.5
CAMERA_LENS_PROTRUSION_X_MM = 5.0

MOUNT_WORLD_X_MM = quadruped_body_lid.LEKIWI_CAMERA_MOUNT_CENTER_X_MM
MOUNT_WORLD_Y_MM = 0.0
MOUNT_WORLD_Z_MM = quadruped_body.BODY_LID_THICKNESS_Z_MM


def _one_valid_solid(shape: Shape, label: str) -> Shape:
    solids = list(shape.solids())
    if len(solids) != 1 or not solids[0].is_valid:
        raise RuntimeError(
            f"{label} must be one connected valid solid; found {len(solids)} solids"
        )
    result = solids[0]
    result.label = label
    return result


def make_mount_reference_proxy() -> Shape:
    """Return a solid proxy with the upstream mount envelope and bolt row."""
    foot = Box(
        LEKIWI_MOUNT_DEPTH_X_MM,
        LEKIWI_MOUNT_WIDTH_Y_MM,
        LEKIWI_MOUNT_FOOT_THICKNESS_Z_MM,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    ).moved(Location((LEKIWI_MOUNT_MIN_X_MM, 0.0, 0.0)))
    panel = Box(
        LEKIWI_MOUNT_PANEL_THICKNESS_X_MM,
        LEKIWI_MOUNT_WIDTH_Y_MM,
        LEKIWI_MOUNT_HEIGHT_Z_MM,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    ).moved(Location((LEKIWI_MOUNT_MIN_X_MM, 0.0, 0.0)))
    mount = foot.fuse(panel)

    foot_holes = tuple(
        Cylinder(
            LEKIWI_MOUNT_HOLE_RADIUS_MM,
            LEKIWI_MOUNT_FOOT_THICKNESS_Z_MM
            + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    0.0,
                    y_mm,
                    -quadruped_body.BOOLEAN_OVERTRAVEL_MM,
                )
            )
        )
        for _, y_mm in quadruped_body_lid.LEKIWI_CAMERA_MOUNT_HOLE_CENTERS_XY_MM
    )
    cable_cutout = Box(
        LEKIWI_MOUNT_PANEL_THICKNESS_X_MM
        + 2.0 * quadruped_body.BOOLEAN_OVERTRAVEL_MM,
        LEKIWI_MOUNT_CABLE_CUTOUT_YZ_MM[0],
        LEKIWI_MOUNT_CABLE_CUTOUT_YZ_MM[1],
        align=(Align.MIN, Align.CENTER, Align.CENTER),
    ).moved(
        Location(
            (
                LEKIWI_MOUNT_MIN_X_MM - quadruped_body.BOOLEAN_OVERTRAVEL_MM,
                0.0,
                LEKIWI_MOUNT_HEIGHT_Z_MM * 0.72,
            )
        )
    )
    return _one_valid_solid(
        mount - (foot_holes + (cable_cutout,)),
        "lekiwi_base_camera_mount_reference_proxy",
    )


def make_camera_reference_proxy() -> Shape:
    """Return the measured camera envelope plus a forward-facing lens."""
    panel_front_x = (
        LEKIWI_MOUNT_MIN_X_MM + LEKIWI_MOUNT_PANEL_THICKNESS_X_MM
    )
    camera = Box(
        CAMERA_REFERENCE_DEPTH_X_MM,
        CAMERA_REFERENCE_WIDTH_Y_MM,
        CAMERA_REFERENCE_HEIGHT_Z_MM,
        align=(Align.MIN, Align.CENTER, Align.MIN),
    ).moved(
        Location(
            (
                panel_front_x,
                0.0,
                CAMERA_REFERENCE_BOTTOM_Z_MM,
            )
        )
    )
    lens = (
        Cylinder(
            CAMERA_LENS_RADIUS_MM,
            CAMERA_LENS_PROTRUSION_X_MM,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        .rotate(Axis.Y, 90.0)
        .moved(
            Location(
                (
                    panel_front_x + CAMERA_REFERENCE_DEPTH_X_MM,
                    0.0,
                    CAMERA_REFERENCE_BOTTOM_Z_MM
                    + CAMERA_REFERENCE_HEIGHT_Z_MM / 2.0,
                )
            )
        )
    )
    return _one_valid_solid(
        camera.fuse(lens),
        "arducam_5mp_reference_proxy",
    )


def mount_world_location() -> Location:
    """Return the upstream mount origin aligned to the lid's camera row."""
    return Location((MOUNT_WORLD_X_MM, MOUNT_WORLD_Y_MM, MOUNT_WORLD_Z_MM))


def gen_step():
    """Return a labeled lid, mount-reference, and camera-reference assembly."""
    from cadpy.assembly import AssemblyHelper

    asm = AssemblyHelper("lekiwi_camera_body_fit_preview")
    asm.add(
        quadruped_body_lid.gen_step(),
        "quadruped_body_lid",
        color=Color(0.38, 0.48, 0.60),
    )
    asm.add(
        make_mount_reference_proxy().moved(mount_world_location()),
        "lekiwi_base_camera_mount_reference",
        color=Color(0.25, 0.72, 0.38),
    )
    asm.add(
        make_camera_reference_proxy().moved(mount_world_location()),
        "arducam_5mp_reference",
        color=Color(0.62, 0.28, 0.78),
    )
    return asm.build()


if __name__ == "__main__":
    from build123d import export_step

    output_path = (
        PROJECT_ROOT / "exports" / "step" / "lekiwi_camera_body_fit_preview.step"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
