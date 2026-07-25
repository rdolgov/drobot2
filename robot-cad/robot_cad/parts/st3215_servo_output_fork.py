"""Reusable positive-X fork for the ST3215 servo output joint.

The fork is extracted from the immutable SO-101 upper-arm reference at the
front-view markup cut.  Its local origin is the center of that cut plane:

    - local +X is normal to the cut and points into the fork
    - local +Y follows the original upper-arm +Y direction
    - the preserved distal revolute axis is the ST3215 output-joint datum

The source cut is intentionally kept in one shared module so the shortened
upper arm and this reusable component cannot drift apart.
"""

from __future__ import annotations

from math import atan, cos, degrees, hypot, sin, sqrt
from pathlib import Path
from warnings import catch_warnings, simplefilter

from build123d import (
    Axis,
    BuildSketch,
    Location,
    Plane,
    Rectangle,
    Shape,
    extrude,
    import_step,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_STEP = (
    PROJECT_ROOT
    / "vendor"
    / "references"
    / "so101"
    / "Upper_arm_SO101.step"
)

# The updated front-view markup requests a true vertical root face.  A constant
# X=12 mm YZ split also moves the datum 2.7 mm into the arm-side material,
# leaving at least the project's 3 mm structural-wall baseline at the narrow
# centerline ligament between the two fork prongs.
CUT_PLANE_X_AT_Z0_MM = 12.0
CUT_PLANE_CENTER_Y_MM = 12.0
CUT_PLANE_SLOPE_X_PER_Z = 0.0
CUT_HALFSPACE_MARGIN_MM = 10.0

# Established positive-X fork/output-joint datum from the SO-101 reference.
DISTAL_OUTPUT_AXIS_GLOBAL_MM = (65.084989, 12.0, 0.0)
DISTAL_OUTPUT_AXIS_GLOBAL_DIRECTION = (0.0, 0.0, 1.0)


def cut_plane_origin_global() -> tuple[float, float, float]:
    """Return the marked cut-plane origin in upper-arm coordinates."""
    return (
        float(CUT_PLANE_X_AT_Z0_MM),
        float(CUT_PLANE_CENTER_Y_MM),
        0.0,
    )


def cut_plane_normal_global() -> tuple[float, float, float]:
    """Return the unit normal pointing from the arm into the output fork."""
    raw = (1.0, 0.0, -float(CUT_PLANE_SLOPE_X_PER_Z))
    magnitude = sqrt(sum(component * component for component in raw))
    return tuple(component / magnitude for component in raw)


def cut_plane_alignment_angle_deg() -> float:
    """Return the +Y rotation that maps the cut normal onto local +X."""
    return degrees(atan(-float(CUT_PLANE_SLOPE_X_PER_Z)))


def _cut_plane() -> Plane:
    return Plane(
        origin=cut_plane_origin_global(),
        x_dir=(0.0, 1.0, 0.0),
        z_dir=cut_plane_normal_global(),
    )


def _halfspace_tool(body: Shape, *, fork_side: bool) -> Shape:
    """Create an oversized prism on one side of the marked cut plane."""
    bounds = body.bounding_box()
    margin = float(CUT_HALFSPACE_MARGIN_MM)
    if margin <= 0.0:
        raise ValueError("CUT_HALFSPACE_MARGIN_MM must be positive")

    in_plane_y = bounds.size.Y + 2.0 * margin
    in_plane_z = 2.0 * hypot(bounds.size.X, bounds.size.Z) + 2.0 * margin
    normal_depth = hypot(bounds.size.X, bounds.size.Z) + 2.0 * margin
    with BuildSketch(_cut_plane()) as section:
        Rectangle(in_plane_y, in_plane_z)
    return extrude(
        section.sketch,
        amount=normal_depth if fork_side else -normal_depth,
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


def retain_arm_side(body: Shape) -> Shape:
    """Return the negative side of the marked cut for the shortened upper arm."""
    return _one_valid_solid(
        body.intersect(_halfspace_tool(body, fork_side=False)),
        "upper_arm_without_servo_output_fork",
    )


def retain_fork_side(body: Shape) -> Shape:
    """Return the connected positive side of the marked cut."""
    return _one_valid_solid(
        body.intersect(_halfspace_tool(body, fork_side=True)),
        "st3215_servo_output_fork_global",
    )


def _load_reference_body() -> Shape:
    imported = import_step(REFERENCE_STEP)
    solids = list(imported.solids())
    if len(solids) != 1:
        raise RuntimeError(
            f"Expected one non-empty upper-arm solid in {REFERENCE_STEP.name}, "
            f"found {len(solids)}"
        )
    return solids[0]


def to_local_shape(shape: Shape) -> Shape:
    """Move upper-arm-coordinate geometry into the reusable fork frame."""
    origin = cut_plane_origin_global()
    translated = shape.moved(Location(tuple(-value for value in origin)))
    localized = translated.rotate(Axis.Y, cut_plane_alignment_angle_deg())
    # Bake the compound Location into the B-rep before STEP/GLB generation.
    # This keeps CAD Viewer from applying the local-frame transform twice.
    with catch_warnings():
        simplefilter("ignore", DeprecationWarning)
        localized.relocate(Location())
    localized.label = "st3215_servo_output_fork"
    return localized


def placement_in_upper_arm_coordinates() -> Location:
    """Return the transform that places the local fork back on the upper arm."""
    return Location(
        cut_plane_origin_global(),
        (0.0, -cut_plane_alignment_angle_deg(), 0.0),
    )


def output_axis_location_local() -> Location:
    """Return a point on the preserved ST3215 output axis in fork coordinates."""
    origin = cut_plane_origin_global()
    relative = tuple(
        point - datum
        for point, datum in zip(
            DISTAL_OUTPUT_AXIS_GLOBAL_MM,
            origin,
            strict=True,
        )
    )
    angle = atan(-float(CUT_PLANE_SLOPE_X_PER_Z))
    return Location(
        (
            cos(angle) * relative[0] + sin(angle) * relative[2],
            relative[1],
            -sin(angle) * relative[0] + cos(angle) * relative[2],
        )
    )


def output_axis_direction_local() -> tuple[float, float, float]:
    """Return the preserved ST3215 output-axis direction in fork coordinates."""
    angle = atan(-float(CUT_PLANE_SLOPE_X_PER_Z))
    return (
        sin(angle),
        0.0,
        cos(angle),
    )


def gen_step() -> Shape:
    """Return the STEP-ready reusable ST3215 servo-output fork."""
    fork_global = retain_fork_side(_load_reference_body())
    return _one_valid_solid(
        to_local_shape(fork_global),
        "st3215_servo_output_fork",
    )


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "st3215_servo_output_fork.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
