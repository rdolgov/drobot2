"""Reusable fork for the ST3215 servo output joint.

The fork is extracted from the immutable SO-101 upper-arm reference at the
front-view markup cut, then extended with a smooth half-oval fusion tongue.
Its local origin remains the center of the legacy split plane:

    - local +X is normal to the split datum and points toward the output axis
    - local +Y follows the original upper-arm +Y direction
    - the preserved distal revolute axis is the ST3215 output-joint datum

The source cut remains available for consumers that need the fork as a
standalone reusable component.  The main upper-arm generator keeps the
positive-X fork as integral geometry.  The standalone fork's negative-X
extension is an intentional overlap envelope for boolean fusion into another
part; it is not a clearance-fit attachment feature.
"""

from __future__ import annotations

from math import atan, cos, degrees, hypot, sin, sqrt
from pathlib import Path
from warnings import catch_warnings, simplefilter

from build123d import (
    Align,
    Axis,
    BuildSketch,
    Ellipse,
    GeomType,
    Location,
    Mode,
    Plane,
    Rectangle,
    Shape,
    extrude,
    fillet,
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

# Smooth half-oval requested in the 2026-07-25 front-view markup and enlarged
# in the follow-up to cover the complete X=0 root edge.  Its nose reaches
# exactly 30 mm into negative X.  It is extruded through the existing 24 mm
# fork width and its two exposed elliptical rim edges are rounded.
ROOT_EXTENSION_CENTER_X_MM = 0.0
ROOT_EXTENSION_CENTER_Z_MM = 0.0
ROOT_EXTENSION_RADIUS_X_MM = 30.0
ROOT_EXTENSION_RADIUS_Z_MM = 31.7
ROOT_EXTENSION_WIDTH_Y_MM = 24.0
ROOT_EXTENSION_EDGE_FILLET_MM = 3.0
ROOT_EXTENSION_PROFILE_MARGIN_MM = 0.1

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


def make_root_extension() -> Shape:
    """Return the rounded half-elliptical negative-X fusion envelope."""
    radius_x = float(ROOT_EXTENSION_RADIUS_X_MM)
    radius_z = float(ROOT_EXTENSION_RADIUS_Z_MM)
    width_y = float(ROOT_EXTENSION_WIDTH_Y_MM)
    fillet_radius = float(ROOT_EXTENSION_EDGE_FILLET_MM)
    profile_margin = float(ROOT_EXTENSION_PROFILE_MARGIN_MM)
    if min(radius_x, radius_z, width_y, fillet_radius, profile_margin) <= 0.0:
        raise ValueError("Root-extension dimensions must be positive")
    if fillet_radius >= width_y / 2.0:
        raise ValueError(
            "ROOT_EXTENSION_EDGE_FILLET_MM must be less than half the Y width"
        )

    profile_plane = Plane(
        origin=(0.0, -width_y / 2.0, 0.0),
        x_dir=(1.0, 0.0, 0.0),
        z_dir=(0.0, 1.0, 0.0),
    )
    with BuildSketch(profile_plane) as profile:
        Ellipse(radius_x, radius_z)
        Rectangle(
            radius_x + profile_margin,
            2.0 * (radius_z + profile_margin),
            align=(Align.MAX, Align.CENTER),
            mode=Mode.INTERSECT,
        )

    raw_extension = extrude(profile.sketch, amount=width_y)
    elliptical_rims = [
        edge
        for edge in raw_extension.edges()
        if edge.geom_type == GeomType.ELLIPSE
    ]
    if len(elliptical_rims) != 2:
        raise RuntimeError(
            "Root extension must expose exactly two elliptical rim edges; "
            f"found {len(elliptical_rims)}"
        )
    rounded_extension = fillet(elliptical_rims, fillet_radius)
    positioned_extension = rounded_extension.moved(
        Location(
            (
                float(ROOT_EXTENSION_CENTER_X_MM),
                0.0,
                float(ROOT_EXTENSION_CENTER_Z_MM),
            )
        )
    )
    return _one_valid_solid(
        positioned_extension,
        "st3215_servo_output_fork_root_extension",
    )


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


def gen_core_step() -> Shape:
    """Return the legacy positive-X fork core without the fusion envelope."""
    fork_global = retain_fork_side(_load_reference_body())
    return _one_valid_solid(
        to_local_shape(fork_global),
        "st3215_servo_output_fork_core",
    )


def gen_step() -> Shape:
    """Return the STEP-ready reusable fork with its full-edge fusion envelope."""
    return _one_valid_solid(
        gen_core_step().fuse(make_root_extension()),
        "st3215_servo_output_fork",
    )


if __name__ == "__main__":
    from build123d import export_step

    output_path = PROJECT_ROOT / "exports" / "step" / "st3215_servo_output_fork.step"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
