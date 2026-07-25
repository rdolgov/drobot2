"""Reusable SO-101 upper arm built from the original reference STEP.

The imported STEP is a boundary-representation model, so its original Fusion
feature history is not recoverable. This file keeps that reference immutable,
removes its perforated mounting panel, imports the shared ST3215 rear motor-bay
generator unchanged, and blends the bay into the arm as one printable solid.

Coordinate convention:
    - Units are millimeters.
    - Coordinates `initially match the immutable SO-101 reference STEP.
    - The perforated mounting panel occupies the negative-X end.
    - The optional box cut is evaluated in the original part coordinates.
    - The shared ST3215 motor bay extends the arm toward negative X.
    - The installed servo's local X axis follows global X, its local Z axis
      follows global -Y, and its local Y axis follows global +Z.
    - Rotations are applied about the configured pivot in X, Y, Z order.
    - Translation is applied after all rotations.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from build123d import (
    Align,
    Axis,
    Box,
    BuildSketch,
    Ellipse,
    Location,
    Locations,
    Plane,
    Rectangle,
    RectangleRounded,
    Shape,
    Vector,
    extrude,
    import_step,
    loft,
)

from robot_cad.parts import st3215_motor_bay

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_STEP = (
    PROJECT_ROOT
    / "vendor"
    / "references"
    / "so101"
    / "Upper_arm_SO101.step"
)

# Remove the complete perforated mounting panel marked in the Back
# orthographic view.  The reference panel joins the main body at
# X=-29.084989 mm; a small positive-X overtravel avoids a coincident boolean.
REMOVE_PERFORATED_MOUNT_PANEL = True
MOUNT_PANEL_CUT_PLANE_X_MM = -29.0
MOUNT_PANEL_CUT_OVERSIZE_MM = 5.0

# The common bay keeps its own fit-critical dimensions and cavity construction.
# This component supplies only placement and the arm-side blending geometry.
ADD_ST3215_REAR_MOTOR_BAY = True
ST3215_MOTOR_BAY_CENTER_Y_MM = 12.0
ST3215_MOTOR_BAY_CENTER_Z_MM = -1.95
ST3215_MOTOR_BAY_JOIN_OVERLAP_MM = 0.8
UPPER_ARM_NEGATIVE_X_EXTENSION_MM = 30.0

# The transition repeats its start/end profiles to make the loft leave the
# straight bay and arrive at the arm gradually. It intentionally closes the
# short gap between the arm's upper and lower edge lobes; the shared bay already
# has a closed attachment wall and therefore still needs a later cable route.
# The negative-X extension moves only the bay-side profiles; the arm-side
# profiles and positive-X body remain fixed in the immutable reference frame.
SMOOTH_TRANSITION_LENGTH_MM = 9.0
SMOOTH_TRANSITION_BAY_EMBED_MM = 0.4
SMOOTH_TRANSITION_END_HOLD_MM = 1.0
SMOOTH_TRANSITION_START_RADIUS_MM = 0.5
SMOOTH_TRANSITION_ARM_SIZE_Y_MM = 24.0
SMOOTH_TRANSITION_ARM_SIZE_Z_MM = 50.0
SMOOTH_TRANSITION_ARM_RADIUS_MM = 5.0

# Enlarge the reference arm's small half-circle into a rounded through-slot
# across the middle of the extended transition. The negative-X tip stays clear
# of the motor-bay attachment datum, so the shared bay is not cut or modified.
ENLARGE_MIDDLE_HALF_CIRCLE = True
MIDDLE_OPENING_LEFT_CAP_CENTER_X_MM = -44.0
MIDDLE_OPENING_RIGHT_CAP_CENTER_X_MM = -20.0
MIDDLE_OPENING_CAP_RADIUS_X_MM = 10.0
MIDDLE_OPENING_HEIGHT_Z_MM = 20.0
MIDDLE_OPENING_CENTER_Z_MM = 0.0
MIDDLE_OPENING_OVERTRAVEL_MM = 1.0

# Optional subtractive edit.  Leave disabled until the intended cut region is
# needed.  The box is centered at CUT_BOX_CENTER_MM and may safely extend
# beyond the imported body.
CUT_ENABLED = False
CUT_BOX_SIZE_MM = (20.0, 30.0, 20.0)
CUT_BOX_CENTER_MM = (0.0, 12.25, 0.0)

# Final pose controls.  The default pivot is near the inspected body center.
# Change it to a measured servo axis or mating datum before posing a real link.
ROTATION_PIVOT_MM = (5.9845195, 12.25, -1.95)
ROTATION_X_DEG = 0.0
ROTATION_Y_DEG = 0.0
ROTATION_Z_DEG = 0.0
TRANSLATION_MM = (0.0, 0.0, 0.0)

# Inspected revolute-joint datums used by the two-link assembly.  The distal
# fork is coaxial about global Z.  The ST3215 output axis is 25.5 mm along
# negative local X from the catalog origin; its installed axis is also global
# Z and is represented at the arm's Z=0 center plane.
DISTAL_FORK_AXIS_MM = (65.084989, 12.0, 0.0)
ST3215_CATALOG_OUTPUT_AXIS_OFFSET_X_MM = -25.5


def _validated_xyz(name: str, values: tuple[float, float, float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return tuple(float(value) for value in values)


def _load_st3215_motor_bay_module() -> ModuleType:
    """Return the reusable package module that owns the fit-critical cavity."""
    return st3215_motor_bay


def _load_reference_body() -> Shape:
    """Load the one non-empty solid from the source STEP assembly."""
    imported = import_step(REFERENCE_STEP)
    solids = list(imported.solids())
    if len(solids) != 1:
        raise RuntimeError(
            f"Expected one non-empty upper-arm solid in {REFERENCE_STEP.name}, "
            f"found {len(solids)}"
        )
    body = solids[0]
    body.label = "upper_arm_so101"
    return body


def _apply_optional_cut(body: Shape) -> Shape:
    result = body

    if REMOVE_PERFORATED_MOUNT_PANEL:
        bounds = result.bounding_box()
        oversize = float(MOUNT_PANEL_CUT_OVERSIZE_MM)
        if oversize <= 0.0:
            raise ValueError("MOUNT_PANEL_CUT_OVERSIZE_MM must be positive")

        cut_min_x = bounds.min.X - oversize
        cut_max_x = float(MOUNT_PANEL_CUT_PLANE_X_MM)
        if not bounds.min.X < cut_max_x < bounds.max.X:
            raise ValueError("MOUNT_PANEL_CUT_PLANE_X_MM must intersect the upper arm")

        cutter_size = (
            cut_max_x - cut_min_x,
            bounds.size.Y + 2.0 * oversize,
            bounds.size.Z + 2.0 * oversize,
        )
        cutter_center = (
            (cut_min_x + cut_max_x) / 2.0,
            (bounds.min.Y + bounds.max.Y) / 2.0,
            (bounds.min.Z + bounds.max.Z) / 2.0,
        )
        mount_panel_cutter = Box(
            *cutter_size,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        ).moved(Location(cutter_center))
        result = result - mount_panel_cutter

    if CUT_ENABLED:
        size = _validated_xyz("CUT_BOX_SIZE_MM", CUT_BOX_SIZE_MM)
        if any(dimension <= 0.0 for dimension in size):
            raise ValueError("Every CUT_BOX_SIZE_MM dimension must be positive")

        center = _validated_xyz("CUT_BOX_CENTER_MM", CUT_BOX_CENTER_MM)
        cutter = Box(
            *size,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        ).moved(Location(center))
        result = result - cutter

    solids = list(result.solids())
    if not solids:
        raise RuntimeError("The configured cut removed the entire upper arm")
    if len(solids) != 1:
        raise RuntimeError(f"The configured cut split the upper arm into {len(solids)} solids")
    result.label = "upper_arm_so101_cut"
    return result


def _motor_bay_location() -> Location:
    """Place the bay's X=0 datum beyond the extended negative-X arm end."""
    overlap = float(ST3215_MOTOR_BAY_JOIN_OVERLAP_MM)
    extension = float(UPPER_ARM_NEGATIVE_X_EXTENSION_MM)
    if overlap <= 0.0:
        raise ValueError("ST3215_MOTOR_BAY_JOIN_OVERLAP_MM must be positive")
    if extension < 0.0:
        raise ValueError("UPPER_ARM_NEGATIVE_X_EXTENSION_MM cannot be negative")
    return Location(
        (
            float(MOUNT_PANEL_CUT_PLANE_X_MM) + overlap - extension,
            float(ST3215_MOTOR_BAY_CENTER_Y_MM),
            float(ST3215_MOTOR_BAY_CENTER_Z_MM),
        )
    )


def _transition_profile(
    *,
    x: float,
    center_y: float,
    center_z: float,
    size_y: float,
    size_z: float,
    radius: float,
) -> Shape:
    """Create one rounded YZ section for the smooth bay-to-arm loft."""
    if min(size_y, size_z, radius) <= 0.0:
        raise ValueError("Transition profile dimensions and radius must be positive")
    if 2.0 * radius >= min(size_y, size_z):
        raise ValueError("Transition radius must be smaller than half the profile")

    yz_plane = Plane(
        origin=(x, center_y, center_z),
        x_dir=(0.0, 1.0, 0.0),
        z_dir=(1.0, 0.0, 0.0),
    )
    with BuildSketch(yz_plane) as profile:
        RectangleRounded(size_y, size_z, radius)
    return profile.sketch.faces()[0]


def _make_motor_bay_transition(motor_bay_module: ModuleType) -> Shape:
    """Loft a gradual, solid blend from the shared bay into the arm."""
    transition_length = float(SMOOTH_TRANSITION_LENGTH_MM)
    bay_embed = float(SMOOTH_TRANSITION_BAY_EMBED_MM)
    end_hold = float(SMOOTH_TRANSITION_END_HOLD_MM)
    start_radius = float(SMOOTH_TRANSITION_START_RADIUS_MM)
    arm_radius = float(SMOOTH_TRANSITION_ARM_RADIUS_MM)
    if min(transition_length, bay_embed, end_hold) <= 0.0:
        raise ValueError("Transition length, embed, and end hold must be positive")
    if end_hold >= transition_length:
        raise ValueError("Transition end hold must be shorter than its total length")

    clearance_y = float(motor_bay_module.SOCKET_CLEARANCE_Y_PER_SIDE_MM)
    clearance_z = float(motor_bay_module.SOCKET_CLEARANCE_Z_TOTAL_MM)
    wall = float(motor_bay_module.SOCKET_WALL_MM)
    bay_size_y = (
        float(motor_bay_module.ST3215_CATALOG_WIDTH_Z_MM)
        + 2.0 * clearance_y
        + 2.0 * wall
    )
    bay_size_z = (
        float(motor_bay_module.ST3215_CATALOG_HEIGHT_Y_MM)
        + clearance_z
        + 2.0 * wall
    )

    cut_x = float(MOUNT_PANEL_CUT_PLANE_X_MM)
    extension = float(UPPER_ARM_NEGATIVE_X_EXTENSION_MM)
    attach_x = _motor_bay_location().position.X
    transition_end_x = cut_x + transition_length
    arm_center_y = float(ST3215_MOTOR_BAY_CENTER_Y_MM)
    arm_center_z = 0.0
    profiles = [
        _transition_profile(
            x=cut_x - extension - bay_embed,
            center_y=arm_center_y,
            center_z=float(ST3215_MOTOR_BAY_CENTER_Z_MM),
            size_y=bay_size_y,
            size_z=bay_size_z,
            radius=start_radius,
        ),
        _transition_profile(
            x=attach_x,
            center_y=arm_center_y,
            center_z=float(ST3215_MOTOR_BAY_CENTER_Z_MM),
            size_y=bay_size_y,
            size_z=bay_size_z,
            radius=start_radius,
        ),
        _transition_profile(
            x=transition_end_x - end_hold,
            center_y=arm_center_y,
            center_z=arm_center_z,
            size_y=float(SMOOTH_TRANSITION_ARM_SIZE_Y_MM),
            size_z=float(SMOOTH_TRANSITION_ARM_SIZE_Z_MM),
            radius=arm_radius,
        ),
        _transition_profile(
            x=transition_end_x,
            center_y=arm_center_y,
            center_z=arm_center_z,
            size_y=float(SMOOTH_TRANSITION_ARM_SIZE_Y_MM),
            size_z=float(SMOOTH_TRANSITION_ARM_SIZE_Z_MM),
            radius=arm_radius,
        ),
    ]
    transition = loft(profiles)
    if len(transition.solids()) != 1 or not transition.is_valid:
        raise RuntimeError("The ST3215 motor-bay transition is not one valid solid")
    transition.label = "st3215_motor_bay_smooth_transition"
    return transition


def _add_st3215_rear_motor_bay(body: Shape) -> Shape:
    """Fuse the unchanged common motor bay and its smooth arm transition."""
    if not ADD_ST3215_REAR_MOTOR_BAY:
        return body

    motor_bay_module = _load_st3215_motor_bay_module()
    motor_bay = motor_bay_module.gen_step()
    if len(motor_bay.solids()) != 1 or not motor_bay.is_valid:
        raise RuntimeError("The shared ST3215 motor bay is not one valid solid")

    placed_motor_bay = motor_bay.moved(_motor_bay_location())
    transition = _make_motor_bay_transition(motor_bay_module)
    result = body.fuse(placed_motor_bay, transition)
    solids = list(result.solids())
    if len(solids) != 1 or not result.is_valid:
        raise RuntimeError(
            "The ST3215 motor bay and transition must fuse into one valid "
            f"upper-arm solid; found {len(solids)} solids"
        )
    result.label = "upper_arm_so101_with_st3215_rear_motor_bay"
    return result


def _enlarge_middle_half_circle(body: Shape) -> Shape:
    """Extend the existing middle half-circle into a large rounded slot."""
    if not ENLARGE_MIDDLE_HALF_CIRCLE:
        return body

    left_cap_x = float(MIDDLE_OPENING_LEFT_CAP_CENTER_X_MM)
    right_cap_x = float(MIDDLE_OPENING_RIGHT_CAP_CENTER_X_MM)
    cap_radius_x = float(MIDDLE_OPENING_CAP_RADIUS_X_MM)
    height_z = float(MIDDLE_OPENING_HEIGHT_Z_MM)
    center_z = float(MIDDLE_OPENING_CENTER_Z_MM)
    overtravel = float(MIDDLE_OPENING_OVERTRAVEL_MM)
    if min(cap_radius_x, height_z, overtravel) <= 0.0:
        raise ValueError("Middle-opening dimensions must be positive")
    straight_length_x = right_cap_x - left_cap_x
    if straight_length_x <= 0.0:
        raise ValueError("Middle-opening cap centers must increase along X")

    negative_x_tip = left_cap_x - cap_radius_x
    bay_attachment_x = _motor_bay_location().position.X
    if negative_x_tip <= bay_attachment_x:
        raise ValueError("Middle opening must remain clear of the motor-bay datum")

    bounds = body.bounding_box()
    through_width_y = bounds.size.Y + 2.0 * overtravel
    xz_plane = Plane(
        origin=(0.0, bounds.max.Y + overtravel, 0.0),
        x_dir=(1.0, 0.0, 0.0),
        z_dir=(0.0, -1.0, 0.0),
    )
    with BuildSketch(xz_plane) as cavity_profile:
        with Locations(((left_cap_x + right_cap_x) / 2.0, center_z)):
            Rectangle(straight_length_x, height_z)
        with Locations((left_cap_x, center_z), (right_cap_x, center_z)):
            Ellipse(cap_radius_x, height_z / 2.0)

    cutter = extrude(cavity_profile.sketch, amount=through_width_y)
    result = body - cutter
    solids = list(result.solids())
    if len(solids) != 1 or not result.is_valid:
        raise RuntimeError(
            "The enlarged middle opening must leave one valid upper-arm solid; "
            f"found {len(solids)} solids"
        )
    result.label = "upper_arm_so101_with_enlarged_middle_opening"
    return result


def st3215_preview_location() -> Location:
    """Return the catalog servo pose from the shared bay in arm coordinates."""
    motor_bay_module = _load_st3215_motor_bay_module()
    return _motor_bay_location() * motor_bay_module.st3215_installed_location()


def st3215_output_axis_location() -> Location:
    """Return a point on the installed servo's global-Z output axis."""
    servo_origin = st3215_preview_location().position
    return Location(
        (
            servo_origin.X + float(ST3215_CATALOG_OUTPUT_AXIS_OFFSET_X_MM),
            servo_origin.Y,
            0.0,
        )
    )


def distal_fork_axis_location() -> Location:
    """Return the placed reusable fork's global-Z revolute axis."""
    return Location(_validated_xyz("DISTAL_FORK_AXIS_MM", DISTAL_FORK_AXIS_MM))


def _apply_pose(body: Shape) -> Shape:
    pivot = Vector(*_validated_xyz("ROTATION_PIVOT_MM", ROTATION_PIVOT_MM))
    posed = body
    for direction, angle in (
        ((1.0, 0.0, 0.0), ROTATION_X_DEG),
        ((0.0, 1.0, 0.0), ROTATION_Y_DEG),
        ((0.0, 0.0, 1.0), ROTATION_Z_DEG),
    ):
        if angle:
            posed = posed.rotate(Axis(pivot, direction), float(angle))

    translation = _validated_xyz("TRANSLATION_MM", TRANSLATION_MM)
    if any(translation):
        posed = posed.moved(Location(translation))
    posed.label = "upper_arm_so101_editable"
    return posed


def gen_step() -> Shape:
    """Return the complete STEP-ready upper arm with its integral output fork."""
    body = _load_reference_body()
    body = _apply_optional_cut(body)
    body = _add_st3215_rear_motor_bay(body)
    body = _enlarge_middle_half_circle(body)
    return _apply_pose(body)


if __name__ == "__main__":
    from build123d import export_step

    output_path = Path("exports/step/upper_arm.step")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
