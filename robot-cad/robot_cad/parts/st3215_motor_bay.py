"""Reusable rear-body bay for a Feetech/Waveshare ST3215 servo.

The printable bay is intentionally independent of any limb.  Its flat
attachment datum is the outer face at X=0, and the bay extends toward negative
X.  A future limb can import ``gen_step()`` and fuse that face to its own
geometry without copying the fit-critical servo cavity.

Coordinate convention:
    - Units are millimeters.
    - The attachment datum is the YZ plane at X=0.
    - The servo inserts from the open negative-X end.
    - Servo local X follows global X, local Z follows global -Y, and servo
      local Y follows global +Z.
"""

from __future__ import annotations

from pathlib import Path

from build123d import (
    Align,
    Box,
    Cylinder,
    Edge,
    Face,
    GeomType,
    Location,
    Shape,
    Vector,
    Wire,
    extrude,
    import_step,
    loft,
)

ST3215_SERVO_STEP = (
    Path(__file__).resolve().parents[2]
    / "vendor"
    / "servos"
    / "waveshare_feetech_st3215_servo.step"
)

# Dimensions inspected from the exact step.parts catalog model.
ST3215_CATALOG_HEIGHT_Y_MM = 37.8
ST3215_CATALOG_WIDTH_Z_MM = 24.7234
ST3215_CATALOG_MAX_X_MM = 9.6117

# Fit parameters. These are preliminary FDM test-fit allowances, not final
# production tolerances.
SOCKET_CLEARANCE_Y_PER_SIDE_MM = 0.25
SOCKET_CLEARANCE_Z_TOTAL_MM = 0.25
SOCKET_WALL_MM = 3.0
SOCKET_LENGTH_X_MM = 16.0
SOCKET_STOP_THICKNESS_MM = 1.5
BOOLEAN_OVERTRAVEL_MM = 0.6
OUTER_SIDE_PERIMETER_FILLET_RADIUS_MM = 2.0

# The stop is a perimeter rim, leaving this opening for the servo cable.
SOCKET_CABLE_WINDOW_Y_MM = 16.0
SOCKET_CABLE_WINDOW_Z_MM = 14.0

# Repeated through-wall ventilation on the two broad +/-Y side walls.  The
# diamonds are defined in the front-view XZ plane, matching CAD Viewer markup.
VENT_DIAMOND_COLUMNS_X_MM = (-11.25, -4.75)
VENT_DIAMOND_ROWS_Z_MM = (-12.0, -4.0, 4.0, 12.0)
VENT_DIAMOND_WIDTH_X_MM = 3.6
VENT_DIAMOND_HEIGHT_Z_MM = 5.2
VENT_WALL_OVERTRAVEL_MM = 0.6

# Four vertical access holes expose the ST3215 mounting positions visible in
# the left/right projections.  Their 20.5 mm spacing agrees with the
# manufacturer drawing; the staggered X locations come from the exact STEP.
SERVO_MOUNT_HOLE_Y_SPACING_MM = 20.5
SERVO_TOP_MOUNT_HOLE_X_MM = -6.8117
SERVO_BOTTOM_MOUNT_HOLE_X_MM = -3.0617
SERVO_MOUNT_THROUGH_DIAMETER_MM = 2.0
SERVO_MOUNT_COUNTERBORE_DIAMETER_MM = 4.0
SERVO_MOUNT_INNER_PILOT_DEPTH_MM = 2.95
# Construction clearance erases the imported servo's reverse-imprinted holes;
# concentric sleeves added afterward establish the smaller finished diameters.
SERVO_MOUNT_PROFILE_CLEARANCE_DIAMETER_MM = 5.4
SERVO_MOUNT_LOCAL_CLEANUP_SIZE_MM = 6.0
# The imported profile leaves two connected screw-tab imprints beside the
# positive-Z fitting floor.  Trim only their measured local envelope.
SERVO_TOP_PROFILE_PROTRUSION_MIN_X_MM = -3.9
SERVO_TOP_PROFILE_PROTRUSION_MAX_X_MM = -1.5
SERVO_TOP_PROFILE_PROTRUSION_CENTER_Y_MM = 10.0
SERVO_TOP_PROFILE_PROTRUSION_SIZE_Y_MM = 5.0
SERVO_TOP_MOUNT_FACE_Z_MM = 19.025
SERVO_BOTTOM_MOUNT_FACE_Z_MM = -19.025
SERVO_TOP_ACCESS_REACH_Z_MM = 14.0
SERVO_BOTTOM_ACCESS_REACH_Z_MM = -13.0
SERVO_LOWER_FITTING_FLOOR_Z_MM = -16.175
SERVO_LOWER_SIDE_RECESS_MIN_X_MM = -6.0617
SERVO_LOWER_SIDE_RECESS_INNER_Y_MM = 7.25
SERVO_UPPER_SIDE_POCKET_FACE_X_MM = -1.5
SERVO_UPPER_SIDE_POCKET_CENTER_Y_MM = 10.0
SERVO_UPPER_SIDE_POCKET_SIZE_Y_MM = 5.0
SERVO_UPPER_SIDE_POCKET_MIN_Z_MM = 13.4
SERVO_UPPER_SIDE_POCKET_MAX_Z_MM = 16.075

# Public datum: the outer flat face used to attach this bay to another part.
ATTACHMENT_DATUM_X_MM = 0.0


def st3215_installed_location() -> Location:
    """Return the exact catalog servo pose seated against the stop rim."""
    stop_x = ATTACHMENT_DATUM_X_MM - SOCKET_STOP_THICKNESS_MM
    translation_x = stop_x - ST3215_CATALOG_MAX_X_MM

    # The source servo's asymmetric local-Y bounds are -28.2 to +9.6 mm.
    # After the +90-degree X rotation, this centers it in global Z while
    # shifting it by half the total vertical clearance.
    translation_z = 9.3 + SOCKET_CLEARANCE_Z_TOTAL_MM / 2.0
    return Location((translation_x, 0.0, translation_z), (90.0, 0.0, 0.0))


def _largest_connected_solid(shape: Shape, operation: str) -> Shape:
    """Keep the connected bay when source-servo holes leave isolated pins."""
    solids = list(shape.solids())
    if not solids:
        raise RuntimeError(f"{operation} removed the entire motor bay")
    return max(solids, key=lambda solid: float(solid.volume))


def _servo_mount_access_cutters(
    outer_z: float,
    overtravel: float,
) -> list[Shape]:
    """Build temporary clearance cutters that erase reverse imprints."""
    spacing_y = float(SERVO_MOUNT_HOLE_Y_SPACING_MM)
    clearance_diameter = float(SERVO_MOUNT_PROFILE_CLEARANCE_DIAMETER_MM)
    top_access_reach_z = float(SERVO_TOP_ACCESS_REACH_Z_MM)
    bottom_access_reach_z = float(SERVO_BOTTOM_ACCESS_REACH_Z_MM)
    outer_top_z = outer_z / 2.0
    outer_bottom_z = -outer_top_z

    if min(spacing_y, clearance_diameter) <= 0.0:
        raise ValueError("Mount-hole dimensions must be positive")
    if not (outer_bottom_z < bottom_access_reach_z < top_access_reach_z < outer_top_z):
        raise ValueError("Mount-hole reach limits must lie inside the bay")

    cutters = []
    for y in (-spacing_y / 2.0, spacing_y / 2.0):
        cutters.append(
            Cylinder(
                clearance_diameter / 2.0,
                outer_top_z + overtravel - top_access_reach_z,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            ).moved(Location((SERVO_TOP_MOUNT_HOLE_X_MM, y, top_access_reach_z)))
        )
        cutters.append(
            Cylinder(
                clearance_diameter / 2.0,
                bottom_access_reach_z - (outer_bottom_z - overtravel),
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            ).moved(
                Location(
                    (
                        SERVO_BOTTOM_MOUNT_HOLE_X_MM,
                        y,
                        outer_bottom_z - overtravel,
                    )
                )
            )
        )
    return cutters


def _servo_mount_local_access_cleanup(
    inner_y: float,
    cavity_min_x: float,
    cavity_max_x: float,
) -> list[Shape]:
    """Clear four local mount pockets, preserving surrounding fitting grooves."""
    spacing_y = float(SERVO_MOUNT_HOLE_Y_SPACING_MM)
    cleanup_size = float(SERVO_MOUNT_LOCAL_CLEANUP_SIZE_MM)
    top_mount_face_z = float(SERVO_TOP_MOUNT_FACE_Z_MM)
    bottom_mount_face_z = float(SERVO_BOTTOM_MOUNT_FACE_Z_MM)
    top_access_reach_z = float(SERVO_TOP_ACCESS_REACH_Z_MM)
    bottom_access_reach_z = float(SERVO_BOTTOM_ACCESS_REACH_Z_MM)
    top_cleanup_depth = top_mount_face_z - top_access_reach_z
    bottom_cleanup_depth = bottom_access_reach_z - bottom_mount_face_z
    cavity_span_x = cavity_max_x - cavity_min_x

    if (
        min(
            inner_y,
            top_cleanup_depth,
            bottom_cleanup_depth,
            cleanup_size,
            cavity_span_x,
        )
        <= 0.0
    ):
        raise ValueError("Local cleanup dimensions must be positive")

    top_clip = Box(
        cavity_span_x,
        inner_y,
        top_cleanup_depth,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).moved(
        Location(
            (
                (cavity_min_x + cavity_max_x) / 2.0,
                0.0,
                top_access_reach_z,
            )
        )
    )
    bottom_clip = Box(
        cavity_span_x,
        inner_y,
        bottom_cleanup_depth,
        align=(Align.CENTER, Align.CENTER, Align.MIN),
    ).moved(
        Location(
            (
                (cavity_min_x + cavity_max_x) / 2.0,
                0.0,
                bottom_mount_face_z,
            )
        )
    )

    cleanup_cutters = []
    for y in (-spacing_y / 2.0, spacing_y / 2.0):
        top_local = Box(
            cleanup_size,
            cleanup_size,
            top_cleanup_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_TOP_MOUNT_HOLE_X_MM,
                    y,
                    top_access_reach_z,
                )
            )
        )
        cleanup_cutters.append(top_local & top_clip)

        bottom_local = Box(
            cleanup_size,
            cleanup_size,
            bottom_cleanup_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_BOTTOM_MOUNT_HOLE_X_MM,
                    y,
                    bottom_mount_face_z,
                )
            )
        )
        cleanup_cutters.append(bottom_local & bottom_clip)
    return cleanup_cutters


def _servo_top_profile_protrusion_cleanup(
    cavity_min_x: float,
    cavity_max_x: float,
    overtravel: float,
) -> list[Shape]:
    """Trim the paired screw-tab imprints flush with the top fitting floor."""
    requested_min_x = float(SERVO_TOP_PROFILE_PROTRUSION_MIN_X_MM)
    requested_max_x = float(SERVO_TOP_PROFILE_PROTRUSION_MAX_X_MM)
    center_y = float(SERVO_TOP_PROFILE_PROTRUSION_CENTER_Y_MM)
    size_y = float(SERVO_TOP_PROFILE_PROTRUSION_SIZE_Y_MM)
    reach_z = float(SERVO_TOP_ACCESS_REACH_Z_MM)
    floor_z = float(SERVO_TOP_MOUNT_FACE_Z_MM) - float(SERVO_MOUNT_INNER_PILOT_DEPTH_MM)
    min_x = max(requested_min_x, cavity_min_x)
    max_x = min(requested_max_x, cavity_max_x)
    span_x = max_x - min_x
    depth_z = floor_z - reach_z

    if min(span_x, size_y, depth_z, overtravel) <= 0.0:
        raise ValueError("Top profile-protrusion cleanup dimensions must be positive")

    cutters = []
    for y in (-center_y, center_y):
        cutters.append(
            Box(
                span_x,
                size_y,
                depth_z + overtravel,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            ).moved(
                Location(
                    (
                        (min_x + max_x) / 2.0,
                        y,
                        reach_z - overtravel,
                    )
                )
            )
        )
    return cutters


def _add_servo_mount_step_sleeves(
    bay: Shape,
    outer_z: float,
    overtravel: float,
) -> Shape:
    """Restore only the intended diameter-4/diameter-2 paths after cleanup."""
    spacing_y = float(SERVO_MOUNT_HOLE_Y_SPACING_MM)
    pilot_radius = float(SERVO_MOUNT_THROUGH_DIAMETER_MM) / 2.0
    access_radius = float(SERVO_MOUNT_COUNTERBORE_DIAMETER_MM) / 2.0
    clearance_radius = float(SERVO_MOUNT_PROFILE_CLEARANCE_DIAMETER_MM) / 2.0
    pilot_depth = float(SERVO_MOUNT_INNER_PILOT_DEPTH_MM)
    top_mount_face_z = float(SERVO_TOP_MOUNT_FACE_Z_MM)
    bottom_mount_face_z = float(SERVO_BOTTOM_MOUNT_FACE_Z_MM)
    top_access_reach_z = float(SERVO_TOP_ACCESS_REACH_Z_MM)
    bottom_access_reach_z = float(SERVO_BOTTOM_ACCESS_REACH_Z_MM)
    outer_top_z = outer_z / 2.0
    outer_bottom_z = -outer_top_z
    radial_overlap = 0.05

    if not 0.0 < pilot_radius < access_radius < clearance_radius:
        raise ValueError("Mount sleeve radii must increase from pilot to clearance")
    if top_mount_face_z - pilot_depth <= top_access_reach_z:
        raise ValueError("Top pilot step exceeds the cleaned cavity reach")
    if bottom_mount_face_z + pilot_depth >= bottom_access_reach_z:
        raise ValueError("Bottom pilot step exceeds the cleaned cavity reach")

    sleeves = []
    for y in (-spacing_y / 2.0, spacing_y / 2.0):
        top_access_outer = Cylinder(
            clearance_radius + radial_overlap,
            outer_top_z - top_mount_face_z,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(Location((SERVO_TOP_MOUNT_HOLE_X_MM, y, top_mount_face_z)))
        top_access_inner = Cylinder(
            access_radius,
            outer_top_z + 2.0 * overtravel - top_mount_face_z,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_TOP_MOUNT_HOLE_X_MM,
                    y,
                    top_mount_face_z - overtravel,
                )
            )
        )
        top_pilot_outer = Cylinder(
            access_radius + radial_overlap,
            pilot_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_TOP_MOUNT_HOLE_X_MM,
                    y,
                    top_mount_face_z - pilot_depth,
                )
            )
        )
        top_pilot_inner = Cylinder(
            pilot_radius,
            pilot_depth + 2.0 * overtravel,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_TOP_MOUNT_HOLE_X_MM,
                    y,
                    top_mount_face_z - pilot_depth - overtravel,
                )
            )
        )
        sleeves.append(
            (top_access_outer - top_access_inner).fuse(top_pilot_outer - top_pilot_inner)
        )

        bottom_access_outer = Cylinder(
            clearance_radius + radial_overlap,
            bottom_mount_face_z - outer_bottom_z,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_BOTTOM_MOUNT_HOLE_X_MM,
                    y,
                    outer_bottom_z,
                )
            )
        )
        bottom_access_inner = Cylinder(
            access_radius,
            bottom_mount_face_z - outer_bottom_z + 2.0 * overtravel,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_BOTTOM_MOUNT_HOLE_X_MM,
                    y,
                    outer_bottom_z - overtravel,
                )
            )
        )
        bottom_pilot_outer = Cylinder(
            access_radius + radial_overlap,
            pilot_depth,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_BOTTOM_MOUNT_HOLE_X_MM,
                    y,
                    bottom_mount_face_z,
                )
            )
        )
        bottom_pilot_inner = Cylinder(
            pilot_radius,
            pilot_depth + 2.0 * overtravel,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        ).moved(
            Location(
                (
                    SERVO_BOTTOM_MOUNT_HOLE_X_MM,
                    y,
                    bottom_mount_face_z - overtravel,
                )
            )
        )
        sleeves.append(
            (bottom_access_outer - bottom_access_inner).fuse(
                bottom_pilot_outer - bottom_pilot_inner
            )
        )

    for sleeve in sleeves:
        bay = _largest_connected_solid(
            bay.fuse(sleeve),
            "ST3215 stepped mount-sleeve fuse",
        )
    return bay


def _extend_top_mount_contact_faces_to_floor(bay: Shape) -> Shape:
    """Extrude the paired top contact profiles down to the fitting floor."""
    top_mount_face_z = float(SERVO_TOP_MOUNT_FACE_Z_MM)
    extension_depth = float(SERVO_MOUNT_INNER_PILOT_DEPTH_MM)
    coordinate_tolerance = 1.0e-5

    contact_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and abs(face.center().Z - top_mount_face_z) <= coordinate_tolerance
        and face.normal_at().Z < -0.99
        and abs(face.bounding_box().min.X + 9.8117) <= coordinate_tolerance
        and abs(face.bounding_box().max.X + 3.8117) <= coordinate_tolerance
        and abs(face.center().Y) > 7.0
    ]
    if len(contact_faces) != 2:
        raise RuntimeError("Could not resolve both top contact profiles for floor extension")

    for face in contact_faces:
        extension = extrude(face, amount=extension_depth)
        bay = _largest_connected_solid(
            bay.fuse(extension),
            "ST3215 top contact-profile extension",
        )
    return bay


def _extend_rear_contact_face_to_lower_floor(bay: Shape) -> Shape:
    """Continue the sloped rear contact wall down to the lower fitting floor."""
    target_floor_z = float(SERVO_LOWER_FITTING_FLOOR_Z_MM)
    coordinate_tolerance = 1.0e-5

    rear_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and face.normal_at().X < -0.99
        and abs(face.bounding_box().min.X + 1.69544) <= 2.0e-4
        and abs(face.bounding_box().max.X + 1.6117) <= coordinate_tolerance
        and abs(face.bounding_box().min.Z - 8.575) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z - 14.574975) <= 2.0e-4
        and face.bounding_box().size.Y > 20.0
    ]
    target_floors = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and face.normal_at().Z > 0.99
        and abs(face.center().Z - target_floor_z) <= coordinate_tolerance
        and abs(face.bounding_box().min.X + 16.0) <= coordinate_tolerance
        and face.bounding_box().max.X < -2.0
        and face.bounding_box().size.Y > 20.0
    ]
    if len(rear_faces) != 1 or len(target_floors) != 1:
        raise RuntimeError("Could not resolve the rear contact face and lower fitting floor")

    rear_face = rear_faces[0]
    target_floor_z = target_floors[0].center().Z
    rear_bbox = rear_face.bounding_box()
    rear_center = rear_face.center()
    rear_normal = rear_face.normal_at()
    if abs(rear_normal.X) <= coordinate_tolerance:
        raise RuntimeError("Rear contact face cannot be extended from a vertical plane")

    def plane_x_at(z: float) -> float:
        return rear_center.X - (rear_normal.Z / rear_normal.X) * (z - rear_center.Z)

    top_z = rear_bbox.min.Z
    top_x = plane_x_at(top_z)
    bottom_x = plane_x_at(target_floor_z)
    profile_wire = Wire.make_polygon(
        [
            Vector(top_x, top_z),
            Vector(ATTACHMENT_DATUM_X_MM, top_z),
            Vector(ATTACHMENT_DATUM_X_MM, target_floor_z),
            Vector(bottom_x, target_floor_z),
        ],
        close=True,
    )
    profile_face = Face(profile_wire).moved(Location((0.0, rear_bbox.min.Y, 0.0), (90.0, 0.0, 0.0)))
    extension = extrude(profile_face, amount=rear_bbox.size.Y)
    return _largest_connected_solid(
        bay.fuse(extension),
        "ST3215 rear contact-wall extension",
    )


def _extend_upper_side_pockets_to_main_wall(
    bay: Shape,
    inner_y: float,
) -> Shape:
    """Replace both upper molded recesses with the neighboring wall planes."""
    pocket_face_x = float(SERVO_UPPER_SIDE_POCKET_FACE_X_MM)
    pocket_center_y = float(SERVO_UPPER_SIDE_POCKET_CENTER_Y_MM)
    pocket_size_y = float(SERVO_UPPER_SIDE_POCKET_SIZE_Y_MM)
    pocket_min_z = float(SERVO_UPPER_SIDE_POCKET_MIN_Z_MM)
    pocket_max_z = float(SERVO_UPPER_SIDE_POCKET_MAX_Z_MM)
    side_wall_y = inner_y / 2.0
    coordinate_tolerance = 1.0e-5

    main_wall_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and face.normal_at().X < -0.99
        and abs(face.bounding_box().min.X + 1.69544) <= 2.0e-4
        and abs(face.bounding_box().max.X + 1.6117) <= coordinate_tolerance
        and abs(face.bounding_box().min.Z - 8.575) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z - 14.574975) <= 2.0e-4
        and face.bounding_box().size.Y > 20.0
    ]
    pocket_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and face.normal_at().X < -0.99
        and abs(face.center().X - pocket_face_x) <= coordinate_tolerance
        and abs(abs(face.center().Y) - pocket_center_y) <= coordinate_tolerance
        and abs(face.bounding_box().size.Y - pocket_size_y) <= coordinate_tolerance
        and abs(face.bounding_box().min.Z - pocket_min_z) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z - pocket_max_z) <= coordinate_tolerance
    ]
    upper_side_wall_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and abs(face.normal_at().Y) > 0.99
        and abs(abs(face.center().Y) - side_wall_y) <= coordinate_tolerance
        and abs(face.bounding_box().min.Z - 14.0) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z - pocket_max_z) <= coordinate_tolerance
        and face.bounding_box().size.X > 1.5
    ]
    upper_recess_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.CYLINDER
        and abs(face.bounding_box().min.Z - 14.0) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z - pocket_max_z) <= coordinate_tolerance
        and abs(abs(face.center().Y) - 12.95) <= coordinate_tolerance
    ]
    upper_protrusion_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and abs(face.normal_at().Y) > 0.99
        and abs(abs(face.center().Y) - 12.5) <= coordinate_tolerance
        and abs(face.bounding_box().min.Z - pocket_min_z) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z - pocket_max_z) <= coordinate_tolerance
        and face.bounding_box().size.X > 2.0
    ]
    if (
        len(main_wall_faces) != 1
        or len(pocket_faces) != 2
        or len(upper_side_wall_faces) != 4
        or len(upper_recess_faces) != 2
        or len(upper_protrusion_faces) != 2
    ):
        raise RuntimeError("Could not resolve both upper recesses and their neighboring walls")

    main_wall = main_wall_faces[0]
    wall_center = main_wall.center()
    wall_normal = main_wall.normal_at()
    if abs(wall_normal.X) <= coordinate_tolerance:
        raise RuntimeError("Main wall cannot define an X position from height")

    def wall_x_at(z: float) -> float:
        return wall_center.X - (wall_normal.Z / wall_normal.X) * (z - wall_center.Z)

    for pocket_face in pocket_faces:
        pocket_bbox = pocket_face.bounding_box()
        source_x = pocket_face.center().X
        target_min_x = wall_x_at(pocket_bbox.min.Z)
        target_max_x = wall_x_at(pocket_bbox.max.Z)
        if min(target_min_x, target_max_x) >= source_x:
            raise RuntimeError("Upper side-pocket extension points outside the cavity")

        profile_wire = Wire.make_polygon(
            [
                Vector(source_x, pocket_bbox.min.Z),
                Vector(source_x, pocket_bbox.max.Z),
                Vector(target_max_x, pocket_bbox.max.Z),
                Vector(target_min_x, pocket_bbox.min.Z),
            ],
            close=True,
        )
        profile_face = Face(profile_wire).moved(
            Location((0.0, pocket_bbox.min.Y, 0.0), (90.0, 0.0, 0.0))
        )
        extension = extrude(
            profile_face,
            amount=pocket_bbox.size.Y,
            dir=(0.0, 1.0, 0.0),
        )
        bay = _largest_connected_solid(
            bay.fuse(extension),
            "ST3215 upper side-pocket extension",
        )

    recess_min_x = min(face.bounding_box().min.X for face in upper_side_wall_faces)
    recess_outer_y = max(
        max(abs(face.bounding_box().min.Y), abs(face.bounding_box().max.Y))
        for face in upper_recess_faces
    )
    trim_inner_y = min(
        min(
            abs(face.bounding_box().min.Y),
            abs(face.bounding_box().max.Y),
        )
        for face in pocket_faces
    )
    if not trim_inner_y < side_wall_y < recess_outer_y:
        raise RuntimeError("Upper recess walls do not bracket the target side wall")

    target_min_x = wall_x_at(pocket_min_z)
    target_max_x = wall_x_at(pocket_max_z)
    side_profile_wire = Wire.make_polygon(
        [
            Vector(recess_min_x, pocket_min_z),
            Vector(recess_min_x, pocket_max_z),
            Vector(target_max_x, pocket_max_z),
            Vector(target_min_x, pocket_min_z),
        ],
        close=True,
    )

    for y_sign in (-1.0, 1.0):
        fill_start_y = -recess_outer_y if y_sign < 0.0 else side_wall_y
        fill_depth_y = recess_outer_y - side_wall_y
        fill_face = Face(side_profile_wire).moved(
            Location((0.0, fill_start_y, 0.0), (90.0, 0.0, 0.0))
        )
        fill_extension = extrude(
            fill_face,
            amount=fill_depth_y,
            dir=(0.0, 1.0, 0.0),
        )
        bay = _largest_connected_solid(
            bay.fuse(fill_extension),
            "ST3215 upper side-wall recess fill",
        )

        trim_start_y = -side_wall_y if y_sign < 0.0 else trim_inner_y
        trim_depth_y = side_wall_y - trim_inner_y
        trim_face = Face(side_profile_wire).moved(
            Location((0.0, trim_start_y, 0.0), (90.0, 0.0, 0.0))
        )
        trim_extension = extrude(
            trim_face,
            amount=trim_depth_y,
            dir=(0.0, 1.0, 0.0),
        )
        bay = _largest_connected_solid(
            bay - trim_extension,
            "ST3215 upper side-wall protrusion trim",
        )
    return bay


def _extend_bottom_mount_contact_faces_to_floor(bay: Shape) -> Shape:
    """Raise the paired lower contact profiles to the lower fitting floor."""
    bottom_mount_face_z = float(SERVO_BOTTOM_MOUNT_FACE_Z_MM)
    target_floor_z = float(SERVO_LOWER_FITTING_FLOOR_Z_MM)
    extension_height = target_floor_z - bottom_mount_face_z
    coordinate_tolerance = 1.0e-5

    contact_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and abs(face.center().Z - bottom_mount_face_z) <= coordinate_tolerance
        and face.normal_at().Z > 0.99
        and abs(face.bounding_box().min.X + 6.0617) <= coordinate_tolerance
        and -0.5 < face.bounding_box().max.X < -0.3
        and abs(face.center().Y) > 7.0
    ]
    if len(contact_faces) != 2:
        raise RuntimeError("Could not resolve both lower contact profiles for floor extension")
    if extension_height <= 0.0:
        raise ValueError("Lower contact-profile extension height must be positive")

    for face in contact_faces:
        extension = extrude(face, amount=extension_height)
        bay = _largest_connected_solid(
            bay.fuse(extension),
            "ST3215 lower contact-profile extension",
        )
    return bay


def _replace_lower_side_recesses_with_wall_continuations(
    bay: Shape,
    outer_y: float,
) -> Shape:
    """Continue the lower rear, side, and rounded walls through both recesses."""
    top_z = float(SERVO_BOTTOM_ACCESS_REACH_Z_MM)
    bottom_z = float(SERVO_BOTTOM_MOUNT_FACE_Z_MM) + float(SERVO_MOUNT_INNER_PILOT_DEPTH_MM)
    min_x = float(SERVO_LOWER_SIDE_RECESS_MIN_X_MM)
    inner_y = float(SERVO_LOWER_SIDE_RECESS_INNER_Y_MM)
    outer_side_y = outer_y / 2.0
    coordinate_tolerance = 2.0e-4

    main_wall_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and face.normal_at().X < -0.99
        and face.normal_at().Z > 0.01
        and abs(face.bounding_box().min.Z + 14.675146) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z + 4.675) <= coordinate_tolerance
        and face.bounding_box().size.Y > 20.0
    ]
    side_wall_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.PLANE
        and abs(face.normal_at().Y) > 0.99
        and face.normal_at().Z > 0.01
        and abs(face.bounding_box().min.Z + 14.675146) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z + 4.675) <= coordinate_tolerance
        and face.bounding_box().size.X > 12.0
    ]
    corner_faces = [
        face
        for face in bay.faces()
        if face.geom_type == GeomType.CYLINDER
        and abs(face.bounding_box().min.Z - top_z) <= coordinate_tolerance
        and abs(face.bounding_box().max.Z + 4.675) <= coordinate_tolerance
        and abs(face.center().Y) > 10.0
        and abs(float(face.radius) - 2.0) <= coordinate_tolerance
    ]
    if len(main_wall_faces) != 1 or len(side_wall_faces) != 2 or len(corner_faces) != 2:
        raise RuntimeError("Could not resolve the lower neighboring rear, side, and corner walls")

    main_wall = main_wall_faces[0]
    positive_side_wall = max(side_wall_faces, key=lambda face: face.center().Y)
    corner_radius = float(corner_faces[0].radius)

    def plane_x_at(face: Face, z: float) -> float:
        center = face.center()
        normal = face.normal_at()
        if abs(normal.X) <= coordinate_tolerance:
            raise RuntimeError("Lower rear wall cannot define X from height")
        return center.X - (normal.Z / normal.X) * (z - center.Z)

    def plane_y_at(face: Face, z: float) -> float:
        center = face.center()
        normal = face.normal_at()
        if abs(normal.Y) <= coordinate_tolerance:
            raise RuntimeError("Lower side wall cannot define Y from height")
        return center.Y - (normal.Z / normal.Y) * (z - center.Z)

    def cavity_profile(z: float, y_sign: float) -> Face:
        main_x = plane_x_at(main_wall, z)
        side_y = abs(plane_y_at(positive_side_wall, z))
        corner_x = main_x - corner_radius
        corner_y = side_y - corner_radius
        diagonal_offset = corner_radius / (2.0**0.5)

        if y_sign > 0.0:
            start = Vector(min_x, inner_y, z)
            rear_start = Vector(main_x, inner_y, z)
            arc_start = Vector(main_x, corner_y, z)
            arc_mid = Vector(
                corner_x + diagonal_offset,
                corner_y + diagonal_offset,
                z,
            )
            arc_end = Vector(corner_x, side_y, z)
            side_end = Vector(min_x, side_y, z)
        else:
            start = Vector(min_x, -inner_y, z)
            rear_start = Vector(min_x, -side_y, z)
            arc_start = Vector(corner_x, -side_y, z)
            arc_mid = Vector(
                corner_x + diagonal_offset,
                -(corner_y + diagonal_offset),
                z,
            )
            arc_end = Vector(main_x, -corner_y, z)
            side_end = Vector(main_x, -inner_y, z)

        profile_wire = Wire(
            [
                Edge.make_line(start, rear_start),
                Edge.make_line(rear_start, arc_start),
                Edge.make_three_point_arc(arc_start, arc_mid, arc_end),
                Edge.make_line(arc_end, side_end),
                Edge.make_line(side_end, start),
            ]
        )
        return Face(profile_wire)

    for y_sign in (-1.0, 1.0):
        cavity = loft(
            [
                cavity_profile(top_z, y_sign),
                cavity_profile(bottom_z, y_sign),
            ],
            ruled=True,
        )
        patch_min_y = inner_y if y_sign > 0.0 else -outer_side_y
        patch = Box(
            -min_x,
            outer_side_y - inner_y,
            top_z - bottom_z,
            align=(Align.MIN, Align.MIN, Align.MIN),
        ).moved(Location((min_x, patch_min_y, bottom_z)))
        wall_continuation = patch - cavity

        bay = _largest_connected_solid(
            bay - cavity,
            "ST3215 lower side-recess clearance",
        )
        bay = _largest_connected_solid(
            bay.fuse(wall_continuation),
            "ST3215 lower wall continuation",
        )
    return bay


def _side_wall_diamond_vent_cutters(
    inner_y: float,
    outer_y: float,
    outer_z: float,
    outer_min_x: float,
) -> list[Shape]:
    """Build a mirrored 2x4 diamond-vent pattern for the broad side walls."""
    columns_x = tuple(float(value) for value in VENT_DIAMOND_COLUMNS_X_MM)
    rows_z = tuple(float(value) for value in VENT_DIAMOND_ROWS_Z_MM)
    width_x = float(VENT_DIAMOND_WIDTH_X_MM)
    height_z = float(VENT_DIAMOND_HEIGHT_Z_MM)
    overtravel = float(VENT_WALL_OVERTRAVEL_MM)
    wall_depth = (outer_y - inner_y) / 2.0
    half_width_x = width_x / 2.0
    half_height_z = height_z / 2.0

    if not columns_x or not rows_z:
        raise ValueError("Diamond ventilation requires at least one row and column")
    if min(width_x, height_z, overtravel, wall_depth) <= 0.0:
        raise ValueError("Diamond ventilation dimensions must be positive")

    outer_max_x = float(ATTACHMENT_DATUM_X_MM)
    outer_half_z = outer_z / 2.0
    for center_x in columns_x:
        if not (outer_min_x < center_x - half_width_x and center_x + half_width_x < outer_max_x):
            raise ValueError("Diamond ventilation must stay inside the X walls")
    for center_z in rows_z:
        if not (
            -outer_half_z < center_z - half_height_z and center_z + half_height_z < outer_half_z
        ):
            raise ValueError("Diamond ventilation must stay inside the Z walls")

    cutter_depth = wall_depth + 2.0 * overtravel
    side_start_y = (
        -outer_y / 2.0 - overtravel,
        inner_y / 2.0 - overtravel,
    )
    cutters = []
    for center_x in columns_x:
        for center_z in rows_z:
            profile = Face(
                Wire.make_polygon(
                    [
                        Vector(center_x - half_width_x, center_z),
                        Vector(center_x, center_z + half_height_z),
                        Vector(center_x + half_width_x, center_z),
                        Vector(center_x, center_z - half_height_z),
                    ],
                    close=True,
                )
            )
            for start_y in side_start_y:
                cutter_face = profile.moved(Location((0.0, start_y, 0.0), (90.0, 0.0, 0.0)))
                cutters.append(
                    extrude(
                        cutter_face,
                        amount=cutter_depth,
                        dir=(0.0, 1.0, 0.0),
                    )
                )
    return cutters


def _outer_side_perimeter_edges(
    bay: Shape,
    outer_y: float,
    outer_z: float,
    outer_min_x: float,
) -> list[Edge]:
    """Resolve the eight broad-side perimeter edges marked for filleting."""
    outer_max_x = float(ATTACHMENT_DATUM_X_MM)
    length_x = outer_max_x - outer_min_x
    tolerance = 1.0e-4
    edges = []

    for edge in bay.edges():
        if edge.geom_type != GeomType.LINE:
            continue

        bounds = edge.bounding_box()
        center = edge.center()
        on_outer_side = (
            abs(abs(center.Y) - outer_y / 2.0) <= tolerance
            and bounds.size.Y <= tolerance
        )
        horizontal_boundary = (
            abs(bounds.size.X - length_x) <= tolerance
            and bounds.size.Z <= tolerance
            and abs(abs(center.Z) - outer_z / 2.0) <= tolerance
        )
        vertical_boundary = (
            bounds.size.X <= tolerance
            and abs(bounds.size.Z - outer_z) <= tolerance
            and (
                abs(center.X - outer_min_x) <= tolerance
                or abs(center.X - outer_max_x) <= tolerance
            )
        )
        if on_outer_side and (horizontal_boundary or vertical_boundary):
            edges.append(edge)

    if len(edges) != 8:
        raise RuntimeError(
            f"Expected eight outer side-perimeter edges for filleting, found {len(edges)}"
        )
    return edges


def gen_step() -> Shape:
    """Return the standalone, STEP-ready ST3215 rear motor bay."""
    clearance_y = float(SOCKET_CLEARANCE_Y_PER_SIDE_MM)
    clearance_z = float(SOCKET_CLEARANCE_Z_TOTAL_MM)
    wall = float(SOCKET_WALL_MM)
    length_x = float(SOCKET_LENGTH_X_MM)
    stop = float(SOCKET_STOP_THICKNESS_MM)
    overtravel = float(BOOLEAN_OVERTRAVEL_MM)
    fillet_radius = float(OUTER_SIDE_PERIMETER_FILLET_RADIUS_MM)

    if min(clearance_y, clearance_z) < 0.0:
        raise ValueError("Socket clearances cannot be negative")
    if min(wall, length_x, stop, overtravel, fillet_radius) <= 0.0:
        raise ValueError("Wall, length, stop, overtravel, and fillet radius must be positive")
    if stop >= length_x:
        raise ValueError("The stop must be thinner than the bay length")
    if fillet_radius >= wall:
        raise ValueError("The cosmetic fillet radius must remain smaller than the wall")

    inner_y = ST3215_CATALOG_WIDTH_Z_MM + 2.0 * clearance_y
    inner_z = ST3215_CATALOG_HEIGHT_Y_MM + clearance_z
    outer_y = inner_y + 2.0 * wall
    outer_z = inner_z + 2.0 * wall
    outer_min_x = ATTACHMENT_DATUM_X_MM - length_x

    bay = Box(
        length_x,
        outer_y,
        outer_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(Location(((outer_min_x + ATTACHMENT_DATUM_X_MM) / 2.0, 0.0, 0.0)))

    cavity_max_x = ATTACHMENT_DATUM_X_MM - stop
    cavity_min_x = outer_min_x - overtravel
    cavity_trim = Box(
        cavity_max_x - cavity_min_x,
        outer_y + 2.0 * wall,
        outer_z + 2.0 * wall,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(Location(((cavity_min_x + cavity_max_x) / 2.0, 0.0, 0.0)))

    installed_servo = import_step(ST3215_SERVO_STEP).moved(st3215_installed_location())
    rear_pieces = []
    for servo_solid in installed_servo.solids():
        trimmed = servo_solid & cavity_trim
        if trimmed is None:
            continue
        rear_pieces.extend(solid for solid in trimmed.solids() if float(solid.volume) > 1.0e-6)
    if not rear_pieces:
        raise RuntimeError("The ST3215 model did not intersect the bay capture zone")

    exact_rear_profile = rear_pieces[0].fuse(*rear_pieces[1:])
    if len(exact_rear_profile.solids()) != 1 or not exact_rear_profile.is_valid:
        raise RuntimeError("Could not consolidate the ST3215 rear profile")

    # Expand the exact negative profile to create the specified allowances.
    for delta_y, delta_z in (
        (0.0, 0.0),
        (clearance_y, 0.0),
        (-clearance_y, 0.0),
        (0.0, clearance_z),
    ):
        bay = _largest_connected_solid(
            bay - exact_rear_profile.moved(Location((0.0, delta_y, delta_z))),
            "ST3215 cavity cut",
        )

    cable_y = float(SOCKET_CABLE_WINDOW_Y_MM)
    cable_z = float(SOCKET_CABLE_WINDOW_Z_MM)
    if min(cable_y, cable_z) <= 0.0:
        raise ValueError("Cable-window dimensions must be positive")
    if cable_y >= inner_y or cable_z >= inner_z:
        raise ValueError("Cable window must remain inside the attachment rim")

    cable_window = Box(
        stop + 2.0 * overtravel,
        cable_y,
        cable_z,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).moved(
        Location(
            (
                ATTACHMENT_DATUM_X_MM - stop / 2.0,
                0.0,
                0.0,
            )
        )
    )
    bay = _largest_connected_solid(bay - cable_window, "Cable-window cut")
    for cleanup_cutter in _servo_mount_local_access_cleanup(
        inner_y,
        cavity_min_x,
        cavity_max_x,
    ):
        bay = _largest_connected_solid(
            bay - cleanup_cutter,
            "ST3215 local mount-zone cleanup",
        )
    for cleanup_cutter in _servo_top_profile_protrusion_cleanup(
        cavity_min_x,
        cavity_max_x,
        overtravel,
    ):
        bay = _largest_connected_solid(
            bay - cleanup_cutter,
            "ST3215 top profile-protrusion cleanup",
        )
    for access_cutter in _servo_mount_access_cutters(outer_z, overtravel):
        bay = _largest_connected_solid(
            bay - access_cutter,
            "ST3215 mount-profile clearance cut",
        )
    bay = _add_servo_mount_step_sleeves(bay, outer_z, overtravel)
    bay = _extend_top_mount_contact_faces_to_floor(bay)
    bay = _extend_rear_contact_face_to_lower_floor(bay)
    bay = _extend_upper_side_pockets_to_main_wall(bay, inner_y)
    bay = _extend_bottom_mount_contact_faces_to_floor(bay)
    bay = _replace_lower_side_recesses_with_wall_continuations(bay, outer_y)
    vent_cutters = _side_wall_diamond_vent_cutters(
        inner_y,
        outer_y,
        outer_z,
        outer_min_x,
    )
    vent_pattern = vent_cutters[0].fuse(*vent_cutters[1:])
    bay = _largest_connected_solid(
        bay - vent_pattern,
        "Diamond ventilation cut",
    )
    perimeter_edges = _outer_side_perimeter_edges(
        bay,
        outer_y,
        outer_z,
        outer_min_x,
    )
    bay = _largest_connected_solid(
        bay.fillet(fillet_radius, perimeter_edges),
        "Outer side-perimeter fillet",
    )

    if len(bay.solids()) != 1 or not bay.is_valid:
        raise RuntimeError("The ST3215 motor bay is not one valid solid")
    bay.label = "st3215_rear_motor_bay"
    return bay


if __name__ == "__main__":
    from build123d import export_step

    output_path = Path("exports/step/st3215_motor_bay.step")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_step(gen_step(), output_path)
    print(f"Generated {output_path}")
