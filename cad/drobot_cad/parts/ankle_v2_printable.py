"""Printable V2.1 ankle kit. Mechanical coordinates: ankle at 0; +X down.

The motor socket is imported unchanged from the existing fit-critical source.
Foot cheeks bolt to actual metal ST3215 wheels; no printed shaft spline.
"""
from __future__ import annotations

from functools import lru_cache
from math import cos, pi

from build123d import (
    Align, Axis, Box, BuildSketch, Color, Cylinder, Ellipse, Location, Plane,
    Rectangle, RectangleRounded, RegularPolygon, Shape, extrude,
)
from cadgen.step_scene import import_step

from drobot_cad.interfaces import UPPER_ARM_INTERFACES
from drobot_cad.parts import st3215_motor_bay as motor_bay

KNEE_TO_ANKLE_MM = (
    UPPER_ARM_INTERFACES["frame_distal_fork_axis"].xyz_mm[0]
    - UPPER_ARM_INTERFACES["frame_st3215_output_axis"].xyz_mm[0]
)
CATALOG_SHAFT_X_MM = -25.5
BAY_TO_SHAFT_X_MM = motor_bay.st3215_installed_location().position.X + CATALOG_SHAFT_X_MM
KNEE_BAY_X_MM = -KNEE_TO_ANKLE_MM - BAY_TO_SHAFT_X_MM
ANKLE_BAY_X_MM = BAY_TO_SHAFT_X_MM
KNEE_BAY_Z_MM = UPPER_ARM_INTERFACES["frame_st3215_bay_attachment"].xyz_mm[2]

# The outer metal-wheel mounting faces measured in the immutable catalog STEP.
WHEEL_POSITIVE_Y_MM = 9.6
WHEEL_NEGATIVE_Y_MM = -27.65
WHEEL_FACE_HALF_SPAN_MM = (WHEEL_POSITIVE_Y_MM - WHEEL_NEGATIVE_Y_MM) / 2
ANKLE_SERVO_Z_MM = -(WHEEL_POSITIVE_Y_MM + WHEEL_NEGATIVE_Y_MM) / 2
ANKLE_BAY_Z_MM = ANKLE_SERVO_Z_MM - motor_bay.st3215_installed_location().position.Z
BAY_WIDTH_MM = motor_bay.ST3215_CATALOG_WIDTH_Z_MM + 2 * motor_bay.SOCKET_CLEARANCE_Y_PER_SIDE_MM + 2 * motor_bay.SOCKET_WALL_MM
BAY_HEIGHT_MM = motor_bay.ST3215_CATALOG_HEIGHT_Y_MM + motor_bay.SOCKET_CLEARANCE_Z_TOTAL_MM + 2 * motor_bay.SOCKET_WALL_MM

SOLE_LENGTH_MM, SOLE_WIDTH_MM, SOLE_THICKNESS_MM = 100.0, 60.0, 6.0
SOLE_TOP_X_MM = 24.0
SOLE_BOTTOM_X_MM = SOLE_TOP_X_MM + SOLE_THICKNESS_MM
SOLE_CORNER_MM = 2.0
CHEEK_THICKNESS_MM = 7.0
CHEEK_RADIUS_MM = 12.5
CHEEK_AXIAL_ALLOWANCE_MM = 0.2
CHEEK_INNER_Z_MM = WHEEL_FACE_HALF_SPAN_MM + CHEEK_AXIAL_ALLOWANCE_MM
FLANGE_BOLT_Z_MM = 4.0
CHEEK_BOLT_Z_MM = CHEEK_INNER_Z_MM + FLANGE_BOLT_Z_MM
WHEEL_PCD_MM = 14.0
WHEEL_HOLES_XY_MM = ((7.0, 0.0), (0.0, 7.0), (-7.0, 0.0), (0.0, -7.0))
M3_CLEARANCE_MM = 3.4
HORN_HEAD_CLEARANCE_MM = 5.8
HORN_HEAD_RECESS_MM = 3.0
SHAFT_SCREW_ACCESS_MM = 6.0
FOOT_BOLT_Y_MM = (-9.0, 9.0)
AXIAL_ADJUSTMENT_MM = 1.2
SOLE_HEAD_RECESS_MM = 2.2
SOLE_HEAD_CLEARANCE_MM = 6.5
FLANGE_TOP_X_MM = 16.0
FLANGE_WIDTH_Y_MM = 30.0
FLANGE_WIDTH_Z_MM = 10.0
NUT_AF_MM = 5.8
NUT_POCKET_FLOOR_X_MM = 21.5
NUT_POCKET_AXIAL_DEPTH_MM = 3.0
EPS = 0.4


def _box(x, y, z, center):
    return Box(x, y, z, align=(Align.CENTER, Align.CENTER, Align.CENTER)).moved(Location(center))


def _cyl_z(radius, depth, x=0.0, y=0.0, z=0.0):
    return Cylinder(radius, depth, align=(Align.CENTER, Align.CENTER, Align.MIN)).moved(Location((x, y, z)))


def _x_prism(length_y, width_z, depth_x, start_x, radius=0.0):
    with BuildSketch(Plane.YZ) as profile:
        RectangleRounded(length_y, width_z, radius) if radius else Rectangle(length_y, width_z)
    return extrude(profile.sketch, amount=depth_x).moved(Location((start_x, 0, 0)))


def _slot_x(diameter, axial_adjustment, start_x, depth, y, z):
    """Capsule through local X; the slot's long dimension is local Z."""
    radius = diameter / 2
    cylinders = [
        _cyl_z(radius, depth).rotate(Axis.Y, 90).moved(Location((start_x, y, z + sign * axial_adjustment)))
        for sign in (-1, 1)
    ]
    middle = _box(depth, diameter, 2 * axial_adjustment, (start_x + depth / 2, y, z))
    return cylinders[0].fuse(cylinders[1], middle)


@lru_cache(maxsize=1)
def _shared_bay():
    return motor_bay.gen_step()


def knee_socket_location():
    return Location((KNEE_BAY_X_MM, 0, KNEE_BAY_Z_MM))


def ankle_socket_location():
    return Location((ANKLE_BAY_X_MM, 0, ANKLE_BAY_Z_MM), (0, 0, 180))


def make_lower_leg() -> Shape:
    """Two unmodified keyed sockets joined by an open, cable-accessible spine."""
    knee = _shared_bay().moved(knee_socket_location())
    ankle = _shared_bay().moved(ankle_socket_location())
    length = ANKLE_BAY_X_MM - KNEE_BAY_X_MM
    center_x = (KNEE_BAY_X_MM + ANKLE_BAY_X_MM) / 2
    center_z = (KNEE_BAY_Z_MM + ANKLE_BAY_Z_MM) / 2
    height = BAY_HEIGHT_MM + abs(KNEE_BAY_Z_MM - ANKLE_BAY_Z_MM)
    spine = _box(length + 2 * EPS, BAY_WIDTH_MM, height, (center_x, 0, center_z))
    cable = _box(length + 4 * EPS, 16.0, 14.0, (center_x, 0, center_z))
    # An elliptical window opens both broad side walls. Ends and upper/lower
    # rails remain continuous; the socket cable windows are not covered.
    with BuildSketch(Plane.XZ) as sketch:
        Ellipse((length - 24.0) / 2, (height - 16.0) / 2)
    window = extrude(sketch.sketch, amount=BAY_WIDTH_MM + 2, both=True).moved(Location((center_x, 0, center_z)))
    spine = spine - cable - window
    result = spine.fuse(knee, ankle).clean()
    result.label = "V2_1_replacement_lower_leg_two_ST3215_sockets"
    result.color = Color(0.08, 0.55, 0.35)
    return result


def make_sole() -> Shape:
    """Flat 100 x 60 sole with four adjustable cheek fixings and recessed heads."""
    sole = _x_prism(SOLE_LENGTH_MM, SOLE_WIDTH_MM, SOLE_THICKNESS_MM, SOLE_TOP_X_MM, SOLE_CORNER_MM)
    # Two low central ribs, separated from the cheek seating areas.
    ribs = [_box(4.4, 80.0, 4.0, (22.2, 0, z)) for z in (-6.0, 6.0)]
    sole = sole.fuse(*ribs)
    for sign in (-1, 1):
        for y in FOOT_BOLT_Y_MM:
            z = sign * CHEEK_BOLT_Z_MM
            sole -= _slot_x(M3_CLEARANCE_MM, AXIAL_ADJUSTMENT_MM, SOLE_TOP_X_MM - EPS, SOLE_THICKNESS_MM + 2 * EPS, y, z)
            sole -= _slot_x(SOLE_HEAD_CLEARANCE_MM, AXIAL_ADJUSTMENT_MM, SOLE_BOTTOM_X_MM - SOLE_HEAD_RECESS_MM, SOLE_HEAD_RECESS_MM + EPS, y, z)
    sole.label = "V2_1_flat_sole_adjustable_axial_cheek_slots"
    sole.color = Color(0.18, 0.22, 0.28)
    return sole.clean()


def make_cheek() -> Shape:
    """One identical flat-printable cheek; inner wheel face at local Z=0.

    Two are used per ankle, the negative-Z cheek rotated 180 degrees about X.
    """
    disk = _cyl_z(CHEEK_RADIUS_MM, CHEEK_THICKNESS_MM)
    stem = _box(SOLE_TOP_X_MM, 2 * CHEEK_RADIUS_MM, CHEEK_THICKNESS_MM,
                (SOLE_TOP_X_MM / 2, 0, CHEEK_THICKNESS_MM / 2))
    flange = _box(SOLE_TOP_X_MM - FLANGE_TOP_X_MM, FLANGE_WIDTH_Y_MM, FLANGE_WIDTH_Z_MM,
                  ((FLANGE_TOP_X_MM + SOLE_TOP_X_MM) / 2, 0, FLANGE_WIDTH_Z_MM / 2))
    cheek = disk.fuse(stem, flange)
    cheek -= _cyl_z(SHAFT_SCREW_ACCESS_MM / 2, CHEEK_THICKNESS_MM + 2 * EPS, z=-EPS)
    for x, y in WHEEL_HOLES_XY_MM:
        cheek -= _cyl_z(M3_CLEARANCE_MM / 2, CHEEK_THICKNESS_MM + 2 * EPS, x, y, -EPS)
        cheek -= _cyl_z(HORN_HEAD_CLEARANCE_MM / 2, HORN_HEAD_RECESS_MM + EPS,
                        x, y, CHEEK_THICKNESS_MM - HORN_HEAD_RECESS_MM)
    for y in FOOT_BOLT_Y_MM:
        cheek -= _slot_x(M3_CLEARANCE_MM, 0.01, FLANGE_TOP_X_MM - EPS,
                         SOLE_TOP_X_MM - FLANGE_TOP_X_MM + 2 * EPS, y, FLANGE_BOLT_Z_MM)
        with BuildSketch(Plane.YZ) as nut_profile:
            RegularPolygon(NUT_AF_MM / (2 * cos(pi / 6)), 6, rotation=30)
        trap = extrude(nut_profile.sketch, amount=NUT_POCKET_AXIAL_DEPTH_MM)
        trap_x = NUT_POCKET_FLOOR_X_MM - NUT_POCKET_AXIAL_DEPTH_MM
        cheek -= trap.moved(Location((trap_x, y, FLANGE_BOLT_Z_MM)))
        # Load the nut from the exposed OUTER Z face. The stem covers the
        # flange's -X face, so a blind hex pocket alone would be inaccessible.
        access_height = FLANGE_WIDTH_Z_MM + EPS - FLANGE_BOLT_Z_MM
        cheek -= _box(NUT_POCKET_AXIAL_DEPTH_MM, NUT_AF_MM, access_height,
                      (trap_x + NUT_POCKET_AXIAL_DEPTH_MM / 2, y,
                       FLANGE_BOLT_Z_MM + access_height / 2))
    cheek.label = "V2_1_identical_metal_horn_cheek_print_two_per_ankle"
    cheek.color = Color(0.93, 0.59, 0.10)
    return cheek.clean()


def positioned_cheeks():
    cheek = make_cheek()
    return (
        cheek.moved(Location((0, 0, CHEEK_INNER_Z_MM))),
        cheek.rotate(Axis.X, 180).moved(Location((0, 0, -CHEEK_INNER_Z_MM))),
    )


def motor_placements():
    installed = motor_bay.st3215_installed_location()
    return (knee_socket_location() * installed, ankle_socket_location() * installed)


def make_assembly():
    from cadgen.assembly import AssemblyHelper
    assembly = AssemblyHelper("ankle_V2_1_one_leg_print_prototype")
    leg = assembly.add(make_lower_leg(), "printed_lower_leg")
    sole = assembly.add(make_sole(), "printed_sole")
    for name, shape in zip(("positive_axle_cheek", "negative_axle_cheek"), positioned_cheeks()):
        assembly.add(shape, name)
    servo = import_step(motor_bay.ST3215_SERVO_STEP)
    for name, location in zip(("existing_knee_ST3215", "added_ankle_ST3215"), motor_placements()):
        assembly.add(servo.moved(location), name)
    # Explicit ankle frame names survive in the source-level joint record.
    fixed = assembly.revolute_frame(leg, "ankle_axis_fixed_to_shank", Axis((0, 0, 0), (0, 0, 1)), angular_range=(-35, 35))
    moving = assembly.rigid_frame(sole, "ankle_axis_of_foot", Location())
    assembly.revolute(fixed, moving, angle=0.0, label="nominal_ankle_axis_pose_not_a_validated_range")
    return assembly.build()


def on_build_plate(shape, rotation=(0, 0, 0)):
    """Orient the generated print artifact, then put its minimum on Z=0."""
    result = shape.moved(Location((0, 0, 0), rotation))
    bounds = result.bounding_box()
    result = result.moved(Location((-bounds.min.X, -bounds.min.Y, -bounds.min.Z)))
    result.label = shape.label
    result.color = shape.color
    return result


def printable_lower_leg():
    return on_build_plate(make_lower_leg(), (90, 0, 0))


def printable_sole():
    return on_build_plate(make_sole(), (0, 90, 0))


def printable_cheek():
    # Continuous flat inner face on the bed; all flange growth is outward.
    return on_build_plate(make_cheek())
