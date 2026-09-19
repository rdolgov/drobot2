"""One motor-socket foot on the user's retained lower-leg output fork."""
from build123d import Axis, Color, Location
from drobot_cad.parts import ankle_v2_printable as v21
from drobot_cad.parts import st3215_motor_bay as bay
from drobot_cad.parts import upper_arm

SERVO_AXIS_TO_BAY_X_MM = -v21.BAY_TO_SHAFT_X_MM
BAY_CENTER_Z_MM = upper_arm.ST3215_MOTOR_BAY_CENTER_Z_MM
FUSION_OVERLAP_MM = .4
SOLE_TOP_X_MM = SERVO_AXIS_TO_BAY_X_MM - FUSION_OVERLAP_MM
SOLE_CONTACT_X_MM = SOLE_TOP_X_MM + v21.SOLE_THICKNESS_MM

def bay_location():
    return Location((SERVO_AXIS_TO_BAY_X_MM,0,BAY_CENTER_Z_MM))

def servo_location():
    return bay_location()*bay.st3215_installed_location()

def make_existing_leg_foot():
    socket = bay.gen_step().moved(bay_location())
    sole = v21._x_prism(v21.SOLE_LENGTH_MM,v21.SOLE_WIDTH_MM,v21.SOLE_THICKNESS_MM,
                        SOLE_TOP_X_MM,v21.SOLE_CORNER_MM)
    # Full-thickness sole with no cable groove, per the user's R3 revision.
    foot = sole.fuse(socket).clean()
    foot.label = 'V2_3_R3_solid_sole_motor_socket_reuse_existing_lower_leg'
    foot.color = Color(.93,.59,.10)
    return foot

def printable_existing_leg_foot():
    return v21.on_build_plate(make_existing_leg_foot(),(0,90,0))

def make_assembly():
    from cadgen.assembly import AssemblyHelper
    from cadgen.step_scene import import_step
    leg_pose = Location(tuple(-v for v in upper_arm.DISTAL_FORK_AXIS_MM))
    retained = upper_arm.gen_step().moved(leg_pose)
    retained.label = 'REUSED_existing_lower_leg_do_not_print_again'
    retained.color = Color(.22,.55,.82)
    asm = AssemblyHelper('ankle_V2_3_existing_lower_leg_and_new_motorized_foot')
    fixed_leg = asm.add(retained,'REUSE_existing_lower_leg')
    foot = asm.add(make_existing_leg_foot(),'PRINT_one_motor_socket_foot')
    servo = import_step(bay.ST3215_SERVO_STEP)
    asm.add(servo.moved(servo_location()),'ADD_ankle_ST3215_case_moves_with_foot')
    asm.add(servo.moved(leg_pose*upper_arm.st3215_preview_location()),'REUSE_existing_knee_ST3215')
    fixed = asm.revolute_frame(fixed_leg,'retained_fork_axis',Axis((0,0,0),(0,0,1)),angular_range=(-35,35))
    moving = asm.rigid_frame(foot,'servo_output_axis_in_foot',Location())
    asm.revolute(fixed,moving,angle=0.,label='nominal_pose_only_case_moves_with_foot_unqualified_range')
    return asm.build()
