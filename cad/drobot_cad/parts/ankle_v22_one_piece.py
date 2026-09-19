"""Fused V2.2 ankle foot; preserve V2.1 sources used by active RL jobs."""
from build123d import Axis, Color, Location
from drobot_cad.parts import ankle_v2_printable as v21

# Fixed wheel mating span, from the immutable ST3215 wheel faces. This replaces
# V2.1's slotted axial adjustment. Tune only this revision after a physical fit.
WHEEL_FACE_GAP_MM = 2 * v21.WHEEL_FACE_HALF_SPAN_MM
ROOT_OVERLAP_MM = 0.4

def _solid_root_cheek():
    disk = v21._cyl_z(v21.CHEEK_RADIUS_MM, v21.CHEEK_THICKNESS_MM)
    stem = v21._box(v21.SOLE_TOP_X_MM + ROOT_OVERLAP_MM,
                    2*v21.CHEEK_RADIUS_MM, v21.CHEEK_THICKNESS_MM,
                    ((v21.SOLE_TOP_X_MM + ROOT_OVERLAP_MM)/2,0,v21.CHEEK_THICKNESS_MM/2))
    root_length = v21.SOLE_TOP_X_MM + ROOT_OVERLAP_MM - v21.FLANGE_TOP_X_MM
    root = v21._box(root_length,v21.FLANGE_WIDTH_Y_MM,v21.FLANGE_WIDTH_Z_MM,
                    (v21.FLANGE_TOP_X_MM+root_length/2,0,v21.FLANGE_WIDTH_Z_MM/2))
    cheek = disk.fuse(stem,root)
    cheek -= v21._cyl_z(v21.SHAFT_SCREW_ACCESS_MM/2,v21.CHEEK_THICKNESS_MM+2*v21.EPS,z=-v21.EPS)
    for x,y in v21.WHEEL_HOLES_XY_MM:
        cheek -= v21._cyl_z(v21.M3_CLEARANCE_MM/2,v21.CHEEK_THICKNESS_MM+2*v21.EPS,x,y,-v21.EPS)
        cheek -= v21._cyl_z(v21.HORN_HEAD_CLEARANCE_MM/2,v21.HORN_HEAD_RECESS_MM+v21.EPS,
                            x,y,v21.CHEEK_THICKNESS_MM-v21.HORN_HEAD_RECESS_MM)
    return cheek

def make_one_piece_foot():
    # A continuous sole: the old slots, counterbores and nut traps are absent.
    sole = v21._x_prism(v21.SOLE_LENGTH_MM,v21.SOLE_WIDTH_MM,v21.SOLE_THICKNESS_MM,
                        v21.SOLE_TOP_X_MM,v21.SOLE_CORNER_MM)
    ribs = [v21._box(4.4,80.,4.,(22.2,0,z)) for z in (-6.,6.)]
    cheek = _solid_root_cheek()
    positive = cheek.moved(Location((0,0,WHEEL_FACE_GAP_MM/2)))
    negative = cheek.rotate(Axis.X,180).moved(Location((0,0,-WHEEL_FACE_GAP_MM/2)))
    foot = sole.fuse(*ribs,positive,negative).clean()
    foot.label = 'V2_2_single_print_sole_and_two_integral_ankle_cheeks'
    foot.color = Color(.93,.59,.10)
    return foot

def printable_one_piece_foot():
    return v21.on_build_plate(make_one_piece_foot(),(0,90,0))

def make_assembly():
    from cadgen.assembly import AssemblyHelper
    from cadgen.step_scene import import_step
    from drobot_cad.parts import st3215_motor_bay
    assembly = AssemblyHelper('ankle_V2_2_one_piece_foot_assembly')
    leg = assembly.add(v21.make_lower_leg(),'existing_V2_1_lower_leg')
    foot = assembly.add(make_one_piece_foot(),'one_piece_printed_foot')
    servo = import_step(st3215_motor_bay.ST3215_SERVO_STEP)
    for name,location in zip(('existing_knee_ST3215','added_ankle_ST3215'),v21.motor_placements()):
        assembly.add(servo.moved(location),name)
    fixed = assembly.revolute_frame(leg,'ankle_axis_fixed_to_shank',Axis((0,0,0),(0,0,1)),angular_range=(-35,35))
    moving = assembly.rigid_frame(foot,'one_piece_foot_axis',Location())
    assembly.revolute(fixed,moving,angle=0.,label='nominal_ankle_pose_range_not_physically_qualified')
    return assembly.build()
