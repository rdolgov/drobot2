"""Retained existing lower leg, in knee-joint coordinates, for simulation."""
from build123d import Location
from drobot_cad.parts import upper_arm

def gen_step():
    origin = upper_arm.st3215_output_axis_location().position
    return upper_arm.gen_step().moved(Location(tuple(-v for v in origin)))
