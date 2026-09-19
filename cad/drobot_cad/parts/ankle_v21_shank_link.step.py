"""Simulation visual in knee-link coordinates; not the print orientation."""
from build123d import Location
from drobot_cad.parts.ankle_v2_printable import make_lower_leg, KNEE_TO_ANKLE_MM

def gen_step():
    return make_lower_leg().moved(Location((KNEE_TO_ANKLE_MM, 0, 0)))
