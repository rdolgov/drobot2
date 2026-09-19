"""Assembled printable foot visual in ankle coordinates, millimetres."""
from build123d import Compound
from drobot_cad.parts.ankle_v2_printable import make_sole, positioned_cheeks

def gen_step():
    return Compound(children=[make_sole(), *positioned_cheeks()])
