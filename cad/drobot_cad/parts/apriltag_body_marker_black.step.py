"""Aligned black-marker build entry for two-STL slicer workflows."""

from build123d import Compound

from drobot_cad.parts.apriltag_body_marker import make_black_tag


def gen_step() -> Compound:
    return make_black_tag()

