"""Aligned white-body build entry for two-STL slicer workflows."""

from build123d import Shape

from drobot_cad.parts.apriltag_body_marker import make_white_plate


def gen_step() -> Shape:
    return make_white_plate()

