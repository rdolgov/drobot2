"""Build entry for the two-material AprilTag robot body marker."""

from build123d import Compound

from drobot_cad.parts.apriltag_body_marker import make_apriltag_body_marker


def gen_step() -> Compound:
    return make_apriltag_body_marker()

