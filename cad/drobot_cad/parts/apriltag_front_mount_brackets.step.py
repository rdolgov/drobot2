"""Build entry for the AprilTag front-wall mounting bracket pair."""

from drobot_cad.parts.apriltag_front_mount_brackets import (
    make_apriltag_front_mount_brackets,
)


def gen_step():
    return make_apriltag_front_mount_brackets()

