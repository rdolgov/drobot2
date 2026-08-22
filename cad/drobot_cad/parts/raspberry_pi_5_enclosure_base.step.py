"""Build entry for the printable Raspberry Pi 5 enclosure base."""

from drobot_cad.parts.raspberry_pi_5_enclosure import make_base


def gen_step():
    return make_base()
