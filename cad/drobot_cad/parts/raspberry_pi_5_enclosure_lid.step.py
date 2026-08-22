"""Build entry for the printable Raspberry Pi 5 enclosure lid."""

from drobot_cad.parts.raspberry_pi_5_enclosure import make_lid


def gen_step():
    return make_lid()
