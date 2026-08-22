"""Build entry for the printable lid-mounted BNO085 protection roof."""

from drobot_cad.parts.raspberry_pi_5_enclosure import make_imu_cover


def gen_step():
    return make_imu_cover()
