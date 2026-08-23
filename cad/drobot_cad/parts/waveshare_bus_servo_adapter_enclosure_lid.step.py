"""Build entry for the Waveshare adapter enclosure lid."""

from drobot_cad.parts.waveshare_bus_servo_adapter_enclosure import make_lid


def gen_step():
    return make_lid()

