"""Build entry for the CM5202 battery box."""

from build123d import Shape

from drobot_cad.parts import cm5202_battery_box


def gen_step() -> Shape:
    """Return the printable main battery box."""
    return cm5202_battery_box.make_cm5202_battery_box()
