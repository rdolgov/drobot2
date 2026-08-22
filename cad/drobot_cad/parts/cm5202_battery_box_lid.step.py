"""Build entry for the CM5202 battery-box lid."""

from build123d import Shape

from drobot_cad.parts import cm5202_battery_box


def gen_step() -> Shape:
    """Return the printable screw-on lid."""
    return cm5202_battery_box.make_cm5202_battery_box_lid()
