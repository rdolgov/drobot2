"""Build entry for the exploded CM5202 battery-box fit preview."""

from build123d import Compound

from drobot_cad.parts import cm5202_battery_box


def gen_step() -> Compound:
    """Return the box, measured battery proxy, and exploded lid."""
    return cm5202_battery_box.make_cm5202_battery_box_fit_preview()
