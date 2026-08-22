"""Build entry for the CM5202 quadruped battery cradle."""

from build123d import Shape

from drobot_cad.parts import cm5202_battery_cradle


def gen_step() -> Shape:
    """Return the source-level CM5202 cradle geometry."""
    return cm5202_battery_cradle.make_cm5202_battery_cradle()
