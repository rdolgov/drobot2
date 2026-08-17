"""Build entry for the installed rectangular PLA fork-shoe preview."""

from build123d import Compound

from drobot_cad.assembly import rectangular_fork_shoe_fit_preview


def gen_step() -> Compound:
    """Return the source-level installed fit preview."""
    return rectangular_fork_shoe_fit_preview.make_fit_preview()
