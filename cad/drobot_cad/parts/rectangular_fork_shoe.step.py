"""Build entry for the rectangular PLA fork shoe."""

from build123d import Shape

from drobot_cad.parts import rectangular_fork_shoe


def gen_step() -> Shape:
    """Return the source-level rectangular shoe geometry."""
    return rectangular_fork_shoe.make_rectangular_fork_shoe()
