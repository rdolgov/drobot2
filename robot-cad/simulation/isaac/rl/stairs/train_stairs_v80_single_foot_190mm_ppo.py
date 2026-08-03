"""Fine-tune the isolated front-left 190 mm lift-and-balance policy.

This bounded prerequisite starts beside the exact 180 mm-rise, 250 mm-tread
staircase.  The robot unloads and raises only its front-left foot while the
other three feet remain loaded and the body stays upright.  Stair placement
and multi-leg transfer are deliberately outside this task.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
TRAINER = SCRIPT_DIR / "train_stairs_ppo.py"
DEFAULT_CONFIG = SCRIPT_DIR / "quadruped_stairs_v15_front_left_stabilized_lift.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "output"
    / "rl"
    / "ppo-stairs-v80-single-foot-190mm-2048-seed1030"
)
DEFAULT_RESUME = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "models"
    / "ppo-stairs-v17-single-foot-190mm-small"
    / "drobot_stairs_ppo_final.zip"
)


def _has_option(arguments: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def main() -> None:
    arguments = list(sys.argv[1:])
    defaults = (
        ("--config", str(DEFAULT_CONFIG)),
        ("--output-dir", str(DEFAULT_OUTPUT)),
        ("--placement-start-level", "front-left-stabilized-190mm-lift-hold"),
        ("--fixed-active-steps", "1"),
        ("--total-timesteps", "2048"),
        ("--seed", "1030"),
        ("--device", "cpu"),
        ("--resume", str(DEFAULT_RESUME)),
        ("--ppo-learning-rate", "0.00005"),
        ("--ppo-entropy-coefficient", "0.0"),
    )
    for option, value in reversed(defaults):
        if not _has_option(arguments, option):
            arguments[:0] = [option, value]
    sys.argv = [str(TRAINER), *arguments]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
