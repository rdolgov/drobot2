"""Fine-tune the isolated 190 mm front-foot balance policy.

This is the deliberately simplified gate before multi-foot stair climbing:
the robot starts on four feet beside the 250 mm-deep stair, unloads the
front-left foot, and must hold a measured 190 mm lift without tipping or
losing the three-foot support margin.
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
    / "ppo-stairs-v17-single-foot-190mm-small"
)
DEFAULT_RESUME = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "models"
    / "ppo-stairs-v15-front-left-190mm-lift-small"
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
        ("--total-timesteps", "4096"),
        ("--seed", "271"),
        ("--device", "cpu"),
        ("--resume", str(DEFAULT_RESUME)),
    )
    for option, value in reversed(defaults):
        if not _has_option(arguments, option):
            arguments[:0] = [option, value]
    sys.argv = [str(TRAINER), *arguments]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
