"""Small PPO fine-tune for a stable 190 mm front-left foot raise.

This is the simplest useful stair prerequisite: start from four-foot support,
raise only the front-left foot at least 190 mm, hold that clearance for 0.50 s,
and fail the episode if body attitude, support contact, or support slip leaves
the real-test safety envelope.  No stair approach or foothold transfer is
trained here.
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
    / "ppo-stairs-v91-front-left-190mm-hold-1024-seed1043"
)
DEFAULT_RESUME = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "models"
    / "ppo-stairs-v80-single-foot-190mm-2048-seed1030"
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
        ("--total-timesteps", "1024"),
        ("--seed", "1043"),
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
