"""Train the blind right-placement then left-lift/placement curriculum."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
TRAINER = SCRIPT_DIR / "train_stairs_ppo.py"
DEFAULT_CONFIG = (
    SCRIPT_DIR / "quadruped_stairs_v16_front_pair_proprioceptive_support.yaml"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "output"
    / "rl"
    / "ppo-stairs-v16-front-pair"
)
DEFAULT_PRECURSOR = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "models"
    / "ppo-stairs-v10-180mm-25cm-front-right-placement-small"
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
        ("--placement-start-level", "left-supported-140mm-lift"),
        ("--phase-train-leg", "front_left"),
        (
            "--precursor-leg-model",
            f"front_right={DEFAULT_PRECURSOR}",
        ),
    )
    for option, value in reversed(defaults):
        if not _has_option(arguments, option):
            arguments[:0] = [option, value]
    if not _has_option(arguments, "--phase-residual-support-only"):
        arguments.insert(0, "--phase-residual-support-only")
    sys.argv = [str(TRAINER), *arguments]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
