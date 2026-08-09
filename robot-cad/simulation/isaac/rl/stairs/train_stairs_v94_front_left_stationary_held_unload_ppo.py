"""Train front-left held unloading from a stationary captured boundary.

The V94 snapshot is captured from the lowest-load state inside a verified
0.10 s stable V90 window, then re-anchored to a zero-displacement COM target.
This wrapper exposes a fresh zero-initialized 12-joint residual and trains the
continuous 0.50 s unload gate through 8 N, 4 N, and the final 1 N threshold.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
TRAINER = SCRIPT_DIR / "train_stairs_ppo.py"
DEFAULT_CONFIG = SCRIPT_DIR / "quadruped_stairs_v14_front_pair_right_then_left.yaml"
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "output"
    / "rl"
    / "front-left-stationary-unload-snapshot-v94-seed1047.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "output"
    / "rl"
    / "ppo-stairs-v94-front-left-stationary-held-unload-8192-seed1048"
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
        (
            "--first-tread-profile",
            "front-pair-preposition-load-advance-forward-floor",
        ),
        ("--output-dir", str(DEFAULT_OUTPUT)),
        ("--total-timesteps", "8192"),
        ("--curriculum-total-timesteps", "8192"),
        ("--seed", "1048"),
        ("--device", "cpu"),
        ("--fixed-placement-level", "left-quarter-tread-load"),
        ("--phase-train-leg", "front_left"),
        ("--phase-train-transfer", ""),
        ("--phase-snapshot", str(DEFAULT_SNAPSHOT)),
        ("--phase-residual-swing-support-all", ""),
        ("--phase-compact-residual-action", ""),
        ("--phase-transfer-unload-successes-per-level", "1"),
        ("--ppo-learning-rate", "0.00003"),
        ("--ppo-initial-log-std", "-3.20"),
        ("--ppo-entropy-coefficient", "0"),
    )
    for option, value in reversed(defaults):
        if not _has_option(arguments, option):
            arguments[:0] = [option] if not value else [option, value]
    if not _has_option(arguments, "--phase-transfer-unload-threshold-n"):
        arguments[:0] = [
            "--phase-transfer-unload-threshold-n",
            "8",
            "--phase-transfer-unload-threshold-n",
            "4",
            "--phase-transfer-unload-threshold-n",
            "1",
        ]
    if not _has_option(arguments, "--phase-transfer-upright-cosine"):
        arguments[:0] = [
            "--phase-transfer-upright-cosine",
            "0.975",
            "--phase-transfer-upright-cosine",
            "0.977",
            "--phase-transfer-upright-cosine",
            "0.9781476",
        ]
    sys.argv = [str(TRAINER), *arguments]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
