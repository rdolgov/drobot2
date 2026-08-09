"""Train a bounded held-unload residual around the frozen V90 policy.

V92 showed that fine-tuning all inherited transfer weights destroys the best
known unload behavior.  V93 instead keeps V90 deterministic and frozen, then
learns a zero-initialized, ten-percent residual across the same twelve joints.
The curriculum requires continuous 0.50 s holds below 6 N, 4 N, and 1 N while
the other three feet remain loaded and the body stays upright.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
TRAINER = SCRIPT_DIR / "train_stairs_ppo.py"
DEFAULT_CONFIG = SCRIPT_DIR / "quadruped_stairs_v14_front_pair_right_then_left.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "output"
    / "rl"
    / "ppo-stairs-v93-front-left-frozen-base-held-unload-8192-seed1046"
)
DEFAULT_PRECURSOR = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "output"
    / "rl"
    / "ppo-stairs-v75-first-strict-foothold-2048-seed1025"
    / "drobot_stairs_ppo_final.zip"
)
DEFAULT_BASE_POLICY = (
    PROJECT_ROOT
    / "simulation"
    / "isaac"
    / "models"
    / "ppo-stairs-v90-frozen-v88-support-knees-sustained-unload-8192-seed1034"
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
        (
            "--first-tread-profile",
            "front-pair-preposition-load-advance-forward-floor",
        ),
        ("--output-dir", str(DEFAULT_OUTPUT)),
        ("--total-timesteps", "8192"),
        ("--curriculum-total-timesteps", "8192"),
        ("--seed", "1046"),
        ("--device", "cpu"),
        ("--fixed-placement-level", "left-quarter-tread-load"),
        ("--phase-train-leg", "front_left"),
        ("--phase-train-transfer", ""),
        ("--precursor-leg-model", f"front_right={DEFAULT_PRECURSOR}"),
        ("--phase-base-model", str(DEFAULT_BASE_POLICY)),
        ("--phase-residual-swing-support-all", ""),
        ("--phase-compact-residual-action", ""),
        ("--phase-residual-scale", "0.10"),
        ("--phase-reset-attempts", "32"),
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
            "6",
            "--phase-transfer-unload-threshold-n",
            "4",
            "--phase-transfer-unload-threshold-n",
            "1",
        ]
    if not _has_option(arguments, "--phase-transfer-upright-cosine"):
        arguments[:0] = [
            "--phase-transfer-upright-cosine",
            "0.976",
            "--phase-transfer-upright-cosine",
            "0.977",
            "--phase-transfer-upright-cosine",
            "0.9781476",
        ]
    sys.argv = [str(TRAINER), *arguments]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
