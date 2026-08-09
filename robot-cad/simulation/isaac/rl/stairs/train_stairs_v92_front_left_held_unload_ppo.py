"""Train a held three-foot unload before the V91 front-left lift.

V92 starts from the verified V75 front-right foothold snapshot and fine-tunes
all twelve transfer actions together.  The curriculum must hold the next
front-left swing foot below successively stricter load limits while the other
three feet stay loaded and the robot remains upright.  The final transfer gate
is continuous for 0.50 s; foot lift and stair landing remain separate skills.
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
    / "ppo-stairs-v92-front-left-held-unload-8192-seed1045"
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
DEFAULT_INITIAL_POLICY = (
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
        ("--seed", "1045"),
        ("--device", "cpu"),
        ("--fixed-placement-level", "left-quarter-tread-load"),
        ("--phase-train-leg", "front_left"),
        ("--phase-train-transfer", ""),
        ("--precursor-leg-model", f"front_right={DEFAULT_PRECURSOR}"),
        ("--phase-residual-swing-support-all", ""),
        ("--phase-compact-residual-action", ""),
        ("--phase-reset-attempts", "32"),
        ("--phase-transfer-unload-successes-per-level", "1"),
        ("--initialize-from-stairs", str(DEFAULT_INITIAL_POLICY)),
        ("--initialize-stairs-source-action-mode", "swing_plus_support_all"),
        ("--ppo-learning-rate", "0.00003"),
        ("--ppo-initial-log-std", "-2.75"),
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
            "6",
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
