"""Train terrain-aware residual PPO over the frozen flat walking policy."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
TRAINER = SCRIPT_DIR / "train_stairs_ppo.py"
DEFAULT_CONFIG = SCRIPT_DIR / "quadruped_stairs_v5.yaml"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "simulation" / "isaac" / "output" / "rl" / "ppo-stairs-v5"
)


def _has_option(arguments: list[str], option: str) -> bool:
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def main() -> None:
    arguments = list(sys.argv[1:])
    if not _has_option(arguments, "--config"):
        arguments[:0] = ["--config", str(DEFAULT_CONFIG)]
    if not _has_option(arguments, "--output-dir"):
        arguments[:0] = ["--output-dir", str(DEFAULT_OUTPUT)]
    sys.argv = [str(TRAINER), *arguments]
    runpy.run_path(str(TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
