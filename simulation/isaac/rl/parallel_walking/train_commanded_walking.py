"""Register and train the Drobot commanded-walking task."""

from __future__ import annotations

import sys
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

import parallel_walking  # noqa: E402, F401

if __name__ == "__main__":
    raise SystemExit(run_train_cli())
