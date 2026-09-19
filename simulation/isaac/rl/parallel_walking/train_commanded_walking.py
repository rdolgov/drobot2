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
    from v44_training_contract import (
        EXPERIMENT, TASK, record_completed_training, training_snapshot,
    )

    selected_task = next(
        (sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1]) if value == "--task"),
        next((value.split("=", 1)[1] for value in sys.argv if value.startswith("--task=")), ""),
    )
    experiment_root = Path.cwd() / "logs" / "rsl_rl" / EXPERIMENT
    before = training_snapshot(experiment_root) if selected_task == TASK else None
    result = run_train_cli()
    if before is not None and result in (None, 0):
        record_completed_training(experiment_root, before)
    raise SystemExit(result)
