"""Run a bounded deterministic RSL-RL evaluation without RGB frame capture.

The unified play backend uses its video-length counter as its only bounded-play
stop condition.  This wrapper preserves that counter while replacing the RGB
recorder with a pass-through wrapper, so hundreds of parallel environments can
be evaluated quickly and the environment's exact episode totals are flushed on
close.  Pass ``--video --video_length N``; no video file is produced here.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import gymnasium as gym
import torch
import warp as wp
from isaaclab import app as isaaclab_app
from isaaclab_rl.entrypoints import common as entrypoint_common
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

wp.config.enable_backward = False

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_PARENT))


def _bounded_without_rgb(env: gym.Env, **_: object) -> gym.Env:
    return env


_create_isaaclab_env = entrypoint_common.create_isaaclab_env
_launch_simulation = isaaclab_app.launch_simulation


@contextlib.contextmanager
def _launch_headless_bounded(env_cfg: object, args_cli: object):
    """Launch without cameras, then restore the backend's bounded-loop flag."""
    video_requested = args_cli.video
    cameras_requested = args_cli.enable_cameras
    args_cli.video = False
    args_cli.enable_cameras = False
    try:
        with _launch_simulation(env_cfg, args_cli):
            args_cli.video = video_requested
            yield
    finally:
        args_cli.video = video_requested
        args_cli.enable_cameras = cameras_requested


def _create_headless_bounded_env(*args: object, **kwargs: object) -> gym.Env:
    """Prevent the backend's stop flag from also requesting rgb_array rendering."""
    args_cli = args[2]
    video_requested = args_cli.video
    args_cli.video = False
    try:
        return _create_isaaclab_env(*args, **kwargs)
    finally:
        args_cli.video = video_requested


# The RSL-RL play loop still uses --video_length to terminate, but this avoids
# RecordVideo calling env.render() at every step.
gym.wrappers.RecordVideo = _bounded_without_rgb  # type: ignore[assignment]
isaaclab_app.launch_simulation = _launch_headless_bounded
entrypoint_common.create_isaaclab_env = _create_headless_bounded_env


def _skip_policy_export(self: OnPolicyRunner, path: str, filename: str, **_: object) -> None:
    """Avoid recompiling identical deployment artifacts during checkpoint screening."""


OnPolicyRunner.export_policy_to_jit = _skip_policy_export
OnPolicyRunner.export_policy_to_onnx = _skip_policy_export

_rsl_step = RslRlVecEnvWrapper.step


def _step_with_success_indices(
    self: RslRlVecEnvWrapper, actions: torch.Tensor
) -> tuple[object, torch.Tensor, torch.Tensor, dict]:
    """Print strict terminal-success indices for reproducible visual review."""

    observations, rewards, dones, extras = _rsl_step(self, actions)
    step = getattr(self, "_drobot_evaluation_step", 0)
    self._drobot_evaluation_step = step + 1
    completion_reward = float(self.unwrapped.cfg.success_completion_reward_scale)
    successful = torch.nonzero(
        dones.bool() & (rewards >= 0.5 * completion_reward), as_tuple=False
    ).flatten()
    if successful.numel() > 0:
        indices = ",".join(str(int(index)) for index in successful.tolist())
        print(f"[DROBOT_SUCCESS_INDICES] step={step} envs={indices}", flush=True)
    return observations, rewards, dones, extras


RslRlVecEnvWrapper.step = _step_with_success_indices

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

import parallel_stairs  # noqa: E402, F401

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
