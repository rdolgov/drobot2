"""Register, evaluate, and optionally record the Drobot pure-RL task."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch
from isaaclab import app as isaaclab_app
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


def _consume_viewer_env_index() -> int | None:
    """Remove the local camera-selection option before unified CLI parsing."""

    flag = "--viewer_env_index"
    if flag not in sys.argv:
        return None
    index = sys.argv.index(flag)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"{flag} requires an integer")
    value = int(sys.argv[index + 1])
    del sys.argv[index : index + 2]
    return value


def _consume_flag(flag: str) -> bool:
    """Remove a review-only boolean option before unified CLI parsing."""

    if flag not in sys.argv:
        return False
    sys.argv.remove(flag)
    return True


VIEWER_ENV_INDEX = _consume_viewer_env_index()
HIDE_OTHER_ROBOTS = _consume_flag("--hide_other_robots")
_launch_simulation = isaaclab_app.launch_simulation


@contextlib.contextmanager
def _launch_with_selected_view(env_cfg: object, args_cli: object):
    """Aim the ordinary third-person viewer at a selected parallel robot."""

    if VIEWER_ENV_INDEX is not None:
        env_cfg.viewer.origin_type = "world"
    with _launch_simulation(env_cfg, args_cli):
        yield


isaaclab_app.launch_simulation = _launch_with_selected_view

_rsl_step = RslRlVecEnvWrapper.step


def _step_with_selected_success(
    self: RslRlVecEnvWrapper, actions: torch.Tensor
) -> tuple[object, torch.Tensor, torch.Tensor, dict]:
    """Report whether the robot followed by the review camera passed."""

    observations, rewards, dones, extras = _rsl_step(self, actions)
    if VIEWER_ENV_INDEX is not None:
        if not getattr(self, "_drobot_camera_aimed", False):
            origins = self.unwrapped._terrain.env_origins
            roots = self.unwrapped._robot.data.root_pos_w.torch
            if HIDE_OTHER_ROBOTS:
                from pxr import UsdGeom

                for env_index in range(self.num_envs):
                    if env_index == VIEWER_ENV_INDEX:
                        continue
                    prim = self.unwrapped.scene.stage.GetPrimAtPath(
                        f"/World/envs/env_{env_index}"
                    )
                    if prim.IsValid():
                        UsdGeom.Imageable(prim).MakeInvisible()
            origin_distances = torch.cdist(origins[:, :2], origins[:, :2])
            root_distances = torch.cdist(roots[:, :2], roots[:, :2])
            diagonal = torch.arange(self.num_envs, device=origins.device)
            origin_distances[diagonal, diagonal] = torch.inf
            root_distances[diagonal, diagonal] = torch.inf
            selected_local = roots[VIEWER_ENV_INDEX] - origins[VIEWER_ENV_INDEX]
            print(
                "[DROBOT_ENV_SPACING_AUDIT] "
                f"envs={self.num_envs} "
                f"min_origin_xy_m={float(origin_distances.min().item()):.6f} "
                f"min_root_xy_m={float(root_distances.min().item()):.6f} "
                f"selected_local_root={selected_local.detach().cpu().tolist()} "
                f"replicate_physics={self.unwrapped.scene.cfg.replicate_physics} "
                f"other_robots_hidden={HIDE_OTHER_ROBOTS}",
                flush=True,
            )
            origin = origins[VIEWER_ENV_INDEX].detach().cpu()
            eye = (float(origin[0] - 0.75), float(origin[1] - 0.65), 0.68)
            target = (float(origin[0] + 0.05), float(origin[1]), 0.30)
            self.unwrapped.sim.set_camera_view(eye, target)
            from isaaclab_physx.renderers.kit_viewport_utils import (
                set_kit_renderer_camera_view,
            )

            set_kit_renderer_camera_view(
                eye=eye,
                target=target,
                camera_prim_path=self.unwrapped.cfg.viewer.cam_prim_path,
            )
            root = roots[VIEWER_ENV_INDEX]
            print(
                f"[DROBOT_REVIEW_CAMERA] origin={origin.tolist()} "
                f"root={root.detach().cpu().tolist()}",
                flush=True,
            )
            self._drobot_camera_aimed = True
        completion_reward = float(self.unwrapped.cfg.success_completion_reward_scale)
        successful = torch.nonzero(
            dones.bool() & (rewards >= 0.5 * completion_reward), as_tuple=False
        ).flatten()
        if successful.numel() > 0:
            indices = ",".join(str(int(index)) for index in successful.tolist())
            selected = bool((successful == VIEWER_ENV_INDEX).any())
            print(
                f"[DROBOT_RECORDED_SUCCESS] envs={indices} selected={selected}",
                flush=True,
            )
    return observations, rewards, dones, extras


RslRlVecEnvWrapper.step = _step_with_selected_success

PACKAGE_PARENT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_PARENT))

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

import parallel_stairs  # noqa: E402, F401

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
