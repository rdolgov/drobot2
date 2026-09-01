"""Left/right symmetry augmentation for the V25 walking policy contract."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from rsl_rl.env import VecEnv


__all__ = ["compute_left_right_symmetric_states"]


_POLICY_OBSERVATION_COUNT = 50
_CRITIC_OBSERVATION_COUNT = 58
_ACTION_COUNT = 12

# Joint/action ordering is joint-kind-major:
#   FL abd, RL abd, FR abd, RR abd,
#   FL hip, RL hip, FR hip, RR hip,
#   FL knee, RL knee, FR knee, RR knee.
_LEFT_RIGHT_JOINT_SOURCE_INDEX = (2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9)
_ABDUCTION_JOINT_COUNT = 4
_LEG_NAMES = ("front_left", "rear_left", "front_right", "rear_right")
_MIRRORED_LEG_NAME = {
    "front_left": "front_right",
    "rear_left": "rear_right",
    "front_right": "front_left",
    "rear_right": "rear_left",
}


def _mirror_joint_data(joint_data: torch.Tensor) -> torch.Tensor:
    """Swap left/right legs and reverse hip-abduction signs."""

    if joint_data.shape[-1] != _ACTION_COUNT:
        raise ValueError(
            f"Expected {_ACTION_COUNT} joint values, got {joint_data.shape[-1]}"
        )
    mirrored = joint_data[..., _LEFT_RIGHT_JOINT_SOURCE_INDEX].clone()
    mirrored[..., :_ABDUCTION_JOINT_COUNT] *= -1.0
    return mirrored


def _mirror_policy_observation(observation: torch.Tensor) -> torch.Tensor:
    """Mirror the fixed 50-value deployable policy observation."""

    if observation.shape[-1] != _POLICY_OBSERVATION_COUNT:
        raise ValueError(
            "V25 left/right symmetry requires a 50-value policy observation, "
            f"got {observation.shape[-1]}"
        )

    mirrored = observation.clone()

    # Command is [forward, lateral, yaw].
    mirrored[..., 1] *= -1.0
    mirrored[..., 2] *= -1.0

    # The analytic crawl order mirrors into the same order half a cycle later,
    # so sin(phase) and cos(phase) both reverse sign after the leg swap.
    mirrored[..., 3:5] *= -1.0

    # Angular velocity is an axial vector under the sagittal-plane reflection.
    mirrored[..., 5] *= -1.0
    mirrored[..., 7] *= -1.0

    # Projected gravity and linear acceleration are polar vectors: y reverses.
    mirrored[..., 9] *= -1.0
    mirrored[..., 12] *= -1.0

    mirrored[..., 14:26] = _mirror_joint_data(observation[..., 14:26])
    mirrored[..., 26:38] = _mirror_joint_data(observation[..., 26:38])
    mirrored[..., 38:50] = _mirror_joint_data(observation[..., 38:50])
    return mirrored


def _foot_contact_mirror_indices(env: VecEnv) -> tuple[int, ...]:
    """Return the source contact index for every mirrored destination foot."""

    unwrapped = env.unwrapped
    sensor_names = tuple(str(name) for name in unwrapped._foot_sensor_names)
    if len(sensor_names) != len(_LEG_NAMES):
        raise ValueError(
            "V25 left/right symmetry requires four foot-contact values, "
            f"got sensors {sensor_names}"
        )

    sensor_leg_names: list[str] = []
    for sensor_name in sensor_names:
        matches = [leg_name for leg_name in _LEG_NAMES if leg_name in sensor_name]
        if len(matches) != 1:
            raise ValueError(
                f"Could not identify exactly one leg in foot sensor {sensor_name!r}"
            )
        sensor_leg_names.append(matches[0])

    sensor_index_by_leg = {
        leg_name: sensor_index
        for sensor_index, leg_name in enumerate(sensor_leg_names)
    }
    if set(sensor_index_by_leg) != set(_LEG_NAMES):
        raise ValueError(
            "Foot sensor names must identify front-left, rear-left, front-right, "
            f"and rear-right exactly once; got {sensor_names}"
        )
    return tuple(
        sensor_index_by_leg[_MIRRORED_LEG_NAME[leg_name]]
        for leg_name in sensor_leg_names
    )


def _mirror_critic_observation(
    env: VecEnv, observation: torch.Tensor
) -> torch.Tensor:
    """Mirror the 58-value critic observation, including privileged state."""

    if observation.shape[-1] != _CRITIC_OBSERVATION_COUNT:
        raise ValueError(
            "V25 left/right symmetry requires a 58-value critic observation, "
            f"got {observation.shape[-1]}"
        )

    mirrored = observation.clone()
    mirrored[..., :_POLICY_OBSERVATION_COUNT] = _mirror_policy_observation(
        observation[..., :_POLICY_OBSERVATION_COUNT]
    )

    # Root linear velocity is a polar vector; height is unchanged.
    mirrored[..., 51] *= -1.0
    contact_source_indices = _foot_contact_mirror_indices(env)
    mirrored[..., 54:58] = observation[..., 54:58][
        ..., contact_source_indices
    ]
    return mirrored


@torch.no_grad()
def compute_left_right_symmetric_states(
    env: VecEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append one left/right-mirrored copy for RSL-RL symmetry training."""

    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_augmented = obs.repeat(2)
        if "policy" not in obs.keys():
            raise KeyError("Symmetry augmentation requires the 'policy' observation group")
        obs_augmented["policy"][batch_size:] = _mirror_policy_observation(
            obs["policy"]
        )
        if "critic" in obs.keys():
            obs_augmented["critic"][batch_size:] = _mirror_critic_observation(
                env, obs["critic"]
            )
    else:
        obs_augmented = None

    if actions is not None:
        if actions.shape[-1] != _ACTION_COUNT:
            raise ValueError(
                f"V25 left/right symmetry requires {_ACTION_COUNT} actions, "
                f"got {actions.shape[-1]}"
            )
        batch_size = actions.shape[0]
        actions_augmented = torch.empty(
            (batch_size * 2, *actions.shape[1:]),
            device=actions.device,
            dtype=actions.dtype,
        )
        actions_augmented[:batch_size] = actions
        actions_augmented[batch_size:] = _mirror_joint_data(actions)
    else:
        actions_augmented = None

    return obs_augmented, actions_augmented
