"""Episode-persistent latent-mode PPO for symmetric pure stair learning.

The actor chooses one of two whole-action policies once, on the first sensor
sample of an episode, and keeps that latent choice until the environment is
reset.  The latent is policy state, not an observation: the robot still sees
only the 70 deployable IMU, joint, load, previous-action, and ToF values.

Unlike a per-control-step mixture, this implements the exact hierarchical
probability ``p(mode | first observation) * product p(action | state, mode)``.
The categorical log probability is included only on the step where a new mode
is chosen.  A tiny PPO override stores the just-chosen latent state alongside
that transition so recurrent mini-batches can reconstruct the same decision.
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict


class PersistentModeGaussianDistribution(Distribution):
    """Categorical episode mode followed by a conditional diagonal Gaussian."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.20,
        num_modes: int = 2,
        std_range: tuple[float, float] = (1.0e-4, 2.0),
    ) -> None:
        super().__init__(output_dim)
        if num_modes < 2:
            raise ValueError("persistent mode policy requires at least two modes")
        self.num_modes = int(num_modes)
        self.std_range = std_range
        self.std_param = nn.Parameter(torch.full((num_modes, output_dim), init_std))
        self._logits: torch.Tensor | None = None
        self._means: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        self._mode_ids: torch.Tensor | None = None
        self._new_mode: torch.Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self.num_modes + self.num_modes * self.output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        self._logits = mlp_output[..., : self.num_modes]
        self._means = mlp_output[..., self.num_modes :].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        self._std = self.std_param.clamp(*self.std_range).expand_as(self._means)

    def set_commitment_context(
        self, mode_ids: torch.Tensor, new_mode: torch.Tensor
    ) -> None:
        """Set the fixed mode and the one-step categorical decision mask."""
        self._mode_ids = mode_ids.long()
        self._new_mode = new_mode.to(dtype=torch.bool)

    def _selected(self, values: torch.Tensor) -> torch.Tensor:
        index = self._mode_ids[..., None, None].expand(
            *self._mode_ids.shape, 1, values.shape[-1]
        )
        return values.gather(-2, index).squeeze(-2)

    def sample(self) -> torch.Tensor:
        return torch.normal(self.mean, self.std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output[..., : self.num_modes]
        means = mlp_output[..., self.num_modes :].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        modes = logits.argmax(dim=-1)
        index = modes[..., None, None].expand(*modes.shape, 1, self.output_dim)
        return means.gather(-2, index).squeeze(-2)

    def as_deterministic_output_module(self) -> nn.Module:
        return _MostProbablePersistentMode(self.output_dim, self.num_modes)

    @property
    def mean(self) -> torch.Tensor:
        return self._selected(self._means)

    @property
    def std(self) -> torch.Tensor:
        return self._selected(self._std)

    @property
    def entropy(self) -> torch.Tensor:
        gaussian = (0.5 + 0.5 * math.log(2.0 * math.pi) + self.std.log()).sum(dim=-1)
        probabilities = torch.softmax(self._logits, dim=-1)
        categorical = -(probabilities * torch.log_softmax(self._logits, dim=-1)).sum(dim=-1)
        return gaussian + self._new_mode.to(gaussian.dtype) * categorical

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        mode_one_hot = torch.nn.functional.one_hot(
            self._mode_ids, num_classes=self.num_modes
        ).to(self._means.dtype)
        return (
            self._logits,
            self._means,
            self._std,
            mode_one_hot,
            self._new_mode.to(self._means.dtype),
        )

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        centered = (outputs - self.mean) / self.std
        gaussian = (
            -0.5 * centered.square() - self.std.log() - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        selected_logit = torch.log_softmax(self._logits, dim=-1).gather(
            -1, self._mode_ids.unsqueeze(-1)
        ).squeeze(-1)
        return gaussian + self._new_mode.to(gaussian.dtype) * selected_logit

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_logits, old_means, old_std, old_mode_one_hot, old_new_mode = old_params
        new_logits, new_means, new_std, _, _ = new_params
        old_mode = old_mode_one_hot.argmax(dim=-1)
        index = old_mode[..., None, None].expand(*old_mode.shape, 1, self.output_dim)
        old_selected_mean = old_means.gather(-2, index).squeeze(-2)
        old_selected_std = old_std.gather(-2, index).squeeze(-2)
        new_selected_mean = new_means.gather(-2, index).squeeze(-2)
        new_selected_std = new_std.gather(-2, index).squeeze(-2)
        gaussian = (
            (old_selected_std.square() + (old_selected_mean - new_selected_mean).square())
            / (2.0 * new_selected_std.square())
            + new_selected_std.log()
            - old_selected_std.log()
            - 0.5
        ).sum(dim=-1)
        old_log_probability = torch.log_softmax(old_logits, dim=-1)
        new_log_probability = torch.log_softmax(new_logits, dim=-1)
        old_probability = old_log_probability.exp()
        categorical = (
            old_probability * (old_log_probability - new_log_probability)
        ).sum(dim=-1)
        return gaussian + old_new_mode * categorical

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        last_linear = next(module for module in reversed(mlp) if isinstance(module, nn.Linear))
        with torch.no_grad():
            last_linear.bias[: self.num_modes].zero_()


class _MostProbablePersistentMode(nn.Module):
    def __init__(self, output_dim: int, num_modes: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_modes = num_modes

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output[..., : self.num_modes]
        means = mlp_output[..., self.num_modes :].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        modes = logits.argmax(dim=-1)
        index = modes[..., None, None].expand(*modes.shape, 1, self.output_dim)
        return means.gather(-2, index).squeeze(-2)


class PersistentModeActor(MLPModel):
    """Feed-forward component policies with an episode-persistent latent state."""

    is_recurrent: bool = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.distribution, PersistentModeGaussianDistribution):
            raise TypeError("PersistentModeActor requires PersistentModeGaussianDistribution")
        self.num_modes = self.distribution.num_modes
        self._commitment_state: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = MLPModel.get_latent(self, obs)
        mlp_output = self.mlp(latent)
        self.distribution.update(mlp_output)

        if masks is None:
            batch_size = mlp_output.shape[0]
            if self._commitment_state is None or self._commitment_state.shape[1] != batch_size:
                self._commitment_state = mlp_output.new_zeros(1, batch_size, self.num_modes + 1)
            one_hot = self._commitment_state[0, :, : self.num_modes]
            needs_mode = one_hot.sum(dim=-1) < 0.5
            modes = one_hot.argmax(dim=-1)
            if needs_mode.any():
                logits = mlp_output[..., : self.num_modes]
                if stochastic_output:
                    proposed = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)
                else:
                    proposed = logits.argmax(dim=-1)
                modes = torch.where(needs_mode, proposed, modes)
            committed = torch.nn.functional.one_hot(modes, self.num_modes).to(mlp_output.dtype)
            self._commitment_state[0, :, : self.num_modes] = committed
            self._commitment_state[0, :, self.num_modes] = needs_mode.to(mlp_output.dtype)
            self.distribution.set_commitment_context(modes, needs_mode)
        else:
            if hidden_state is None:
                raise ValueError("persistent mode state is required for PPO trajectory replay")
            trajectory_modes = hidden_state[0, :, : self.num_modes].argmax(dim=-1)
            padded_modes = trajectory_modes.unsqueeze(0).expand(mlp_output.shape[0], -1)
            padded_new_mode = torch.zeros_like(padded_modes, dtype=torch.bool)
            padded_new_mode[0] = hidden_state[0, :, self.num_modes] > 0.5
            mlp_output = unpad_trajectories(mlp_output, masks)
            modes = unpad_trajectories(padded_modes.unsqueeze(-1), masks).squeeze(-1)
            new_mode = unpad_trajectories(padded_new_mode.unsqueeze(-1), masks).squeeze(-1)
            self.distribution.update(mlp_output)
            self.distribution.set_commitment_context(modes, new_mode)

        if stochastic_output:
            return self.distribution.sample()
        return self.distribution.mean

    def reset(
        self, dones: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None
    ) -> None:
        if dones is None:
            self._commitment_state = hidden_state
        elif self._commitment_state is not None:
            self._commitment_state[:, dones == 1, :] = 0.0

    def get_hidden_state(self) -> torch.Tensor | None:
        return self._commitment_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._commitment_state is not None:
            self._commitment_state = self._commitment_state.detach()

    def as_jit(self) -> nn.Module:
        return _TorchPersistentModeActor(self)


class _TorchPersistentModeActor(nn.Module):
    """Single-robot deterministic export retaining one argmax mode per episode."""

    def __init__(self, model: PersistentModeActor) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.output_dim = model.distribution.output_dim
        self.num_modes = model.num_modes
        self.register_buffer("mode", torch.zeros(1, dtype=torch.long))
        self.register_buffer("has_mode", torch.zeros(1, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(self.obs_normalizer(x))
        logits = raw[..., : self.num_modes]
        means = raw[..., self.num_modes :].reshape(-1, self.num_modes, self.output_dim)
        if not bool(self.has_mode[0]):
            self.mode[0] = logits[0].argmax()
            self.has_mode[0] = True
        index = self.mode.view(1, 1, 1).expand(means.shape[0], 1, self.output_dim)
        return means.gather(1, index).squeeze(1)

    @torch.jit.export
    def reset(self) -> None:
        self.mode.zero_()
        self.has_mode.zero_()


class PersistentModePPO(PPO):
    """Capture the newly sampled latent before saving the first transition."""

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.hidden_states = (
            self.actor.get_hidden_state(),
            self.critic.get_hidden_state(),
        )
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(
            self.transition.actions
        ).detach()
        self.transition.distribution_params = tuple(
            parameter.detach() for parameter in self.actor.output_distribution_params
        )
        self.transition.observations = obs
        return self.transition.actions


class PersistentBiasGaussianDistribution(Distribution):
    """A learned episode bias plus small independent per-step exploration.

    The persistent bias makes an entire unload attempt explore in one coherent
    joint-space direction.  Its learned center is the deterministic deployment
    bias, so successful episode-level exploration can move the deployed policy
    instead of remaining a sequence of unrelated Gaussian tail samples.
    """

    def __init__(
        self,
        output_dim: int,
        init_action_std: float = 0.08,
        init_bias_std: float = 0.20,
        num_modes: int = 2,
        commitment_credit_scale: float = 1.0,
        std_range: tuple[float, float] = (1.0e-4, 2.0),
    ) -> None:
        super().__init__(output_dim)
        self.num_modes = int(num_modes)
        self.commitment_credit_scale = float(commitment_credit_scale)
        if self.commitment_credit_scale <= 0.0:
            raise ValueError("commitment_credit_scale must be positive")
        self.std_range = std_range
        self.action_std_param = nn.Parameter(
            torch.full((num_modes, output_dim), init_action_std)
        )
        self.bias_std_param = nn.Parameter(
            torch.full((num_modes, output_dim), init_bias_std)
        )
        self._logits: torch.Tensor | None = None
        self._control_means: torch.Tensor | None = None
        self._bias_centers: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._bias_std: torch.Tensor | None = None
        self._mode_ids: torch.Tensor | None = None
        self._episode_bias: torch.Tensor | None = None
        self._new_mode: torch.Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self.num_modes + 2 * self.num_modes * self.output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        mode_end = self.num_modes
        control_end = mode_end + self.num_modes * self.output_dim
        self._logits = mlp_output[..., :mode_end]
        self._control_means = mlp_output[..., mode_end:control_end].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        self._bias_centers = mlp_output[..., control_end:].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        self._action_std = self.action_std_param.clamp(*self.std_range).expand_as(
            self._control_means
        )
        self._bias_std = self.bias_std_param.clamp(*self.std_range).expand_as(
            self._bias_centers
        )

    def set_commitment_context(
        self,
        mode_ids: torch.Tensor,
        episode_bias: torch.Tensor,
        new_mode: torch.Tensor,
    ) -> None:
        self._mode_ids = mode_ids.long()
        self._episode_bias = episode_bias
        self._new_mode = new_mode.bool()

    def selected(self, values: torch.Tensor) -> torch.Tensor:
        index = self._mode_ids[..., None, None].expand(
            *self._mode_ids.shape, 1, values.shape[-1]
        )
        return values.gather(-2, index).squeeze(-2)

    def selected_for(self, values: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        index = modes[..., None, None].expand(*modes.shape, 1, values.shape[-1])
        return values.gather(-2, index).squeeze(-2)

    @property
    def bias_center(self) -> torch.Tensor:
        return self.selected(self._bias_centers)

    @property
    def bias_std(self) -> torch.Tensor:
        return self.selected(self._bias_std)

    @property
    def mean(self) -> torch.Tensor:
        return self.selected(self._control_means) + self._episode_bias

    @property
    def std(self) -> torch.Tensor:
        return self.selected(self._action_std)

    def sample(self) -> torch.Tensor:
        return torch.normal(self.mean, self.std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        mode_end = self.num_modes
        control_end = mode_end + self.num_modes * self.output_dim
        logits = mlp_output[..., :mode_end]
        control = mlp_output[..., mode_end:control_end].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        centers = mlp_output[..., control_end:].reshape_as(control)
        modes = logits.argmax(dim=-1)
        index = modes[..., None, None].expand(*modes.shape, 1, self.output_dim)
        return (control + centers).gather(-2, index).squeeze(-2)

    def as_deterministic_output_module(self) -> nn.Module:
        return _MostProbablePersistentBias(self.output_dim, self.num_modes)

    @property
    def entropy(self) -> torch.Tensor:
        action_entropy = (
            0.5 + 0.5 * math.log(2.0 * math.pi) + self.std.log()
        ).sum(dim=-1)
        bias_entropy = (
            0.5 + 0.5 * math.log(2.0 * math.pi) + self.bias_std.log()
        ).sum(dim=-1)
        probabilities = torch.softmax(self._logits, dim=-1)
        categorical = -(
            probabilities * torch.log_softmax(self._logits, dim=-1)
        ).sum(dim=-1)
        return action_entropy + self._new_mode.to(action_entropy.dtype) * (
            bias_entropy + categorical
        )

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        mode_one_hot = torch.nn.functional.one_hot(
            self._mode_ids, num_classes=self.num_modes
        ).to(self._control_means.dtype)
        return (
            self._logits,
            self._control_means,
            self._bias_centers,
            self._action_std,
            self._bias_std,
            mode_one_hot,
            self._episode_bias,
            self._new_mode.to(self._control_means.dtype),
        )

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        residual = (outputs - self.mean) / self.std
        action_log_prob = (
            -0.5 * residual.square()
            - self.std.log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        bias_residual = (self._episode_bias - self.bias_center) / self.bias_std
        bias_log_prob = (
            -0.5 * bias_residual.square()
            - self.bias_std.log()
            - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        categorical_log_prob = torch.log_softmax(self._logits, dim=-1).gather(
            -1, self._mode_ids.unsqueeze(-1)
        ).squeeze(-1)
        return action_log_prob + self._new_mode.to(action_log_prob.dtype) * (
            self.commitment_credit_scale * (bias_log_prob + categorical_log_prob)
        )

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        (
            old_logits,
            old_control,
            old_centers,
            old_action_std,
            old_bias_std,
            old_mode_one_hot,
            old_episode_bias,
            old_new_mode,
        ) = old_params
        (
            new_logits,
            new_control,
            new_centers,
            new_action_std,
            new_bias_std,
            _,
            _,
            _,
        ) = new_params
        old_mode = old_mode_one_hot.argmax(dim=-1)
        old_action_mean = self.selected_for(old_control, old_mode) + old_episode_bias
        new_action_mean = self.selected_for(new_control, old_mode) + old_episode_bias
        old_selected_action_std = self.selected_for(old_action_std, old_mode)
        new_selected_action_std = self.selected_for(new_action_std, old_mode)
        action_kl = _diagonal_normal_kl(
            old_action_mean,
            old_selected_action_std,
            new_action_mean,
            new_selected_action_std,
        )
        old_log_probability = torch.log_softmax(old_logits, dim=-1)
        new_log_probability = torch.log_softmax(new_logits, dim=-1)
        categorical_kl = (
            old_log_probability.exp() * (old_log_probability - new_log_probability)
        ).sum(dim=-1)
        bias_kl = _diagonal_normal_kl(
            self.selected_for(old_centers, old_mode),
            self.selected_for(old_bias_std, old_mode),
            self.selected_for(new_centers, old_mode),
            self.selected_for(new_bias_std, old_mode),
        )
        return action_kl + old_new_mode * self.commitment_credit_scale * (
            categorical_kl + bias_kl
        )

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        last_linear = next(module for module in reversed(mlp) if isinstance(module, nn.Linear))
        with torch.no_grad():
            last_linear.bias[: self.num_modes].zero_()


def _diagonal_normal_kl(
    old_mean: torch.Tensor,
    old_std: torch.Tensor,
    new_mean: torch.Tensor,
    new_std: torch.Tensor,
) -> torch.Tensor:
    return (
        (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
        + new_std.log()
        - old_std.log()
        - 0.5
    ).sum(dim=-1)


class _MostProbablePersistentBias(nn.Module):
    def __init__(self, output_dim: int, num_modes: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_modes = num_modes

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        mode_end = self.num_modes
        control_end = mode_end + self.num_modes * self.output_dim
        modes = raw[..., :mode_end].argmax(dim=-1)
        control = raw[..., mode_end:control_end].reshape(
            *raw.shape[:-1], self.num_modes, self.output_dim
        )
        centers = raw[..., control_end:].reshape_as(control)
        index = modes[..., None, None].expand(*modes.shape, 1, self.output_dim)
        return (control + centers).gather(-2, index).squeeze(-2)


class PersistentBiasActor(MLPModel):
    """Sensor policy with learned discrete and continuous episode commitment."""

    is_recurrent: bool = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.distribution, PersistentBiasGaussianDistribution):
            raise TypeError("PersistentBiasActor requires PersistentBiasGaussianDistribution")
        self.num_modes = self.distribution.num_modes
        self._commitment_state: torch.Tensor | None = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: torch.Tensor | None = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        raw = self.mlp(MLPModel.get_latent(self, obs))
        self.distribution.update(raw)
        bias_start = self.num_modes
        state_size = self.num_modes + self.distribution.output_dim + 1

        if masks is None:
            batch_size = raw.shape[0]
            if self._commitment_state is None or self._commitment_state.shape[1] != batch_size:
                self._commitment_state = raw.new_zeros(1, batch_size, state_size)
            one_hot = self._commitment_state[0, :, : self.num_modes]
            needs_mode = one_hot.sum(dim=-1) < 0.5
            modes = one_hot.argmax(dim=-1)
            logits = raw[..., : self.num_modes]
            if stochastic_output:
                proposed_modes = torch.multinomial(
                    torch.softmax(logits, dim=-1), 1
                ).squeeze(-1)
            else:
                proposed_modes = logits.argmax(dim=-1)
            modes = torch.where(needs_mode, proposed_modes, modes)
            self.distribution.set_commitment_context(
                modes,
                raw.new_zeros(batch_size, self.distribution.output_dim),
                needs_mode,
            )
            center = self.distribution.bias_center
            if stochastic_output:
                proposed_bias = torch.normal(center, self.distribution.bias_std)
            else:
                proposed_bias = center
            previous_bias = self._commitment_state[
                0, :, bias_start : bias_start + self.distribution.output_dim
            ]
            episode_bias = torch.where(needs_mode.unsqueeze(-1), proposed_bias, previous_bias)
            self._commitment_state[0, :, : self.num_modes] = torch.nn.functional.one_hot(
                modes, self.num_modes
            ).to(raw.dtype)
            self._commitment_state[
                0, :, bias_start : bias_start + self.distribution.output_dim
            ] = episode_bias
            self._commitment_state[0, :, -1] = needs_mode.to(raw.dtype)
            self.distribution.set_commitment_context(modes, episode_bias, needs_mode)
        else:
            if hidden_state is None:
                raise ValueError("persistent bias state is required for PPO replay")
            trajectory_modes = hidden_state[0, :, : self.num_modes].argmax(dim=-1)
            trajectory_bias = hidden_state[
                0, :, bias_start : bias_start + self.distribution.output_dim
            ]
            padded_modes = trajectory_modes.unsqueeze(0).expand(raw.shape[0], -1)
            padded_bias = trajectory_bias.unsqueeze(0).expand(raw.shape[0], -1, -1)
            padded_new = torch.zeros_like(padded_modes, dtype=torch.bool)
            padded_new[0] = hidden_state[0, :, -1] > 0.5
            raw = unpad_trajectories(raw, masks)
            modes = unpad_trajectories(padded_modes.unsqueeze(-1), masks).squeeze(-1)
            episode_bias = unpad_trajectories(padded_bias, masks)
            new_mode = unpad_trajectories(padded_new.unsqueeze(-1), masks).squeeze(-1)
            self.distribution.update(raw)
            self.distribution.set_commitment_context(modes, episode_bias, new_mode)

        if stochastic_output:
            return self.distribution.sample()
        return self.distribution.mean

    def reset(
        self, dones: torch.Tensor | None = None, hidden_state: torch.Tensor | None = None
    ) -> None:
        if dones is None:
            self._commitment_state = hidden_state
        elif self._commitment_state is not None:
            self._commitment_state[:, dones == 1, :] = 0.0

    def get_hidden_state(self) -> torch.Tensor | None:
        return self._commitment_state

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self._commitment_state is not None:
            self._commitment_state = self._commitment_state.detach()

    def as_jit(self) -> nn.Module:
        return _TorchPersistentBiasActor(self)


class _TorchPersistentBiasActor(nn.Module):
    """Single-robot deterministic export retaining mode and bias per episode."""

    def __init__(self, model: PersistentBiasActor) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        self.output_dim = model.distribution.output_dim
        self.num_modes = model.num_modes
        self.register_buffer("mode", torch.zeros(1, dtype=torch.long))
        self.register_buffer("bias", torch.zeros(1, self.output_dim))
        self.register_buffer("has_commitment", torch.zeros(1, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(self.obs_normalizer(x))
        mode_end = self.num_modes
        control_end = mode_end + self.num_modes * self.output_dim
        logits = raw[:, :mode_end]
        control = raw[:, mode_end:control_end].reshape(
            -1, self.num_modes, self.output_dim
        )
        centers = raw[:, control_end:].reshape(
            -1, self.num_modes, self.output_dim
        )
        if not bool(self.has_commitment[0]):
            self.mode[0] = logits[0].argmax()
            selected_center = centers[0, self.mode[0]]
            self.bias[0].copy_(selected_center)
            self.has_commitment[0] = True
        index = self.mode.view(1, 1, 1).expand(control.shape[0], 1, self.output_dim)
        selected_control = control.gather(1, index).squeeze(1)
        return selected_control + self.bias

    @torch.jit.export
    def reset(self) -> None:
        self.mode.zero_()
        self.bias.zero_()
        self.has_commitment.zero_()


class PersistentBiasPPO(PersistentModePPO):
    """Persistent-mode PPO with one-time optimizer reset for widened heads."""

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        infos = loaded_dict.get("infos")
        if isinstance(infos, dict) and infos.get("persistent_bias_transplant"):
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": False,
                "iteration": True,
                "rnd": False,
            }
        return super().load(loaded_dict, load_cfg, strict)
