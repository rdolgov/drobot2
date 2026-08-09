"""Two-mode Gaussian action distribution for pure-PPO mode commitment.

The standard diagonal Gaussian used by PPO has one action mean.  When two
different one-foot lifts are both useful, that mean can land between them and
produce a stable four-foot stance.  This distribution keeps two complete
action means and lets PPO learn their probabilities.  Training samples from
the exact mixture; deterministic deployment selects the most probable mode
instead of averaging incompatible actions.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import Distribution


class _MostProbableMode(nn.Module):
    """Export-friendly deterministic selector for mixture MLP output."""

    def __init__(self, output_dim: int, num_modes: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_modes = num_modes

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output[..., : self.num_modes]
        means = mlp_output[..., self.num_modes :].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        mode = logits.argmax(dim=-1, keepdim=True)
        gather_index = mode.unsqueeze(-1).expand(*mode.shape, self.output_dim)
        return means.gather(-2, gather_index).squeeze(-2)


class TwoModeGaussianDistribution(Distribution):
    """A small categorical mixture of diagonal Gaussian action policies."""

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.20,
        num_modes: int = 2,
        std_range: tuple[float, float] = (1.0e-4, 2.0),
    ) -> None:
        super().__init__(output_dim)
        if num_modes < 2:
            raise ValueError("TwoModeGaussianDistribution requires at least two modes")
        self.num_modes = num_modes
        self.std_range = std_range
        self.std_param = nn.Parameter(torch.full((num_modes, output_dim), init_std))
        self._logits: torch.Tensor | None = None
        self._means: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    @property
    def input_dim(self) -> int:
        return self.num_modes + self.num_modes * self.output_dim

    def _split(self, mlp_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = mlp_output[..., : self.num_modes]
        means = mlp_output[..., self.num_modes :].reshape(
            *mlp_output.shape[:-1], self.num_modes, self.output_dim
        )
        return logits, means

    def update(self, mlp_output: torch.Tensor) -> None:
        self._logits, self._means = self._split(mlp_output)
        std = self.std_param.clamp(*self.std_range)
        self._std = std.expand_as(self._means)

    def sample(self) -> torch.Tensor:
        probabilities = torch.softmax(self._logits, dim=-1)
        flat_probabilities = probabilities.reshape(-1, self.num_modes)
        flat_modes = torch.multinomial(flat_probabilities, 1).squeeze(-1)
        modes = flat_modes.reshape(probabilities.shape[:-1])
        gather_index = modes[..., None, None].expand(*modes.shape, 1, self.output_dim)
        selected_mean = self._means.gather(-2, gather_index).squeeze(-2)
        selected_std = self._std.gather(-2, gather_index).squeeze(-2)
        return torch.normal(selected_mean, selected_std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return _MostProbableMode(self.output_dim, self.num_modes)(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        return _MostProbableMode(self.output_dim, self.num_modes)

    @property
    def mean(self) -> torch.Tensor:
        probabilities = torch.softmax(self._logits, dim=-1)
        return (probabilities[..., None] * self._means).sum(dim=-2)

    @property
    def std(self) -> torch.Tensor:
        probabilities = torch.softmax(self._logits, dim=-1)[..., None]
        second_moment = (probabilities * (self._std.square() + self._means.square())).sum(dim=-2)
        return (second_moment - self.mean.square()).clamp_min(1.0e-12).sqrt()

    @property
    def entropy(self) -> torch.Tensor:
        probabilities = torch.softmax(self._logits, dim=-1)
        log_probabilities = torch.log_softmax(self._logits, dim=-1)
        categorical = -(probabilities * log_probabilities).sum(dim=-1)
        component = (0.5 + 0.5 * math.log(2.0 * math.pi) + self._std.log()).sum(dim=-1)
        # This tractable upper bound is used only as an exploration regularizer.
        return categorical + (probabilities * component).sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return self._logits, self._means, self._std

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        centered = (outputs.unsqueeze(-2) - self._means) / self._std
        component_log_prob = (
            -0.5 * centered.square() - self._std.log() - 0.5 * math.log(2.0 * math.pi)
        ).sum(dim=-1)
        return torch.logsumexp(torch.log_softmax(self._logits, dim=-1) + component_log_prob, dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Return an aligned-component KL upper bound for PPO scheduling."""
        old_logits, old_means, old_std = old_params
        new_logits, new_means, new_std = new_params
        old_log_prob = torch.log_softmax(old_logits, dim=-1)
        new_log_prob = torch.log_softmax(new_logits, dim=-1)
        old_probability = old_log_prob.exp()
        categorical_kl = (old_probability * (old_log_prob - new_log_prob)).sum(dim=-1)
        normal_kl = (
            (old_std.square() + (old_means - new_means).square()) / (2.0 * new_std.square())
            + new_std.log()
            - old_std.log()
            - 0.5
        ).sum(dim=-1)
        return categorical_kl + (old_probability * normal_kl).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        last_linear = next(module for module in reversed(mlp) if isinstance(module, nn.Linear))
        with torch.no_grad():
            last_linear.bias[: self.num_modes].zero_()

