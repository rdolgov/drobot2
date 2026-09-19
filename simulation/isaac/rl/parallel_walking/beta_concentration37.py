"""Study37 only: consistent concentration scaling, unchanged deterministic mean.

No production registration or trained parameters. Explicit study launcher only.
Native sampling/log-probability/entropy/KL methods consume scaled parameters.
"""
import torch
from rsl_rl.modules.distribution import BetaDistribution


class ConcentratedBeta37(BetaDistribution):
    def __init__(self, output_dim, action_range=(-1., 1.), *, concentration_factor):
        if (type(concentration_factor) not in (int, float)
                or concentration_factor not in (1., 2.) or output_dim != 12
                or tuple(action_range) != (-1., 1.)):
            raise ValueError('Study37 requires12 bounded actions and explicit factor1 or2')
        super().__init__(output_dim, action_range)
        self.concentration_factor = float(concentration_factor)
        self._native_alpha37 = self._native_beta37 = None

    def update(self, mlp_output):
        super().update(mlp_output)
        self._native_alpha37, self._native_beta37 = self._alpha, self._beta
        if self.concentration_factor == 1.:
            return
        self._alpha = self._alpha * self.concentration_factor
        self._beta = self._beta * self.concentration_factor
        self._distribution = torch.distributions.Beta(self._alpha, self._beta)
