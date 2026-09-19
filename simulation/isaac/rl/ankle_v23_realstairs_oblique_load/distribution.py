"""Concentration changes sampling variance, preserving the deterministic mean."""
import torch
from rsl_rl.modules.distribution import BetaDistribution

class StairBeta(BetaDistribution):
    def __init__(self, output_dim, action_range=(-1., 1.), concentration_factor=4.):
        if output_dim != 16 or concentration_factor != 4. or tuple(action_range) != (-1., 1.):
            raise ValueError('This isolated arm requires 16 actions and concentration factor 4')
        super().__init__(output_dim, action_range)
        self.concentration_factor = float(concentration_factor)

    def update(self, mlp_output):
        super().update(mlp_output)
        self._alpha = self._alpha * self.concentration_factor
        self._beta = self._beta * self.concentration_factor
        self._distribution = torch.distributions.Beta(self._alpha, self._beta)
