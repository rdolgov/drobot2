"""Keep learned corrections within the useful swing-control range."""
import json
import torch
from ankle_v26_balanced.env import IKDiagonalEnv as BalancedEnv


class IKDiagonalEnv(BalancedEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        contract = {'schema': 'drobot.ankle-v26-feasible.v1',
            'initialization': getattr(cfg, 'continuation_description', 'See run checkpoint provenance'),
            'projection_cost': '1.5 times squared normalized hip/knee correction discarded during swing, integrated over time',
            'mirror_loss_coefficient': .25,
            'physics_controller_observations_limits': 'unchanged from ankle_v25_pair',
            'qualification': 'unchanged; reward shaping does not count as success',
            'hardware_qualified': False}
        if self.metrics_file:
            self.metrics_file.with_suffix('.feasible-contract.json').write_text(json.dumps(contract, indent=2)+'\n')
        print('FEASIBLE_RL_CONTRACT='+json.dumps(contract), flush=True)

    def _get_rewards(self):
        reward = super()._get_rewards()
        selected = self.pairs[self.pair]
        active = self.started & ~self.parked & ((self.stage >= 1) & (self.stage <= 3))
        mask = torch.cat((selected, selected), dim=1) & active[:, None]
        # Sagittal hip/knee channels only: ankle compensation intentionally
        # differs from the raw joint residual and must not be penalized here.
        discarded = ((self.unconstrained_target[:, 4:12]-self.requested_target[:, 4:12])
                     /self.residual_scale[4:12]).square().clamp(max=4.)
        projection_cost = (discarded*mask).sum(1)/4.
        self.last_diagnostics['discarded_swing_correction_cost'] = projection_cost.clone()
        return reward-1.5*projection_cost*self.step_dt
