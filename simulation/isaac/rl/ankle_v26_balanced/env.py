"""Continue the pair-focused actor with airborne stride credit and diagnostics."""
import json
import torch
from ankle_v25_pair.env import IKDiagonalEnv as PairEnv


class IKDiagonalEnv(PairEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.airborne_stride_best = torch.zeros((self.num_envs, 4), device=self.device)
        self.pair_stride_best = torch.zeros(self.num_envs, device=self.device)
        contract = {
            'schema': 'drobot.ankle-v26-balanced.v1',
            'initialization': 'continue pair-focused checkpoint; preserve actor and critic, fresh optimizer',
            'curriculum': 'full stride from the first reset',
            'symmetry': 'soft lateral-reflection actor consistency only; no synthetic mirrored rewards',
            'new_shaping': 'bounded once-per-placement credit for actual airborne forward foot travel',
            'physics_controller_observations_limits': 'unchanged from ankle_v25_pair',
            'qualification': 'unchanged strict paired landings; shaping is not success',
            'hardware_qualified': False}
        if self.metrics_file:
            self.metrics_file.with_suffix('.balanced-contract.json').write_text(json.dumps(contract, indent=2)+'\n')
        print('BALANCED_RL_CONTRACT='+json.dumps(contract), flush=True)

    def _curriculum(self):
        return 1.

    def _advance_scheduler(self):
        old_pair = self.pair.clone()
        super()._advance_scheduler()
        if hasattr(self, 'airborne_stride_best'):
            changed = old_pair != self.pair
            self.airborne_stride_best[changed] = 0.
            self.pair_stride_best[changed] = 0.

    def _get_rewards(self):
        # Capture flight evidence before the parent clears it on touchdown.
        corners, sole, low, _ = self._geometry()
        _, margin, force, weight = self._support(corners)
        contact = self._contact()
        selected = self.pairs[self.pair]
        swing = selected & ((self.stage >= 1) & (self.stage <= 3))[:, None]
        valid = self.history_valid
        liftoff = ~contact & self.previous_contact & valid[:, None]
        touchdown = contact & ~self.previous_contact & valid[:, None]
        liftoff_position = torch.where(liftoff[:, :, None], self.previous_sole, self.liftoff_sole)
        distance = sole[..., 0]-liftoff_position[..., 0]
        sideways = (sole[..., 1]-liftoff_position[..., 1]).abs()
        air_time = self.air_time+(~contact)*self.step_dt
        peak = torch.maximum(self.flight_peak, low*~contact)
        flight_valid = torch.where(liftoff, swing, self.flight_valid)
        tilt = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(1)
        support_contact = (contact | selected).all(1)
        load_good = (force*~selected).sum(1) > .60*weight
        supported = support_contact & load_good & (tilt < .16**2) & (margin > 0.) & ~self.fallen
        flight_supported = torch.where(liftoff, supported[:, None], self.flight_supported)
        flight_supported &= contact | supported[:, None]
        evidence = {
            'sole_position_w': sole.clone(), 'foot_touchdown': touchdown.clone(),
            'flight_forward_m': distance.clone(), 'flight_sideways_m': sideways.clone(),
            'flight_air_time_s': air_time.clone(), 'flight_peak_m': peak.clone(),
            'flight_was_selected': flight_valid.clone(), 'flight_support_good': flight_supported.clone(),
            'flight_stride_required_m': (.65*self.stride*torch.where(self.first_placement, .5, 1.)).clone(),
            'support_contact_good': support_contact.clone(), 'support_load_good': load_good.clone(),
            'body_gravity': self.robot.data.projected_gravity_b.torch.clone()}
        reward = super()._get_rewards()
        # Measured air motion earns bounded intermediate credit. This cannot
        # accrue from floor dragging or repeated back-and-forth oscillation.
        airborne = (swing & ~contact & flight_valid & (low > .008)
                    & (support_contact & load_good & (tilt < .16**2))[:, None])
        progress = (distance/self.stride[:, None].clamp(min=.01)).clamp(0., 1.)*airborne
        best = torch.maximum(self.airborne_stride_best, progress)
        pair_progress = torch.where(selected, best, torch.inf).amin(1)
        pair_best = torch.maximum(self.pair_stride_best, pair_progress)
        shaping = .35*(best-self.airborne_stride_best).sum(1)+2.*(pair_best-self.pair_stride_best)
        self.airborne_stride_best.copy_(best)
        self.pair_stride_best.copy_(pair_best)
        self.last_diagnostics.update(evidence)
        self.last_diagnostics['airborne_stride_progress'] = best.clone()
        return reward+shaping

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if hasattr(self, 'airborne_stride_best'):
            self.airborne_stride_best[env_ids] = 0.
            self.pair_stride_best[env_ids] = 0.
