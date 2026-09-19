"""Research-inspired rewards; shared robot, controller, sensors and milestones.

The potential is a bounded, obstacle-aware distance to the first tread. It
replaces the old forward-delta and sustained-lift rewards. Terrain geometry is
used here only: actor observations and joint references remain unchanged.
"""
import json
import torch
from ankle_v23_dense_turn.env import DenseTurnEnv


class ResearchEnv(DenseTurnEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.episode_serial = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.event_file = self.metrics_file.with_suffix(".placements.jsonl")
        self.reward_sums = torch.zeros(4, device=self.device)
        self.reward_samples = 0

    def _potential(self, local):
        cfg = self.cfg
        sole = local.mean(2)
        goal_x = cfg.stair_start_m + .070
        clear_z = cfg.stair_rise_m + .035
        low_z = local[..., 2].amin(2)
        # Distance along a lift, advance, lower path, rather than a straight
        # line that would point a grounded foot into the riser.
        before = local[..., 0].amin(2) < cfg.stair_start_m + .010
        horizontal = (goal_x - sole[..., 0]).clamp(min=0)
        over_riser = (clear_z - low_z).clamp(min=0) + (torch.maximum(
            low_z, torch.full_like(low_z, clear_z)) - cfg.stair_rise_m).abs()
        remaining = horizontal + torch.where(before, over_riser, (low_z - cfg.stair_rise_m).abs())
        proximity = (1 - remaining / .4).clamp(0, 1)
        weights = torch.tensor([6., 2., 6., 2.], device=self.device)  # FL RL FR RR
        advance = (self.robot.data.root_pos_w.torch[:, 0] - self.start_position[:, 0]).clamp(0, 1.2)
        return torch.nan_to_num(15 * advance + (proximity * weights).sum(1), nan=0., posinf=0., neginf=0.)

    def _pre_physics_step(self, actions):
        # Capture the actual pre-transition state, including immediately after
        # reset, so the first transition does not get a fabricated potential.
        if self.weights["potential"]:
            self.phi_before = self._potential(self._geometry()[0])
        super()._pre_physics_step(actions)

    def _terrain_costs(self, local, velocity):
        cfg = self.cfg
        edges = cfg.stair_start_m + torch.arange(cfg.stair_count, device=self.device) * cfg.stair_tread_m
        tops = (torch.arange(cfg.stair_count, device=self.device) + 1) * cfg.stair_rise_m
        x_min, x_max = local[..., 0].amin(2), local[..., 0].amax(2)
        z_min = local[..., 2].amin(2)
        distance = edges - x_max[..., None]
        proximity = (1 - distance / .060).clamp(0, 1) * (distance >= 0)
        too_low = ((tops + .010 - z_min[..., None]) / .020).clamp(0, 1)
        approaching = (velocity[..., 0] / .030).clamp(0, 3)
        riser = (proximity * too_low).amax(2) * approaching
        # Penalize a footprint straddling an edge only near the upper surface.
        # A safely elevated swing across the edge receives zero edge penalty.
        straddles = (x_min[..., None] < edges) & (x_max[..., None] > edges)
        overhang = ((edges - x_min[..., None]) / (x_max - x_min)[..., None].clamp(min=.001)).clamp(0, 1)
        near_surface = (1 - (z_min[..., None] - tops).abs() / .020).clamp(0, 1)
        edge = (straddles * overhang * near_surface).amax(2)
        return riser.sum(1), edge.sum(1)

    def _get_rewards(self):
        local, gap, _, contact, _, _ = self._geometry()
        sole = local.mean(2)
        velocity = (sole - self.previous_sole) / self.step_dt * self.history_valid[:, None, None]
        old_placed = self.placed.clone()
        upright = (self.robot.data.projected_gravity_b.torch[:, 2] < -.90) & (~self.fallen)
        other_support = contact.sum(1)[:, None] - contact.long()
        low = gap.amin(2)
        supported_lift = ((~contact) & (other_support >= 3)
            & (self.air_time + (~contact) * self.step_dt > .08)
            & (torch.maximum(self.swing_peak, low * (~contact)) > .012) & upright[:, None])
        old_lift = .5 * (torch.exp(-((low - .045) / .025).square()) * supported_lift).sum(1) * self.step_dt
        delta = (self.robot.data.root_pos_w.torch[:, 0] - self.previous_position[:, 0]) * self.history_valid
        old_progress = 30 * delta.clamp(-.005, .005) * upright * (contact.sum(1) >= 2)
        reward = super()._get_rewards()
        shaping = torch.zeros_like(reward)
        riser, edge = self._terrain_costs(local, velocity)
        if self.weights["potential"]:
            # Use PPO gamma; true absorbing terminals have zero potential.
            # Time-limit truncations retain next-state potential/bootstrap.
            terminal = self.fallen | self.success
            next_phi = torch.where(terminal, 0., self._potential(local))
            shaping = self.cfg.potential_gamma * next_phi - self.phi_before
            reward += shaping - old_progress - old_lift
        if self.weights["contact_penalties"]:
            reward -= (self.cfg.riser_cost_weight * riser + self.cfg.edge_cost_weight * edge) * self.step_dt
        self.reward_sums += torch.stack((shaping, old_progress + old_lift,
            riser * self.step_dt, edge * self.step_dt), 1).sum(0)
        self.reward_samples += self.num_envs
        new = (self.placed & ~old_placed).nonzero().cpu().tolist()
        if new:
            yaw = self._yaw()
            records = []
            for env_id, leg in new:
                records.append(dict(event="held_placement", step=self.common_step_counter,
                    env=env_id, episode=int(self.episode_serial[env_id]),
                    age_s=float(self.episode_length_buf[env_id] * self.step_dt),
                    leg=["front_left", "rear_left", "front_right", "rear_right"][leg],
                    level=int(self.reached[env_id, leg]), yaw_deg=float(torch.rad2deg(yaw[env_id])),
                    sole_m=sole[env_id, leg].cpu().tolist(),
                    corner_gap_m=gap[env_id, leg].cpu().tolist(),
                    hold_s=float(self.placement_hold[env_id, leg])))
            self._write_events(records)
        return reward

    def _write_events(self, records):
        if records:
            with self.event_file.open("a", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record) + "\n")

    def _reset_idx(self, env_ids):
        if hasattr(self, "episode_serial"):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            completed = env_ids[self.episode_length_buf[env_ids] > 0]
            placed_ids = completed[self.placed[completed].any(1)].cpu().tolist()
            self._write_events([dict(event="placement_episode_end", step=self.common_step_counter,
                env=i, episode=int(self.episode_serial[i]), fall=bool(self.fallen[i]),
                full_climb=bool(self.success[i]), age_s=float(self.episode_length_buf[i] * self.step_dt),
                legs=self.placed[i].cpu().tolist()) for i in placed_ids])
            self.episode_serial[completed] += 1
        super()._reset_idx(env_ids)

    def _publish(self):
        super()._publish()
        if hasattr(self, "reward_sums"):
            record = dict(samples=self.reward_samples, mean_per_transition=dict(zip(
                ("potential_added", "old_progress_and_lift", "unweighted_riser_dt", "unweighted_edge_dt"),
                (self.reward_sums / max(self.reward_samples, 1)).cpu().tolist())))
            self.metrics_file.with_suffix(".reward_components.json").write_text(json.dumps(record, indent=2) + "\n")
