"""Second candidate: real paired clearance before forward-motion reward.

The first candidate's independent evaluation improved translation by dragging
rear feet. Keep that evidence and use a separate, fresh-actor experiment.
Physics, torque, mass, limits, and strict qualification are unchanged.
"""
import json
import torch
from ankle_v25_ik.env import IKDiagonalEnv as ReferenceEnv


class IKDiagonalEnv(ReferenceEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        contract = {
            "schema": "drobot.ankle-v25-pair.v2",
            "changes_from_ik_r1": {
                "forward_reward": "gated by actual pair clearance during lift or strict completed pair during final push",
                "swing": "stronger minimum-pair clearance, synchronization and level-body rewards",
                "residual_scales_rad_joint_major": cfg.residual_scales,
                "extra_lift_m": cfg.swing_extra_lift_m,
                "forward_window_m": cfg.swing_forward_window_m,
                "curriculum_steps": cfg.curriculum_steps},
            "physics_unchanged": True, "qualification_unchanged": True,
            "fresh_actor": True, "hardware_qualified": False}
        if self.metrics_file:
            self.metrics_file.with_suffix(".pair-contract.json").write_text(json.dumps(contract, indent=2)+"\n")
        print("PAIR_RL_CONTRACT="+json.dumps(contract), flush=True)

    def _get_rewards(self):
        reward = super()._get_rewards()
        d = self.last_diagnostics
        selected = self.pairs[self.pair]
        moving = self.started & ~self.parked & (self.commands[:, 0] > .005)
        lifting = moving & (self.stage == 1)
        swinging = moving & ((self.stage == 1) | (self.stage == 2))
        support_contact = (d["contacts"] | selected).all(1)
        pair_clearance = (d["pair_clearance_m"] / self.cfg.lift_m).clamp(0., 1.)
        physical_clearance = (d["pair_clearance_m"] / .008).clamp(0., 1.)
        speed = d["body_velocity"][:, 0]/self.commands[:, 0].clamp(min=.012)
        old_push = 1.5*speed.clamp(-1., 2.)*moving*(self.stage == 5)
        old_lift = 1.5*speed.clamp(-1., 1.5)*lifting*support_contact
        # Preserve costs for moving backward; positive speed alone cannot pay
        # for a dragged rear leg. Completed-pair counts stay strict and sparse.
        push_credit = torch.where(speed > 0., self.pair_paid.float(), 1.)
        lift_credit = torch.where(speed > 0., physical_clearance, 1.)
        tilt = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(1)
        sync = (d["pair_height_difference_m"]/.015).square().clamp(max=9.)
        extra = (old_push*(push_credit-1.) + old_lift*(lift_credit-1.)
                 +8.*pair_clearance*swinging*support_contact
                 -2.*sync*swinging
                 -2.*(tilt/.05**2).clamp(max=9.)*swinging)
        d["paired_forward_credit"] = physical_clearance.clone()
        return reward+extra*self.step_dt
