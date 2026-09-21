"""Diagnostic-only constraint on learned swing corrections; no physical changes.

Keep stance corrections available. During lift/advance/lower, constrain the
selected ankle-pivot request around its nominal planar path, then solve the
same two-link IK. This is not a floor-contact or world-clearance measurement.
"""
import torch


def install_swing_projection(raw, support_compression=False):
    original_reference = raw._reference
    compression = torch.zeros(raw.num_envs, device=raw.device)

    def inverse(forward, down):
        length = .159896689
        cosine = ((forward.square() + down.square() - 2 * length**2) / (2 * length**2)).clamp(-.999, .999)
        knee = -raw.front_sign * torch.acos(cosine)
        hip = torch.atan2(forward, down) - torch.atan2(torch.sin(knee), 1 + torch.cos(knee))
        return hip, knee

    def reference_with_projected_residual():
        nominal = original_reference()
        selected = raw.pairs[raw.pair]
        swing = selected & ((raw.stage >= 1) & (raw.stage <= 3))[:, None]
        if support_compression:
            # Planar encoder estimate only. It assumes the support soles stay
            # planted and level; actual contacts remain reward/diagnostic data.
            joints = raw.robot.data.joint_pos.torch[:, raw.ids]
            _, measured_down = raw._planar_ankle_position(joints)
            support_down = (measured_down * ~selected).sum(1) / 2.
            target = (raw.cfg.stance_down_m - support_down).clamp(0., .025)
            compression[raw._age() < raw.step_dt * 1.5] = 0.
            compression.add_(raw.step_dt / (.15 + raw.step_dt) * (target - compression))
            amount = compression[:, None] * (raw.ref_clearance / raw.cfg.lift_m).clamp(0., 1.)
            down_reference = raw.ref_down - amount * swing
            hip, knee = inverse(raw.ref_forward, down_reference)
            nominal = nominal.clone()
            nominal[:, 4:8] = torch.where(swing, hip, nominal[:, 4:8])
            nominal[:, 8:12] = torch.where(swing, knee, nominal[:, 8:12])
            raw.reference_clipped |= (nominal < raw.limits[:, :, 0]) | (nominal > raw.limits[:, :, 1])
            nominal = torch.maximum(torch.minimum(nominal, raw.limits[:, :, 1]), raw.limits[:, :, 0])
            raw.ref = nominal
            raw.ref_down = down_reference
        residual = raw.filtered * raw.residual_scale
        requested = nominal + residual
        forward, down = raw._planar_ankle_position(requested)
        forward = torch.maximum(torch.minimum(forward, raw.ref_forward + .008), raw.ref_forward - .008)
        down = torch.maximum(torch.minimum(down, raw.ref_down + .002), raw.ref_down - .010)
        hip, knee = inverse(forward, down)
        projected = requested.clone()
        projected[:, 4:8] = torch.where(swing, hip, requested[:, 4:8])
        projected[:, 8:12] = torch.where(swing, knee, requested[:, 8:12])
        # _pre_physics_step adds the filtered residual immediately after this
        # call. Subtract it here to substitute only the requested pose. The
        # original nominal raw.ref, actuator limits, output limiter, gates,
        # observations, rewards and success criteria remain unchanged.
        return projected - residual

    raw._reference = reference_with_projected_residual
    return {
        "mode": "diagnostic swing projection",
        "forward_window_m": [-.008, .008],
        "down_window_m": [-.010, .002],
        "stages": ["lift", "advance", "lower"],
        "physical_limits_changed": False,
        "qualification_thresholds_changed": False,
        "requires_retraining_if_adopted": True,
        "support_compression_compensation": support_compression,
        "compression_cap_m": .025 if support_compression else 0.,
        "compression_filter_s": .15 if support_compression else 0.,
    }
