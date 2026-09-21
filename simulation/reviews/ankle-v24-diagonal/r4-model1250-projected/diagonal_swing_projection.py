"""Diagnostic-only constraint on learned swing corrections; no physical changes.

Keep stance corrections available. During lift/advance/lower, constrain the
selected ankle-pivot request around its nominal planar path, then solve the
same two-link IK. This is not a floor-contact or world-clearance measurement.
"""
import math
import torch


def install_swing_projection(raw):
    original_reference = raw._reference

    def reference_with_projected_residual():
        nominal = original_reference()
        residual = raw.filtered * raw.residual_scale
        requested = nominal + residual
        selected = raw.pairs[raw.pair]
        swing = selected & ((raw.stage >= 1) & (raw.stage <= 3))[:, None]
        forward, down = raw._planar_ankle_position(requested)
        forward = torch.maximum(torch.minimum(forward, raw.ref_forward + .008), raw.ref_forward - .008)
        down = torch.maximum(torch.minimum(down, raw.ref_down + .002), raw.ref_down - .010)
        length = .159896689
        cosine = ((forward.square() + down.square() - 2 * length**2) / (2 * length**2)).clamp(-.999, .999)
        knee = -raw.front_sign * torch.acos(cosine)
        hip = torch.atan2(forward, down) - torch.atan2(torch.sin(knee), 1 + torch.cos(knee))
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
    }
