"""Diagnostic-only staggered lift within the existing selected diagonal."""
import torch


def install_rear_first(raw):
    if raw.num_envs != 4 or not raw.cfg.evaluation:
        raise ValueError("Restricted to the four-environment diagnostic")
    original_reference = raw._reference
    rear = torch.tensor([False, True, False, True], device=raw.device)

    def reference():
        ref = original_reference().clone()
        u, _ = raw._stage_progress()
        active = (raw.stage == 1) & raw.started & ~raw.parked
        selected = raw.pairs[raw.pair] & active[:, None]
        # Rear foot takes the first half of lift, then remains raised while
        # the front partner lifts. Both must pass the original end-of-lift
        # gate before advance. The other diagonal remains the support pair.
        phase = torch.where(rear[None, :], 2. * u[:, None], 2. * u[:, None] - 1.).clamp(0., 1.)
        profile = phase**3 * (10. + phase * (-15. + 6. * phase))
        clearance = torch.where(selected, raw.cfg.lift_m * profile, raw.ref_clearance)
        down = raw.ref_down + raw.ref_clearance - clearance
        length = .159896689
        cosine = ((raw.ref_forward.square() + down.square() - 2 * length**2) / (2 * length**2)).clamp(-.999, .999)
        knee = -raw.front_sign * torch.acos(cosine)
        hip = torch.atan2(raw.ref_forward, down) - knee / 2.
        ref[:, 4:8] = torch.where(selected, hip, ref[:, 4:8])
        ref[:, 8:12] = torch.where(selected, knee, ref[:, 8:12])
        raw.ref_down, raw.ref_clearance = down, clearance
        # During the delay, the selected front foot is still intended to stand.
        # This changes the scheduled stance bit, never uses measured contact.
        raw.expected_stance |= selected & ~rear[None, :] & (u[:, None] <= .5)
        raw.reference_clipped |= (ref < raw.limits[:, :, 0]) | (ref > raw.limits[:, :, 1])
        raw.ref = torch.maximum(torch.minimum(ref, raw.limits[:, :, 1]), raw.limits[:, :, 0])
        return raw.ref

    raw._reference = reference
    return {"mode": "rear-first lift within the selected diagonal",
            "lift_stage": "rear in first half, front in second half, same total stage duration",
            "other_stages": "unchanged", "physical_limits_changed": False,
            "qualification_thresholds_changed": False,
            "residual_projection": "unchanged; sampled motion must establish actual order",
            "requires_retraining_if_adopted": True}
