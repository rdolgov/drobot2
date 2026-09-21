"""Isolated lower-load reference candidate; never changes the active trainer."""
import math
import torch


def configure_load_stance(cfg):
    cfg.stance_forward_m = .030
    cfg.stance_down_m = .305
    cfg.body_forward_shift_m = 0.
    cfg.target_base_height_m = cfg.stance_down_m + .088908389
    cfg.robot.init_state.pos = (0., 0., cfg.target_base_height_m + .014)
    positions = {".*_hip_abduction": 0.}
    for prefix, forward, sign in (("front", .030, 1.), ("rear", -.020, -1.)):
        knee = -sign * math.acos((forward**2 + cfg.stance_down_m**2 - 2 * .159896689**2) / (2 * .159896689**2))
        hip = math.atan2(forward, cfg.stance_down_m) - knee / 2.
        positions.update({prefix + "_.*_hip_flexion": hip,
                          prefix + "_.*_knee": knee,
                          prefix + "_.*_ankle_pitch": -hip-knee})
    cfg.robot.init_state.joint_pos = positions


def install_load_stance(raw):
    original_reference = raw._reference
    side_sign = torch.tensor([1., 1., -1., -1.], device=raw.device)

    def reference():
        original_reference()
        forward = raw.ref_forward.clone()
        forward[:, (1, 3)] += .010
        down = raw.cfg.stance_down_m - raw.ref_clearance
        down = torch.minimum(down, (raw.cfg.maximum_leg_reach_m**2 - forward.square()).clamp(min=.01).sqrt())
        cosine = ((forward.square() + down.square() - 2 * .159896689**2) / (2 * .159896689**2)).clamp(-.999, .999)
        knee = -raw.front_sign * torch.acos(cosine)
        hip = torch.atan2(forward, down) - knee / 2.
        u, _ = raw._stage_progress()
        blend = u**3 * (10. + u * (-15. + 6. * u))
        startup = (raw._age() / raw.cfg.startup_s).clamp(0., 1.)
        startup = startup**3 * (10. + startup * (-15. + 6. * startup))
        direction = torch.where(raw.pair == 0, 1., -1.)
        transfer = torch.where((raw.stage == 0) & ~raw.first_placement, 2. * blend - 1., 1.)
        shift = .015 * direction * transfer * startup
        ref = raw.ref.clone()
        ref[:, :4] = -side_sign * torch.atan2(shift[:, None], torch.full_like(shift[:, None], raw.cfg.target_base_height_m + .05))
        ref[:, 4:8], ref[:, 8:12] = hip, knee
        raw.reference_clipped = (ref < raw.limits[:, :, 0]) | (ref > raw.limits[:, :, 1])
        raw.ref = torch.maximum(torch.minimum(ref, raw.limits[:, :, 1]), raw.limits[:, :, 0])
        raw.ref_forward, raw.ref_down = forward, down
        return raw.ref

    raw._reference = reference
    return {"mode": "lower-load reference prototype", "front_stance_m": .030,
            "rear_stance_m": -.020, "down_m": .305, "lateral_transfer_m": .015,
            "physical_limits_changed": False, "qualification_thresholds_changed": False,
            "initial_pose_changed": True, "trained_for_this_reference": False}
