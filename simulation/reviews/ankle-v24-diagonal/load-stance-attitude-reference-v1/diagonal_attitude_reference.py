"""Diagnostic IMU compensation for nominal foot vectors; no physical changes."""
import torch


def install_attitude_reference(raw):
    original_reference = raw._reference
    side = torch.tensor([1., 1., -1., -1.], device=raw.device)

    def reference():
        ref = original_reference().clone()
        gravity = raw.robot.data.projected_gravity_b.torch
        pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        # A body-pitch change otherwise rotates the fixed hip/knee foot-vector
        # reference in world space while the ankle tries to keep the sole flat.
        # Express the nominal world-oriented planar vector in the pitched body.
        forward = raw.ref_forward * pitch.cos()[:, None] + raw.ref_down * pitch.sin()[:, None]
        down = raw.ref_down * pitch.cos()[:, None] - raw.ref_forward * pitch.sin()[:, None]
        ref[:, 4:8] += pitch[:, None]
        ref[:, :4] -= side * roll[:, None]
        raw.reference_clipped |= (ref < raw.limits[:, :, 0]) | (ref > raw.limits[:, :, 1])
        raw.ref = torch.maximum(torch.minimum(ref, raw.limits[:, :, 1]), raw.limits[:, :, 0])
        raw.ref_forward, raw.ref_down = forward, down
        return raw.ref

    raw._reference = reference
    return {"mode": "IMU attitude compensation prototype", "hip_pitch_gain": 1.,
            "abduction_roll_gain": 1., "physical_limits_changed": False,
            "qualification_thresholds_changed": False, "trained_for_this_reference": False}
