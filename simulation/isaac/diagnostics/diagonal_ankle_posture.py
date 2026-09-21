"""Diagnostic support-ankle posture feedback; no privileged contact inputs."""
import torch


def install_ankle_posture(raw):
    original_reference = raw._reference
    filtered_pitch = torch.zeros(raw.num_envs, device=raw.device)

    def reference():
        ref = original_reference().clone()
        pitch = torch.asin(raw.robot.data.projected_gravity_b.torch[:, 0].clamp(-1., 1.))
        filtered_pitch[raw._age() < raw.step_dt * 1.5] = 0.
        filtered_pitch.add_(raw.step_dt / (.06 + raw.step_dt) * (pitch - filtered_pitch))
        # A planted flat sole has q_ankle ~= pitch-q_hip-q_knee. Matching
        # that exactly supplies no static position-error torque. Add pitch
        # restoring feedback only on scheduler-designated supporting feet.
        ref[:, 12:16] -= filtered_pitch[:, None] * raw.expected_stance
        raw.reference_clipped |= (ref < raw.limits[:, :, 0]) | (ref > raw.limits[:, :, 1])
        raw.ref = torch.maximum(torch.minimum(ref, raw.limits[:, :, 1]), raw.limits[:, :, 0])
        return raw.ref

    raw._reference = reference
    return {"mode": "support ankle posture feedback prototype", "pitch_gain": 1.,
            "pitch_filter_s": .06, "stance_detection": "reference scheduler, not contacts",
            "physical_limits_changed": False, "qualification_thresholds_changed": False}
