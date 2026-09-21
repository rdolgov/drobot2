"""Diagnostic model-based support preload through bounded position offsets.

Uses nominal joint geometry and scheduled support, never measured simulator
contacts/forces. The vertical-force static model is optimistic and cannot
certify dynamic feasibility. This is not a hardware controller or training default.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch

from diagonal_static_load import URDF, solve


def install_support_preload(raw):
    if raw.num_envs != 4 or not raw.cfg.evaluation:
        raise ValueError("Restricted to the four-environment diagnostic")
    original_reference = raw._reference
    desired = np.zeros((4, 16))
    filtered = torch.zeros_like(raw.targets)
    peak = np.zeros(4)
    valid = np.zeros(4, dtype=bool)
    last_step = -1000
    raw.support_preload_diagnostics = {
        "support_preload_rad": filtered.clone(),
        "support_preload_static_peak_nm": torch.zeros(4, device=raw.device),
        "support_preload_solver_valid": torch.zeros(4, dtype=torch.bool, device=raw.device),
    }

    def reference():
        nonlocal last_step
        nominal = original_reference().clone()
        active = raw.started & ~raw.parked
        if raw.common_step_counter - last_step >= 6:
            last_step = raw.common_step_counter
            poses = nominal.detach().cpu().numpy().astype(float)
            pairs = raw.pair.cpu().tolist()
            enabled = active.cpu().tolist()
            desired[:] = 0.
            valid[:] = False
            peak[:] = 0.
            for index, (pose, pair, enable) in enumerate(zip(poses, pairs, enabled)):
                if not enable:
                    continue
                # Solve the intended level-body, flat-sole pose. It is a
                # feedforward approximation, not an estimate of actual contact.
                pose[12:16] = -pose[4:8] - pose[8:12]
                solved = solve(dict(zip(raw.action_order, pose)), pair)
                if not solved.get("feasible_vertical_equilibrium"):
                    continue
                valid[index] = True
                peak[index] = solved["minimum_peak_torque_nm"]
                support = [1, 2] if pair == 0 else [0, 3]
                for leg in support:
                    for offset in (0, 4, 8, 12):
                        joint = leg + offset
                        torque = solved["joint_torque_nm"][raw.action_order[joint]]
                        # Existing simulated position stiffness is 30 Nm/rad.
                        # This only biases position; the unchanged conservative
                        # actuator torque-speed/strength caps still act in physics.
                        desired[index, joint] = np.clip(torque, -.8825985, .8825985) / 30.
        filtered[raw._age() < raw.step_dt * 1.5] = 0.
        target = torch.as_tensor(desired, device=raw.device, dtype=nominal.dtype) * active[:, None]
        filtered.add_(raw.step_dt / (.15 + raw.step_dt) * (target - filtered))
        ref = nominal + filtered
        raw.reference_clipped |= (ref < raw.limits[:, :, 0]) | (ref > raw.limits[:, :, 1])
        raw.ref = torch.maximum(torch.minimum(ref, raw.limits[:, :, 1]), raw.limits[:, :, 0])
        # Preserve the nominal physical foot-path gate: the offset accounts
        # for load error, it is not a new desired loaded foot position.
        raw.support_preload_diagnostics.update(
            support_preload_rad=filtered.clone(),
            support_preload_static_peak_nm=torch.as_tensor(peak, device=raw.device, dtype=nominal.dtype),
            support_preload_solver_valid=torch.as_tensor(valid, device=raw.device, dtype=torch.bool))
        return raw.ref

    raw._reference = reference
    return {"mode": "static support-load position preload prototype",
            "inputs": "nominal reference geometry, URDF mass/inertia and scheduled support pair",
            "model_assumptions": "level body, flat supporting soles, vertical-only contact forces",
            "position_stiffness_nm_rad": 30., "nominal_torque_used_for_offset_cap_nm": .8825985,
            "filter_s": .15, "solve_every_control_steps": 6,
            "physical_limits_changed": False, "qualification_thresholds_changed": False,
            "requires_retraining_if_adopted": True, "hardware_stiffness_calibrated": False,
            "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
            "static_solver_sha256": hashlib.sha256(Path(__file__).with_name("diagonal_static_load.py").read_bytes()).hexdigest()}
