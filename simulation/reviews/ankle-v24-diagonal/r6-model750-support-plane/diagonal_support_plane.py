"""Isolated encoder/IMU swing-clearance experiment, never a training default.

Estimate floor height assuming the scheduled support soles remain planted.
No simulated body position, link pose, contact or force enters the reference.
"""
import hashlib
import itertools
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
URDF = ROOT / "simulation/exports/isaac/ankle_v23_stride/active/training_input.urdf"
LEGS = ("front_left", "rear_left", "front_right", "rear_right")


def origin(element):
    matrix = np.eye(4)
    if element is None:
        return matrix
    matrix[:3, 3] = [float(v) for v in element.get("xyz", "0 0 0").split()]
    r, p, y = [float(v) for v in element.get("rpy", "0 0 0").split()]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    matrix[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                      [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]]
    return matrix


class SoleKinematics:
    """Batched exact URDF FK, including fixed frames and ankle orientation."""
    def __init__(self, action_order):
        self.chain = []
        pending = list(ET.parse(URDF).getroot().findall("joint"))
        resolved = {"base_link"}
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in resolved]
            if not ready:
                raise ValueError("Unresolved URDF tree")
            for joint in ready:
                parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
                index, skew = None, None
                if joint.get("type") != "fixed":
                    index = action_order.index(joint.get("name"))
                    axis = np.array([float(v) for v in joint.find("axis").get("xyz").split()])
                    x, y, z = axis / np.linalg.norm(axis)
                    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                self.chain.append((parent, child, origin(joint.find("origin")), index, skew))
                resolved.add(child)
                pending.remove(joint)
        self.corners = np.array([[.0432117, y, z, 1.]
                                for y, z in itertools.product((-.047, .047), (-.027, .027))])

    def minimum_heights(self, q, up):
        transforms = {"base_link": np.broadcast_to(np.eye(4), (len(q), 4, 4))}
        for parent, child, frame, index, skew in self.chain:
            transform = transforms[parent] @ frame
            if index is not None:
                angle = q[:, index, None, None]
                motion = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
                motion[:, :3, :3] = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
                transform = transform @ motion
            transforms[child] = transform
        feet = np.stack([transforms[leg + "_foot_link"] for leg in LEGS], axis=1)
        points = np.einsum("nlij,cj->nlci", feet, self.corners)[..., :3]
        return np.einsum("nlci,ni->nlc", points, up).min(axis=2)


def install_support_plane(raw):
    if raw.num_envs != 4 or not raw.cfg.evaluation:
        raise ValueError("This prototype is restricted to the four-environment diagnostic")
    fk = SoleKinematics(raw.action_order)
    original_reference = raw._reference
    filtered = np.zeros((raw.num_envs, 4))
    raw.support_plane_diagnostics = {
        "estimated_body_height_m": torch.zeros(raw.num_envs, device=raw.device),
        "extra_swing_lift_m": torch.zeros((raw.num_envs, 4), device=raw.device),
        "nominal_estimated_sole_clearance_m": torch.zeros((raw.num_envs, 4), device=raw.device),
    }

    def reference():
        ref = original_reference().clone()
        selected = raw.pairs[raw.pair]
        swing = selected & ((raw.stage >= 1) & (raw.stage <= 3))[:, None]
        q = raw.robot.data.joint_pos.torch[:, raw.ids].detach().cpu().numpy().astype(float)
        up = -raw.robot.data.projected_gravity_b.torch.detach().cpu().numpy().astype(float)
        up /= np.maximum(np.linalg.norm(up, axis=1, keepdims=True), 1.e-6)
        supporting = (~selected).cpu().numpy()
        height = -(fk.minimum_heights(q, up) * supporting).sum(axis=1) / 2.
        nominal = ref.detach().cpu().numpy().astype(float)
        pitch = np.arcsin(np.clip(-up[:, 0], -1., 1.))
        mask = swing.cpu().numpy()
        nominal[:, 12:16] = np.where(mask, pitch[:, None] - nominal[:, 4:8] - nominal[:, 8:12], nominal[:, 12:16])
        predicted = height[:, None] + fk.minimum_heights(nominal, up)
        clearance = raw.ref_clearance.detach().cpu().numpy()
        # Cap and filter compensation. The estimate is conditional on planted
        # support soles and cannot certify contact or support stability.
        extra = np.clip((clearance - predicted) / np.maximum(up[:, 2:3], .8), 0., .025)
        extra *= np.clip(clearance / raw.cfg.lift_m, 0., 1.) * mask
        filtered[raw._age().cpu().numpy() < raw.step_dt * 1.5] = 0.
        filtered[:] += raw.step_dt / (.12 + raw.step_dt) * (extra - filtered)
        amount = torch.as_tensor(filtered * mask, device=raw.device, dtype=ref.dtype)
        down = raw.ref_down - amount
        length = .159896689
        cosine = ((raw.ref_forward.square() + down.square() - 2 * length**2) / (2 * length**2)).clamp(-.999, .999)
        knee = -raw.front_sign * torch.acos(cosine)
        hip = torch.atan2(raw.ref_forward, down) - knee / 2.
        ref[:, 4:8] = torch.where(swing, hip, ref[:, 4:8])
        ref[:, 8:12] = torch.where(swing, knee, ref[:, 8:12])
        flat_ankle = torch.as_tensor(pitch, device=raw.device, dtype=ref.dtype)[:, None] - hip - knee
        ref[:, 12:16] = torch.where(swing, flat_ankle, ref[:, 12:16])
        raw.reference_clipped |= (ref < raw.limits[:, :, 0]) | (ref > raw.limits[:, :, 1])
        raw.ref = torch.maximum(torch.minimum(ref, raw.limits[:, :, 1]), raw.limits[:, :, 0])
        raw.ref_down = torch.where(swing, down, raw.ref_down)
        raw.support_plane_diagnostics.update(
            estimated_body_height_m=torch.as_tensor(height, device=raw.device, dtype=ref.dtype),
            extra_swing_lift_m=amount,
            nominal_estimated_sole_clearance_m=torch.as_tensor(predicted, device=raw.device, dtype=ref.dtype))
        return raw.ref

    raw._reference = reference
    return {"mode": "encoder/IMU support-plane swing compensation prototype",
            "inputs": "joint encoders, IMU gravity, fixed URDF and scheduled support pair",
            "assumption": "both support soles remain planted on a horizontal floor",
            "compensation_cap_m": .025, "filter_s": .12,
            "swing_ankle_reference": "flat feedforward from requested hip/knee and IMU pitch",
            "support_ankles": "unchanged", "physical_limits_changed": False,
            "qualification_thresholds_changed": False, "requires_retraining_if_adopted": True,
            "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest()}
