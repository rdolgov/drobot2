"""Drobot-specific URDF adapters and named foot operations."""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .leg import LegModel
from .rigid_body import PointContact, URDFRobot, _origin, _xyz
from .transforms import inverse_transform, rx, rz, vector

LEGS = ("front_left", "front_right", "rear_left", "rear_right")
KINDS = ("hip_abduction", "hip_flexion", "knee")
VARIANTS = {
    "original": "quadruped_robot.urdf",
    "rigid_shoe": "quadruped_robot_v2_ankle_baseline.urdf",
    "active_ankle": "quadruped_robot_v2_ankle_active.urdf",
}


class DrobotModel(URDFRobot):
    @classmethod
    def from_repository(cls, repository_root, *, variant="rigid_shoe"):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {tuple(VARIANTS)}")
        path = Path(repository_root) / "cad" / "exports" / "urdf" / VARIANTS[variant]
        model = cls(path)
        model.variant = variant
        return model

    def __init__(self, urdf_path):
        source = ET.parse(urdf_path).getroot()
        active = {j.attrib["name"] for j in source.findall("joint") if j.get("type") != "fixed"}
        names = []
        for leg in LEGS:
            names.extend(f"{leg}_{kind}" for kind in KINDS)
            if f"{leg}_ankle_pitch" in active:
                names.append(f"{leg}_ankle_pitch")
        super().__init__(urdf_path, joint_order=names)
        self.variant = "custom"
        self.endpoints = {}
        self._leg_models = {}
        for leg in LEGS:
            foot = source.find(f"link[@name='{leg}_foot_link']")
            if foot is not None:
                tread = foot.find("collision[@name='tread']")
                if tread is None or tread.find("geometry/box") is None:
                    raise ValueError(f"{leg}: explicit box tread is needed to locate the sole center")
                T = _origin(tread.find("origin"))
                size = _xyz(tread.find("geometry/box").get("size"))
                if not np.allclose(T[:3, :3], np.eye(3), atol=1e-7, rtol=0):
                    raise ValueError("analytical sole endpoint assumes axis-aligned local tread")
                point = T[:3, 3] + [size[0] / 2, 0, 0]
                self.endpoints[leg] = (f"{leg}_foot_link", point)
            else:
                distal = source.find(f"link[@name='{leg}_distal_link']")
                proxy = distal.find("collision[@name='simulation_only_fork_tip_contact_proxy']")
                if proxy is None:
                    raise ValueError(f"{leg}: fork endpoint is not defined by the source")
                point = _origin(proxy.find("origin"))[:3, 3]
                self.endpoints[leg] = (f"{leg}_distal_link", point)
            self._leg_models[leg] = self._read_leg(leg)

    def leg_indices(self, leg) -> np.ndarray:
        if leg not in LEGS:
            raise KeyError(f"unknown leg {leg}")
        return np.array([self.index[name] for name in self.joint_names if name.startswith(leg + "_")])

    def leg(self, leg) -> LegModel:
        return self._leg_models[leg]

    def _read_leg(self, leg):
        root, hip, knee = [self.joint_by_name[f"{leg}_{kind}"] for kind in KINDS]
        s = 1 if root.origin[1, 3] > 0 else -1
        def require_close(actual, expected, description):
            if not np.allclose(actual, expected, atol=2e-7, rtol=0):
                raise ValueError(f"{leg}: {description} differs from the analytical family; use URDFRobot for arbitrary geometry")
        require_close(root.origin[:3, :3], rz(s * np.pi / 2) @ rx(np.pi / 2), "root orientation")
        require_close(root.axis, [0, 0, 1], "abduction axis")
        require_close(hip.origin[:3, :3], rz(-np.pi / 2) @ rx(-np.pi / 2), "hip orientation")
        require_close(hip.axis, [0, 0, -s], "hip axis")
        require_close(knee.axis, [0, 0, -s], "knee axis")
        require_close(knee.origin[:3, :3], np.eye(3), "knee orientation")
        require_close(knee.origin[1:3, 3], [0, 0], "knee off-axis translation")
        if root.parent != self.root_link or hip.parent != root.child or knee.parent != hip.child:
            raise ValueError("unexpected leg topology")
        endpoint_link, endpoint = self.endpoints[leg]
        require_close(endpoint[1:], [0, 0], "endpoint off-axis displacement")
        ankle = self.joint_by_name.get(f"{leg}_ankle_pitch")
        if ankle is None:
            if endpoint_link != knee.child:
                raise ValueError("unexpected fork parent")
            distal_length, extension, active = endpoint[0], 0., False
        else:
            if ankle.parent != knee.child or ankle.child != endpoint_link:
                raise ValueError("unexpected ankle topology")
            require_close(ankle.origin[:3, :3], np.eye(3), "ankle origin orientation")
            require_close(ankle.origin[1:3, 3], [0, 0], "ankle off-axis translation")
            active = ankle.kind != "fixed"
            if active:
                require_close(ankle.axis, [0, 0, -s], "ankle axis")
            distal_length, extension = ankle.origin[0, 3], endpoint[0]
        indices = self.leg_indices(leg)
        a, minus_b, c = hip.origin[:3, 3]
        return LegModel(s, root.origin[:3, 3], a, -minus_b, c, knee.origin[0, 3],
                        distal_length, extension, self.lower[indices], self.upper[indices], active)

    def foot_pose(self, q, leg, *, base_pose=None):
        link, point = self.endpoints[leg]
        return self.point_pose(q, link, point, base_pose=base_pose)

    def foot_positions(self, q, *, base_pose=None):
        state = self.kinematics(q, base_pose=base_pose)
        return {leg: state.poses[link][:3, 3] + state.poses[link][:3, :3] @ point
                for leg, (link, point) in self.endpoints.items()}

    def foot_jacobian(self, q, leg, *, base_pose=None, floating=False):
        link, point = self.endpoints[leg]
        return self.point_jacobian(q, link, point, base_pose=base_pose, floating=floating)

    def feet_ik(self, targets, seed, *, base_pose=None, sole_pitches=None):
        """Solve named foot targets, preserving other joints; targets in world if base_pose supplied."""
        q = vector(seed, self.n, "seed").copy()
        world_to_body = inverse_transform(np.eye(4) if base_pose is None else base_pose)
        for leg, target in targets.items():
            p = vector(target, 3, "foot target")
            p = world_to_body[:3, :3] @ p + world_to_body[:3, 3]
            indices = self.leg_indices(leg)
            pitch = None if sole_pitches is None else sole_pitches.get(leg)
            q[indices] = self.leg(leg).ik(p, seed=q[indices], sole_pitch=pitch)
        return q

    def stance(self, *, fore_aft_m=.080):
        """Flat-sole zero-abduction stance, with front/rear knees bent oppositely.

        Returns q and a level base pose placing each endpoint center at z=0.
        For the original variant the endpoint is the proxy CENTER, not its
        ground contact. This method is geometry, not a stability guarantee.
        """
        if not np.isfinite(fore_aft_m) or fore_aft_m < 0:
            raise ValueError("fore_aft_m must be finite and nonnegative")
        q, heights = np.zeros(self.n), []
        for leg in LEGS:
            model = self.leg(leg)
            if fore_aft_m >= model.proximal_length:
                raise ValueError("stance reach must be below proximal link length")
            beta = np.arcsin(fore_aft_m / model.proximal_length) * (1 if leg.startswith("front") else -1)
            angles = np.array([0., beta, -beta] + ([0.] if model.active_ankle else []))
            if np.any(angles < model.lower) or np.any(angles > model.upper):
                raise ValueError("stance exceeds stored joint limits")
            q[self.leg_indices(leg)] = angles
            heights.append(-model.fk(angles)[2, 3])
        if np.ptp(heights) > 1e-7:
            raise ValueError("one level base height cannot place all endpoints at the floor")
        base = np.eye(4)
        base[2, 3] = float(np.mean(heights))
        return q, base

    def sole_contacts(self, *, friction_coefficient=None):
        """One point per sole center; finite-area sole moments are not modeled."""
        contacts = []
        for leg, (link, point) in self.endpoints.items():
            if not link.endswith("_foot_link"):
                raise ValueError("original fork sphere center is not a material ground point; use rigid_shoe for this contact example")
            contacts.append(PointContact(link, point.copy(), friction_coefficient=friction_coefficient))
        return tuple(contacts)
