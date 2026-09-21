"""Offline URDF FK and minimum peak static torque with two supporting sole patches.

No simulator, robot, asset modification or learning. Vertical contact forces
only; friction, acceleration, actuator dynamics and collision are not modeled.
The result is a quasi-static feasibility bound, not gait qualification.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
LEGS = ("front_left", "rear_left", "front_right", "rear_right")
URDF = ROOT / "simulation/exports/isaac/ankle_v23_stride/active/training_input.urdf"


def vector(value):
    return np.array([float(v) for v in value.split()])


def origin(element):
    matrix = np.eye(4)
    if element is not None:
        matrix[:3, :3] = Rotation.from_euler("xyz", vector(element.get("rpy", "0 0 0"))).as_matrix()
        matrix[:3, 3] = vector(element.get("xyz", "0 0 0"))
    return matrix


tree = ET.parse(URDF).getroot()
links = {link.get("name"): link for link in tree.findall("link")}
joints = tree.findall("joint")


def kinematics(q):
    transforms = {"base_link": np.eye(4)}
    joint_frames, descendants = {}, {}
    pending = list(joints)
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in transforms]
        if not ready:
            raise ValueError("Unresolved URDF tree")
        for j in ready:
            name, parent, child = j.get("name"), j.find("parent").get("link"), j.find("child").get("link")
            frame = transforms[parent] @ origin(j.find("origin"))
            if j.get("type") != "fixed":
                axis = vector(j.find("axis").get("xyz"))
                axis /= np.linalg.norm(axis)
                joint_frames[name] = (frame[:3, 3], frame[:3, :3] @ axis)
                motion = np.eye(4)
                motion[:3, :3] = Rotation.from_rotvec(axis * q.get(name, 0.)).as_matrix()
                transforms[child] = frame @ motion
            else:
                transforms[child] = frame
            for members in descendants.values():
                if parent in members:
                    members.add(child)
            descendants[name] = {child}
            pending.remove(j)
    mass, centers = {}, {}
    for name, link in links.items():
        inertial = link.find("inertial")
        if inertial is not None:
            mass[name] = float(inertial.find("mass").get("value"))
            centers[name] = (transforms[name] @ origin(inertial.find("origin")))[:3, 3]
    total_mass = sum(mass.values())
    com = sum(mass[n] * centers[n] for n in mass) / total_mass
    return transforms, joint_frames, descendants, mass, centers, com, total_mass


def pose(front, rear, down, shift_x, shift_y, pair, lift=.022):
    q = {}
    selected = {0, 3} if pair == 0 else {1, 2}
    for i, leg in enumerate(LEGS):
        sign = 1 if i in (0, 2) else -1
        x = (front if sign == 1 else -rear) - shift_x
        z = down - (lift if i in selected else 0.)
        cosine = (x*x + z*z - 2 * .159896689**2) / (2 * .159896689**2)
        if not -1 < cosine < 1:
            return None
        knee = -sign * math.acos(cosine)
        hip = math.atan2(x, z) - knee / 2
        side = 1 if i < 2 else -1
        abduction = -side * math.atan2(shift_y, down + .088908389)
        q.update({leg + "_hip_abduction": abduction, leg + "_hip_flexion": hip,
                  leg + "_knee": knee, leg + "_ankle_pitch": -hip-knee})
    for j in joints:
        if j.get("name") in q:
            limit = j.find("limit")
            if not .95 * float(limit.get("lower")) <= q[j.get("name")] <= .95 * float(limit.get("upper")):
                return None
    return q


def solve(q, pair):
    transforms, frames, descendants, mass, centers, com, total_mass = kinematics(q)
    support = [1, 2] if pair == 0 else [0, 3]
    points, feet = [], []
    for i in support:
        foot = LEGS[i] + "_foot_link"
        for y, z in itertools.product((-.047, .047), (-.027, .027)):
            points.append((transforms[foot] @ np.array([.0432117, y, z, 1.]))[:3])
            feet.append(foot)
    points = np.array(points)
    equality = np.zeros((3, 9))
    equality[:, :8] = np.stack((np.ones(8), points[:, 0], points[:, 1]))
    rhs = total_mass * 9.81 * np.array([1, com[0], com[1]])
    inequalities, bounds_rhs, gravity, contact_moments = [], [], {}, {}
    for name, (pivot, axis) in frames.items():
        g = sum(np.dot(np.cross(centers[link] - pivot, [0, 0, -9.81 * mass[link]]), axis)
                for link in descendants[name] if link in mass)
        c = np.array([np.dot(np.cross(point-pivot, [0, 0, 1]), axis)
                      if foot in descendants[name] else 0. for point, foot in zip(points, feet)])
        inequalities += [np.r_[c, -1.], np.r_[-c, -1.]]
        bounds_rhs += [-g, g]
        gravity[name], contact_moments[name] = g, c
    result = linprog(np.r_[np.zeros(8), 1.], A_ub=inequalities, b_ub=bounds_rhs,
                     A_eq=equality, b_eq=rhs, bounds=[(0, None)] * 9, method="highs")
    if not result.success:
        return {"feasible_vertical_equilibrium": False, "solver_message": result.message}
    torques = {name: float(-(gravity[name] + contact_moments[name] @ result.x[:8])) for name in frames}
    return {"feasible_vertical_equilibrium": True, "minimum_peak_torque_nm": float(result.x[-1]),
            "body_com_m": com.tolist(), "mass_kg": total_mass,
            "support_force_n": [float(result.x[:4].sum()), float(result.x[4:8].sum())],
            "joint_torque_nm": torques,
            "support_corner_height_range_m": float(np.ptp(points[:, 2]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    current = []
    for pair in (0, 1):
        q = pose(.080, .080, .280, .014, 0., pair)
        current.append({"pair": pair, **solve(q, pair)})
    # Both pairs, first lift endpoint only; a bounded posture screen, not a gait.
    candidates = []
    for front, rear, down, shift in itertools.product((.03, .05, .07), (.02, .04, .06), (.295, .305, .310), (0., .015, .03)):
        pairs = []
        for pair in (0, 1):
            q = pose(front, rear, down, 0., shift * (1 if pair == 0 else -1), pair)
            if q is None:
                break
            pairs.append(solve(q, pair))
        if len(pairs) == 2 and all(p["feasible_vertical_equilibrium"] for p in pairs):
            candidates.append({"front_m": front, "rear_m": rear, "down_m": down,
                "pair_lateral_shift_m": shift, "worst_pair_peak_torque_nm": max(p["minimum_peak_torque_nm"] for p in pairs),
                "pairs": pairs})
    candidates.sort(key=lambda c: c["worst_pair_peak_torque_nm"])
    report = {"scope": __doc__, "urdf": str(URDF), "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
              "nominal_effort_cap_nm": .8825985, "weak_motor_effort_cap_nm": .85 * .8825985,
              "current_pose": current, "candidate_count": len(candidates), "best_candidates": candidates[:10]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"current": [{"pair": p["pair"], "peak_nm": p.get("minimum_peak_torque_nm")} for p in current],
                      "best": [{k: c[k] for k in ("front_m", "rear_m", "down_m", "pair_lateral_shift_m", "worst_pair_peak_torque_nm")} for c in candidates[:5]]}, indent=2))


if __name__ == "__main__":
    main()
