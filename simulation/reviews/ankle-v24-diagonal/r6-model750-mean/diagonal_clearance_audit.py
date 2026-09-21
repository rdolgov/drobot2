"""Offline encoder/IMU support-plane audit of a saved diagnostic trace.

This only evaluates recorded poses and hypothetical IK targets. It does not
launch physics, move a robot, change a controller or establish dynamic success.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from diagonal_static_load import LEGS, URDF, joints, kinematics


CORNERS = np.array([[.0432117, y, z, 1.]
                    for y, z in itertools.product((-.047, .047), (-.027, .027))])


def sole_heights(q, action_order, up):
    transforms = kinematics(dict(zip(action_order, q)))[0]
    return np.array([((transforms[leg + "_foot_link"] @ CORNERS.T).T[:, :3] @ up).min()
                     for leg in LEGS])


def audit(folder):
    report = json.loads((folder / "report.json").read_text())
    trace = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
    order = report["action_order"]
    limits = {j.get("name"): (.95 * float(j.find("limit").get("lower")),
                              .95 * float(j.find("limit").get("upper")))
              for j in joints if j.get("name") in order}
    trials = []
    for outcome in sorted(report["trial_outcomes"], key=lambda row: row["trial"]):
        index = outcome["trial"]
        samples = [row for row in trace if row["trial"] == index]
        lift = [row for row in samples if row["stage"] == 1]
        if not lift:
            continue
        row = lift[-1]
        before = min(samples, key=lambda s: abs(s["time_s"] - 2.4))
        selected = [0, 3] if row["pair"] == 0 else [1, 2]
        support = [i for i in range(4) if i not in selected]
        up = -np.array(row["projected_gravity_body"])
        up /= np.linalg.norm(up)
        measured_relative = sole_heights(row["joint_position"], order, up)
        # Assumes the scheduled support feet are planted on one horizontal floor.
        # Contact is checked below for this audit, but is not an estimator input.
        height = -measured_relative[support].mean()
        predicted = height + measured_relative
        nominal = np.array(row["nominal_joint_reference"])
        nominal_clearance = height + sole_heights(nominal, order, up)
        pitch = math.asin(float(np.clip(-up[0], -1., 1.)))
        legs = []
        for leg in selected:
            target_clear = row["reference_clearance_m"][leg]
            forward, down = row["reference_forward_m"][leg], row["reference_down_m"][leg]
            sign = 1 if leg in (0, 2) else -1

            def proposed(amount):
                z = down - amount
                cosine = (forward**2 + z**2 - 2 * .159896689**2) / (2 * .159896689**2)
                knee = -sign * math.acos(max(-.999, min(.999, cosine)))
                hip = math.atan2(forward, z) - knee / 2.
                pose = nominal.copy()
                pose[leg + 4], pose[leg + 8] = hip, knee
                # Hypothetical swing-only ankle feedforward; supporting ankles
                # are untouched. This has not been adopted in the controller.
                pose[leg + 12] = pitch - hip - knee
                clearance = height + sole_heights(pose, order, up)[leg]
                return pose, float(clearance)

            lo, hi = 0., .035
            if proposed(lo)[1] >= target_clear:
                hi = 0.
            elif proposed(hi)[1] >= target_clear:
                for _ in range(18):
                    mid = (lo + hi) / 2.
                    if proposed(mid)[1] >= target_clear:
                        hi = mid
                    else:
                        lo = mid
            pose, clear = proposed(hi)
            bounded = np.array([np.clip(q, *limits[name]) for q, name in zip(pose, order)])
            indices = [leg + offset for offset in (0, 4, 8, 12)]
            legs.append({
                "leg": LEGS[leg],
                "actual_sole_clearance_mm": row["sole_clearance_m"][leg] * 1000,
                "predicted_from_encoders_imu_and_assumed_support_mm": predicted[leg] * 1000,
                "nominal_pose_predicted_sole_clearance_mm": nominal_clearance[leg] * 1000,
                "reference_clearance_mm": target_clear * 1000,
                "rear_or_front_hip_drop_since_pre_lift_mm":
                    (before["hip_pivot_world"][leg][2] - row["hip_pivot_world"][leg][2]) * 1000,
                "hypothetical_extra_planar_lift_mm": hi * 1000,
                "hypothetical_clearance_mm": clear * 1000,
                "hypothetical_clearance_after_joint_clamp_mm":
                    (height + sole_heights(bounded, order, up)[leg]) * 1000,
                "hypothetical_joint_degrees": {order[j]: math.degrees(pose[j]) for j in indices},
                "joint_limits_exceeded": [order[j] for j in indices if abs(pose[j] - bounded[j]) > 1.e-7],
            })
        trials.append({"trial": index, "time_s": row["time_s"], "pair": row["pair"],
                       "support_contacts_present": all(row["contacts"][i] for i in support),
                       "estimated_root_height_m": float(height),
                       "measured_root_height_m": row["body_position"][2],
                       "root_height_error_mm": (height - row["body_position"][2]) * 1000,
                       "body_pitch_degrees": math.degrees(pitch), "selected_legs": legs})
    return {"scope": "Offline first-lift snapshots, not a simulation or a feasible-gait claim",
            "checkpoint_sha256": report["checkpoint_sha256"],
            "urdf_sha256": hashlib.sha256(URDF.read_bytes()).hexdigest(),
            "estimator_inputs": "16 encoders, IMU gravity, fixed URDF, scheduled support pair",
            "assumptions": ["Both scheduled support soles are planted on a horizontal floor",
                            "No foot slip or unmodeled geometry", "Recorded pose is held fixed for hypothetical IK"],
            "limitations": "Ignores dynamic load redistribution and future tracking. Extra lift could worsen body sag. Joint feasibility does not establish torque feasibility. Flat swing ankle feedforward is an untested change.",
            "trials": trials}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.folder)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for trial in result["trials"]:
        print("Trial", trial["trial"], "root estimate error mm", round(trial["root_height_error_mm"], 2))
        for leg in trial["selected_legs"]:
            print(leg["leg"], "actual mm", round(leg["actual_sole_clearance_mm"], 2),
                  "extra lift mm", round(leg["hypothetical_extra_planar_lift_mm"], 2),
                  "clamped clearance mm", round(leg["hypothetical_clearance_after_joint_clamp_mm"], 2),
                  "joint bounds exceeded", leg["joint_limits_exceeded"])
