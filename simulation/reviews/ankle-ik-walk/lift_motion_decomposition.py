"""Offline decomposition of saved sole-center displacement; no rollout or control changes."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
trace = Path(__file__).with_name("20260920-current-video") / "trace.jsonl"
output = Path(__file__).with_name("lift-motion-decomposition-20260920.json")
if output.exists():
    raise FileExistsError(output)
spec = importlib.util.spec_from_file_location("saved_fk", ROOT / "simulation/isaac/diagnostics/diagonal_static_load.py")
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)
keys = [f"{leg}_{kind}" for kind in ("hip_abduction", "hip_flexion", "knee", "ankle_pitch") for leg in fk.LEGS]
center = np.array([.0432117, 0., 0., 1.])
rows = [json.loads(line) for line in trace.read_text().splitlines()]


def geometry(row):
    frames = fk.kinematics(dict(zip(keys, row["joint_position"])))[0]
    local = np.stack([(frames[leg + "_foot_link"] @ center)[:3] for leg in fk.LEGS])
    world = np.array(row["sole_world"])
    u, _, vt = np.linalg.svd((local-local.mean(0)).T @ (world-world.mean(0)))
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ fix @ u.T
    translation = world.mean(0)-rotation @ local.mean(0)
    return local, rotation, translation


results = []
for trial in range(6):
    samples = [r for r in rows if r["trial"] == trial]
    end = samples[-1]
    start = next(r for r in reversed(samples) if r["time_s"] > 2.5 and r["phase"] == "weight_transfer")
    l0, r0, t0 = geometry(start)
    l1, r1, t1 = geometry(end)
    selected = end["swing_pair"] or [end["swing_corner"]]
    contributions = {}
    for leg in selected:
        i = fk.LEGS.index(leg)
        observed = (np.array(end["sole_world"])[i, 2]-np.array(start["sole_world"])[i, 2])*1000.
        translation = (t1[2]-t0[2])*1000.
        articulation = (r0 @ (l1[i]-l0[i]))[2]*1000.
        rotation = ((r1-r0) @ l1[i])[2]*1000.
        contributions[leg] = {"actual_sole_center_height_change_mm": observed,
            "root_vertical_translation_mm": translation,
            "measured_joint_motion_at_initial_attitude_mm": articulation,
            "body_rotation_at_final_joint_pose_mm": rotation,
            "sum_error_mm": observed-translation-articulation-rotation,
            "final_lowest_sole_corner_mm": end["sole_clearance_m_FL_RL_FR_RR"][i]*1000.}
    results.append({"trial": trial, "baseline_time_s": start["time_s"], "final_time_s": end["time_s"],
                    "baseline_requested_lift_mm": start["requested_lift_m"]*1000.,
                    "final_requested_lift_mm": end["requested_lift_m"]*1000., "selected_legs": contributions})
report = {"scope": __doc__, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
          "formula": "delta world sole-center Z = delta root Z + [R0*(L1-L0)]Z + [(R1-R0)*L1]Z",
          "baseline": "Last saved weight-transfer sample before the first lift; about 10 Hz sampling.",
          "limitations": "Algebraic decomposition of recorded motion, not causal isolation. Terms depend on the stated ordering. Sole-center change differs from lowest-corner clearance. No counterfactual physics was run.",
          "trials": results}
output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))
