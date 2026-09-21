"""Offline calculations on saved joint/sole traces; no simulator or controller execution."""
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_name("20260920-current-video")
OUTPUT = Path(__file__).with_name("sole-attitude-audit-20260920.json")
if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
helper = ROOT / "simulation/isaac/diagnostics/diagonal_static_load.py"
spec = importlib.util.spec_from_file_location("saved_urdf_fk", helper)
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)
legs = fk.LEGS
joint_kinds = ("hip_abduction", "hip_flexion", "knee", "ankle_pitch")
keys = [f"{leg}_{kind}" for kind in joint_kinds for leg in legs]
center = np.array([.0432117, 0., 0., 1.])
corners = np.array([[.0432117, y, z, 1.] for y, z in itertools.product((-.047, .047), (-.027, .027))])


def tilt(normal):
    normal = normal/np.linalg.norm(normal)
    return {"total_deg": math.degrees(math.acos(np.clip(normal[2], -1., 1.))),
            "longitudinal_deg": math.degrees(math.atan2(normal[0], normal[2])),
            "lateral_deg": math.degrees(math.atan2(normal[1], normal[2]))}


rows = []
for line in (SOURCE / "trace.jsonl").read_text().splitlines():
    data = json.loads(line)
    measured = fk.kinematics(dict(zip(keys, data["joint_position"])))[0]
    desired = fk.kinematics(dict(zip(keys, data["joint_target"])))[0]
    local = np.stack([(measured[leg + "_foot_link"] @ center)[:3] for leg in legs])
    world = np.array(data["sole_world"])
    a, b = local-local.mean(0), world-world.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    handedness = np.eye(3)
    handedness[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ handedness @ u.T
    translation = world.mean(0)-rotation @ local.mean(0)
    reconstructed = local @ rotation.T + translation
    error = np.linalg.norm(reconstructed-world, axis=1).max()
    cap = .8825985 * (1. if data["trial"] % 2 == 0 else .85)
    selected = data["swing_pair"] or [data["swing_corner"]]
    row = {"time_s": data["time_s"], "trial": data["trial"], "phase": data["phase"],
           "gate": data["gate"], "failure": data["failure"],
           "root_rigid_fit_error_m": float(error),
           "root_translation_vs_saved_position_m": float(np.linalg.norm(translation-data["root_position"])),
           "body_up_tilt": tilt(rotation[:, 2]), "legs": {}}
    if error > 1e-5:
        raise ValueError(f"Saved geometry does not support rigid pose reconstruction: {error}")
    for i, leg in enumerate(legs):
        transform = measured[leg + "_foot_link"]
        actual_normal = -transform[:3, 0]
        requested_normal = -desired[leg + "_foot_link"][:3, 0]
        points_world = (corners @ transform.T)[:, :3] @ rotation.T + translation
        low, high = points_world[:, 2].min(), points_world[:, 2].max()
        geometry_error = max(abs(low-data["sole_clearance_m_FL_RL_FR_RR"][i]), abs(high-data["sole_high_m"][i]))
        if geometry_error > 1e-5:
            raise ValueError(f"Reconstructed corners disagree with saved geometry: {geometry_error}")
        row["legs"][leg] = {
            "selected": leg in selected,
            "normal_force_n": data["normal_force_n"][i],
            "clearance_mm": data["sole_clearance_m_FL_RL_FR_RR"][i]*1000.,
            "sole_corner_height_span_mm": float((high-low)*1000.),
            "corner_reconstruction_error_m": float(geometry_error),
            "actual_sole_world_tilt": tilt(rotation @ actual_normal),
            "actual_sole_chassis_tilt": tilt(actual_normal),
            "requested_sole_world_tilt_at_actual_body_attitude": tilt(rotation @ requested_normal),
            "joint_error_deg": {kind: math.degrees(data["joint_position"][i+4*j]-data["joint_target"][i+4*j]) for j, kind in enumerate(joint_kinds)},
            "joint_effort_nm": {kind: data["joint_effort_nm"][i+4*j] for j, kind in enumerate(joint_kinds)},
            "joint_at_cap": {kind: abs(data["joint_effort_nm"][i+4*j]) >= .98*cap for j, kind in enumerate(joint_kinds)},
        }
    rows.append(row)

report = {"scope": __doc__,
    "method": "URDF forward kinematics at saved measured and delivered joint positions. Fit one rigid transform from four sole centers into recorded world coordinates using proper-rotation SVD. Reconstruct all sole corner heights and compare to saved low/high values. Desired sole orientation is evaluated at the measured body attitude; it is not a new controller rollout.",
    "limitations": ["Sampled trace is about 10 Hz; transient forces between samples are unavailable.",
                    "Only net vertical foot forces are saved. Actual pressure distribution, contact point wrench and center of pressure cannot be recovered.",
                    "World/body orientation comes from geometric reconstruction, not a new IMU measurement.",
                    "Correlated tilt/torque/track error does not prove a feedback correction will solve unloading."],
    "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (SOURCE / "trace.jsonl", SOURCE / "report.json", fk.URDF, helper, Path(__file__))},
    "maximum_root_rigid_fit_error_m": max(r["root_rigid_fit_error_m"] for r in rows),
    "maximum_root_translation_error_m": max(r["root_translation_vs_saved_position_m"] for r in rows),
    "maximum_corner_height_error_m": max(v["corner_reconstruction_error_m"] for r in rows for v in r["legs"].values()),
    "final_samples": [next(r for r in reversed(rows) if r["trial"] == i) for i in range(6)],
    "rows": rows}
OUTPUT.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps({k: report[k] for k in ("maximum_root_rigid_fit_error_m", "maximum_corner_height_error_m")}, indent=2))
for row in report["final_samples"]:
    print(json.dumps({"trial": row["trial"], "body_tilt": row["body_up_tilt"], "legs": {
        leg: {"selected": v["selected"], "world_tilt": v["actual_sole_world_tilt"],
              "target_world_tilt": v["requested_sole_world_tilt_at_actual_body_attitude"],
              "height_span_mm": v["sole_corner_height_span_mm"], "errors": v["joint_error_deg"],
              "ankle_effort_nm": v["joint_effort_nm"]["ankle_pitch"]}
        for leg, v in row["legs"].items()}}, indent=2))
