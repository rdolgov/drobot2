"""Read recorded experiment data only; never launch training or simulation."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
args = parser.parse_args()
run = args.run
result = {"run": str(run), "status": json.loads((run/"status.json").read_text())}
metrics = run/"metrics.jsonl"
if metrics.exists():
    rows = [json.loads(x) for x in metrics.read_text().splitlines()]
    diagonal = [x for x in rows if "Metrics/train_paired_landings" in x]
    base = [x for x in rows if "Metrics/train_forward_speed_m_s" in x]
    if base:
        result["latest_training_motion"] = base[-1]
    if diagonal:
        result["latest_training_steps"] = {k: v for k, v in diagonal[-1].items()
            if any(s in k for s in ("step", "paired_landings", "qualified_touchdowns", "lift_clear", "slip"))}
trace = run/"trace.jsonl"
report = run/"report.json"
if trace.exists() and report.exists():
    rows = [json.loads(x) for x in trace.read_text().splitlines()]
    result["evaluation"] = json.loads(report.read_text())
    trials = []
    for trial in result["evaluation"]["trials"]:
        e = trial["environment"]
        pairs = ((0, 3), (1, 2))
        motion = [r for r in rows if r["active"][e] and not r["parked"][e] and r["time_s"] > 2.5]
        swing = [r for r in motion if r["stage"][e] in (1, 2, 3)]
        advance = [r for r in motion if r["stage"][e] == 2]
        lifting = [r for r in motion if r["stage"][e] == 1]
        by_leg = {}
        for leg, name in enumerate(("FL", "RL", "FR", "RR")):
            selected = [r for r in swing if leg in pairs[r["pair"][e]]]
            by_leg[name] = {
                "peak_swing_clearance_mm": max((r["sole_clearance_m"][e][leg]*1000 for r in selected), default=0),
                "swing_samples_clear_above_8mm": mean(r["sole_clearance_m"][e][leg] > .008 and not r["contacts"][e][leg]
                                                      for r in selected) if selected else 0,
                "mean_knee_target_lag_deg": mean((r["joint_target"][e][leg+8]-r["joint_position"][e][leg+8])*180/math.pi
                                                   for r in selected) if selected else 0}
        trials.append({"environment": e, "evidence": "10 Hz saved trace; may miss brief transients", "legs": by_leg,
            "motion_tilt_rms_deg": math.sqrt(mean(r["tilt_rad"][e]**2 for r in motion))*180/math.pi if motion else 0,
            "swing_pair_simultaneously_clear_above_8mm_fraction": mean(
                all(r["sole_clearance_m"][e][l] > .008 and not r["contacts"][e][l] for l in pairs[r["pair"][e]])
                for r in advance) if advance else 0,
            "lift_support_ankle_progress_mean": mean(r["ankle_onset_progress"][e] for r in lifting) if lifting else 0})
    result["trace_analysis"] = trials
print(json.dumps(result, indent=2))
