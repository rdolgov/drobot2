"""Read-only comparison of saved traces and optimizer health, no rollouts."""
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
health = [json.loads(s) for s in (HERE / "optimizer-health.jsonl").read_text().splitlines()]
result = {
    "optimizer_iterations": len(health),
    "learning_rates": sorted({r["learning_rate"] for r in health}),
    "uniform_action_fraction_range": [min(r["uniform_action_fraction"] for r in health), max(r["uniform_action_fraction"] for r in health)],
    "last_50_mean_uniform_action_fraction": statistics.mean(r["uniform_action_fraction"] for r in health[-50:]),
    "last_50_mean_absolute_action_mean": statistics.mean(r["mean_abs_action_mean"] for r in health[-50:]),
    "collapse_stop_triggered": any(r["consecutive_collapsed_updates"] >= 10 for r in health),
    "comparison_scope": "Saved approximately 10 Hz traces, selected swing feet after startup. Torque cap comparison uses instantaneous trace speed rather than preceding physics substep; implicit drive effort is approximate telemetry, not measured hardware torque.",
    "conditions": {},
}
for folder_name in ("r6-model1500-mean", "r7-model249-mean"):
    folder = ROOT / folder_name
    report = json.loads((folder / "report.json").read_text())
    samples = [json.loads(s) for s in (folder / "trace.jsonl").read_text().splitlines()]
    trials = []
    for trial in range(4):
        strength = 1. if trial < 2 else .85
        trace = [s for s in samples if s["trial"] == trial and s["time_s"] > 2.5]
        peaks = []
        for leg in range(4):
            selected = [s for s in trace if s["stage"] in (1, 2, 3) and leg in ((0, 3) if s["pair"] == 0 else (1, 2))]
            peaks.append(max((s["sole_clearance_m"][leg] for s in selected), default=None))
        supporting = []
        for leg in range(4):
            selected = [s for s in trace if s["stage"] in (1, 2) and leg not in ((0, 3) if s["pair"] == 0 else (1, 2))]
            if not selected:
                continue
            joint = leg + 8
            ratios = []
            errors = []
            for s in selected:
                cap = strength * min(.8825985, 2.6477955 * max(0., 1. - abs(s["joint_velocity"][joint]) / 4.712389))
                ratios.append(abs(s["joint_effort_nm"][joint]) / max(cap, 1e-8))
                errors.append(math.degrees(abs(s["joint_target"][joint] - s["joint_position"][joint])))
            supporting.append({"leg_FL_RL_FR_RR": leg, "samples": len(selected),
                               "near_cap_fraction": statistics.mean(r >= .95 for r in ratios),
                               "mean_tracking_error_deg": statistics.mean(errors)})
        outcome = next(o for o in report["trial_outcomes"] if o["trial"] == trial)
        trials.append({"trial": trial, "motor_strength_fraction": strength,
                       "result": outcome["result"], "paired_landings": outcome["paired_landings"],
                       "peak_selected_clearance_m_FL_RL_FR_RR": peaks,
                       "supporting_knees": supporting})
    result["conditions"][folder_name] = trials
(HERE / "recovery-comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
