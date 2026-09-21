"""Summarize saved evaluation evidence without exercising a controller."""
import json
import math
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent
experiments = (
    ("IK reference", "20260920-reference-full"),
    ("Initial RL, checkpoint 100", "20260920-eval-100"),
    ("Initial RL, checkpoint 300", "20260920-eval-300"),
    ("Pair-focused RL, checkpoint 100", "20260920-pair-eval-100"),
    ("Pair-focused RL, checkpoint 200", "20260920-pair-eval-200"),
    ("Pair-focused RL, checkpoint 299", "20260920-pair-eval-299"),
)
results = []
for title, directory in experiments:
    path = root/directory/"report.json"
    if not path.exists():
        continue
    report = json.loads(path.read_text())
    report.update(title=title, directory=directory)
    rows = [json.loads(x) for x in (path.parent/"trace.jsonl").read_text().splitlines()]
    for trial in report["trials"]:
        e = trial["environment"]
        motion = [r for r in rows if r["active"][e] and not r["parked"][e] and r["time_s"] > 2.5]
        trial["motion_mean_body_height_m"] = mean(r["body_position"][e][2] for r in motion) if motion else None
        trial["motion_tilt_rms_deg"] = math.sqrt(mean(r["tilt_rad"][e]**2 for r in motion))*180/math.pi if motion else None
    results.append(report)
(root/"comparison.json").write_text(json.dumps({"simulation_only": True, "hardware_qualified": False,
    "trials": "32 seconds, both starting diagonals at nominal and 85% motor strength; deterministic policy; first episodes only",
    "qualification": "all legs at least two strict completed steps, >=10 cm forward, <5 cm lateral, <10 deg RMS tilt, <20 deg peak tilt, no termination",
    "results": results}, indent=2)+"\n")
lines = ["# IK-guided diagonal RL review", "",
    "Two residual PPO candidates were trained from the diagonal IK reference. Training produced checkpoints, but no evaluated checkpoint achieved reliable long paired strides. The final candidate keeps the trunk flatter while rear-foot flight, forward placement and paired landings remain inconsistent. No candidate has been deployed to the Pi.", "",
    "Each four-trial evaluation uses 32 seconds, both starting diagonals, and nominal/85% motor strength. The final portion asks the robot to stop. No post-reset episodes count.", "",
    "| Candidate | Forward range | RMS tilt range | Qualified trials | Completed pair events across trials |",
    "|---|---:|---:|---:|---:|"]
for r in results:
    t = r["trials"]
    distance = [x["forward_m"]*100 for x in t]
    tilt = [x["tilt_rms_deg"] for x in t]
    pairs = sum(sum(x["full_steps_by_leg_FL_RL_FR_RR"])/2 for x in t)
    lines.append(f"| [{r['title']}]({r['directory']}/report.json) | {min(distance):.1f}–{max(distance):.1f} cm | {min(tilt):.2f}–{max(tilt):.2f}° | {sum(x['qualified'] for x in t)}/{len(t)} | {pairs:.0f} |")
lines += ["", "A paired event requires real flight and long forward placement of both selected feet followed by a quiet plant. Phase advancement and forward sliding are not qualifying steps. A few events do not establish reliable walking.", "",
    "## Per-trial evidence", "",
    "| Candidate / start | Strength | Forward | Lateral | Motion body height | Full steps FL/RL/FR/RR | Peak swing clearance FL/RL/FR/RR |",
    "|---|---:|---:|---:|---:|---|---|"]
for r in results:
    for t in r["trials"]:
        pair = "FL+RR" if t["start_pair"] == 0 else "FR+RL"
        counts = "/".join(str(int(x)) for x in t["full_steps_by_leg_FL_RL_FR_RR"])
        heights = "/".join(f"{x*1000:.1f}" for x in t["peak_swing_clearance_m_by_leg"])
        lines.append(f"| {r['title']} / {pair} | {t['motor_strength']:.0%} | {t['forward_m']*100:.1f} cm | {t['lateral_m']*100:.1f} cm | {t['motion_mean_body_height_m']*100:.1f} cm | {counts} | {heights} mm |")
lines += ["", "## Training and model assumptions", "",
    "- Initial run: fresh Gaussian residual PPO; saved checkpoint 300 retained; stopped after iteration 313 rather than completing its planned 500 iterations.",
    "- Second run: separate fresh actor, stronger real-pair clearance and level-body rewards, forward-reward gating, and larger bounded corrections. Its status and completion files are the authority for completion.",
    "- 16-servo model with estimated 3.395 kg mass. Mass distribution and loaded actuator dynamics have not been measured on the updated physical robot.",
    "- Original 0.883 Nm working torque cap, torque-speed envelope, calibration and joint limits retained. Neither run uses a stair policy.",
    "- Videos use actual simulation frames, with phase/contact/clearance/displacement annotations. Frames after an episode reset are excluded.",
    "- Source and calibration snapshots, checkpoint hashes, exact arguments, optimizer logs and evaluation traces are retained in each run directory.",
    "", "See [the controller/reward description](../../docs/rl-ankle-v25-ik.md) and comparison.json for exact values.", ""]
lines += ["## Videos and checkpoints", "",
    "[36-second comparison](diagonal-rl-comparison.mp4) shows the same 2.5–14.5 second interval of the IK reference, initial RL checkpoint 100, and the final pair-focused checkpoint. Each segment plays at real speed. These clips illustrate the tradeoff between forward motion and body stability; neither is presented as qualified walking.", "",
    "Full single-environment videos and their own measured reports:", "",
    "- [IK reference](20260920-reference-video/walk.mp4)",
    "- [Initial RL checkpoint 100](20260920-progress-video/walk.mp4)",
    "- [Final pair-focused checkpoint 299](20260920-pair-final-video/walk.mp4)", "",
    "Recordings use one nominal-strength robot with FL+RR first. The four-environment evaluations above include both first diagonals and weaker motors; results can differ from the recording because of scene size and simulation numerics.", "",
    "Saved trained weights:", "",
    "- [Initial RL checkpoint 100](20260920-train-r1/checkpoints/model_100.pt), used in the forward-progress clip.",
    "- [Initial RL checkpoint 300](20260920-train-r1/checkpoints/model_300.pt), retained from the capped first run.",
    "- [Final pair-focused checkpoint 299](20260920-train-r2/checkpoints/model_299.pt), from the completed second run.", "",
    "These are simulator checkpoints with their matching controller packages and frozen provenance. They are not directly compatible with the currently deployed Pi policy runtime.", ""]
(root/"README.md").write_text("\n".join(lines), encoding="utf-8")
print(root/"README.md")
