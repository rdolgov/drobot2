"""Summarize saved training/diagnostic evidence; does not run the policy."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--diagnostic", type=Path)
args = parser.parse_args()
run = args.run
status = json.loads((run / "status.json").read_text())
contract = json.loads((run / "contract.json").read_text())
reports = []
current = None
for line in (run / "train.log").read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
    if match:
        current = {"iteration": int(match[1])}
        reports.append(current)
    match = re.search(r"Metrics/(train_\w+):\s+([-+\d.e]+)", line)
    if match and current is not None:
        current[match[1]] = float(match[2])
keys = sorted({key for row in reports[-100:] for key in row if key != "iteration"})
totals_path = run / "train.metrics.gate-totals.json"
result = {
    "training_status": status["status"], "checkpoint": status.get("checkpoint"),
    "checkpoint_sha256": status.get("checkpoint_sha256"),
    "iteration_reports": len(reports),
    "max_logged_paired_landings": max((r.get("train_paired_landings", 0.) for r in reports), default=None),
    "last_100_iteration_means": {key: statistics.mean(r[key] for r in reports[-100:] if key in r) for key in keys},
    "statistical_scope": "Unweighted means of rounded logged iteration summaries; training evidence only. Cumulative counts cover completed episodes only.",
    "completed_episode_totals": json.loads(totals_path.read_text()) if totals_path.exists() else None,
    "source_hashes_still_match": all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in contract["source_hashes"].items()),
    "asset_hash_still_matches": hashlib.sha256(Path(contract["asset"]).read_bytes()).hexdigest() == contract["asset_sha256"],
    "hardware_qualified": False,
}
if args.diagnostic:
    folder = args.diagnostic
    report = json.loads((folder / "report.json").read_text())
    samples = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines() if line.strip()]
    result["independent_diagnostic"] = {
        "path": str(folder.resolve()), "status": report["status"],
        "scope": report["scope"], "checkpoint_sha256": report["checkpoint_sha256"],
        "matches_training_final_checkpoint": report["checkpoint_sha256"] == status.get("checkpoint_sha256"),
        "actor_max_absolute_sampled_action": max(abs(a) for row in samples for a in row["policy_action"]),
        "trace_samples": len(samples),
        "trials": [{"trial": t["trial"], "result": t["result"],
                    "last_time_s": t["last_sample"]["time_s"],
                    "last_stage": t["last_sample"]["stage"],
                    "paired_landings": t["paired_landings"],
                    "qualified_touchdowns_by_leg_FL_RL_FR_RR": t["last_sample"]["qualified_touchdowns_by_leg"],
                    "physical_fall": t["last_sample"]["physical_fall"]}
                   for t in sorted(report["trial_outcomes"], key=lambda row: row["trial"])],
    }
    result["gait_result"] = (
        "failed: all independent trials timed out with zero full paired landings"
        if all(t["result"] == "gate_timeout" and not t["paired_landings"] for t in report["trial_outcomes"])
        else "review independent trial details; no automatic hardware qualification")
(run / "result-summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in result.items() if key not in ("last_100_iteration_means", "completed_episode_totals")}, indent=2))
