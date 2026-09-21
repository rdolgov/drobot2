"""Explicitly requested, bounded reference-only diagnostic; no learning/hardware.

The user authorized local diagnostics and training for this walking-policy work.
Four first-episode trials cover both diagonal starts with nominal and 85% motors.
Reset episodes are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=48.)
parser.add_argument("--speed", type=float, default=.015)
args = parser.parse_args()
if not 8. <= args.seconds <= 60. or not .012 <= args.speed <= .018:
    parser.error("Use 8–60 seconds and 0.012–0.018 m/s")
args.output.mkdir(parents=True, exist_ok=False)
report_path = args.output / "report.json"
report = {"status": "starting", "scope": "reference only; zero learned residuals",
          "seconds": args.seconds, "speed_m_s": args.speed, "hardware_qualified": False,
          "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in Path(__file__).parent.glob("*.py")},
          "first_episode_only": True, "training_run": False, "trial_outcomes": []}
report_path.write_text(json.dumps(report, indent=2) + "\n")

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
env = None
try:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import ankle_v24_diagonal
    from ankle_v24_diagonal.env_cfg import DiagonalCfg

    cfg = DiagonalCfg()
    cfg.evaluation = True
    cfg.evaluation_speed = args.speed
    cfg.evaluation_stop_s = args.seconds - 8.
    cfg.episode_length_s = args.seconds + 2.
    cfg.scene.num_envs = 4
    cfg.seed = 2420
    # This diagnostic writes its own trace; training metric labels stay absent.
    cfg.metrics_path = ""
    env = RslRlVecEnvWrapper(gym.make("Drobot-Ankle-V24-Diagonal", cfg=cfg), clip_actions=1.)
    raw = env.unwrapped
    env.reset()
    raw.pair[:] = torch.tensor([0, 1, 0, 1], device=raw.device)
    raw.strength[:2], raw.rate_strength[:2] = 1., 1.
    raw.strength[2:], raw.rate_strength[2:] = .85, .85
    action = torch.zeros((4, 16), device=raw.device)
    finished = [False] * 4
    stages_seen = [set() for _ in range(4)]
    paired_landings = [0] * 4
    last_rows = [None] * 4
    previous_stage = [-1] * 4
    report.update(status="running", contract_revision=cfg.contract_revision,
                  reference_config={key: getattr(cfg, key) for key in
                    ("stance_forward_m", "stance_down_m", "body_forward_shift_m", "maximum_leg_reach_m", "lift_m")},
                  calibration=raw.calibration_record,
                  robot_asset_sha256=hashlib.sha256(Path(cfg.robot.spawn.usd_path).read_bytes()).hexdigest(),
                  trial_order=[{"pair": pair, "motor_fraction": strength}
                               for pair, strength in ((0, 1.), (1, 1.), (0, .85), (1, .85))])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    keys = ("pair", "stage", "stage_time_s", "gate_wait_s", "gate_values", "gate_blocked",
            "gate_blockers_seen_during_wait", "gate_ready_hold_s", "gate_failed", "parked",
            "contacts", "flat", "sole_clearance_m", "support_margin_m", "normal_force_n",
            "requested_target", "joint_target", "nominal_joint_reference", "joint_position",
            "joint_velocity", "reference_clipped", "reference_forward_m", "reference_clearance_m",
            "encoder_forward_m", "encoder_down_m", "body_angular_velocity", "body_velocity",
            "joint_effort_nm", "body_position", "reference_down_m", "first_placement", "startup_landing",
            "qualified_steps_by_leg", "qualified_touchdowns_by_leg")
    with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for step in range(round(args.seconds / raw.step_dt)):
            with torch.no_grad():
                _, _, dones, _ = env.step(action)
            d = raw.last_diagnostics
            done = dones.cpu().tolist()
            stages = d["stage"].cpu().tolist()
            events = d["paired_landing"].cpu().tolist()
            for index in range(4):
                if finished[index]:
                    continue
                stages_seen[index].add(stages[index])
                paired_landings[index] += int(events[index])
                row = {"step": step + 1, "time_s": (step + 1) * raw.step_dt, "trial": index,
                       **{key: d[key][index].cpu().tolist() for key in keys}}
                last_rows[index] = row
                if step % 6 == 0 or done[index] or stages[index] != previous_stage[index]:
                    trace.write(json.dumps(row) + "\n")
                previous_stage[index] = stages[index]
                if done[index]:
                    finished[index] = True
                    report["trial_outcomes"].append({
                        "trial": index, "result": "gate_timeout" if row["gate_failed"] else "terminated",
                        "stages_seen": sorted(stages_seen[index]), "paired_landings": paired_landings[index],
                        "last_sample": row})
            if all(finished):
                break
    for index in range(4):
        if not finished[index]:
            report["trial_outcomes"].append({"trial": index, "result": "reached_duration",
                "stages_seen": sorted(stages_seen[index]), "paired_landings": paired_landings[index],
                "last_sample": last_rows[index]})
    report.update(status="completed", simulated_steps=step + 1,
                  gate_names=raw.gate_names, stage_names=raw.stage_names, action_order=raw.action_order,
                  interpretation="Diagnostic execution completed; gait/hardware success is not implied")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("REFERENCE_DIAGNOSTIC=" + str(report_path), flush=True)
except BaseException:
    error = traceback.format_exc()
    report.update(status="failed", error=error)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    try:
        if env is not None:
            env.close()
    finally:
        app.close()
