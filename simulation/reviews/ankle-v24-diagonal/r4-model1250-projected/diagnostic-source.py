"""Bounded saved-checkpoint diagnostic; no learning or hardware.

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
import shutil
from importlib import metadata

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / "simulation/isaac/rl/ankle_v24_diagonal"

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=48.)
parser.add_argument("--speed", type=float, default=.015)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--curriculum", type=float, default=1.)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--swing-projection", action="store_true")
args = parser.parse_args()
if not 8. <= args.seconds <= 60. or not .012 <= args.speed <= .018:
    parser.error("Use 8–60 seconds and 0.012–0.018 m/s")
if not 0. <= args.curriculum <= 1.:
    parser.error("Curriculum must be in [0,1]")
contract = json.loads((args.run / "contract.json").read_text())
for relative, expected in contract["source_hashes"].items():
    if relative.endswith(".py") and hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
        raise ValueError("Training source changed: " + relative)
args.output.mkdir(parents=True, exist_ok=False)
shutil.copy2(args.checkpoint, args.output / "checkpoint.pt")
shutil.copy2(__file__, args.output / "diagnostic-source.py")
if args.swing_projection:
    shutil.copy2(Path(__file__).with_name("diagonal_swing_projection.py"), args.output / "diagonal_swing_projection.py")
shutil.copytree(args.run / "provenance", args.output / "provenance")
report_path = args.output / "report.json"
report = {"status": "starting", "scope": "saved checkpoint; stochastic" if args.stochastic else "saved checkpoint; deterministic mean",
          "seconds": args.seconds, "speed_m_s": args.speed, "hardware_qualified": False,
          "curriculum_fraction": args.curriculum, "checkpoint": str(args.checkpoint.resolve()),
          "checkpoint_sha256": hashlib.sha256((args.output / "checkpoint.pt").read_bytes()).hexdigest(),
          "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in FOLDER.glob("*.py")},
          "first_episode_only": True, "training_run": False, "trial_outcomes": []}
report_path.write_text(json.dumps(report, indent=2) + "\n")

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
env = None
try:
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    sys.path.insert(0, str(FOLDER.parent))
    import ankle_v24_diagonal
    from ankle_v24_diagonal.env_cfg import DiagonalCfg
    from ankle_v24_diagonal.ppo_cfg import DiagonalPPOCfg

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
    if args.swing_projection:
        from diagonal_swing_projection import install_swing_projection
        report["controller_intervention"] = install_swing_projection(raw)
    # Explicit diagnostic curriculum override; training source is unchanged.
    raw._curriculum = lambda: args.curriculum
    env.reset()
    raw.pair[:] = torch.tensor([0, 1, 0, 1], device=raw.device)
    raw.strength[:2], raw.rate_strength[:2] = 1., 1.
    raw.strength[2:], raw.rate_strength[2:] = .85, .85
    agent = handle_deprecated_rsl_rl_cfg(DiagonalPPOCfg(), metadata.version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
    runner.load(str(args.output / "checkpoint.pt"), load_cfg={"actor": True}, map_location=raw.device)
    policy = runner.get_inference_policy(device=raw.device)
    obs = env.get_observations()
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
            "qualified_steps_by_leg", "qualified_touchdowns_by_leg", "policy_action")
    with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for step in range(round(args.seconds / raw.step_dt)):
            with torch.no_grad():
                action = policy(obs, stochastic_output=args.stochastic)
                action_snapshot = action.detach().clone()
                obs, _, dones, _ = env.step(action)
            d = {**raw.last_diagnostics, "policy_action": action_snapshot}
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
    print("CHECKPOINT_DIAGNOSTIC=" + str(report_path), flush=True)
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
