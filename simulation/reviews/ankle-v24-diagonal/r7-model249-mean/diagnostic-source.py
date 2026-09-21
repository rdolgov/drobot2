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
parser.add_argument("--support-compression", action="store_true")
parser.add_argument("--load-stance", action="store_true")
parser.add_argument("--zero-actions", action="store_true")
parser.add_argument("--attitude-reference", action="store_true")
parser.add_argument("--ankle-posture", action="store_true")
parser.add_argument("--support-plane", action="store_true")
parser.add_argument("--rear-first", action="store_true")
parser.add_argument("--support-preload", action="store_true")
parser.add_argument("--candidate-source", type=Path)
args = parser.parse_args()
if not 8. <= args.seconds <= 60. or not .012 <= args.speed <= .018:
    parser.error("Use 8–60 seconds and 0.012–0.018 m/s")
if not 0. <= args.curriculum <= 1.:
    parser.error("Curriculum must be in [0,1]")
if args.support_compression and not args.swing_projection:
    parser.error("Support compression requires swing projection")
if args.support_plane and any((args.swing_projection, args.load_stance, args.attitude_reference, args.ankle_posture, args.candidate_source)):
    parser.error("Support-plane diagnostic must be isolated from other prototypes")
if args.rear_first and any((args.swing_projection, args.load_stance, args.attitude_reference, args.ankle_posture, args.support_plane, args.candidate_source)):
    parser.error("Rear-first diagnostic must be isolated from other prototypes")
if args.support_preload and any((args.swing_projection, args.load_stance, args.attitude_reference, args.ankle_posture, args.support_plane, args.rear_first, args.candidate_source)):
    parser.error("Support-preload diagnostic must be isolated from other prototypes")
contract = json.loads((args.run / "contract.json").read_text())
source_folder = args.candidate_source.resolve() if args.candidate_source else FOLDER
if not (source_folder / "env.py").is_file() or source_folder.name != "ankle_v24_diagonal":
    parser.error("Candidate source must be an ankle_v24_diagonal package")
for relative, expected in contract["source_hashes"].items():
    if relative.endswith(".py") and hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
        raise ValueError("Training source changed: " + relative)
args.output.mkdir(parents=True, exist_ok=False)
shutil.copy2(args.checkpoint, args.output / "checkpoint.pt")
shutil.copy2(__file__, args.output / "diagnostic-source.py")
if args.swing_projection:
    shutil.copy2(Path(__file__).with_name("diagonal_swing_projection.py"), args.output / "diagonal_swing_projection.py")
if args.load_stance:
    shutil.copy2(Path(__file__).with_name("diagonal_load_reference.py"), args.output / "diagonal_load_reference.py")
if args.attitude_reference:
    shutil.copy2(Path(__file__).with_name("diagonal_attitude_reference.py"), args.output / "diagonal_attitude_reference.py")
if args.ankle_posture:
    shutil.copy2(Path(__file__).with_name("diagonal_ankle_posture.py"), args.output / "diagonal_ankle_posture.py")
if args.support_plane:
    shutil.copy2(Path(__file__).with_name("diagonal_support_plane.py"), args.output / "diagonal_support_plane.py")
if args.rear_first:
    shutil.copy2(Path(__file__).with_name("diagonal_rear_first.py"), args.output / "diagonal_rear_first.py")
if args.support_preload:
    for name in ("diagonal_support_preload.py", "diagonal_static_load.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    shutil.copy2(ROOT / "simulation/exports/isaac/ankle_v23_stride/active/training_input.urdf", args.output / "preload-training-input.urdf")
shutil.copytree(args.run / "provenance", args.output / "provenance")
if args.candidate_source:
    shutil.copytree(source_folder, args.output / "candidate-source")
report_path = args.output / "report.json"
report = {"status": "starting", "scope": "saved checkpoint; stochastic" if args.stochastic else "saved checkpoint; deterministic mean",
          "seconds": args.seconds, "speed_m_s": args.speed, "hardware_qualified": False,
          "curriculum_fraction": args.curriculum, "checkpoint": str(args.checkpoint.resolve()),
          "checkpoint_sha256": hashlib.sha256((args.output / "checkpoint.pt").read_bytes()).hexdigest(),
          "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in source_folder.glob("*.py")},
          "candidate_source": str(source_folder) if args.candidate_source else None,
          "first_episode_only": True, "training_run": False, "trial_outcomes": []}
report["actor_used"] = not args.zero_actions
if args.zero_actions:
    report["scope"] = "reference only; copied checkpoint is not used"
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
    if args.candidate_source:
        sys.path.insert(0, str(source_folder.parent))
    import ankle_v24_diagonal
    from ankle_v24_diagonal.env_cfg import DiagonalCfg
    from ankle_v24_diagonal.ppo_cfg import DiagonalPPOCfg

    cfg = DiagonalCfg()
    if args.load_stance:
        from diagonal_load_reference import configure_load_stance, install_load_stance
        configure_load_stance(cfg)
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
    # Capture physical link geometry before auto-reset mutates the live state.
    # These additional fields are diagnostics only, never actor inputs.
    original_rewards = raw._get_rewards
    hip_bodies = [raw.robot.body_names.index(leg + "_proximal_link")
                  for leg in ("front_left", "rear_left", "front_right", "rear_right")]

    def rewards_with_geometry():
        reward = original_rewards()
        _, sole, _, high = raw._geometry()
        physical = {
            "physical_fall": raw.fallen,
            "body_quaternion_xyzw": raw.robot.data.root_quat_w.torch,
            "projected_gravity_body": raw.robot.data.projected_gravity_b.torch,
            "sole_high_m": high, "sole_center_world": sole,
            "hip_pivot_world": raw.robot.data.body_link_pos_w.torch[:, hip_bodies],
            "ankle_pivot_world": raw.robot.data.body_link_pos_w.torch[:, raw.feet],
            "foot_quaternion_xyzw": raw.robot.data.body_link_quat_w.torch[:, raw.feet],
        }
        raw.last_diagnostics.update({key: value.detach().clone() for key, value in physical.items()})
        if args.support_plane:
            raw.last_diagnostics.update({key: value.detach().clone() for key, value in raw.support_plane_diagnostics.items()})
        if args.support_preload:
            raw.last_diagnostics.update({key: value.detach().clone() for key, value in raw.support_preload_diagnostics.items()})
        return reward

    raw._get_rewards = rewards_with_geometry
    if args.load_stance:
        report["reference_intervention"] = install_load_stance(raw)
    if args.attitude_reference:
        from diagonal_attitude_reference import install_attitude_reference
        report["attitude_intervention"] = install_attitude_reference(raw)
    if args.ankle_posture:
        from diagonal_ankle_posture import install_ankle_posture
        report["ankle_posture_intervention"] = install_ankle_posture(raw)
    if args.support_plane:
        from diagonal_support_plane import install_support_plane
        report["support_plane_intervention"] = install_support_plane(raw)
    if args.rear_first:
        from diagonal_rear_first import install_rear_first
        report["rear_first_intervention"] = install_rear_first(raw)
    if args.support_preload:
        from diagonal_support_preload import install_support_preload
        report["support_preload_intervention"] = install_support_preload(raw)
    if args.swing_projection:
        from diagonal_swing_projection import install_swing_projection
        report["controller_intervention"] = install_swing_projection(raw, args.support_compression)
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
    keys += ("body_quaternion_xyzw", "projected_gravity_body", "sole_high_m",
             "sole_center_world", "hip_pivot_world", "ankle_pivot_world", "foot_quaternion_xyzw", "physical_fall")
    if args.support_plane:
        keys += ("estimated_body_height_m", "extra_swing_lift_m", "nominal_estimated_sole_clearance_m")
    if args.support_preload:
        keys += ("support_preload_rad", "support_preload_static_peak_nm", "support_preload_solver_valid")
    with (args.output / "trace.jsonl").open("w", encoding="utf-8") as trace:
        for step in range(round(args.seconds / raw.step_dt)):
            with torch.no_grad():
                action = torch.zeros((4, 16), device=raw.device) if args.zero_actions else policy(obs, stochastic_output=args.stochastic)
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
