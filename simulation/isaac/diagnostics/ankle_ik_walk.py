"""Historical V1 IK flat-floor diagnostic and recording.

No RL, hardware, state corrections during walking, or changed torque limits.
Six first episodes: crawl and both diagonal starts, nominal and 85% motors.
Uses [ankle_ik] for all trials, without IMU/encoder ankle leveling. It does
not evaluate the newer [ankle_ik_diagonal] V2 runtime controller.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD = ROOT / "hardware/test-apps/four-leg-dashboard/src"
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=60.)
parser.add_argument("--record", action="store_true", help="Record nominal crawl and both diagonal starts")
args = parser.parse_args()
if not 40. <= args.seconds <= 90.:
    parser.error("Use 40–90 seconds")
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(DASHBOARD))
from drobot_hardware_test_apps.ankle_ik_gait import AnkleIKConfig, AnkleIKClock, ankle_ik_gait_degrees
import tomllib
manifest = ROOT / "hardware/robot-runtime/four-leg.toml"
config = AnkleIKConfig(**tomllib.loads(manifest.read_text())["ankle_ik"])
trials = [(mode, start, strength) for mode, start in
          (("distributed", 0), ("diagonal_pair", 0), ("diagonal_pair", 1)) for strength in (1., .85)]
sources = [Path(__file__), manifest,
           DASHBOARD / "drobot_hardware_test_apps/ankle_ik_gait.py",
           DASHBOARD / "drobot_hardware_test_apps/four_leg_control.py"]
sources += list((ROOT / "simulation/isaac/rl/ankle_v24_diagonal").glob("*.py"))
if args.record:
    sources.append(Path(__file__).with_name("walk_video.py"))
for path in sources:
    dest = out / "provenance" / path.relative_to(ROOT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
report = {"status": "starting", "scope": "historical V1 flat-ground IK trial; no RL; not diagonal V2",
          "config": vars(config), "trial_definitions": trials, "seconds": args.seconds,
          "stop_command_s": args.seconds-12., "mass_kg": 3.39514659,
          "mass_method": "uniform mass/inertia sensitivity scaling; same COM distribution",
          "source_hashes": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
          "qualification": "air >=0.1s, clearance >=8mm, forward advance >=65% stride, sideways <35mm; per-leg counts; no automatic hardware qualification",
          "hardware_qualified": False, "outcomes": []}
def save():
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
save()
import warp as wp
wp.config.enable_backward = False
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=args.record)
app = launcher.app
env = None
videos = {}
try:
    import gymnasium as gym
    import torch
    sys.path.insert(0, str(ROOT / "simulation/isaac/rl"))
    import ankle_v24_diagonal
    from ankle_v24_diagonal.env_cfg import DiagonalCfg
    cfg = DiagonalCfg()
    cfg.scene.num_envs = len(trials)
    cfg.seed = 2422
    cfg.evaluation = True
    cfg.metrics_path = ""
    cfg.episode_length_s = args.seconds+5.
    cfg.stance_down_m = config.down_m
    cfg.target_base_height_m = config.down_m + .088908389
    if args.record:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        cfg.video_recorder = VideoRecorderCfg(window_width=960, window_height=540, backend_source="renderer")
        cfg.viewer.eye = (1.15, -1.4, .90)
        cfg.viewer.lookat = (0., 0., .25)
    env = gym.make("Drobot-Ankle-V24-Diagonal", cfg=cfg, render_mode="rgb_array" if args.record else None)
    raw = env.unwrapped
    n = len(trials)
    clocks = [AnkleIKClock(config, mode) for mode, _, _ in trials]
    for clock, (_, start, _) in zip(clocks, trials):
        clock.time_s = start * config.period_s/2.
    ordered_legs = ("front_left", "rear_left", "front_right", "rear_right")
    action_keys = [(leg, joint) for joint in ("hip_abduction", "hip_flexion", "knee", "ankle_pitch") for leg in ordered_legs]
    desired = []
    for clock in clocks:
        targets, _ = ankle_ik_gait_degrees(clock.time_s, config, clock.mode)
        desired.append([math.radians(targets[key]) for key in action_keys])
    initial = torch.tensor(desired, device=raw.device)
    env.reset()
    body_ids = torch.arange(len(raw.robot.body_names), dtype=torch.int32, device=raw.device)
    env_ids = torch.arange(n, dtype=torch.int32, device=raw.device)
    masses = raw.robot.data.body_mass.torch.clone()
    factor = 3.39514659/masses.sum(1)
    raw.robot.set_masses_index(masses=masses*factor[:, None], body_ids=body_ids, env_ids=env_ids)
    raw.robot.set_inertias_index(inertias=raw.robot.data.body_inertia.torch.clone()*factor[:, None, None], body_ids=body_ids, env_ids=env_ids)
    report["actual_mass_kg"] = raw.robot.data.body_mass.torch.sum(1).cpu().tolist()
    native = raw.robot.data.default_joint_pos.torch.clone()
    native[:, raw.ids] = initial
    raw.robot.write_joint_position_to_sim_index(position=native, env_ids=env_ids)
    raw.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(native), env_ids=env_ids)
    raw.targets.copy_(initial)
    strength = torch.tensor([s for _, _, s in trials], device=raw.device)
    raw.strength[:] = strength[:, None]
    raw.rate_strength[:] = strength[:, None]
    rates = torch.tensor([60.]*4+[100.]*12, device=raw.device)*math.pi/180.
    failed = [None]*n
    sim_time = 0.
    stopped_at = [None]*n
    feedback_time = None
    feedback_positions = feedback_velocities = None
    def pre_step(actions):
        global feedback_time, feedback_positions, feedback_velocities
        if feedback_time is None or sim_time-feedback_time >= .05-1e-9:
            current = raw.robot.data.joint_pos.torch[:, raw.ids].cpu().tolist()
            feedback_velocities = ([[0.]*16 for _ in trials] if feedback_time is None else
                [[(q-p)/(sim_time-feedback_time) for q, p in zip(row, old)]
                 for row, old in zip(current, feedback_positions)])
            feedback_positions, feedback_time = current, sim_time
        measured, velocities = feedback_positions, feedback_velocities
        delivered = raw.targets.cpu().tolist()
        requested = raw.targets.clone()
        for i, clock in enumerate(clocks):
            if failed[i] is not None or sim_time < 2.5:
                continue
            if sim_time >= args.seconds-12.:
                clock.stop_requested = True
            try:
                pose = clock.tick(raw.step_dt,
                    {k: math.degrees(v) for k, v in zip(action_keys, measured[i])},
                    {k: math.degrees(v) for k, v in zip(action_keys, velocities[i])},
                    {k: math.degrees(v) for k, v in zip(action_keys, delivered[i])})
                requested[i] = torch.tensor([math.radians(pose[k]) for k in action_keys], device=raw.device)
                if ((requested[i] < raw.limits[i, :, 0]) | (requested[i] > raw.limits[i, :, 1])).any():
                    raise RuntimeError("IK target exceeds unchanged physical joint limits")
                if clock.stopped and stopped_at[i] is None:
                    stopped_at[i] = sim_time
            except (ValueError, RuntimeError) as error:
                failed[i] = str(error)
        cap = torch.minimum(rates[None, :]*strength[:, None]*raw.step_dt,
                            torch.full_like(raw.targets, math.radians(2.)))
        raw.targets += (requested-raw.targets).clamp(-cap, cap)
    raw._pre_physics_step = pre_step
    raw._get_observations = lambda: {"policy": torch.zeros((n, 82), device=raw.device), "critic": torch.zeros((n, 101), device=raw.device)}
    raw._get_rewards = lambda: torch.zeros(n, device=raw.device)
    raw._get_dones = lambda: (torch.zeros(n, dtype=torch.bool, device=raw.device), torch.zeros(n, dtype=torch.bool, device=raw.device))
    previous_contact = [[True]*4 for _ in trials]
    liftoff = [[None]*4 for _ in trials]
    air = [[0.]*4 for _ in trials]
    peaks = [[0.]*4 for _ in trials]
    qualified = [[0]*4 for _ in trials]
    finished = [False]*n
    origins = raw.scene.env_origins.cpu().tolist()
    start_positions = None
    stop_speeds = [[] for _ in trials]
    report["status"] = "running"
    if args.record:
        from walk_video import WalkVideo
        videos = {i: WalkVideo(raw, out, i, title) for i, title in
                  ((0, "CRAWL / rear-right lift"), (2, "DIAGONAL / FL + RR"), (4, "DIAGONAL / FR + RL"))}
        for _ in range(24):
            raw.render()
    save()
    with (out / "trace.jsonl").open("w") as trace:
        for step in range(round(args.seconds/raw.step_dt)):
            sim_time = step*raw.step_dt
            with torch.no_grad():
                env.step(torch.zeros((n, 16), device=raw.device))
            _, sole_t, low_t, high_t = raw._geometry()
            soles, lows, highs = sole_t.cpu().tolist(), low_t.cpu().tolist(), high_t.cpu().tolist()
            forces = raw.contacts.data.net_forces_w.torch[:, raw.foot_sensors, 2].cpu().tolist()
            positions = raw.robot.data.root_pos_w.torch.cpu().tolist()
            gravity = raw.robot.data.projected_gravity_b.torch.cpu().tolist()
            speed = raw.robot.data.root_lin_vel_w.torch[:, :2].norm(dim=1).cpu().tolist()
            joints = raw.robot.data.joint_pos.torch[:, raw.ids].cpu().tolist()
            efforts = raw.robot.data.applied_torque.torch[:, raw.ids].cpu().tolist()
            if sim_time >= 2.5 and start_positions is None:
                start_positions = positions
            for i, clock in enumerate(clocks):
                if finished[i]:
                    continue
                _, state = ankle_ik_gait_degrees(clock.time_s, config, clock.mode)
                if sim_time >= 2.5:
                    for leg in range(4):
                        contact = forces[i][leg] > 1.
                        if not contact:
                            if previous_contact[i][leg]:
                                liftoff[i][leg] = soles[i][leg][:]
                                air[i][leg] = peaks[i][leg] = 0.
                            air[i][leg] += raw.step_dt
                            peaks[i][leg] = max(peaks[i][leg], lows[i][leg])
                        elif not previous_contact[i][leg] and liftoff[i][leg] is not None:
                            forward = soles[i][leg][0]-liftoff[i][leg][0]
                            lateral = abs(soles[i][leg][1]-liftoff[i][leg][1])
                            if (air[i][leg] >= .1 and peaks[i][leg] >= .008
                                and .65*config.stride_m <= forward <= 1.8*config.stride_m
                                and lateral < .035 and highs[i][leg]-lows[i][leg] < .008
                                and state["phase"] in ("lower", "firm_plant", "all_feet_push")):
                                qualified[i][leg] += 1
                        previous_contact[i][leg] = contact
                tilt = math.degrees(math.asin(min(1., math.hypot(*gravity[i][:2]))))
                if tilt > 20. or positions[i][2]-origins[i][2] < .20:
                    failed[i] = failed[i] or "physical fall/tilt limit"
                if stopped_at[i] is not None:
                    stop_speeds[i].append(speed[i])
                row = {"step": step+1, "time_s": sim_time+raw.step_dt, "trial": i,
                       "trajectory_time_s": clock.time_s, "mode": clock.mode, **state,
                       "gate": clock.gate, "gate_wait_s": clock.wait_s,
                       "root_position": positions[i], "tilt_deg": tilt,
                       "sole_clearance_m_FL_RL_FR_RR": lows[i], "sole_high_m": highs[i],
                       "sole_world": soles[i], "normal_force_n": forces[i],
                       "joint_position": joints[i], "joint_target": raw.targets[i].cpu().tolist(),
                       "joint_effort_nm": efforts[i], "qualified_steps_FL_RL_FR_RR": qualified[i][:],
                       "failure": failed[i], "stopped": clock.stopped}
                if i in videos and (step % 2 == 1 or failed[i] is not None):
                    videos[i].capture(row)
                if step % 6 == 0 or failed[i] is not None or step == round(args.seconds/raw.step_dt)-1:
                    trace.write(json.dumps(row)+"\n")
                if failed[i] is not None or step == round(args.seconds/raw.step_dt)-1:
                    report["outcomes"].append({"trial": i, "mode": clock.mode,
                        "result": "failed" if failed[i] else "reached_duration", "failure": failed[i],
                        "qualified_steps_FL_RL_FR_RR": qualified[i][:], "stopped_at_s": stopped_at[i],
                        "mean_stopped_speed_m_s": sum(stop_speeds[i])/len(stop_speeds[i]) if stop_speeds[i] else None,
                        "forward_displacement_m": positions[i][0]-start_positions[i][0] if start_positions else 0.,
                        "last_sample": row})
                    finished[i] = True
                    if i in videos:
                        videos[i].finish()
            if all(finished):
                break
    report["status"] = "completed"
    report["videos"] = [v.info() for v in videos.values()]
    save()
    print("IK_DIAGNOSTIC="+str(out / "report.json"), flush=True)
except BaseException:
    report.update(status="failed", error=traceback.format_exc())
    save()
    print(report["error"], file=sys.stderr, flush=True)
    raise
finally:
    try:
        for video in videos.values():
            video.finish()
        if env is not None:
            env.close()
    finally:
        app.close()
