"""Authorized training/evaluation/video entry point. No hardware interfaces."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("train", "evaluate"))
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--iterations", type=int, default=250)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--seed", type=int, default=2520)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--resume-optimizer", action="store_true")
parser.add_argument("--learning-rate", type=float)
parser.add_argument("--record", action="store_true")
parser.add_argument("--seconds", type=float, default=32.)
parser.add_argument("--pair", type=int, choices=(0, 1), default=0)
args = parser.parse_args()
if args.resume_optimizer and (args.mode != "train" or not args.checkpoint):
    parser.error("--resume-optimizer requires training from a checkpoint")
if args.learning_rate is not None and not 0. < args.learning_rate <= .001:
    parser.error("learning rate must be positive and at most 0.001")
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(Path(__file__).parent.parent))
from ankle_v24_diagonal.progress_io import best_effort_json

def save(name, value):
    best_effort_json(out/name, value)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

sources = [p for package in ("ankle_v26_feasible", "ankle_v26_balanced", "ankle_v25_pair", "ankle_v25_ik", "ankle_v24_diagonal", "ankle_v23_transfer", "ankle_v21", "ankle_v2")
           for p in (Path(__file__).parent.parent/package).glob("*.py")]
sources += [ROOT/p for p in ("hardware/robot-runtime/four-leg.toml",
    "hardware/robot-runtime/servos/calibration-ankles.json", "hardware/robot-runtime/servos/ankle-travel-observations.json",
    "hardware/test-apps/four-leg-dashboard/src/drobot_hardware_test_apps/ankle_ik_gait.py")]
for source in sources:
    target = out/"provenance"/source.relative_to(ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
contract = {"created_at": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
            "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
            "asset_sha256": digest(ROOT/"simulation/exports/isaac/ankle_v23_stride/active/robot.usdc"),
            "checkpoint_sha256": digest(args.checkpoint) if args.checkpoint else None,
            "optimizer_resumed": args.resume_optimizer,
            "learning_rate_override": args.learning_rate,
            "authorization": "User requested training, evaluation and video based on diagonal IK walking",
            "hardware_deployed": False}
save("contract.json", json.loads(json.dumps(contract, default=str)))
state = {"status": "starting", "pid": os.getpid(), "mode": args.mode, "started_unix": time.time()}
save("status.json", state)
app = env = encoder = None
try:
    import warp as wp
    wp.config.enable_backward = False
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=True, enable_cameras=args.record).app
    import gymnasium as gym
    import torch
    import numpy as np
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    import ankle_v26_feasible
    from ankle_v26_feasible.env_cfg import IKDiagonalCfg
    from ankle_v26_feasible.ppo_cfg import IKPPOCfg
    cfg = IKDiagonalCfg()
    cfg.seed = args.seed
    cfg.continuation_description = (f"checkpoint {args.checkpoint}; optimizer "
        + ("resumed" if args.resume_optimizer else "fresh")) if args.checkpoint else "fresh actor"
    cfg.scene.num_envs = args.num_envs if args.mode == "train" else (1 if args.record else 4)
    cfg.evaluation = args.mode != "train"
    cfg.evaluation_pair = args.pair
    cfg.metrics_path = str(out/"metrics.jsonl") if args.mode == "train" else ""
    if cfg.evaluation:
        cfg.evaluation_stop_s = args.seconds-8.
        cfg.episode_length_s = args.seconds+2.
    if args.record:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        cfg.video_recorder = VideoRecorderCfg(window_width=1280, window_height=720, backend_source="renderer")
        cfg.viewer.eye, cfg.viewer.lookat = (1., -1.2, .8), (0., 0., .25)
    agent = handle_deprecated_rsl_rl_cfg(IKPPOCfg(), metadata.version("rsl-rl-lib"))
    agent.seed = args.seed
    agent.max_iterations = args.iterations
    if args.learning_rate is not None:
        agent.algorithm.learning_rate = args.learning_rate
    save("agent.json", agent.to_dict())
    env = RslRlVecEnvWrapper(gym.make("Drobot-Ankle-V26-Feasible", cfg=cfg,
            render_mode="rgb_array" if args.record else None), clip_actions=1.)
    raw = env.unwrapped
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(out/"checkpoints") if args.mode == "train" else None,
                            device=agent.device)
    if args.checkpoint:
        runner.load(str(args.checkpoint.resolve()), load_cfg={
            "actor": True, "critic": True, "optimizer": args.resume_optimizer, "iteration": True})
        # Loading optimizer state restores its previous rate. Apply an explicit
        # continuation rate to both PPO's reported rate and actual groups.
        if args.learning_rate is not None:
            runner.alg.learning_rate = args.learning_rate
            for group in runner.alg.optimizer.param_groups:
                group["lr"] = args.learning_rate
        if args.mode == "train":
            runner.current_learning_iteration += 1
    elif args.mode == "train":
        linear = [m for m in runner.alg._raw_actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.zero_()
    state.update(status="running", started_simulation_unix=time.time(),
                 start_iteration=runner.current_learning_iteration)
    save("status.json", state)
    if args.mode == "train":
        original_log = runner.logger.log
        def logged(**kwargs):
            original_log(**kwargs)
            state.update(last_iteration=kwargs["it"], updated_unix=time.time())
            if kwargs["it"] % 10 == 0:
                save("status.json", state)
            policy = runner.alg.get_policy()
            row = {"iteration": kwargs["it"], "unix": time.time(),
                   "learning_rate": float(runner.alg.learning_rate),
                   "action_std": float(policy.output_std.mean().item()),
                   "loss": {k: float(v) for k,v in kwargs["loss_dict"].items()}}
            with (out/"optimizer.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row)+"\n")
        runner.logger.log = logged
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        final = out/"checkpoints"/f"model_{runner.current_learning_iteration}.pt"
        if not final.is_file():
            raise RuntimeError("Training returned without expected final checkpoint")
        state.update(status="completed", checkpoint=str(final), checkpoint_sha256=digest(final),
                     training_returned=True, finished_unix=time.time())
        save("status.json", state)
        save("completion.json", state)
    else:
        policy = runner.get_inference_policy(device=raw.device) if args.checkpoint else None
        obs, _ = env.reset()
        strengths = torch.tensor([1.] if args.record else [1., 1., .85, .85], device=raw.device)
        raw.strength[:] = strengths[:, None]
        raw.rate_strength[:] = strengths[:, None]
        n = raw.num_envs
        active = np.ones(n, dtype=bool)
        origin = raw.robot.data.root_pos_w.torch.cpu().numpy().copy()
        last = origin.copy()
        steps = np.zeros((n, 4)); touchdowns = steps.copy(); peaks = steps.copy()
        tilt_sum = np.zeros(n); tilt_max = np.zeros(n); samples = np.zeros(n)
        outcomes = [None]*n
        trace = (out/"trace.jsonl").open("w", encoding="utf-8")
        events = (out/"touchdown-events.jsonl").open("w", encoding="utf-8")
        if args.record:
            from ankle_v25_ik.video import ReviewVideo
            encoder = ReviewVideo(raw, out/"walk.mp4", "Trained PPO" if policy else "IK reference only")
        for step in range(round(args.seconds/raw.step_dt)):
            with torch.no_grad():
                action = policy(obs) if policy else torch.zeros((n, 16), device=raw.device)
                obs, _, dones, _ = env.step(action)
            d = {k: v.detach().cpu().numpy() for k,v in raw.last_diagnostics.items()}
            done = dones.cpu().numpy().astype(bool)
            for i, leg in np.argwhere(d["foot_touchdown"] & active[:, None]):
                events.write(json.dumps({"time_s": (step+1)*raw.step_dt, "environment": int(i), "leg": int(leg),
                    "pair": int(d["pair"][i]), "stage": int(d["stage"][i]),
                    **{k: d[k][i, leg].item() for k in ("flight_forward_m", "flight_sideways_m", "flight_air_time_s", "flight_peak_m", "flight_was_selected", "flight_support_good")},
                    "required_stride_m": float(d["flight_stride_required_m"][i]),
                    "support_margin_m": float(d["support_margin_m"][i])})+"\n")
            last[active] = d["body_position"][active]
            steps[active] = np.maximum(steps[active], d["qualified_steps_by_leg"][active])
            touchdowns[active] = np.maximum(touchdowns[active], d["qualified_touchdowns_by_leg"][active])
            swing = (d["stage"] >= 1) & (d["stage"] <= 3)
            swing_legs = np.asarray([[1, 0, 0, 1], [0, 1, 1, 0]], dtype=bool)[d["pair"]]
            peaks[active] = np.maximum(peaks[active],
                    np.where(swing[:, None] & swing_legs, d["sole_clearance_m"], 0.)[active])
            tilt_sum += d["tilt_rad"]**2*active
            tilt_max = np.maximum(tilt_max, d["tilt_rad"]*active)
            samples += active
            if step % 6 == 0 or (done & active).any():
                trace.write(json.dumps({"time_s": (step+1)*raw.step_dt, "active": active.tolist(),
                        **{k:v.tolist() for k,v in d.items()}})+"\n")
            for i in np.flatnonzero(done & active):
                outcomes[i] = {"terminated_s": (step+1)*raw.step_dt, "failed": True}
            if encoder and active[0]:
                if done[0]:
                    encoder.failure("Episode terminated; reset frames excluded")
                elif step % 2 == 1:
                    encoder.capture((step+1)*raw.step_dt, {k:v[0] for k,v in d.items()}, last[0]-origin[0])
            active &= ~done
            if not active.any():
                break
        trace.close()
        events.close()
        if encoder:
            encoder.close()
            encoder = None
        trials = []
        for i in range(n):
            rms = float(np.sqrt(tilt_sum[i]/max(samples[i],1))*180/np.pi)
            trials.append({"environment": i, "start_pair": args.pair if args.record else i%2,
                           "motor_strength": float(strengths[i].item()),
                           "outcome": outcomes[i] or {"failed": False, "duration_s": args.seconds},
                           "forward_m": float(last[i,0]-origin[i,0]), "lateral_m": float(last[i,1]-origin[i,1]),
                           "tilt_rms_deg": rms, "tilt_max_deg": float(tilt_max[i]*180/np.pi),
                           "full_steps_by_leg_FL_RL_FR_RR": steps[i].tolist(),
                           "qualified_touchdowns_by_leg": touchdowns[i].tolist(),
                           "peak_swing_clearance_m_by_leg": peaks[i].tolist(),
                           "qualified": bool(active[i] and steps[i].min() >= 2 and last[i,0]-origin[i,0] >= .10
                                             and abs(last[i,1]-origin[i,1]) < .05 and rms < 10.
                                             and tilt_max[i]*180/np.pi < 20.)})
        report = {"status": "completed", "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                  "reference_only": policy is None, "first_episodes_only": True,
                  "all_trials_qualified": all(x["qualified"] for x in trials),
                  "hardware_qualified": False, "trials": trials,
                  "video": str(out/"walk.mp4") if args.record else None}
        save("report.json", report)
        state.update(status="completed", finished_unix=time.time(), report=str(out/"report.json"))
        save("status.json", state)
except BaseException:
    error = traceback.format_exc()
    (out/"error.txt").write_text(error, encoding="utf-8")
    state.update(status="failed", error=error, finished_unix=time.time())
    save("status.json", state)
    print(error, file=sys.stderr, flush=True)
    raise
finally:
    if encoder:
        encoder.close()
    if env:
        env.unwrapped.write_diagnostic_summary()
        env.close()
    if app:
        app.close()

