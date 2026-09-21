"""Bounded optimizer recovery experiment with the unchanged revision-4 robot.

Restore a pre-collapse actor/critic, use fresh Adam at fixed 1e-4, and record
Beta saturation on actual update minibatches. This does not deploy a policy.
"""
import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--base-run", type=Path, required=True)
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
# Exclusive creation prevents duplicate workers sharing an output directory.
with (out / "pilot-claim.json").open("x", encoding="utf-8") as stream:
    json.dump({"pid": os.getpid(), "started_unix": time.time()}, stream)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")

state = {"status": "starting", "worker_pid": os.getpid(),
         "started_unix": time.time(), "last_iteration": -1,
         "training_returned": False, "deployed": False}
write(out / "status.json", state)
app = env = hook = None
try:
    base = json.loads((args.base_run / "contract.json").read_text())
    for relative, expected in base["source_hashes"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError("Base source changed: " + relative)
    if digest(Path(base["asset"])) != base["asset_sha256"]:
        raise ValueError("Robot asset changed")
    warm = args.checkpoint.resolve()
    expected_warm = "b9165cb4d43e6fe4cf41752378f2aa1728dd8201a654fbe237b3f879ffc626c9"
    if digest(warm) != expected_warm:
        raise ValueError("Pilot requires the reviewed R6 checkpoint 1500")
    shutil.copytree(args.base_run / "provenance", out / "provenance")
    relative = Path(__file__).resolve().relative_to(ROOT)
    snapshot = out / "provenance" / relative
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, snapshot)
    shutil.copy2(warm, out / "initial-checkpoint.pt")
    contract = {**base, "seed": 2421, "iterations": 250, "num_envs": 256,
                "planned_transitions": 250 * 256 * 64,
                "run_name": "diagonal-optimizer-pilot-2421-r7",
                "base_run": str(args.base_run.resolve()), "warm_start": str(warm),
                "warm_start_sha256": expected_warm,
                "restore": "actor, critic and their normalizers; fresh optimizer and iteration",
                "overrides": {"learning_rate": .0001, "schedule": "fixed",
                              "curriculum_fraction": 1., "save_interval": 50},
                "collapse_stop": {"both_beta_parameters_exactly_one_fraction": .95,
                                  "consecutive_iterations_last_update_minibatch": 10},
                "qualification": "independent matching-controller evaluation still required"}
    contract["source_hashes"] = {**base["source_hashes"], relative.as_posix(): digest(Path(__file__))}
    write(out / "contract.json", contract)

    import warp as wp
    wp.config.enable_backward = False
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    app = launcher.app
    import gymnasium as gym
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from rsl_rl.runners import OnPolicyRunner
    sys.path.insert(0, str(ROOT / "simulation/isaac/rl"))
    import ankle_v24_diagonal
    from ankle_v24_diagonal.env_cfg import DiagonalCfg
    from ankle_v24_diagonal.ppo_cfg import DiagonalPPOCfg
    from ankle_v24_diagonal.progress_io import best_effort_json

    cfg = DiagonalCfg()
    cfg.scene.num_envs = 256
    cfg.seed = 2421
    cfg.metrics_path = str(out / "train.metrics.jsonl")
    agent = handle_deprecated_rsl_rl_cfg(DiagonalPPOCfg(), metadata.version("rsl-rl-lib"))
    agent.seed = 2421
    agent.max_iterations = 250
    agent.save_interval = 50
    agent.run_name = contract["run_name"]
    agent.algorithm.learning_rate = .0001
    agent.algorithm.schedule = "fixed"
    folder = ROOT / "logs/rsl_rl" / agent.experiment_name / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + agent.run_name)
    folder.mkdir(parents=True, exist_ok=False)
    write(out / "run-path.json", {"directory": str(folder)})
    write(out / "effective-agent-config.json", agent.to_dict())
    env = RslRlVecEnvWrapper(gym.make("Drobot-Ankle-V24-Diagonal", cfg=cfg), clip_actions=1.)
    raw = env.unwrapped
    raw._curriculum = lambda: 1.
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(folder), device=agent.device)
    runner.load(str(out / "initial-checkpoint.pt"), load_cfg={
        "actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False})
    state.update(status="running", checkpoint_directory=str(folder))
    best_effort_json(out / "status.json", state)
    latest = {}
    def capture_head(module, inputs, output):
        latest["raw"] = output.detach()
    hook = runner.alg._raw_actor.mlp.register_forward_hook(capture_head)
    original_log = runner.logger.log
    consecutive = 0
    def log_health(**kwargs):
        global consecutive
        original_log(**kwargs)
        policy = runner.alg.get_policy()
        alpha, beta = policy.distribution.params
        fraction = float(((alpha == 1.) & (beta == 1.)).float().mean().item())
        consecutive = consecutive + 1 if fraction >= .95 else 0
        head = latest["raw"]
        row = {"iteration": kwargs["it"], "unix": time.time(),
               "sample": "last actual PPO update minibatch, before its optimizer step",
               "learning_rate": float(runner.alg.learning_rate),
               "uniform_action_fraction": fraction,
               "consecutive_collapsed_updates": consecutive,
               "mean_abs_action_mean": float(policy.distribution.mean.detach().abs().mean().item()),
               "action_std_mean": float(policy.output_std.detach().mean().item()),
               "raw_head_min": float(head.min().item()), "raw_head_max": float(head.max().item()),
               "loss": {key: float(value) for key, value in kwargs["loss_dict"].items()}}
        with (out / "optimizer-health.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        state.update(last_iteration=kwargs["it"], updated_unix=time.time(), optimizer_health=row)
        best_effort_json(out / "status.json", state)
        if not torch.isfinite(head).all().item():
            raise RuntimeError("Nonfinite policy head; pilot stopped")
        if consecutive >= 10:
            runner.save(str(folder / f"model_{kwargs['it']}_collapse.pt"))
            raise RuntimeError("Uniform Beta collapse persisted for 10 updates; pilot stopped")
    runner.logger.log = log_health
    runner.learn(num_learning_iterations=250, init_at_random_ep_len=False)
    final = folder / "model_249.pt"
    if state["last_iteration"] != 249 or not final.is_file():
        raise RuntimeError("Training returned without its final iteration/checkpoint")
    state.update(status="completed", training_returned=True, finished_unix=time.time(),
                 checkpoint=str(final), checkpoint_sha256=digest(final))
    best_effort_json(out / "status.json", state)
    write(out / "completion.json", state)
    print(f"Training time: {time.time() - state['started_unix']:.2f} seconds", flush=True)
except BaseException:
    failure = traceback.format_exc()
    (out / "training-error.txt").write_text(failure, encoding="utf-8")
    state.update(status="failed", finished_unix=time.time(), error=failure)
    write(out / "failure.json", state)
    try:
        write(out / "status.json", state)
    except OSError:
        pass
    print(failure, file=sys.stderr, flush=True)
    raise
finally:
    if hook is not None:
        hook.remove()
    try:
        if env is not None:
            env.unwrapped.write_diagnostic_summary()
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if app is not None:
                app.close()
