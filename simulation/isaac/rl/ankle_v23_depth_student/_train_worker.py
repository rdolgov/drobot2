"""One authorized training job. No evaluation or recording is launched."""
import argparse
import copy
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import sys
import time
import warp as wp

wp.config.enable_backward = False
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Drobot-Ankle-V23-DepthStudent")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=2501)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--run_name", required=True)
parser.add_argument("--rise", type=float, default=.02)
parser.add_argument("--warm-start")
parser.add_argument("--strategy", default="free")
parser.add_argument("--guidance", type=int, choices=(0, 1), default=1)
parser.add_argument("--phases", type=int, choices=(1, 2, 4, 6, 8, 10, 14, 18, 22, 26), default=6)
parser.add_argument("--teacher", type=int, choices=(0,1), default=0)
args, overrides = parser.parse_known_args()
metrics = next((v.split("=", 1)[1] for v in overrides if v.startswith("env.metrics_path=")), None)
if not metrics or len(overrides) != 1:
    raise ValueError("Supply exactly one env.metrics_path=... and no unsupported options")
if not .005 <= args.rise <= .18:
    raise ValueError("Stair rise must be between 5 and 180 mm")
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import DistillationRunner
from isaaclab.utils.io import dump_yaml
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ankle_v23_depth_student
from ankle_v23_depth_student.env_cfg import DepthStudentCfg
from ankle_v23_depth_student.ppo_cfg import DepthStudentPPOCfg

root = Path(__file__).resolve().parents[4]
cfg = DepthStudentCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.metrics_path = metrics
cfg.stair_rise_m = args.rise
cfg.strategy = args.strategy
cfg.guidance = bool(args.guidance)
cfg.phase_count = args.phases
cfg.episode_length_s = cfg.phase_count*cfg.phase_s+4.
if args.teacher or not args.warm_start:
    raise ValueError('Student training requires --teacher 0 and --warm-start TEACHER_CHECKPOINT')
cfg.privileged_actor = False
cfg.observation_space = 508 if cfg.privileged_actor else 485
agent = handle_deprecated_rsl_rl_cfg(DepthStudentPPOCfg(), metadata.version("rsl-rl-lib"))
agent.seed, agent.max_iterations, agent.run_name = args.seed, args.max_iterations, args.run_name
folder = root / "logs/rsl_rl" / agent.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
folder.mkdir(parents=True, exist_ok=False)
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg), clip_actions=1.)
# Use the installed recurrent distillation algorithm; the student acts during
# collection and the frozen privileged teacher only supplies supervised labels.
run_cfg = agent.to_dict()
actor_cfg = run_cfg.pop('actor')
run_cfg.pop('critic')
run_cfg['student'] = copy.deepcopy(actor_cfg)
run_cfg['teacher'] = copy.deepcopy(actor_cfg)
run_cfg['obs_groups'] = {'student':['policy'], 'teacher':['teacher']}
run_cfg['algorithm'] = {'class_name':'ankle_v23_depth_student.distillation:ContinuousTeacherDistillation', 'num_learning_epochs':1,
    'gradient_length':16, 'learning_rate':3e-4, 'max_grad_norm':.5, 'loss_type':'mse'}
saved_run_cfg = copy.deepcopy(run_cfg)
runner = DistillationRunner(env, run_cfg, log_dir=str(folder), device=agent.device)
import torch
loaded = torch.load(str(Path(args.warm_start).resolve()), map_location=agent.device, weights_only=False)
runner.alg.load(loaded, load_cfg={'teacher':True, 'student':False, 'optimizer':False, 'iteration':False}, strict=True)
teacher_state = loaded['actor_state_dict']
student_state = runner.alg._raw_student.state_dict()
for key, target_value in student_state.items():
    value = teacher_state[key]
    if value.shape == target_value.shape:
        student_state[key] = value.clone()
    elif key == 'rnn.rnn.weight_ih_l0' and value.shape[1] == 508 and target_value.shape[1] == 485:
        student_state[key] = value[:, :485].clone()
    elif key in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std') and value.shape[-1] == 508 and target_value.shape[-1] == 485:
        student_state[key] = value[:, :485].clone()
    else:
        raise ValueError(f'Unsupported student transfer: {key} {value.shape} -> {target_value.shape}')
runner.alg._raw_student.load_state_dict(student_state, strict=True)
print('STUDENT_INITIALIZATION=teacher weights with privileged input columns removed; frozen teacher; student actions drive training', flush=True)
dump_yaml(str(folder / "params/env.yaml"), cfg)
dump_yaml(str(folder / "params/agent.yaml"), saved_run_cfg)
(folder / "run.json").write_text(json.dumps({"arguments": vars(args), "metrics_path": metrics,
    "init_at_random_ep_len": False, "evidence": "training_only"}, indent=2) + "\n")
# Capture student weights at the full-climb event, before the next update.
# Earlier chunks of this long training episode may have used earlier weights.
original_step = env.step
last_full_event = [0]
def training_step(actions):
    result = original_step(actions)
    count = env.unwrapped.full_event_count
    if count > last_full_event[0]:
        last_full_event[0] = count
        runner.save(str(folder / f"full_climb_behavior_{count}.pt"))
        (folder / "full_climb_behavior.json").write_text(json.dumps({
            "count":count, "step":env.unwrapped.common_step_counter,
            "checkpoint":f"full_climb_behavior_{count}.pt", "privileged_actor":cfg.privileged_actor,
            "evidence":"student_weights_at_full_climb_training_event",
            "episode_policy_updates":"weights may change between rollout chunks; not frozen-policy evaluation"}, indent=2)+"\n")
    return result
env.step = training_step
try:
    start = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=False)
    print(f"Training time: {time.time() - start:.2f} seconds", flush=True)
finally:
    env.close()
    app.close()


