"""Isolated full PPO worker with optional actor-only initialization."""
import argparse
from datetime import datetime
from importlib import metadata
import json
from pathlib import Path
import sys
import time
import warp as wp
wp.config.enable_backward=False

parser=argparse.ArgumentParser()
parser.add_argument('--rl_library',default='rsl_rl')
parser.add_argument('--task',default='Drobot-Ankle-V23-Stride')
parser.add_argument('--num_envs',type=int,default=256)
parser.add_argument('--seed',type=int,required=True)
parser.add_argument('--max_iterations',type=int,default=3000)
parser.add_argument('--run_name',required=True)
parser.add_argument('--warm-start')
parser.add_argument('--visualizer',default='none')
args,overrides=parser.parse_known_args()
metrics=next((v.split('=',1)[1] for v in overrides if v.startswith('env.metrics_path=')),None)
if metrics is None: raise ValueError('env.metrics_path is required')
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True)
app=launcher.app
import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from rsl_rl.runners import OnPolicyRunner
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import ankle_v23_stride
from ankle_v23_stride.env_cfg import StrideCfg,ROOT
from ankle_v23_stride.ppo_cfg import StridePPOCfg
cfg=StrideCfg()
cfg.scene.num_envs=args.num_envs
cfg.seed=args.seed
cfg.metrics_path=metrics
agent=handle_deprecated_rsl_rl_cfg(StridePPOCfg(),metadata.version('rsl-rl-lib'))
agent.seed=args.seed
agent.max_iterations=args.max_iterations
agent.run_name=args.run_name
folder=ROOT/'logs/rsl_rl'/agent.experiment_name/(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'_'+args.run_name)
folder.mkdir(parents=True,exist_ok=False)
env=RslRlVecEnvWrapper(gym.make(args.task,cfg=cfg),clip_actions=1.)
runner=OnPolicyRunner(env,agent.to_dict(),log_dir=str(folder),device=agent.device)
warm=None
if args.warm_start:
    warm=str(Path(args.warm_start).resolve())
    runner.load(warm,load_cfg={'actor':True,'critic':False,'optimizer':False,'iteration':False,'rnd':False})
    print('V23_ACTOR_ONLY_INITIALIZATION='+warm,flush=True)
(folder/'stride-run.json').write_text(json.dumps({'arguments':vars(args),'metrics_path':metrics,
    'actor_initialization':warm,'critic_optimizer_iteration':'fresh','config':agent.to_dict()},indent=2,default=str)+'\n')
try:
    start=time.time()
    runner.learn(num_learning_iterations=args.max_iterations,init_at_random_ep_len=True)
    print(f'Training time: {time.time()-start:.2f} seconds',flush=True)
finally:
    env.close()
    app.close()
