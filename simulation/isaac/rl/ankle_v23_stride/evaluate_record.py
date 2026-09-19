"""Checkpoint selection and requested walk recording, preserving failed trials.

Usage: --checkpoints PATH ... --output DIR [--record --speed .04]
Selection trials are separate from training telemetry. No hardware interfaces.
"""
import argparse
import json
import sys
from pathlib import Path
from importlib import metadata
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoints',nargs='+',required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--record',action='store_true')
parser.add_argument('--speed',type=float,default=.03)
parser.add_argument('--seed',type=int,default=2401)
parser.add_argument('--seconds',type=float,default=48.)
parser.add_argument('--ankle-mode',default='learned',choices=['learned','reference_only','locked_target'])
args = parser.parse_args()
args.output.mkdir(parents=True,exist_ok=True)

from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True,enable_cameras=args.record)
app = launcher.app
import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper,handle_deprecated_rsl_rl_cfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from rsl_rl.runners import OnPolicyRunner
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import ankle_v23_stride
from ankle_v23_stride.env_cfg import StrideCfg
from ankle_v23_stride.ppo_cfg import StridePPOCfg

cfg = StrideCfg()
cfg.evaluation = True
cfg.evaluation_stop_s = 40.
cfg.episode_length_s = args.seconds+2
cfg.scene.num_envs = 1 if args.record else 16
cfg.seed = args.seed
cfg.ankle_mode = args.ankle_mode
if args.record:
    cfg.viewer.eye = (1.4,-1.5,.9)
    cfg.viewer.lookat = (.2,0,.20)
    cfg.video_recorder = VideoRecorderCfg(window_width=1280,window_height=720,backend_source='renderer')
agent = handle_deprecated_rsl_rl_cfg(StridePPOCfg(),metadata.version('rsl-rl-lib'))
env = RslRlVecEnvWrapper(gym.make('Drobot-Ankle-V23-Stride',cfg=cfg,render_mode='rgb_array' if args.record else None),clip_actions=1.)
raw = env.unwrapped
runner = OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=agent.device)
report = {'status':'running','seed':args.seed,'recording':args.record,'trials':[],
          'scope':'nominal recording' if args.record else 'selection: nominal plus independently weaker motors (.85-1.0)',
          'flat_contact':'force plus corner-height proxy; no pressure sensor',
          'mass_kg':cfg.nominal_mass_kg,'ankle_mode':args.ankle_mode,
          'model':'V2.3_R3_solid_sole_retained_leg','evaluation_commands':[.024,.034]}
report_path = args.output/'report.json'
encoder = None
encoding = False

try:
    for checkpoint in args.checkpoints:
        runner.load(str(Path(checkpoint).resolve()))
        policy = runner.get_inference_policy(device=raw.device)
        for speed in ([args.speed] if args.record else [.024,.034]):
            raw.cfg.evaluation_speed = speed
            obs,_ = env.reset()
            if not args.record:
                generator = torch.Generator(device=raw.device).manual_seed(args.seed)
                raw.strength[:] = .85+.15*torch.rand(raw.strength.shape,device=raw.device,generator=generator)
                raw.rate_strength[:] = .85+.15*torch.rand(raw.rate_strength.shape,device=raw.device,generator=generator)
                raw.strength[0] = 1.
                raw.rate_strength[0] = 1.
            n = raw.num_envs
            failed = np.zeros(n,dtype=bool)
            totals = {k:np.zeros(n) for k in ['move_time','distance','stance','planted','tilt','slip','ankle_motion','stop_speed','stop_steps','expected_stance','contact_stance']}
            totals.update({k:np.zeros(n) for k in ['mid_swing','swing_clear','step_distance','reference_sweep','rate_limited','moving_steps']})
            steps_per_leg = np.zeros((n,4))
            start = raw.robot.data.root_pos_w.torch.detach().cpu().numpy().copy()
            previous = start.copy()
            last = start.copy()
            ankle_min = np.full((n,4),np.inf)
            ankle_max = np.full((n,4),-np.inf)
            frames = 0
            trace = []
            video = args.output/(Path(checkpoint).stem+f'-{speed:.2f}mps-{args.ankle_mode}.mp4')
            if args.record:
                from video_encoding import get_video_encoding_interface
                from PIL import Image,ImageDraw,ImageFont
                encoder = get_video_encoding_interface()
                if encoder is None or not encoder.start_encoding(str(video),30.,0,True):
                    raise RuntimeError('H264 encoder could not start')
                encoding = True
                # Render-only warmup; no policy or physics stepping.
                for _ in range(24): raw.render()
            for step in range(round(args.seconds/raw.step_dt)):
                # reset() mutates buffers created by step(); ordinary no-grad
                # tensors remain writable between the checkpoint trials.
                with torch.no_grad():
                    action = policy(obs)
                    obs,_,dones,_ = env.step(action)
                d = {k:v.cpu().numpy() for k,v in raw.last_diagnostics.items()}
                valid = ~failed
                moving = d['moving'] & valid
                p = d['base_xyz']
                totals['move_time'] += moving*raw.step_dt
                totals['distance'] += (p[:,0]-previous[:,0])*moving
                sc = d['stance'] & d['contact'] & moving[:,None]
                totals['stance'] += sc.sum(axis=1)
                totals['expected_stance'] += (d['stance'] & moving[:,None]).sum(axis=1)
                totals['contact_stance'] += sc.sum(axis=1)
                totals['planted'] += (d['planted'] & sc).sum(axis=1)
                totals['tilt'] += (d['tilt_deg']*sc).sum(axis=1)
                totals['slip'] += (d['slip_m_s']*sc).sum(axis=1)
                totals['ankle_motion'] += np.abs(d['ankle_velocity_rad_s']).mean(axis=1)*raw.step_dt*moving
                ankle_min[moving] = np.minimum(ankle_min[moving],d['ankle_rad'][moving])
                ankle_max[moving] = np.maximum(ankle_max[moving],d['ankle_rad'][moving])
                swing = d['mid_swing'] & moving[:,None]
                totals['mid_swing'] += swing.sum(axis=1)
                totals['swing_clear'] += (d['swing_clear'] & swing).sum(axis=1)
                qualified = d['qualified_touchdown'] & moving[:,None]
                steps_per_leg += qualified
                totals['step_distance'] += (d['touchdown_stride_m']*qualified).sum(axis=1)
                totals['reference_sweep'] += d['reference_stance_sweep_m']*moving*raw.step_dt
                totals['rate_limited'] += d['rate_limited']*moving
                totals['moving_steps'] += moving
                stopped = (step*raw.step_dt>=42.) & valid
                totals['stop_speed'] += np.linalg.norm(d['base_velocity'][:,:2],axis=1)*stopped
                totals['stop_steps'] += stopped
                last[valid] = p[valid]
                previous = p.copy()
                failed |= d['fallen'] | (dones.cpu().numpy().astype(bool))
                if args.record:
                    trace.append({'time':(step+1)*raw.step_dt,**{k:v[0].tolist() for k,v in d.items()}})
                    # step() already resets terminated bodies: never render that
                    # reset pose as if it were continued successful locomotion.
                    if failed[0]: break
                    if step%2==1:
                        from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
                        center = p[0]+np.array([.12,0,-.10])
                        set_kit_renderer_camera_view(center+np.array([1.2,-1.6,.75]),center)
                        rgb = np.ascontiguousarray(raw.render()[...,:3])
                        frame = Image.fromarray(rgb)
                        draw = ImageDraw.Draw(frame)
                        draw.rectangle((0,0,1280,65),fill=(15,23,35))
                        draw.text((20,12),f'V2.3 R3 solid sole | longer-stride PPO {Path(checkpoint).stem} | simulated {(step+1)*raw.step_dt:.1f}s',fill='white')
                        draw.text((20,35),f'command {d["command"][0]:.3f} m/s | actual {d["base_velocity"][0,0]:.3f} m/s | planted feet {int(d["planted"][0].sum())}/4 (swing feet lift)',fill='white')
                        rgb = np.asarray(frame,dtype=np.uint8)
                        rgba = np.ascontiguousarray(np.concatenate([rgb,np.full((720,1280,1),255,dtype=np.uint8)],axis=2))
                        if not encoder.encode_next_frame_from_buffer(rgba,1280,720): raise RuntimeError('Frame encoding failed')
                        frames += 1
                        if frames in (120,300,600,900,1200): frame.save(args.output/f'frame-{frames:04d}.png')
            if encoding:
                encoder.finalize_encoding()
                encoding = False
                (args.output/'trace.json').write_text(json.dumps(trace)+'\n')
            def div(a,b): return np.divide(a,np.maximum(b,1e-9))
            per_env = []
            for i in range(n):
                per_env.append({'env':i,'nominal':i==0,'failed':bool(failed[i]),
                   'forward_m_s':float(div(totals['distance'],totals['move_time'])[i]),
                   'forward_displacement_m':float(last[i,0]-start[i,0]),
                   'lateral_drift_m':float(last[i,1]-start[i,1]),
                   'lateral_drift_abs_m':float(abs(last[i,1]-start[i,1])),
                   'qualified_stride_m':float(totals['step_distance'][i]/max(1,steps_per_leg[i].sum())),
                   'qualified_steps_min_leg':float(steps_per_leg[i].min()),
                   'qualified_steps_per_leg':steps_per_leg[i].tolist(),
                   'swing_clear_fraction':float(div(totals['swing_clear'],totals['mid_swing'])[i]),
                   'reference_stance_sweep_m':float(div(totals['reference_sweep'],totals['move_time'])[i]),
                   'target_rate_limited_fraction':float(div(totals['rate_limited'],totals['moving_steps'])[i]),
                   'planted_stance_fraction':float(div(totals['planted'],totals['stance'])[i]),
                   'stance_contact_fraction':float(div(totals['contact_stance'],totals['expected_stance'])[i]),
                   'stance_tilt_deg':float(div(totals['tilt'],totals['stance'])[i]),
                   'stance_slip_m_s':float(div(totals['slip'],totals['stance'])[i]),
                   'ankle_abs_velocity_rad_s':float(div(totals['ankle_motion'],totals['move_time'])[i]),
                   'ankle_range_deg':[float(v) if np.isfinite(v) else None for v in (ankle_max[i]-ankle_min[i])*180/np.pi],
                   'stop_speed_m_s':float(div(totals['stop_speed'],totals['stop_steps'])[i])})
            mean = {key:float(np.mean([r[key] for r in per_env])) for key in ['forward_m_s','planted_stance_fraction','stance_contact_fraction','stance_tilt_deg','stance_slip_m_s','stop_speed_m_s','lateral_drift_abs_m','qualified_stride_m','qualified_steps_min_leg','swing_clear_fraction','reference_stance_sweep_m','target_rate_limited_fraction']}
            mean['fall_fraction'] = float(failed.mean())
            score = (3*min(mean['forward_m_s']/speed,1.2)+mean['planted_stance_fraction']+mean['stance_contact_fraction']
                     +2*mean['swing_clear_fraction']+min(mean['qualified_stride_m']/.12,1.3)
                     -10*mean['fall_fraction']-60*mean['stance_slip_m_s']-4*mean['lateral_drift_abs_m']-10*mean['stop_speed_m_s'])
            trial = {'checkpoint':str(Path(checkpoint).resolve()),'speed_command_m_s':speed,'mean':mean,'score':score,'episodes':per_env}
            if args.record: trial.update(video=str(video.resolve()),frames=frames,duration_s=frames/30.)
            report['trials'].append(trial)
            report_path.write_text(json.dumps(report,indent=2)+'\n')
            print('V23_STRIDE_TRIAL='+json.dumps({k:v for k,v in trial.items() if k!='episodes'}),flush=True)
    report['status'] = 'completed'
except BaseException as exc:
    import traceback
    report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    raise
finally:
    if encoding: encoder.finalize_encoding()
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    env.close()
    app.close()
