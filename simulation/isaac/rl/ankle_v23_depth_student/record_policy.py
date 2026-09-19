"""Record one requested frozen-student episode; no training or trial selection."""
import argparse
import copy
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seed', type=int, default=2961)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True, enable_cameras=True)
app = launcher.app
import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
from rsl_rl.runners import DistillationRunner
from PIL import Image, ImageDraw, ImageFont
from video_encoding import get_video_encoding_interface
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ankle_v23_depth_student
from ankle_v23_depth_student.env_cfg import DepthStudentCfg
from ankle_v23_depth_student.ppo_cfg import DepthStudentPPOCfg

cfg = DepthStudentCfg()
cfg.scene.num_envs = 1
cfg.seed = args.seed
cfg.phase_count = 26
cfg.episode_length_s = 316.
cfg.privileged_actor = False
cfg.observation_space = 485
cfg.ground_start_fraction = 1.
cfg.metrics_path = str(args.output/'recording.metrics.jsonl')
cfg.viewer.eye = (1.6, -1.7, 1.05)
cfg.viewer.lookat = (.65, 0, .24)
cfg.video_recorder = VideoRecorderCfg(window_width=1280, window_height=720, backend_source='renderer')
agent = handle_deprecated_rsl_rl_cfg(DepthStudentPPOCfg(), metadata.version('rsl-rl-lib'))
run_cfg = agent.to_dict()
actor_cfg = run_cfg.pop('actor')
run_cfg.pop('critic')
run_cfg['student'] = copy.deepcopy(actor_cfg)
run_cfg['teacher'] = copy.deepcopy(actor_cfg)
run_cfg['obs_groups'] = {'student':['policy'], 'teacher':['teacher']}
run_cfg['algorithm'] = {'class_name':'ankle_v23_depth_student.distillation:ContinuousTeacherDistillation',
    'num_learning_epochs':1, 'gradient_length':16, 'learning_rate':3e-4, 'max_grad_norm':.5, 'loss_type':'mse'}
env = RslRlVecEnvWrapper(gym.make('Drobot-Ankle-V23-DepthStudent', cfg=cfg, render_mode='rgb_array'), clip_actions=1.)
raw = env.unwrapped
runner = DistillationRunner(env, run_cfg, log_dir=None, device=agent.device)
runner.load(str(args.checkpoint.resolve()), load_cfg={'student':True,'teacher':False,'optimizer':False,'iteration':False})
policy = runner.get_inference_policy(device=raw.device)
obs,_ = env.reset()
policy.reset()
# Hold the terminal pose for the last frame instead of showing an auto-reset.
original_reset = raw._reset_idx
def preserve_terminal_pose(env_ids):
    if len(env_ids):
        return
    original_reset(env_ids)
raw._reset_idx = preserve_terminal_pose
font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 22)
small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
encoder = get_video_encoding_interface()
video = args.output/'frozen-student-20mm.mp4'
report = {'status':'recording','checkpoint':str(args.checkpoint.resolve()),
    'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    'scope':'one frozen deterministic student recording; not the earlier changing-policy training episode',
    'seed':args.seed,'stair_rise_m':.02,'stair_tread_m':.25,'stair_count':3,
    'actor':'student_state_dict, 485 inputs; no teacher actions', 'video':str(video.resolve())}
report_path = args.output/'report.json'
report_path.write_text(json.dumps(report,indent=2)+'\n')
encoding = False
frames = 0
trace = []
try:
    if encoder is None or not encoder.start_encoding(str(video),30.,0,True):
        raise RuntimeError('H264 encoder could not start')
    encoding = True
    for _ in range(24): raw.render()
    for step in range(round(cfg.episode_length_s/raw.step_dt)):
        with torch.no_grad():
            obs,_,done,_ = env.step(policy(obs))
        seconds = (step+1)*raw.step_dt
        finished = bool(done[0])
        if step % 2 == 1 or finished:
            center = raw.robot.data.root_pos_w.torch[0].cpu().numpy()+np.array([.18,0,-.10])
            set_kit_renderer_camera_view(center+np.array([1.05,-1.50,.75]),center)
            rgb = np.ascontiguousarray(raw.render()[...,:3])
            frame = Image.fromarray(rgb)
            draw = ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,94),fill=(15,23,35))
            draw.text((20,10),'V2.3 ankle robot | frozen student policy | 20 mm risers, 250 mm treads',font=font,fill='white')
            draw.text((20,42),f'{seconds:.1f} s | SMALL-STEP RECORDING, NOT THE 180 mm TARGET | real-time playback',font=small,fill='white')
            draw.text((20,68),'Known-stair reference + simulated D405 depth | no training updates',font=small,fill=(180,205,230))
            rgba = np.ascontiguousarray(np.asarray(frame.convert('RGBA'),dtype=np.uint8))
            if not encoder.encode_next_frame_from_buffer(rgba,1280,720):
                raise RuntimeError('Frame encoding failed')
            frames += 1
            if frames in (1,150,600) or finished: frame.save(args.output/f'frame-{frames:05d}.png')
        if step % 60 == 0 or finished:
            trace.append({'time_s':seconds,'root':raw.robot.data.root_pos_w.torch[0].cpu().tolist(),
                'joints':raw.robot.data.joint_pos.torch[0,raw.ids].cpu().tolist(),
                'full_climb':bool(raw.ever_full[0]),'fallen':bool(raw.fallen[0]),
                'range_excess_rad':float(raw.range_peak[0])})
            report.update(time_s=seconds,frames=frames,full_climb=bool(raw.ever_full[0]),fallen=bool(raw.fallen[0]))
            report_path.write_text(json.dumps(report,indent=2)+'\n')
        if finished: break
    for _ in range(30):
        encoder.encode_next_frame_from_buffer(rgba,1280,720)
        frames += 1
    encoder.finalize_encoding()
    encoding = False
    report.update(status='completed',frames=frames,video_duration_s=frames/30.,peak_joint_range_excess_rad=float(raw.range_peak[0]))
    (args.output/'trace.json').write_text(json.dumps(trace)+'\n')
except BaseException as error:
    report.update(status='failed',error=repr(error))
    raise
finally:
    if encoding: encoder.finalize_encoding()
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    env.close()
    app.close()
