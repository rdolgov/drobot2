"""Compare actual old/new simulation recordings at their original speed."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--label', required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
segments = [
    (root.parent/'ankle-v25-ik/20260920-pair-final-video/walk.mp4', 'Previous pair-focused model - SIMULATION'),
    (args.candidate.resolve()/'walk.mp4', args.label+' - SIMULATION'),
]
output = root/'continued-diagonal-comparison.mp4'
command = [shutil.which('ffmpeg'), '-hide_banner', '-loglevel', 'warning', '-n']
filters = []
for i, (path, title) in enumerate(segments):
    command += ['-i', str(path)]
    if any(c in title for c in ("'", ':', '[', ']', '\\')):
        raise ValueError('Use plain alphanumeric title text')
    filters.append(f'[{i}:v]trim=start=2.5:end=20.5,setpts=PTS-STARTPTS,'
        'drawbox=x=0:y=0:w=iw:h=47:color=0x101722:t=fill,'
        "drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':"
        f"text='{title}':fontsize=27:fontcolor=white:x=18:y=7,"
        'drawbox=x=0:y=116:w=iw:h=30:color=0x26374B:t=fill,'
        "drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':"
        "text='Normal speed | Same seed and starting diagonal | Physical foot motion and step counts shown below':"
        f'fontsize=18:fontcolor=white:x=18:y=120[v{i}]')
filters.append('[v0][v1]concat=n=2:v=1:a=0[review]')
command += ['-filter_complex', ';'.join(filters), '-map', '[review]', '-an', '-r', '30',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(output)]
subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
manifest = {'output': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
    'playback_speed': 1, 'source_time_s': [2.5, 20.5],
    'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'title': t} for p,t in segments],
    'changes': 'trim, concatenate and annotate titles; no alteration to robot frames or motion',
    'qualification': 'Simulation evidence; see independent four-trial reports for results.'}
(root/'video-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(output)
