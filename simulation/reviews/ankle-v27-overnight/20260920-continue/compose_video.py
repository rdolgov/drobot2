"""Actual approved/selected simulation frames, matched excerpt and real speed."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

out = Path(__file__).resolve().parent
root = out.parents[3]
selection = json.loads((out/'selected/selection.json').read_text())
selected = selection['selected']['label']
segments = [(root/'simulation/reviews/ankle-v26-balanced/20260920-video-r2-549/walk.mp4', 'User-approved starting model'),
            (out/'video-pair-0/walk.mp4', 'Selected after overnight training')]
output = out/'comparison.mp4'
if not output.exists():
    command = [shutil.which('ffmpeg'),'-hide_banner','-loglevel','warning','-n']
    filters = []
    for i,(video,title) in enumerate(segments):
        command += ['-i',str(video)]
        filters.append(f'[{i}:v]trim=start=2.5:end=20.5,setpts=PTS-STARTPTS,'
            'drawbox=x=0:y=0:w=iw:h=47:color=0x101722:t=fill,'
            "drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':"
            f"text='{title} - SIMULATION':fontsize=27:fontcolor=white:x=18:y=7,"
            'drawbox=x=0:y=116:w=iw:h=30:color=0x26374B:t=fill,'
            "drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':"
            "text='Normal speed | Same seed and first diagonal | Separate reports retain all four evaluation conditions':"
            f'fontsize=18:fontcolor=white:x=18:y=120[v{i}]')
    filters.append('[v0][v1]concat=n=2:v=1:a=0[review]')
    command += ['-filter_complex',';'.join(filters),'-map','[review]','-an','-r','30',
                '-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(output)]
    subprocess.run(command,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
manifest = {'output':str(output),'sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
    'selected':selected,'playback_speed':1,'source_time_s':[2.5,20.5],
    'sources':[{'path':str(p),'title':t,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,t in segments],
    'changes':'trim, titles and concatenation only; no altered robot frames or motion'}
(out/'video-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
frame = out/'comparison-frame.png'
if not frame.exists():
    subprocess.run([shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-ss','25','-i',str(output),
                    '-frames:v','1','-update','1',str(frame)],check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW)
print(output)
