"""Make a labeled review from actual recorded frames, at original speed."""
import json
from pathlib import Path
import shutil
import subprocess

root = Path(__file__).resolve().parent
segments = [
    (root/"20260920-reference-video/walk.mp4", "1 - IK reference - SIMULATION"),
    (root/"20260920-progress-video/walk.mp4", "2 - Forward-progress RL candidate - SIMULATION"),
    (root/"20260920-pair-final-video/walk.mp4", "3 - Pair-focused RL candidate - SIMULATION"),
]
output = root/"diagonal-rl-comparison.mp4"
command = [shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "warning", "-y"]
filters = []
for i, (path, title) in enumerate(segments):
    if not path.is_file():
        raise FileNotFoundError(path)
    command += ["-i", str(path)]
    filters.append(f"[{i}:v]trim=start=2.5:end=14.5,setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=0:w=iw:h=47:color=0x101722:t=fill,"
        "drawtext=fontfile='C\\:/Windows/Fonts/segoeuib.ttf':"
        f"text='{title}':fontsize=27:fontcolor=white:x=18:y=7,"
        "drawbox=x=0:y=116:w=iw:h=30:color=0x26374B:t=fill,"
        "drawtext=fontfile='C\\:/Windows/Fonts/segoeui.ttf':"
        "text='Real-time excerpt 2.5-14.5 s | Same first diagonal | Full trials and measured results saved separately':"
        f"fontsize=18:fontcolor=white:x=18:y=120[v{i}]")
filters.append("[v0][v1][v2]concat=n=3:v=1:a=0[review]")
command += ["-filter_complex", ";".join(filters), "-map", "[review]", "-an", "-r", "30",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output)]
subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
(root/"video-manifest.json").write_text(json.dumps({"output": str(output), "playback_speed": 1,
    "source_time_s": [2.5, 14.5], "sources": [{"path": str(p), "title": t} for p,t in segments],
    "changes": "trim, concatenate, annotate titles; no change to robot frames or motion",
    "qualification": "review evidence, not a qualified walking demonstration"}, indent=2)+"\n")
print(output)
