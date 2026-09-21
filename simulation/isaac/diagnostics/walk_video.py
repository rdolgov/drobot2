"""Render-only annotation for requested IK failure recordings; no physics writes."""
from collections import deque
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class WalkVideo:
    def __init__(self, raw, output, trial, title):
        self.raw, self.trial, self.title = raw, trial, title
        self.path = Path(output) / f"trial-{trial}-{title.split()[0].lower()}.mp4"
        self.width, self.height, self.fps = 1280, 720, 30
        self.frames = 0
        self.last_frame = None
        self.replay = deque(maxlen=120)
        self.closed = False
        self.font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 20)
        self.small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 17)
        self.large = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 28)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for the requested recording")
        self.log = self.path.with_suffix(".encoder.log").open("wb")
        self.process = subprocess.Popen([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720",
            "-r", "30", "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(self.path)], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, stderr=self.log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def _write(self, frame):
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        self.frames += 1

    def capture(self, row):
        from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
        center = np.array(row["root_position"]) + np.array([0., 0., -.15])
        # Front is +X. Right side is -Y. The close right-side oblique view
        # makes the selected rear-right foot and front-left foot distinguishable.
        set_kit_renderer_camera_view(center + np.array([1.15, -1.40, .65]), center)
        # Flush the previous trial's camera without advancing physics.
        for _ in range(3):
            rgb = self.raw.render()
        frame = Image.new("RGB", (1280, 720), (16, 23, 34))
        frame.paste(Image.fromarray(np.ascontiguousarray(rgb[..., :3])), (0, 98))
        draw = ImageDraw.Draw(frame)
        draw.text((22, 12), "16-SERVO IK  /  " + self.title, font=self.large, fill="white")
        draw.text((22, 54), f'Simulation {row["time_s"]:.2f} s  |  nominal motors  |  estimated mass 3.395 kg',
                  font=self.font, fill=(182, 198, 218))
        draw.rectangle((960, 98, 1279, 719), fill=(24, 34, 47))
        phase = "STARTUP SETTLE" if row["time_s"] < 2.5 else ("WAIT: " + row["gate"] if row["gate"] else row["phase"])
        draw.text((982, 116), phase.upper().replace("_", " "), font=self.font, fill=(255, 204, 113))
        moving = row["swing_pair"] or [row["swing_corner"]]
        short = {"front_left": "FL", "rear_left": "RL", "front_right": "FR", "rear_right": "RR"}
        draw.text((982, 151), "Selected: " + " + ".join(short[x] for x in moving), font=self.font, fill="white")
        draw.text((982, 188), f'Requested lift: {row["requested_lift_m"]*1000:.1f} mm', font=self.small, fill="white")
        legs = ("front_left", "rear_left", "front_right", "rear_right")
        for i, leg in enumerate(legs):
            y = 230 + i*66
            force = row["normal_force_n"][i]
            clear = row["sole_clearance_m_FL_RL_FR_RR"][i]*1000
            color = (255, 147, 116) if leg in moving else (155, 202, 233)
            draw.text((982, y), f'{short[leg]}  {"< selected" if leg in moving else "support"}', font=self.font, fill=color)
            draw.text((982, y+27), f'{clear:+.1f} mm clear   |   {force:.1f} N', font=self.small, fill="white")
        error = max(abs(q-t) for q, t in zip(row["joint_position"], row["joint_target"])) * 180/math.pi
        capped = sum(abs(x) >= .98*.8825985 for x in row["joint_effort_nm"])
        draw.text((982, 513), f'Joint error: {error:.1f} deg', font=self.font, fill="white")
        draw.text((982, 548), f'Body tilt: {row["tilt_deg"]:.1f} deg', font=self.font, fill="white")
        draw.text((982, 583), f'Joints at torque cap: {capped}/16', font=self.small, fill=(255, 204, 113))
        draw.text((22, 649), "Front = +X   |   right-side oblique view   |   FL / FR = front left / right; RL / RR = rear left / right",
                  font=self.small, fill=(182, 198, 218))
        draw.text((22, 680), "Physics simulation, not robot footage. Negative clearance is contact penetration in the simulator.",
                  font=self.small, fill=(182, 198, 218))
        if row["failure"]:
            draw.rectangle((10, 103, 951, 178), fill=(104, 29, 32))
            draw.text((24, 110), "FIRST LIFT FAILED  /  tracking timeout", font=self.large, fill="white")
            draw.text((24, 149), "Rear swing foot stays loaded; the sequence stops before forward swing.", font=self.small, fill="white")
            frame.save(self.path.with_suffix(".png"))
        self._write(frame)
        self.last_frame = frame
        if row["time_s"] >= 2.5:
            self.replay.append(frame)

    def finish(self):
        if self.closed:
            return
        if self.last_frame is not None:
            for original in self.replay:
                replay_frame = original.copy()
                draw = ImageDraw.Draw(replay_frame)
                draw.rectangle((700, 12, 1270, 49), fill=(16, 23, 34))
                draw.text((720, 15), "0.5x REPLAY  /  same recorded frames", font=self.font, fill=(255, 204, 113))
                self._write(replay_frame)
                self._write(replay_frame)
            for _ in range(90):
                self._write(self.last_frame)
        self.process.stdin.close()
        result = self.process.wait(timeout=45)
        self.log.close()
        self.closed = True
        if result:
            raise RuntimeError(f"Video encoder failed: {self.path}")

    def info(self):
        return {"trial": self.trial, "title": self.title, "video": str(self.path),
                "frames": self.frames, "fps": self.fps, "duration_s": self.frames/self.fps,
                "replay": "last up to 4 seconds at half speed, then 3-second final freeze",
                "camera": "right-side oblique; rendered actual simulated state"}
