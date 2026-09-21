"""Annotated actual simulation frames; no scene or physics changes."""
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ReviewVideo:
    def __init__(self, raw, path, title):
        self.raw, self.path, self.title = raw, Path(path), title
        self.font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 21)
        self.large = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 28)
        self.last_frame = None
        self.closed = False
        self.log = self.path.with_suffix(".encoder.log").open("wb")
        self.process = subprocess.Popen([shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "warning", "-n",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x720", "-r", "30", "-i", "pipe:0",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=self.log, creationflags=subprocess.CREATE_NO_WINDOW)

    def write(self, frame):
        self.process.stdin.write(np.asarray(frame, dtype=np.uint8).tobytes())
        self.last_frame = frame

    def capture(self, time_s, d, displacement):
        from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view
        center = d["body_position"]+np.array([0., 0., -.12])
        set_kit_renderer_camera_view(center+np.array([.90, -1.05, .50]), center)
        rgb = self.raw.render()
        frame = Image.fromarray(np.ascontiguousarray(rgb[..., :3]))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, 1279, 116), fill=(16, 23, 34))
        draw.text((18, 9), f"DIAGONAL IK + RL  |  {self.title}  |  SIMULATION", font=self.large, fill="white")
        stage = ("transfer", "lift + ankle push", "swing", "lower", "plant", "push + settle")[int(d["stage"])]
        pair = "FL + RR" if int(d["pair"]) == 0 else "FR + RL"
        draw.text((18, 47), f"{time_s:.1f}s  |  {stage.upper()}  |  pair {pair}  |  forward {displacement[0]*100:.1f} cm", font=self.font, fill="white")
        draw.text((18, 80), f"Body tilt {float(d['tilt_rad'])*180/np.pi:.1f} deg  |  complete steps FL/RL/FR/RR: {d['qualified_steps_by_leg'].astype(int).tolist()}", font=self.font, fill=(190, 215, 240))
        draw.rectangle((0, 660, 1279, 719), fill=(16, 23, 34))
        clearance = [round(float(v)*1000, 1) for v in d["sole_clearance_m"]]
        draw.text((18, 669), f"Measured foot clearance FL/RL/FR/RR: {clearance} mm  |  ground contact: {d['contacts'].astype(int).tolist()}", font=self.font, fill="white")
        self.write(frame)

    def failure(self, reason):
        if self.last_frame is None:
            return
        frame = self.last_frame.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle((20, 145, 1260, 206), fill=(112, 25, 35))
        draw.text((35, 157), reason, font=self.large, fill="white")
        self.last_frame = frame

    def close(self):
        if self.closed:
            return
        if self.last_frame is not None:
            self.last_frame.save(self.path.with_suffix(".png"))
            for _ in range(60):
                self.write(self.last_frame)
        self.process.stdin.close()
        result = self.process.wait(timeout=45)
        self.log.close()
        self.closed = True
        if result:
            raise RuntimeError("Video encoding failed")
