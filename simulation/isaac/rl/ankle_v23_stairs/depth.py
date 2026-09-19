"""Sparse pinhole depth, not a stereo image renderer or a hardware driver.

Optical Z depth is the ray parameter because camera rays have forward X=1.
Nearest intersections include the ground and the same solid stair boxes used
by physics. Terrain occlusion is exact for these boxes. Robot self-occlusion,
stereo correspondence, texture and lighting failures are not modeled.
"""
import math
import torch
from ankle_v2.env import rotate


class D405Depth:
    def __init__(self, cfg, num_envs, device, boxes):
        self.cfg, self.device, self.boxes = cfg, device, boxes
        self.period = round(60.0 / cfg.camera_hz)
        cols = (torch.arange(cfg.camera_cols, device=device) + .5) / cfg.camera_cols
        rows = (torch.arange(cfg.camera_rows, device=device) + .5) / cfg.camera_rows
        vertical, horizontal = torch.meshgrid(
            (1 - 2 * rows) * math.tan(math.radians(cfg.camera_vfov_deg / 2)),
            (1 - 2 * cols) * math.tan(math.radians(cfg.camera_hfov_deg / 2)), indexing="ij")
        self.rays = torch.stack((torch.ones_like(horizontal), horizontal, vertical), -1).reshape(-1, 3)
        size = cfg.camera_rows * cfg.camera_cols
        self.frame = torch.zeros((num_envs, 2 * size), device=device)
        self.pending = torch.zeros_like(self.frame)
        self.age = torch.zeros((num_envs, 1), device=device)
        self.pitch = torch.zeros(num_envs, device=device)
        self.offset = torch.zeros((num_envs, 3), device=device)
        self.bias = torch.zeros((num_envs, 1), device=device)
        self.last_step = -1

    def reset(self, ids):
        cfg = self.cfg
        self.frame[ids] = 0
        self.pending[ids] = 0
        self.age[ids] = 1
        self.pitch[ids] = math.radians(cfg.camera_pitch_deg) + (
            2 * torch.rand(len(ids), device=self.device) - 1) * math.radians(cfg.camera_pitch_jitter_deg)
        self.offset[ids] = torch.tensor(cfg.camera_position_m, device=self.device) + (
            2 * torch.rand((len(ids), 3), device=self.device) - 1) * cfg.camera_position_jitter_m
        self.bias[ids] = (2 * torch.rand((len(ids), 1), device=self.device) - 1) * .002

    def read(self, position, quaternion):
        pitch_q = torch.zeros_like(quaternion)
        pitch_q[:, 1] = torch.sin(self.pitch / 2)
        pitch_q[:, 3] = torch.cos(self.pitch / 2)
        directions = rotate(quaternion[:, None], rotate(pitch_q[:, None], self.rays[None]))
        origin = position + rotate(quaternion, self.offset)
        safe = torch.where(directions.abs() < 1e-8, torch.full_like(directions, 1e-8), directions)
        floor = -origin[:, None, 2] / safe[..., 2]
        nearest = torch.where((directions[..., 2] < -1e-8) & (floor > 0), floor, torch.inf)
        for low, high in self.boxes:
            first = (low - origin[:, None]) / safe
            second = (high - origin[:, None]) / safe
            near = torch.minimum(first, second).amax(-1)
            far = torch.maximum(first, second).amin(-1)
            hit = (far >= near.clamp(min=0)) & (near > 0)
            nearest = torch.minimum(nearest, torch.where(hit, near, torch.inf))
        cfg = self.cfg
        valid = torch.isfinite(nearest) & (nearest >= cfg.camera_min_m) & (nearest <= cfg.camera_max_m)
        depth = torch.where(valid, nearest, torch.zeros_like(nearest))
        depth += (2 * torch.rand_like(depth) - 1) * (cfg.camera_noise_fraction * depth + .001) + self.bias
        valid &= (torch.rand_like(depth) >= cfg.camera_dropout)
        valid &= (torch.rand((len(depth), 1), device=self.device) >= cfg.camera_frame_dropout)
        valid &= (depth >= cfg.camera_min_m) & (depth <= cfg.camera_max_m)
        return torch.cat((torch.where(valid, depth / cfg.camera_max_m, 0), valid.float()), 1)

    def update(self, step, position, quaternion, dt):
        # Reading observations twice cannot advance the camera or its latency.
        if step != self.last_step:
            self.age += dt
            if step % self.period == 0:
                self.frame.copy_(self.pending)
                self.pending.copy_(self.read(position, quaternion))
                self.age[:] = 1 / self.cfg.camera_hz
            self.last_step = step
        return torch.cat((self.frame, self.age.clamp(max=1)), 1)
