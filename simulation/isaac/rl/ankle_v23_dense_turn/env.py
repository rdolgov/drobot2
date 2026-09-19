"""Replace only the narrow heading reward and add commanded turn-rate reward."""
import math
import torch
from ankle_v23_experiments.env import StrategyEnv


class DenseTurnEnv(StrategyEnv):
    def _get_rewards(self):
        reward = super()._get_rewards()
        age = self.episode_length_buf.float() * self.step_dt
        target = math.radians(self.weights["yaw_deg"]) * ((age - 1) / 5).clamp(0, 1)
        error = self._yaw() - target
        _, _, _, contact, _, _ = self._geometry()
        supported = ((self.robot.data.projected_gravity_b.torch[:, 2] < -.90)
                     & (~self.fallen) & (contact.sum(1) >= 2))
        gate = supported * (age < 32) * self.step_dt
        old_heading = torch.exp(-(error / .35).square())
        reward += self.weights["yaw_weight"] * (error.cos() - old_heading) * gate
        angular_speed = self.robot.data.root_ang_vel_b.torch[:, 2]
        rate_error = (angular_speed - self.commands[:, 2]) / self.weights["turn_rate_width_rad_s"]
        reward += self.weights["turn_rate_weight"] * torch.exp(-rate_error.square()) * gate
        return reward
