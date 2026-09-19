from isaaclab.utils.configclass import configclass
from ankle_v23_stairs.env_cfg import StairsCfg


@configclass
class StrategyCfg(StairsCfg):
    strategy = "free"
    observation_space = 469  # common D405/proprioception + relative heading sin/cos
    state_space = 485
    episode_length_s = 36.0
    command_min = .020
    command_max = .030
    frequency_hz = .125    # one eight-second placement phase
    residual_scales = (.38, .75, 1.0, .60)
    filter_time_s = .12
    placement_hold_s = .25
    summary_window = 256
