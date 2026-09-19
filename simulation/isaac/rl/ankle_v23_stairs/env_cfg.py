"""The first curriculum stage is a real three-riser collision staircase."""
from isaaclab.utils.configclass import configclass
from ankle_v23_stride.env_cfg import StrideCfg


@configclass
class StairsCfg(StrideCfg):
    observation_space = 467  # 82 proprioception + 192 depth + 192 valid + age
    state_space = 483       # actor + velocity, relative base height, contacts, foot heights, progress
    episode_length_s = 60.0
    command_min = .025
    command_max = .035
    frequency_hz = .22
    duty_factor = .78
    lift_m = .045
    residual_scales = (.16, .50, .65, .50)
    filter_time_s = .12
    stair_rise_m = .020
    stair_tread_m = .250
    stair_count = 3
    stair_start_m = .340
    stair_width_m = 1.4
    platform_depth_m = 1.2
    camera_position_m = (.14, .0, .02)
    camera_pitch_deg = 40.0
    camera_hfov_deg = 87.0
    camera_vfov_deg = 58.0
    camera_rows = 12
    camera_cols = 16
    camera_min_m = .07
    camera_max_m = .50
    camera_hz = 30.0
    camera_dropout = .05
    camera_frame_dropout = .02
    camera_noise_fraction = .02
    camera_pitch_jitter_deg = 3.0
    camera_position_jitter_m = .005
    success_hold_s = 1.0

