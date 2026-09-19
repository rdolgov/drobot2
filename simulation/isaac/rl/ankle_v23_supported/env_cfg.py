from isaaclab.utils.configclass import configclass
from ankle_v23_footplan.env_cfg import FootplanCfg


@configclass
class SupportedCfg(FootplanCfg):
    phase_count = 6
    phase_s = 12.
    episode_length_s = 76.
    frequency_hz = 1/12.
    observation_space = 485  # append desired joint-reference error
    state_space = 524
    residual_scales = (.12, .20, .25, .20)
    ground_start_fraction = .70
    privileged_actor = False
