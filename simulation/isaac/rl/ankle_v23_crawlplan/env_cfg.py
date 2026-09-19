from isaaclab.utils.configclass import configclass
from ankle_v23_footplan.env_cfg import FootplanCfg


@configclass
class CrawlplanCfg(FootplanCfg):
    phase_count = 4
    phase_s = 9.
    episode_length_s = 40.
    frequency_hz = 1/9.
    observation_space = 485  # append desired joint-reference error
    state_space = 501
    residual_scales = (.12, .20, .25, .20)
