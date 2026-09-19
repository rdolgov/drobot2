from isaaclab.utils.configclass import configclass
from ankle_v23_footplan.env_cfg import FootplanCfg


@configclass
class GuidedCfg(FootplanCfg):
    phase_count = 6
    phase_s = 12.
    episode_length_s = 76.
    frequency_hz = 1/12.
    observation_space = 485  # append desired joint-reference error
    state_space = 524
    residual_scales = (.06, .10, .12, .10)
    ground_start_fraction = .30
    privileged_actor = False
