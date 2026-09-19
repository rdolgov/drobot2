from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.env_cfg import StrategyCfg


@configclass
class FootholdCfg(StrategyCfg):
    strategy = "free"
    guidance = True
    stance_forward_m = .025
    stance_down_m = .300
    # Exact URDF FK: hip origin offset plus ankle-to-tread offset.
    target_base_height_m = .388908389
    start_x_m = .200
    start_jitter_m = .002
    placement_forward_m = .114
    placement_height_m = .020
    swing_height_m = .045
    episode_length_s = 12.
    frequency_hz = 1. / 12.
    command_min = 0.
    command_max = 0.
    residual_scales = (.12, .20, .25, .20)
    goal_hold_s = 1.

    def __post_init__(self):
        super().__post_init__()
        self.robot.init_state.pos = (self.start_x_m, 0., self.target_base_height_m + .002)
