from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class DepthRefinePPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_depth_refine"
    save_interval = 16

    def __post_init__(self):
        self.algorithm.learning_rate = 1e-4
        self.algorithm.schedule = "fixed"
