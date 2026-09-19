from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class BalancedSmallPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_balanced_small"
    save_interval = 64
