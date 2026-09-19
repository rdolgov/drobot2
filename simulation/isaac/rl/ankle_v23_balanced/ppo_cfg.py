from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class BalancedPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_balanced"
    save_interval = 64
