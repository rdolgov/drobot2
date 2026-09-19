from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class MarginPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_margin"
    save_interval = 16
