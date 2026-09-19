from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class RealStairsContactPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_realstairs_contact"
    save_interval = 16
