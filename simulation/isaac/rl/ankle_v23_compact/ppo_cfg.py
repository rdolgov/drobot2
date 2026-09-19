from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class CompactPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_compact"
    save_interval = 16
