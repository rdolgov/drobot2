from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class ResolvedPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_resolved"
    save_interval = 16
