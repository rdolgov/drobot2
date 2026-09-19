from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class ShortstepPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_shortstep"
    save_interval = 64
