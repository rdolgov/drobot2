from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg


@configclass
class DepthStudentPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_depth_student"
    save_interval = 16
