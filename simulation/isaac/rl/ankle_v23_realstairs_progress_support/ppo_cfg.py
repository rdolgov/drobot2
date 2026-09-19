from isaaclab.utils.configclass import configclass
from ankle_v2.ppo_cfg import BoundedDistributionCfg
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg

@configclass
class StairDistributionCfg(BoundedDistributionCfg):
    class_name: str = "ankle_v23_realstairs_progress_support.distribution:MeanStairBeta"

@configclass
class RealStairsProgressSupportPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_realstairs_progress_support"
    save_interval = 8
    num_steps_per_env = 256

    def __post_init__(self):
        self.actor.distribution_cfg = StairDistributionCfg()
        self.algorithm.gamma = .9997
        self.algorithm.lam = .995
