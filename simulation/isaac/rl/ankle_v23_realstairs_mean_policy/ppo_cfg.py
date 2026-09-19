from isaaclab.utils.configclass import configclass
from ankle_v2.ppo_cfg import BoundedDistributionCfg
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg

@configclass
class StairDistributionCfg(BoundedDistributionCfg):
    class_name: str = "ankle_v23_realstairs_mean_policy.distribution:MeanStairBeta"

@configclass
class RealStairsMeanPolicyPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_realstairs_mean_policy"
    save_interval = 16

    def __post_init__(self):
        self.actor.distribution_cfg = StairDistributionCfg()
