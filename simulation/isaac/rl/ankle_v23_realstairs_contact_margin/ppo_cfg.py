from isaaclab.utils.configclass import configclass
from ankle_v2.ppo_cfg import BoundedDistributionCfg
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg

@configclass
class StairDistributionCfg(BoundedDistributionCfg):
    class_name: str = "ankle_v23_realstairs_contact_margin.distribution:MeanStairBeta"

@configclass
class RealStairsContactMarginPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_realstairs_contact_margin"
    save_interval = 8
    num_steps_per_env = 128

    def __post_init__(self):
        self.actor.distribution_cfg = StairDistributionCfg()
        self.algorithm.gamma = .999100269973
        self.algorithm.lam = .995
