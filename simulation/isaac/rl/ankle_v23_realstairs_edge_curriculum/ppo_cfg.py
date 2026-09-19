from isaaclab.utils.configclass import configclass
from ankle_v2.ppo_cfg import BoundedDistributionCfg
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg

@configclass
class StairDistributionCfg(BoundedDistributionCfg):
    class_name: str = "ankle_v23_realstairs_edge_curriculum.distribution:MeanStairBeta"

@configclass
class RealStairsEdgeCurriculumPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_realstairs_edge_curriculum"
    save_interval = 8
    num_steps_per_env = 128

    def __post_init__(self):
        self.actor.distribution_cfg = StairDistributionCfg()
        self.algorithm.gamma = .999687548824
        self.algorithm.lam = .995
