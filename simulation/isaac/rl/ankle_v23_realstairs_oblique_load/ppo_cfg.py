from isaaclab.utils.configclass import configclass
from ankle_v2.ppo_cfg import BoundedDistributionCfg
from ankle_v23_experiments.ppo_cfg import StrategyPPOCfg

@configclass
class StairDistributionCfg(BoundedDistributionCfg):
    class_name: str = "ankle_v23_realstairs_oblique_load.distribution:StairBeta"
    concentration_factor: float = 4.

@configclass
class RealStairsObliqueLoadPPOCfg(StrategyPPOCfg):
    experiment_name = "drobot_ankle_v23_realstairs_oblique_load"
    save_interval = 16

    def __post_init__(self):
        self.actor.distribution_cfg = StairDistributionCfg()
