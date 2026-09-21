from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from ankle_v21.ppo_cfg import PlantedPPOCfg


@configclass
class ResidualGaussianCfg(RslRlMLPModelCfg.DistributionCfg):
    class_name = "GaussianDistribution"
    init_std = .30
    std_type = "log"
    std_range = (.08, .60)


@configclass
class IKPPOCfg(PlantedPPOCfg):
    max_iterations = 500
    save_interval = 100
    experiment_name = "drobot_ankle_v25_ik"
    actor = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu",
                           obs_normalization=True, distribution_cfg=ResidualGaussianCfg())

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.e-4
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = .001
