from isaaclab.utils.configclass import configclass
from ankle_v26_balanced.ppo_cfg import IKPPOCfg as BalancedPPOCfg


@configclass
class IKPPOCfg(BalancedPPOCfg):
    max_iterations = 250
    save_interval = 100
    experiment_name = 'drobot_ankle_v26_feasible'

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.e-4
        self.algorithm.symmetry_cfg.mirror_loss_coeff = .25
