from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlSymmetryCfg
from ankle_v25_pair.ppo_cfg import IKPPOCfg as PairPPOCfg


@configclass
class IKPPOCfg(PairPPOCfg):
    max_iterations = 400
    save_interval = 100
    experiment_name = 'drobot_ankle_v26_balanced'

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 5.e-5
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False, use_mirror_loss=True,
            data_augmentation_func='ankle_v26_balanced.symmetry:symmetric_states',
            mirror_loss_coeff=.05)
