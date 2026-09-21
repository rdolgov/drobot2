from isaaclab.utils.configclass import configclass
from ankle_v2.ppo_cfg import ActivePPOCfg, BoundedDistributionCfg
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

@configclass
class PlantedPPOCfg(ActivePPOCfg):
    max_iterations = 3000
    save_interval = 250
    experiment_name = 'drobot_ankle_v21_planted'
    actor = RslRlMLPModelCfg(hidden_dims=[256,256,128],activation='elu',
                           obs_normalization=True,distribution_cfg=BoundedDistributionCfg())
    critic = RslRlMLPModelCfg(hidden_dims=[256,256,128],activation='elu',obs_normalization=True)

