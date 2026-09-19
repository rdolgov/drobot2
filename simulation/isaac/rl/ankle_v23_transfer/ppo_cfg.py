from isaaclab.utils.configclass import configclass
from ankle_v21.ppo_cfg import PlantedPPOCfg

@configclass
class TransferPPOCfg(PlantedPPOCfg):
    max_iterations = 3000
    save_interval = 250
    experiment_name = 'drobot_ankle_v23_transfer'
