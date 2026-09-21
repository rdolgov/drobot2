from isaaclab.utils.configclass import configclass
from ankle_v25_ik.ppo_cfg import IKPPOCfg as ReferencePPOCfg


@configclass
class IKPPOCfg(ReferencePPOCfg):
    max_iterations = 300
    experiment_name = "drobot_ankle_v25_pair"
