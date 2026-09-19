from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.env_cfg import StrategyCfg


@configclass
class DenseTurnCfg(StrategyCfg):
    strategy = "sideways_dense"
