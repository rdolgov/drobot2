from isaaclab.utils.configclass import configclass
from ankle_v23_experiments.env_cfg import StrategyCfg


@configclass
class ResearchCfg(StrategyCfg):
    strategy = "continue_turn"
    potential_gamma = .999
    riser_cost_weight = 1.5
    edge_cost_weight = 1.0
