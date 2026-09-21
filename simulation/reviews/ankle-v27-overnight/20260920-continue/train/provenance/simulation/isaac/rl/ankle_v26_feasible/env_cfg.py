from isaaclab.utils.configclass import configclass
from ankle_v26_balanced.env_cfg import IKDiagonalCfg as BalancedCfg


@configclass
class IKDiagonalCfg(BalancedCfg):
    contract_revision = 4
