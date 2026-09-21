from isaaclab.utils.configclass import configclass
from ankle_v25_pair.env_cfg import IKDiagonalCfg as PairCfg


@configclass
class IKDiagonalCfg(PairCfg):
    contract_revision = 3
    # This experiment continues a trained full-stride policy. Do not restart
    # the short-stride curriculum when a new simulator process is created.
    curriculum_steps = 1
