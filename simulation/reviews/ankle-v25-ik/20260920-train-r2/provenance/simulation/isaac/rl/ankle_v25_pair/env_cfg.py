from isaaclab.utils.configclass import configclass
from ankle_v25_ik.env_cfg import IKDiagonalCfg as ReferenceCfg


@configclass
class IKDiagonalCfg(ReferenceCfg):
    contract_revision = 2
    curriculum_steps = 6400
    residual_scales = (.14, .22, .35, .15)
    swing_extra_lift_m = .025
    swing_forward_window_m = .012
