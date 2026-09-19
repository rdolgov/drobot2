"""Explicit study-only structured config factory; never registered by default."""
from isaaclab.utils.configclass import configclass
from .agents.rsl_rl_ppo_cfg import (
    DrobotBoundedBetaDistributionCfg,
    DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadPPORunnerCfg,
)


@configclass
class Concentration37Cfg(DrobotBoundedBetaDistributionCfg):
    class_name: str = 'parallel_walking.beta_concentration37:ConcentratedBeta37'
    concentration_factor: float = 1.0


def agent_config37():
    cfg = DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadPPORunnerCfg()
    cfg.actor.distribution_cfg = Concentration37Cfg()
    return cfg
