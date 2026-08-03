"""Small first-pass PPO configuration for pure stair learning."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DrobotPureStairsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO with enough capacity for depth plus proprioception, without oversizing."""

    num_steps_per_env = 24
    max_iterations = 80
    save_interval = 20
    experiment_name = "drobot_pure_stairs_180x250_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )


@configclass
class DrobotPureStairsHipPPORunnerCfg(DrobotPureStairsPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_hip_180x250_direct"

    def __post_init__(self) -> None:
        super().__post_init__()
        # Once tread contact emerges, consolidate instead of continually
        # inflating action noise around the real joint limits.
        self.algorithm.entropy_coef = 0.002


@configclass
class DrobotPureStairsLowHipPPORunnerCfg(DrobotPureStairsPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_low_hip_180x250_direct"


@configclass
class DrobotPureStairsSidewaysHipPPORunnerCfg(DrobotPureStairsPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_sideways_hip_180x250_direct"
