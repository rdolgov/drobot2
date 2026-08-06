"""PPO configurations for command-conditioned parallel walking."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DrobotBoundedBetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """Native bounded policy: training and deployment both stay in [-1, 1]."""

    class_name: str = "BetaDistribution"
    action_range: tuple[float, float] = (-1.0, 1.0)


@configclass
class DrobotCommandedWalkingForwardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Compact policy with ample capacity for the 48-value hardware input."""

    num_steps_per_env = 32
    max_iterations = 500
    save_interval = 25
    experiment_name = "drobot_commanded_walk_forward_v3_direct"
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=DrobotBoundedBetaDistributionCfg(),
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
        entropy_coef=0.0005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=1.0,
    )


@configclass
class DrobotCommandedWalkingDirectionalPPORunnerCfg(
    DrobotCommandedWalkingForwardPPORunnerCfg
):
    """Same network shape, expanded to forward/backward/turn commands."""

    experiment_name = "drobot_commanded_walk_directional_v3_direct"
