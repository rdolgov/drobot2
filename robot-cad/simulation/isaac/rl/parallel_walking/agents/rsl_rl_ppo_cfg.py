"""PPO configurations for command-conditioned parallel walking."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DrobotGaussianDistributionCfg(RslRlMLPModelCfg.GaussianDistributionCfg):
    """Exploration distribution matching the successful SB3 PPO baseline."""

    init_std: float = 0.1
    std_type: str = "log"


@configclass
class DrobotCommandedWalkingForwardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Parallel reproduction of the independently validated walking PPO."""

    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 25
    experiment_name = "drobot_commanded_walk_forward_v15_rl_transfer_direct"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        # Keep the exact raw 48-value hardware observation contract used by the
        # proven SB3 actor.  The privileged critic remains normalized.
        obs_normalization=False,
        distribution_cfg=DrobotGaussianDistributionCfg(),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.015,
        max_grad_norm=0.5,
    )


@configclass
class DrobotCommandedWalkingDirectionalPPORunnerCfg(
    DrobotCommandedWalkingForwardPPORunnerCfg
):
    """Same network shape, expanded to forward/backward/turn commands."""

    experiment_name = "drobot_commanded_walk_directional_v15_rl_transfer_direct"
