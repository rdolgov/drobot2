"""PPO configurations for command-conditioned parallel walking."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class DrobotBoundedBetaDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    """Native bounded distribution: sampled and deployed actions stay usable."""

    class_name: str = "BetaDistribution"
    action_range: tuple[float, float] = (-1.0, 1.0)


@configclass
class DrobotCommandedWalkingForwardPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Parallel reproduction of the independently validated walking PPO."""

    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 25
    experiment_name = "drobot_commanded_walk_forward_v18_coordinated_trot_selected"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256],
        activation="elu",
        # The deployable observation adds only a sine/cosine gait clock to the
        # proven command, IMU, joint-state, and previous-action contract.
        obs_normalization=False,
        distribution_cfg=DrobotBoundedBetaDistributionCfg(),
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

    experiment_name = "drobot_commanded_walk_directional_v18_coordinated_trot_selected"


@configclass
class DrobotCommandedWalkingSmoothPayloadPPORunnerCfg(
    DrobotCommandedWalkingForwardPPORunnerCfg
):
    """V19 continuation for the rear battery and low-acceleration gait."""

    max_iterations = 600
    save_interval = 25
    experiment_name = "drobot_commanded_walk_v19_smooth_rear_payload"


@configclass
class DrobotCommandedWalkingExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingSmoothPayloadPPORunnerCfg
):
    """V20 continuation for the externally mounted rear battery assembly."""

    max_iterations = 300
    experiment_name = "drobot_commanded_walk_v20_external_rear_payload_straight"

    def __post_init__(self) -> None:
        super().__post_init__()
        # A conservative continuation protects the already-smooth V19 gait
        # while the critic and actor adapt to the rearward inertia shift.
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.005


@configclass
class DrobotCommandedWalkingLowSpeedExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingExternalRearPayloadPPORunnerCfg
):
    """V21 continuation for speed-scaled, very-low-speed smooth walking."""

    max_iterations = 800
    experiment_name = "drobot_commanded_walk_v21_low_speed_external_rear_payload"
