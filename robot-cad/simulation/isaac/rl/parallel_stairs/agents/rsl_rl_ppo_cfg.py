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


@configclass
class DrobotPureStairsFirstStepHipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_first_step_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepLandingHipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_first_step_landing_hip_180x250_direct"
    save_interval = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.01


@configclass
class DrobotPureStairsFirstStepLandingLongHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    """Longer on-policy batches retain complete rare landing sequences."""

    experiment_name = "drobot_pure_stairs_first_step_landing_long_hip_180x250_direct"
    num_steps_per_env = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.num_learning_epochs = 10
        self.algorithm.num_mini_batches = 8


@configclass
class DrobotPureStairsFirstStepLandingConsolidateHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    """Low-entropy PPO stage for retaining rare exact landing sequences."""

    experiment_name = (
        "drobot_pure_stairs_first_step_landing_consolidate_hip_180x250_direct"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.005


@configclass
class DrobotPureStairsFirstStepLandingLowHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_first_step_landing_low_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepLandingSidewaysHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_first_step_landing_sideways_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepClose1HipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_first_step_close1_hip_180x250_direct"
    save_interval = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.01


@configclass
class DrobotPureStairsFirstStepClose2HipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_first_step_close2_hip_180x250_direct"
    save_interval = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.desired_kl = 0.01


@configclass
class DrobotPureStairsFirstStepClose4HipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_first_step_close4_hip_180x250_direct"
    save_interval = 10


@configclass
class DrobotPureStairsFirstStepClose6HipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_first_step_close6_hip_180x250_direct"
    save_interval = 10


@configclass
class DrobotPureStairsFootLiftHipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_foot_lift_hip_direct"
    save_interval = 10


@configclass
class DrobotPureStairsFootLiftConsolidateHipPPORunnerCfg(
    DrobotPureStairsFootLiftHipPPORunnerCfg
):
    """Low-entropy PPO stage for moving rare supported lifts into the mean."""

    experiment_name = "drobot_pure_stairs_foot_lift_consolidate_hip_direct"
    save_interval = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.005


@configclass
class DrobotPureStairsFootLift10HipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_foot_lift_10_hip_direct"
    save_interval = 10


@configclass
class DrobotPureStairsFootLift14HipPPORunnerCfg(DrobotPureStairsHipPPORunnerCfg):
    experiment_name = "drobot_pure_stairs_foot_lift_14_hip_direct"
    save_interval = 10
