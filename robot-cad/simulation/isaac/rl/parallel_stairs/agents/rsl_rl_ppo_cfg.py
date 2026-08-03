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

    experiment_name = "drobot_pure_stairs_first_step_landing_consolidate_hip_180x250_direct"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.005


@configclass
class DrobotPureStairsFirstStepContactRetentionHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    """Explore centered touchdown while retaining all three other supports."""

    experiment_name = "drobot_pure_stairs_first_step_contact_retention_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepBroadSupportRetentionHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    """Intermediate PPO stage: broad tread contact with all four supports."""

    experiment_name = "drobot_pure_stairs_first_step_broad_support_retention_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105SupportRetentionHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    """Narrow broad four-support contacts to a 210 mm centered tread band."""

    experiment_name = "drobot_pure_stairs_first_step_width105_support_retention_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth90SupportRetentionHipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    """Narrow four-support contacts to a 180 mm centered tread band."""

    experiment_name = "drobot_pure_stairs_first_step_width90_support_retention_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Rise10HipPPORunnerCfg(
    DrobotPureStairsFirstStepLandingHipPPORunnerCfg
):
    """Transfer the verified +/-105 mm landing into 10 mm of body rise."""

    experiment_name = "drobot_pure_stairs_first_step_width105_rise10_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105SupportRetentionHipPPORunnerCfg
):
    """Conservative PPO transfer into the first lower reset posture."""

    experiment_name = "drobot_pure_stairs_first_step_width105_low25_hip_180x250_direct"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.desired_kl = 0.005


@configclass
class DrobotPureStairsFirstStepWidth105Low50HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_first_step_width105_low50_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Low25To37HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Transfer model 734 across correlated Low25-to-Low37.5 resets."""

    experiment_name = "drobot_pure_stairs_first_step_width105_low25_to_37_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Low37HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Transfer the mixed-reset winner to the fixed Low37.5 posture."""

    experiment_name = "drobot_pure_stairs_first_step_width105_low37_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Low25To37HardBiasHipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Continue model 776 with 75% of resets in the bridge's harder half."""

    experiment_name = (
        "drobot_pure_stairs_first_step_width105_low25_to_37_hard_bias_hip_180x250_direct"
    )


@configclass
class DrobotPureStairsFirstStepWidth105Low25To37HardBiasRise10HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Conservatively continue model 895 into supported 10 mm body rise."""

    experiment_name = (
        "drobot_pure_stairs_first_step_width105_low25_to_37_hard_bias_rise10_hip_180x250_direct"
    )


@configclass
class DrobotPureStairsLow25To37HardBiasStandRise10HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Learn a symmetric four-support 10 mm stand-up before tread transfer."""

    experiment_name = "drobot_pure_stairs_low25_to_37_hard_bias_stand_rise10_hip_180x250_direct"


@configclass
class DrobotPureStairsLow25To37HardBiasThreeSupportRise10HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Learn 10 mm body rise with any stable three-foot support set."""

    experiment_name = (
        "drobot_pure_stairs_low25_to_37_hard_bias_three_support_rise10_hip_180x250_direct"
    )


@configclass
class DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Bridge into a held 5 mm rise with any symmetric two-foot support set."""

    experiment_name = (
        "drobot_pure_stairs_low25_to_37_hard_bias_two_support_rise5_hip_180x250_direct"
    )


@configclass
class DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipConsolidatePPORunnerCfg(
    DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipPPORunnerCfg
):
    """Conservatively move rare held two-support rises into the policy mean."""

    experiment_name = (
        "drobot_pure_stairs_low25_to_37_hard_bias_two_support_rise5_hip_consolidate_180x250_direct"
    )
    num_steps_per_env = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.desired_kl = 0.003
        self.algorithm.num_learning_epochs = 10
        self.algorithm.num_mini_batches = 8


@configclass
class DrobotPureStairsFullFoldTwoSupportRise5HipPPORunnerCfg(
    DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipPPORunnerCfg
):
    """Learn the held two-support rise from the verified full-fold reset."""

    experiment_name = "drobot_pure_stairs_full_fold_two_support_rise5_hip_180x250_direct"


@configclass
class DrobotPureStairsYaw45FullFoldTwoSupportRise5HipPPORunnerCfg(
    DrobotPureStairsFullFoldTwoSupportRise5HipPPORunnerCfg
):
    """Continue the full-fold policy at a fixed 45-degree stair approach."""

    experiment_name = "drobot_pure_stairs_yaw45_full_fold_two_support_rise5_hip_180x250_direct"


@configclass
class DrobotPureStairsYaw67p5FullFoldTwoSupportRise5HipPPORunnerCfg(
    DrobotPureStairsFullFoldTwoSupportRise5HipPPORunnerCfg
):
    """Continue the full-fold policy at a fixed 67.5-degree approach."""

    experiment_name = "drobot_pure_stairs_yaw67p5_full_fold_two_support_rise5_hip_180x250_direct"


@configclass
class DrobotPureStairsYaw90FullFoldTwoSupportRise5HipPPORunnerCfg(
    DrobotPureStairsFullFoldTwoSupportRise5HipPPORunnerCfg
):
    """Evaluate and continue the gradual-yaw policy at 90 degrees."""

    experiment_name = "drobot_pure_stairs_yaw90_full_fold_two_support_rise5_hip_180x250_direct"


@configclass
class DrobotPureStairsYaw90FullFoldFootLift10HipPPORunnerCfg(
    DrobotPureStairsFullFoldTwoSupportRise5HipPPORunnerCfg
):
    """Start a staged force-backed lift from the retained lateral policy."""

    experiment_name = "drobot_pure_stairs_yaw90_full_fold_foot_lift10_hip_180x250_direct"
    save_interval = 1


@configclass
class DrobotPureStairsYaw90FullFoldFootLift5HipPPORunnerCfg(
    DrobotPureStairsYaw90FullFoldFootLift10HipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_yaw90_full_fold_foot_lift5_hip_180x250_direct"


@configclass
class DrobotPureStairsYaw90FullFoldFootLift5ConsolidateHipPPORunnerCfg(
    DrobotPureStairsYaw90FullFoldFootLift5HipPPORunnerCfg
):
    """Retain rare complete lateral unloads with long, low-noise PPO batches."""

    experiment_name = "drobot_pure_stairs_yaw90_full_fold_foot_lift5_consolidate_hip_180x250_direct"
    num_steps_per_env = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.desired_kl = 0.003
        self.algorithm.num_learning_epochs = 10
        self.algorithm.num_mini_batches = 8


@configclass
class DrobotPureStairsYaw90FullFoldFootLift10ConsolidateHipPPORunnerCfg(
    DrobotPureStairsYaw90FullFoldFootLift5ConsolidateHipPPORunnerCfg
):
    experiment_name = (
        "drobot_pure_stairs_yaw90_full_fold_foot_lift10_consolidate_hip_180x250_direct"
    )


@configclass
class DrobotPureStairsYaw90FullFoldFootLift14HipPPORunnerCfg(
    DrobotPureStairsYaw90FullFoldFootLift10HipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_yaw90_full_fold_foot_lift14_hip_180x250_direct"


@configclass
class DrobotPureStairsYaw90FullFoldFootLift19HipPPORunnerCfg(
    DrobotPureStairsYaw90FullFoldFootLift10HipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_yaw90_full_fold_foot_lift19_hip_180x250_direct"


@configclass
class DrobotPureStairsSidewaysTwoSupportRise5HipPPORunnerCfg(
    DrobotPureStairsLow25To37HardBiasTwoSupportRise5HipPPORunnerCfg
):
    """Learn the held two-support rise with a lateral body and hip leverage."""

    experiment_name = "drobot_pure_stairs_sideways_two_support_rise5_hip_180x250_direct"


@configclass
class DrobotPureStairsLow25To37HardBiasUprightRise10HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Learn a held 10 mm upright body rise without prescribing a contact pattern."""

    experiment_name = "drobot_pure_stairs_low25_to_37_hard_bias_upright_rise10_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Low75HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_first_step_width105_low75_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Low100HipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    experiment_name = "drobot_pure_stairs_first_step_width105_low100_hip_180x250_direct"


@configclass
class DrobotPureStairsFirstStepWidth105Low25ConsolidateHipPPORunnerCfg(
    DrobotPureStairsFirstStepWidth105Low25HipPPORunnerCfg
):
    """Use complete rare landing sequences to move the lower-reset policy mean."""

    experiment_name = "drobot_pure_stairs_first_step_width105_low25_consolidate_hip_180x250_direct"
    num_steps_per_env = 64

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.entropy_coef = 0.0
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.desired_kl = 0.003
        self.algorithm.num_learning_epochs = 10
        self.algorithm.num_mini_batches = 8


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
class DrobotPureStairsFootLiftConsolidateHipPPORunnerCfg(DrobotPureStairsFootLiftHipPPORunnerCfg):
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
