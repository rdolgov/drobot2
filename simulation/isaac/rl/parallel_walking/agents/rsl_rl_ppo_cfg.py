"""PPO configurations for command-conditioned parallel walking."""

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)

from ..left_right_symmetry import compute_left_right_symmetric_states


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


@configclass
class DrobotCommandedWalkingLowSpeedCrawlExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingExternalRearPayloadPPORunnerCfg
):
    """V22 continuation for a deployment-matched sequential crawl."""

    max_iterations = 1000
    experiment_name = "drobot_commanded_walk_v22_low_speed_crawl_external_rear_payload"

    def __post_init__(self) -> None:
        super().__post_init__()
        # The crawl changes contact topology completely.  Baseline-sized
        # updates and extra exploration are required when training it fresh;
        # deployment gating protects against accepting an unstable checkpoint.
        self.algorithm.learning_rate = 3.0e-4
        self.algorithm.entropy_coef = 0.003
        self.algorithm.desired_kl = 0.015


@configclass
class DrobotCommandedWalkingHigherSpeedStraightCrawlExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingLowSpeedCrawlExternalRearPayloadPPORunnerCfg
):
    """V23 V22-continuation for faster, straighter residual crawling."""

    max_iterations = 1200
    experiment_name = (
        "drobot_commanded_walk_v23_higher_speed_straight_crawl_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Continuation starts from the selected V22 residual actor, so use
        # smaller updates than the fresh V22 topology search.
        self.algorithm.learning_rate = 1.0e-4
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.008


@configclass
class DrobotCommandedWalkingPaddedFeetForwardBiasExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingHigherSpeedStraightCrawlExternalRearPayloadPPORunnerCfg
):
    """V24 conservative V23 continuation for padded feet and forward bias."""

    max_iterations = 1000
    experiment_name = (
        "drobot_commanded_walk_v24_padded_feet_forward_bias_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 7.5e-5
        self.algorithm.entropy_coef = 0.0005
        self.algorithm.desired_kl = 0.006


@configclass
class DrobotCommandedWalkingRobustStraightLowStanceExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingPaddedFeetForwardBiasExternalRearPayloadPPORunnerCfg
):
    """V25 V24-continuation for asymmetric hardware and straight travel."""

    max_iterations = 1600
    experiment_name = (
        "drobot_commanded_walk_v25_robust_straight_low_stance_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # V25 retains V24's actor contract but changes the stance, analytic gait,
        # and physical-randomization distribution.  Keep continuation updates
        # conservative while restoring enough exploration to adapt to mirrored
        # per-joint asymmetry rather than preserving one nominal compensation.
        self.algorithm.learning_rate = 7.5e-5
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.006
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_left_right_symmetric_states,
        )


@configclass
class DrobotCommandedWalkingBalancedFourLegStraightCrawlExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingRobustStraightLowStanceExternalRearPayloadPPORunnerCfg
):
    """V26 controlled V25 ablation with explicit bilateral symmetry loss."""

    experiment_name = (
        "drobot_commanded_walk_v26_balanced_four_leg_straight_crawl_"
        "external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=compute_left_right_symmetric_states,
            mirror_loss_coeff=1.0,
        )


@configclass
class DrobotCommandedWalkingAdaptiveAsymmetricFourLegStraightCrawlExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingBalancedFourLegStraightCrawlExternalRearPayloadPPORunnerCfg
):
    """V27 corrected-reference policy that permits learned leg asymmetry."""

    experiment_name = (
        "drobot_commanded_walk_v27_adaptive_asymmetric_four_leg_straight_crawl_"
        "external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # The real robot, rear payload, printed link tolerances, and independent
        # actuators need not be exactly bilateral.  Disable action-level mirror
        # constraints in both phases so the policy can compensate an observed
        # bias.  Mirrored domain samples still cover both directions and prevent
        # one fixed hardware bias from becoming the only solution.
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=False,
            data_augmentation_func=compute_left_right_symmetric_states,
        )


@configclass
class DrobotCommandedWalkingForwardBiasedCycleGatedFourLegStraightCrawlExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingAdaptiveAsymmetricFourLegStraightCrawlExternalRearPayloadPPORunnerCfg
):
    """V28 deterministic-mean consolidation around the forward-biased crawl."""

    experiment_name = (
        "drobot_commanded_walk_v28_forward_biased_cycle_gated_four_leg_"
        "straight_crawl_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # V27's sampled Beta tails occasionally released the rear shoes while
        # its deployable mean action did not.  Remove the explicit entropy bonus
        # so return improvements consolidate into the deterministic policy mean.
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.004


@configclass
class DrobotCommandedWalkingScheduleMatchedSupportStraightCrawlExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingForwardBiasedCycleGatedFourLegStraightCrawlExternalRearPayloadPPORunnerCfg
):
    """V29 support-identity correction and gradual speed continuation."""

    experiment_name = (
        "drobot_commanded_walk_v29_schedule_matched_support_straight_crawl_"
        "external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # The reward landscape changes sharply when wrong-foot support loses
        # progress credit. Use a smaller continuation step while keeping the
        # deployable deterministic mean consolidated.
        self.algorithm.learning_rate = 5.0e-5
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.004


@configclass
class DrobotCommandedWalkingSymmetryGatedRobustStraightCrawlExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingScheduleMatchedSupportStraightCrawlExternalRearPayloadPPORunnerCfg
):
    """V30 straight-path gating with symmetry data augmentation."""

    experiment_name = (
        "drobot_commanded_walk_v30_symmetry_gated_robust_straight_crawl_"
        "external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Mirrored samples teach an unbiased geometric prior while mirror loss
        # stays off, allowing the actor to compensate each randomized assembly.
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_left_right_symmetric_states,
        )
