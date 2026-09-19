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


@configclass
class DrobotCommandedRunningFastStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedWalkingSymmetryGatedRobustStraightCrawlExternalRearPayloadPPORunnerCfg
):
    """V31 topology adaptation for a fast diagonal run and stable stop."""

    max_iterations = 2000
    experiment_name = "drobot_commanded_run_v31_fast_stop_external_rear_payload"

    def __post_init__(self) -> None:
        super().__post_init__()
        # V31 changes the gait topology and adds within-episode command changes.
        # Restore exploration and a moderate learning rate for the nominal phase;
        # robust continuation can consolidate with smaller CLI overrides.
        self.algorithm.learning_rate = 2.0e-4
        self.algorithm.entropy_coef = 0.002
        self.algorithm.desired_kl = 0.010
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_left_right_symmetric_states,
        )


@configclass
class DrobotCommandedRunningHardwareMatchedStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningFastStopExternalRearPayloadPPORunnerCfg
):
    """V32 conservative continuation under measured actuator constraints."""

    max_iterations = 1800
    experiment_name = (
        "drobot_commanded_run_v32_hardware_matched_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Preserve useful V31 contact timing while allowing the actor to adapt
        # to half the target slew, broader actuator response, and a slower gait.
        self.algorithm.learning_rate = 8.0e-5
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.006


@configclass
class DrobotCommandedRunningAppliedTargetHistoryStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningHardwareMatchedStopExternalRearPayloadPPORunnerCfg
):
    """V33 continuation with controller-applied action history."""

    max_iterations = 1800
    experiment_name = (
        "drobot_commanded_run_v33_applied_target_history_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # Only the final 12 observation semantics change.  Use a small learning
        # rate so useful V32 gait timing survives while the actor relearns its
        # feedback response around the executable target history.
        self.algorithm.learning_rate = 6.0e-5
        self.algorithm.entropy_coef = 0.001
        self.algorithm.desired_kl = 0.005


@configclass
class DrobotCommandedRunningIntegratedTargetVelocityStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningAppliedTargetHistoryStopExternalRearPayloadPPORunnerCfg
):
    """V34 bounded target-velocity policy with a reset output head."""

    max_iterations = 1800
    experiment_name = (
        "drobot_commanded_run_v34_integrated_target_velocity_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # The reusable hidden layers already encode the V33 observation and
        # gait features.  Early V34 checkpoints were useful, but RSL-RL's
        # adaptive scheduler raised 1.5e-4 as high as 1.71e-3 and a continuation
        # rapidly forgot rear-foot release.  Use a conservative fixed rate;
        # prepare_ppo_continuation.py also removes any oversized rate and stale
        # Adam moments restored from an older checkpoint.
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.005
        # A controlled 0.25-coefficient mirror-loss continuation reduced fixed
        # randomized speed from 0.051 to 0.030-0.043 m/s and damaged the stopped
        # four-foot stance.  The real robot's fixed motor/load asymmetry is hidden
        # state at gait startup, so hard output equivariance is too restrictive.
        # Retain mirrored samples for broad coverage, but let joint/IMU feedback
        # drive asymmetric compensation.  The smaller rate and zero entropy bonus
        # consolidate the deterministic mean around the corrected pair gate.
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=compute_left_right_symmetric_states,
        )


@configclass
class DrobotCommandedRunningCommandScaledHighSpeedStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningIntegratedTargetVelocityStopExternalRearPayloadPPORunnerCfg
):
    """V35 command-scaled high-speed continuation of the qualified V34 gait."""

    max_iterations = 1200
    experiment_name = (
        "drobot_commanded_run_v35_command_scaled_high_speed_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.004


@configclass
class DrobotCommandedRunningFlatSoleHighSpeedStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningCommandScaledHighSpeedStopExternalRearPayloadPPORunnerCfg
):
    """V36 flat-sole reference continuation for the rectangular shoes."""

    max_iterations = 1400
    experiment_name = (
        "drobot_commanded_run_v36_flat_sole_high_speed_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.003


@configclass
class DrobotCommandedRunningFlatSoleResidualRunStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningFlatSoleHighSpeedStopExternalRearPayloadPPORunnerCfg
):
    """V37 low-authority residual adaptation around the corrected run gait."""

    max_iterations = 1400
    experiment_name = (
        "drobot_commanded_run_v37_flat_sole_residual_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.004


@configclass
class DrobotCommandedRunningStraightLowAbductionLongStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningFlatSoleHighSpeedStopExternalRearPayloadPPORunnerCfg
):
    """V38 low-abduction continuation with longer straight-path exposure."""

    max_iterations = 1400
    experiment_name = (
        "drobot_commanded_run_v38_straight_low_abduction_long_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # The inherited actor already has a useful forward gait. Use a small,
        # fixed continuation step so the new abduction rate contract and longer
        # path cost reshape that gait without erasing rear-foot release.
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.003


@configclass
class DrobotCommandedRunningReferenceDrivenStraightStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningFlatSoleResidualRunStopExternalRearPayloadPPORunnerCfg
):
    """V39 reference-driven run with a reset residual action head."""

    max_iterations = 1200
    experiment_name = (
        "drobot_commanded_run_v39_reference_driven_straight_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # V39 starts with a zero residual head, so the deterministic policy is
        # exactly the analytic flat-sole gait.  Retain V38's useful observation
        # features but adapt cautiously around that already executable baseline.
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.004


@configclass
class DrobotCommandedRunningSpeedFaithfulStraightStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningStraightLowAbductionLongStopExternalRearPayloadPPORunnerCfg
):
    """V40 small-step speed-fidelity continuation of V38."""

    max_iterations = 1200
    experiment_name = (
        "drobot_commanded_run_v40_speed_faithful_straight_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.003


@configclass
class DrobotCommandedRunningPhaseNormalizedSpeedStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningSpeedFaithfulStraightStopExternalRearPayloadPPORunnerCfg
):
    """V41 frequency-normalized continuation of the selected V40 actor."""

    max_iterations = 1200
    experiment_name = (
        "drobot_commanded_run_v41_phase_normalized_speed_stop_external_rear_payload"
    )


@configclass
class DrobotCommandedRunningLearnedCycleResidualStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningReferenceDrivenStraightStopExternalRearPayloadPPORunnerCfg
):
    """V42 residual adaptation around the distilled stable V40 phase cycle."""

    max_iterations = 1600
    experiment_name = (
        "drobot_commanded_run_v42_learned_cycle_residual_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 2.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.003


@configclass
class DrobotCommandedRunningFootWarpTrotResidualStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningLearnedCycleResidualStopExternalRearPayloadPPORunnerCfg
):
    """V43 cautious residual adaptation around the qualified foot-space trot."""

    max_iterations = 1600
    experiment_name = (
        "drobot_commanded_run_v43_foot_warp_trot_residual_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.5e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.0025


@configclass
class DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadPPORunnerCfg(
    DrobotCommandedRunningFootWarpTrotResidualStopExternalRearPayloadPPORunnerCfg
):
    """V44 separately trained flat-contact gait under measured hardware limits."""

    max_iterations = 1600
    experiment_name = (
        "drobot_commanded_run_v44_flat_contact_hardware_matched_stop_external_rear_payload"
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        # No implicit V43 bootstrap: the reference and output contract changed.
        self.algorithm.learning_rate = 3.0e-5
        self.algorithm.schedule = "fixed"
        self.algorithm.entropy_coef = 0.0
        self.algorithm.desired_kl = 0.004
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=False,
            data_augmentation_func=compute_left_right_symmetric_states,
        )
