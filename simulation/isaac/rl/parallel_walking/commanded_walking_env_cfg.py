"""Configuration for command-conditioned parallel Drobot walking."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROBOT_USD = (
    PROJECT_ROOT
    / "simulation"
    / "exports"
    / "isaac"
    / "quadruped_robot_floating.usdc"
)
EFFORT_CAP_NM = 0.8825985
SERVO_VELOCITY_LIMIT_RAD_S = 4.5836625
# The rectangular shoe controller uses an 80 mm fore/aft stance.  Equal and
# opposite hip/knee angles point the distal link (and therefore the shoe's
# contact-face normal) vertically down instead of balancing on a pitched edge.
RECTANGULAR_SHOE_STANCE_ANGLE_RAD = 0.5239596454
STABLE_NEUTRAL_FRONT_HIP_RAD = RECTANGULAR_SHOE_STANCE_ANGLE_RAD
STABLE_NEUTRAL_REAR_HIP_RAD = -RECTANGULAR_SHOE_STANCE_ANGLE_RAD
STABLE_NEUTRAL_FRONT_KNEE_RAD = -RECTANGULAR_SHOE_STANCE_ANGLE_RAD
STABLE_NEUTRAL_REAR_KNEE_RAD = RECTANGULAR_SHOE_STANCE_ANGLE_RAD

# 2026-08-13 CAD configuration.  The thin tread is modeled separately so its
# friction can be changed later without altering the structural shoe.
DISTAL_LINK_LENGTH_M = 0.159896689
RECTANGULAR_SHOE_SOLE_BACK_FROM_FORK_M = 0.024
RECTANGULAR_SHOE_SOLE_THICKNESS_M = 0.006
RECTANGULAR_SHOE_LENGTH_FORE_AFT_M = 0.100
RECTANGULAR_SHOE_WIDTH_LATERAL_M = 0.060
RECTANGULAR_SHOE_TREAD_PROJECTION_M = 0.001
RECTANGULAR_SHOE_TREAD_LENGTH_M = 0.094
RECTANGULAR_SHOE_TREAD_WIDTH_M = 0.054
RECTANGULAR_SHOE_MASS_KG = 0.070237

# The base-link inertia currently includes a provisional 450 g pack centered
# low in the chassis. V19 replaced that assumption with the measured CM5202
# pack plus CAD-derived box/lid mass on the body floor. V20 rotates and moves
# that same assembly outside the rear wall to match the physical installation.
# The exact bracket offsets remain estimates until measured on the robot.
ORIGINAL_BASE_MASS_KG = 2.049119
ORIGINAL_BASE_COM_M = (0.0, 0.0, 0.046485537)
ORIGINAL_BASE_INERTIA_KG_M2 = (
    0.0132456004,
    0.0000110328,
    0.0,
    0.0097442040,
    0.0,
    0.0196972212,
)
REPLACED_BATTERY_MASS_KG = 0.450
REPLACED_BATTERY_CENTER_M = (0.0, 0.0, 0.024)
REPLACED_BATTERY_SIZE_M = (0.070, 0.066, 0.040)


@configclass
class DrobotCommandedWalkingForwardEnvCfg(DirectRLEnvCfg):
    """Flat-ground, forward-first walking with hardware-reproducible inputs."""

    # The validated single-environment walking policy used a 60 Hz controller.
    # Thirty hertz made the parallel policy learn discrete foot stamping instead.
    decimation = 2
    # Allocate the final horizon up front. Training timeouts ramp from 8 to 32
    # seconds; preview may disable them entirely.
    episode_length_s = 32.0
    # Preview may override this through Hydra. Training leaves it false so PPO
    # continues to receive fixed-horizon episode boundaries.
    disable_time_limit = False
    initial_training_horizon_s = 8.0
    final_training_horizon_s = 32.0
    # 64 controller steps per PPO iteration -> 1,000 iterations to full horizon.
    episode_horizon_curriculum_steps = 64_000
    action_space = 12
    # command 3 + gait clock 2 + IMU 9 + joint position error 12 + velocity 12
    # + last action 12.  The clock is deployable from the controller timestep.
    observation_space = 50
    # policy observation 50 + privileged base velocity 3 + height 1 + contacts 4
    state_space = 58

    viewer: ViewerCfg = ViewerCfg(
        eye=(-1.40, -1.25, 0.65),
        lookat=(0.0, 0.0, -0.15),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**19),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.10,
            dynamic_friction=0.90,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        # A 32-second target-speed rollout covers 4.8 m.
        env_spacing=8.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.10,
            dynamic_friction=0.90,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.23, 0.20)),
        debug_vis=False,
    )
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROBOT_USD),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=True,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=5.0,
                max_angular_velocity=12.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3750),
            joint_pos={
                ".*_hip_abduction": 0.0,
                "front_.*_hip_flexion": STABLE_NEUTRAL_FRONT_HIP_RAD,
                "rear_.*_hip_flexion": STABLE_NEUTRAL_REAR_HIP_RAD,
                "front_.*_knee": STABLE_NEUTRAL_FRONT_KNEE_RAD,
                "rear_.*_knee": STABLE_NEUTRAL_REAR_KNEE_RAD,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            # Use PhysX's implicit drive, matching the validated manual world.
            # The equivalent explicit PD controller bang-banged at the velocity
            # limit under the measured 0.8826 N*m effort cap, even at zero action.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=EFFORT_CAP_NM,
                effort_limit_sim=EFFORT_CAP_NM,
                velocity_limit=SERVO_VELOCITY_LIMIT_RAD_S,
                velocity_limit_sim=SERVO_VELOCITY_LIMIT_RAD_S,
                stiffness=30.0,
                damping=SERVO_VELOCITY_LIMIT_RAD_S,
                friction=0.0,
                armature=0.0,
            )
        },
    )
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/.*",
        history_length=2,
        update_period=1.0 / 120.0,
        track_air_time=True,
        force_threshold=1.0,
    )
    imu_sensor: ImuCfg = ImuCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base_link",
        update_period=1.0 / 120.0,
    )

    command_profile = "forward"
    # Match the command used by the independently validated 1.16 m / 8 s policy.
    # V6-V9's slow command curriculum converged to standing and foot chatter.
    initial_forward_speed_min_m_s = 0.15
    initial_forward_speed_max_m_s = 0.15
    forward_speed_min_m_s = 0.15
    forward_speed_max_m_s = 0.15
    command_curriculum_steps = 1
    command_curriculum_offset_steps = 0
    backward_speed_min_m_s = 0.06
    backward_speed_max_m_s = 0.16
    turn_forward_speed_max_m_s = 0.10
    turn_rate_min_rad_s = 0.35
    turn_rate_max_rad_s = 0.80

    action_scale_abduction_rad = 0.12
    action_scale_hip_rad = 0.30
    action_scale_knee_rad = 0.40
    action_mode = "direct"
    residual_action_scale = 1.0
    # The deployable controller may impose a stricter target slew rate than the
    # actuator itself.  Keeping this configurable lets training reproduce that
    # packet-level behavior exactly.
    target_velocity_limit_rad_s = SERVO_VELOCITY_LIMIT_RAD_S
    reset_joint_position_noise_rad = 0.015
    reset_xy_jitter_m = 0.02
    # These are the validated walking termination limits.  V9's 0.28 m height
    # cutoff and 100-point failure cost made safe standing a strong local optimum.
    minimum_base_height_m = 0.22
    minimum_upright_cosine = 0.78
    base_contact_grace_steps = 10
    maximum_distance_from_origin_m = 6.5
    target_base_height_m = 0.3750
    velocity_tracking_sigma_m_s = 0.04
    distance_success_fraction = 0.75

    # Smooth diagonal-pair trot.  Front-left/rear-right share a phase and the
    # opposite diagonal pair is half a cycle away.  A 65% duty factor includes
    # short four-foot support intervals around pair transitions.
    gait_period_s = 0.80
    # Existing checkpoints use a fixed 1.25 Hz clock. A future policy must opt
    # into command-scaled cadence so deployment cannot change the meaning of a
    # trained observation silently.
    gait_clock_mode = "fixed"
    gait_standstill_deadband_m_s = 0.0
    gait_speed_min_m_s = 0.04
    gait_speed_max_m_s = 0.10
    gait_frequency_min_hz = 1.25
    gait_frequency_max_hz = 1.25
    gait_stride_scale_min = 1.0
    gait_duty_factor = 0.65
    gait_phase_offsets = (0.0, 0.5, 0.5, 0.0)
    gait_reference_mode = "continuous"
    gait_weight_shift_forward_m = 0.0
    gait_stride_m = 0.080
    gait_lift_m = 0.025
    gait_start_ramp_s = 0.60
    gait_reference_sigma = 0.75

    # Deliberately reproduce quadruped_walk_v1.yaml's successful reward instead
    # of continuing to tune a novel objective.  Net displacement remains a hard
    # evaluation metric, but adding it to the reward caused unsafe lunge exploits.
    reward_forward_velocity_tracking = 4.0
    # A two-second trailing speed window prevents a reset-dependent burst from
    # being mistaken for sustained walking.
    sustained_speed_window_s = 2.0
    minimum_sustained_speed_m_s = 0.08
    active_translation_threshold_m_s = 0.03
    scale_sustained_speed_with_command = False
    minimum_sustained_speed_fraction = 0.50
    # The prior policy remained smooth and stable but plateaued near 0.02 m/s.
    # Give PPO a non-vanishing linear gradient away from standing, then keep
    # the two-second term as the stronger sustained-motion signal.
    reward_instant_progress = 2.00
    reward_sustained_progress = 4.00
    penalty_sustained_stall = 4.00
    reward_gait_reference = 2.00
    reward_scheduled_stance = 0.50
    reward_scheduled_swing = 1.50
    reward_upright = 1.00
    # Keep survival far below the motion terms so standing cannot dominate, but
    # retain a small incentive to avoid deliberately terminating an episode.
    reward_alive = 0.10
    # The older baseline drifted about 1.6 m sideways over a 30-second trial.  The
    # broad shoes make heading retention physically achievable, so V18 prices
    # lateral and yaw motion more strongly instead of accepting the drift.
    penalty_lateral_velocity = 5.00
    penalty_lateral_displacement = 8.00
    lateral_corridor_half_width_m = 0.05
    penalty_lateral_corridor = 0.0
    penalty_vertical_velocity = 0.20
    penalty_roll_pitch_rate = 0.10
    penalty_body_tilt = 1.50
    penalty_yaw_rate = 5.00
    heading_error_normalizer_rad = 0.20
    penalty_heading_error = 0.0
    penalty_body_height = 2.0
    penalty_action_rate = 0.03
    penalty_action_acceleration = 0.12
    penalty_action_saturation = 0.50
    penalty_target_limiter_gap = 0.0
    # Contact timing and the analytic gait reference provide the primary
    # diagonal-pair coordination.  This stays deliberately light: forcing the
    # normalized joint actions to match made the learned robot freeze because
    # its mirrored joint frames and load distribution need small asymmetries.
    penalty_diagonal_sync = 0.10
    penalty_action_magnitude = 0.002
    penalty_joint_velocity = 0.010
    penalty_support_foot_slip = 0.30
    penalty_touchdown_impact = 0.05
    touchdown_force_soft_limit_n = 18.0
    reward_qualified_touchdown = 0.025
    reward_least_active_swing = 0.0
    reward_three_foot_support = 0.0
    penalty_excess_airborne_feet = 0.0
    penalty_termination = 350.0
    # Keep value targets conditioned while preserving all reward ratios.
    reward_scale = 0.10
    qualified_foot_air_time_s = 0.10

    rear_payload_enabled = False
    rear_payload_mass_kg = 0.0
    rear_payload_center_m = (0.0, 0.0, 0.0)
    rear_payload_size_m = (0.0, 0.0, 0.0)
    rear_payload_combined_mass_scale_range = (1.0, 1.0)
    rear_payload_combined_com_jitter_m = (0.0, 0.0, 0.0)
    smoothness_curriculum_steps = 0
    smoothness_initial_scale = 1.0
    joint_acceleration_normalizer_rad_s2 = 40.0
    body_linear_acceleration_normalizer_m_s2 = 5.0
    body_angular_acceleration_normalizer_rad_s2 = 10.0
    penalty_joint_acceleration = 0.0
    penalty_body_linear_acceleration = 0.0
    penalty_body_angular_acceleration = 0.0


@configclass
class DrobotCommandedWalkingSmoothPayloadEnvCfg(
    DrobotCommandedWalkingForwardEnvCfg
):
    """Rear-battery continuation prioritizing low-acceleration locomotion."""

    rear_payload_enabled = True
    # 416 g measured battery + 80.196 g box + 26.984 g lid. Fasteners, foam,
    # and wire bulges remain omitted until they are weighed.
    rear_payload_mass_kg = 0.523179545
    rear_payload_center_m = (-0.039104, 0.0, 0.024521)
    rear_payload_size_m = (0.144, 0.068, 0.043)
    # Approximate 450-600 g payload uncertainty without changing the policy
    # inputs. The COM jitter corresponds to roughly +/-12 mm uncertainty in the
    # payload placement after accounting for the dry chassis mass.
    rear_payload_combined_mass_scale_range = (0.965, 1.040)
    rear_payload_combined_com_jitter_m = (0.0030, 0.0020, 0.0020)

    # Train the deployable low-speed range directly. Continuing from V18 and
    # retaining a hard stall cost prevents smooth standing from becoming the
    # preferred solution.
    initial_forward_speed_min_m_s = 0.04
    initial_forward_speed_max_m_s = 0.10
    forward_speed_min_m_s = 0.04
    forward_speed_max_m_s = 0.10
    command_curriculum_steps = 1
    minimum_sustained_speed_m_s = 0.025
    distance_success_fraction = 0.55

    gait_stride_m = 0.055
    gait_lift_m = 0.018
    gait_start_ramp_s = 1.20

    # Motion remains required, but speed is deliberately subordinate to smooth
    # gait tracking, low accelerations, gentle contacts, and stable posture.
    reward_forward_velocity_tracking = 1.50
    reward_instant_progress = 2.00
    reward_sustained_progress = 3.00
    penalty_sustained_stall = 6.00
    reward_gait_reference = 3.00
    reward_scheduled_stance = 0.75
    reward_scheduled_swing = 1.75
    reward_upright = 1.50
    penalty_roll_pitch_rate = 0.30
    penalty_body_tilt = 2.00
    penalty_body_height = 3.00
    penalty_action_rate = 0.12
    penalty_action_acceleration = 0.65
    penalty_joint_velocity = 0.030
    penalty_joint_acceleration = 0.20
    penalty_body_linear_acceleration = 0.30
    penalty_body_angular_acceleration = 0.25
    penalty_touchdown_impact = 0.12
    touchdown_force_soft_limit_n = 14.0
    penalty_support_foot_slip = 0.45
    smoothness_curriculum_steps = 19_200
    smoothness_initial_scale = 0.25


@configclass
class DrobotCommandedWalkingExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingSmoothPayloadEnvCfg
):
    """Smooth walking with the battery box mounted outside the rear plate."""

    # The 144 x 68 mm box face is centered on the 170 x 100 mm rear plate.
    # Its 43 mm box-and-lid depth projects behind the body whose rear face is
    # X=-110 mm. This rotates the holder's CAD axes so its long axis spans Y,
    # its tab width spans Z, and its lid/depth spans X.
    rear_payload_center_m = (-0.1315, 0.0, 0.0500)
    rear_payload_size_m = (0.043, 0.144, 0.068)
    # Keep the measured 523.18 g nominal assembly and 450--600 g mass range,
    # but allow extra fore/aft and vertical placement uncertainty because the
    # physical adapter between the box and rear-wall grid is not yet measured.
    rear_payload_combined_com_jitter_m = (0.0045, 0.0020, 0.0030)

    # The first external-payload continuation recovered commanded speed but
    # learned a persistent lateral arc. Preserve V19's smooth gait and make
    # straight, low-acceleration progress the dominant adaptation target.
    penalty_lateral_velocity = 10.00
    penalty_lateral_displacement = 20.00
    penalty_yaw_rate = 8.00
    penalty_action_acceleration = 0.75
    penalty_joint_acceleration = 0.25
    penalty_body_linear_acceleration = 0.40
    penalty_body_angular_acceleration = 0.30


@configclass
class DrobotCommandedWalkingLowSpeedExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingExternalRearPayloadEnvCfg
):
    """V21 continuation with a deployable clock for very slow smooth walking."""

    # Begin inside V20's learned range, then introduce slower commands over 600
    # 64-step PPO rollouts. The workflow intentionally resets this new
    # curriculum even when bootstrapping the network weights from model_900.pt.
    initial_forward_speed_min_m_s = 0.04
    initial_forward_speed_max_m_s = 0.10
    forward_speed_min_m_s = 0.005
    forward_speed_max_m_s = 0.10
    command_curriculum_steps = 38_400
    episode_horizon_curriculum_steps = 38_400

    # A slower command produces both a slower clock and a shorter step. The
    # lowest mapping is about one 19 mm stride every 2.86 seconds, rather than
    # asking the old 1.25 Hz gait to shuffle at a tiny velocity command.
    gait_clock_mode = "speed_scaled"
    gait_standstill_deadband_m_s = 0.002
    gait_speed_min_m_s = 0.005
    gait_speed_max_m_s = 0.10
    gait_frequency_min_hz = 0.35
    gait_frequency_max_hz = 1.25
    gait_stride_scale_min = 0.35

    # A fixed 0.025 m/s stall threshold would label every correct 0.005 m/s
    # rollout as a failure. Preserve the strong anti-freeze reward while making
    # its threshold proportional to the command at low speed.
    active_translation_threshold_m_s = 0.002
    scale_sustained_speed_with_command = True
    minimum_sustained_speed_fraction = 0.50


@configclass
class DrobotCommandedWalkingLowSpeedCrawlExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingExternalRearPayloadEnvCfg
):
    """V22 deployment-matched, slow sequential crawl with three-foot support."""

    # Train around the speeds used by the hardware UI, then gradually expose
    # very small commands.  Speed is intentionally secondary to stable support
    # and smooth target motion.
    initial_forward_speed_min_m_s = 0.008
    initial_forward_speed_max_m_s = 0.015
    forward_speed_min_m_s = 0.003
    forward_speed_max_m_s = 0.015
    command_curriculum_steps = 51_200
    episode_horizon_curriculum_steps = 51_200

    gait_clock_mode = "speed_scaled"
    gait_standstill_deadband_m_s = 0.002
    gait_speed_min_m_s = 0.003
    gait_speed_max_m_s = 0.015
    gait_frequency_min_hz = 0.06
    gait_frequency_max_hz = 0.30
    gait_stride_scale_min = 1.0
    gait_duty_factor = 0.8625
    # LEG_NAMES is FL, RL, FR, RR.  These offsets schedule the proven hardware
    # crawl order RR -> FR -> RL -> FL with only one leg in swing at a time.
    gait_phase_offsets = (0.07, 0.32, 0.57, 0.82)
    gait_stride_m = 0.050
    gait_lift_m = 0.016
    gait_start_ramp_s = 1.50
    gait_reference_mode = "distributed_push"
    gait_weight_shift_forward_m = 0.006
    action_mode = "gait_residual"
    residual_action_scale = 0.25

    # The Raspberry Pi sends at most 2 degrees per 60 Hz control tick.  At
    # 120 deg/s this simulator limiter has the identical discrete-time cap.
    target_velocity_limit_rad_s = math.radians(120.0)

    active_translation_threshold_m_s = 0.002
    scale_sustained_speed_with_command = True
    minimum_sustained_speed_fraction = 0.40
    minimum_sustained_speed_m_s = 0.006

    reward_forward_velocity_tracking = 1.50
    reward_instant_progress = 2.00
    reward_sustained_progress = 3.00
    penalty_sustained_stall = 6.00
    reward_gait_reference = 12.00
    reward_scheduled_stance = 1.50
    reward_scheduled_swing = 4.00
    reward_qualified_touchdown = 0.10
    reward_least_active_swing = 4.00
    reward_three_foot_support = 2.00
    penalty_excess_airborne_feet = 5.00
    penalty_diagonal_sync = 0.0
    penalty_action_rate = 0.18
    penalty_action_acceleration = 0.90
    penalty_action_saturation = 0.75
    penalty_target_limiter_gap = 1.50
    penalty_joint_velocity = 0.040
    penalty_joint_acceleration = 0.30
    penalty_body_linear_acceleration = 0.55
    penalty_body_angular_acceleration = 0.45
    penalty_body_tilt = 3.00
    penalty_roll_pitch_rate = 0.50
    penalty_lateral_velocity = 12.00
    penalty_lateral_displacement = 24.00
    penalty_yaw_rate = 10.00
    penalty_support_foot_slip = 0.60
    penalty_touchdown_impact = 0.18


@configclass
class DrobotCommandedWalkingHigherSpeedStraightCrawlExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingLowSpeedCrawlExternalRearPayloadEnvCfg
):
    """V23 faster residual crawl with explicit lateral and heading retention."""

    # Start close to V22, then expose the full range over 800 PPO rollouts. The
    # 65 mm reference and speed-scaled clock correspond to roughly 0.005-0.049
    # m/s before learned residual corrections and contact losses.
    initial_forward_speed_min_m_s = 0.010
    initial_forward_speed_max_m_s = 0.030
    forward_speed_min_m_s = 0.005
    forward_speed_max_m_s = 0.050
    command_curriculum_steps = 51_200
    episode_horizon_curriculum_steps = 51_200

    gait_speed_min_m_s = 0.005
    gait_speed_max_m_s = 0.050
    gait_frequency_min_hz = 0.12
    gait_frequency_max_hz = 0.75
    gait_stride_scale_min = 0.65
    gait_stride_m = 0.065
    gait_weight_shift_forward_m = 0.008

    reward_forward_velocity_tracking = 2.50
    reward_instant_progress = 3.00
    reward_sustained_progress = 4.00
    penalty_sustained_stall = 7.00

    # V22 already discouraged sideways motion, but its raw squared-displacement
    # term was weak for a few centimeters of real drift. V23 adds a normalized
    # 20 mm corridor penalty and prices yaw/heading departure independently.
    penalty_lateral_velocity = 20.00
    penalty_lateral_displacement = 40.00
    lateral_corridor_half_width_m = 0.020
    penalty_lateral_corridor = 2.50
    penalty_yaw_rate = 18.00
    heading_error_normalizer_rad = 0.12
    penalty_heading_error = 2.00

    # Preserve V22's strong anti-jerk objective at the higher cadence. Candidate
    # selection may reject later/faster checkpoints if acceleration rises.
    penalty_action_rate = 0.18
    penalty_action_acceleration = 0.90
    penalty_joint_acceleration = 0.30
    penalty_body_linear_acceleration = 0.55
    penalty_body_angular_acceleration = 0.45


@configclass
class DrobotCommandedWalkingDirectionalEnvCfg(DrobotCommandedWalkingForwardEnvCfg):
    """Episode commands include forward, backward, left, right, and stop."""

    command_profile = "directional"
