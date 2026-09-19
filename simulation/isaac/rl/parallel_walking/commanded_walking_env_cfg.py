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

from simulation.isaac._quadruped_runtime import (
    stance_forward_bias_for_body_pitch_m,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROBOT_USD = (
    PROJECT_ROOT
    / "simulation"
    / "exports"
    / "isaac"
    / "quadruped_robot_floating.usdc"
)
EFFORT_CAP_NM = 0.8825985
SUSTAINABLE_RATED_EFFORT_NM = 0.980665
STALL_EFFORT_NM = 2.941995
HARDWARE_TORQUE_LIMIT_FRACTION = 0.90
TRANSIENT_EFFORT_CAP_NM = STALL_EFFORT_NM * HARDWARE_TORQUE_LIMIT_FRACTION
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
LOW_SWEPT_STANCE_FORE_AFT_M = 0.092
LOW_SWEPT_STANCE_ANGLE_RAD = math.asin(
    LOW_SWEPT_STANCE_FORE_AFT_M / DISTAL_LINK_LENGTH_M
)
LOW_SWEPT_STANCE_DOWN_M = (
    math.sqrt(
        DISTAL_LINK_LENGTH_M**2 - LOW_SWEPT_STANCE_FORE_AFT_M**2
    )
    + DISTAL_LINK_LENGTH_M
    + 0.031
)
LOW_SWEPT_BASE_HEIGHT_M = LOW_SWEPT_STANCE_DOWN_M + (0.375 - 0.329341447)
FORWARD_BIASED_BODY_PITCH_RAD = math.radians(2.0)
# V29 spends five percent more of every quarter-step transferring load before
# lift.  Smooth distributed propulsion already occurs throughout the airborne
# interval, so the corresponding all-feet-push hold can be shortened safely.
SCHEDULE_MATCHED_DISTRIBUTED_PUSH_PHASE_FRACTIONS = (
    0.20,  # weight_transfer
    0.15,  # lift
    0.20,  # swing
    0.20,  # lower
    0.10,  # firm_plant
    0.08,  # weight_return
    0.02,  # all_feet_push
    0.05,  # step_settle
)
FORWARD_BIASED_STANCE_FORWARD_M = stance_forward_bias_for_body_pitch_m(
    FORWARD_BIASED_BODY_PITCH_RAD,
    fore_aft_m=LOW_SWEPT_STANCE_FORE_AFT_M,
)
FORWARD_BIASED_FRONT_HIP_RAD = math.asin(
    (LOW_SWEPT_STANCE_FORE_AFT_M + FORWARD_BIASED_STANCE_FORWARD_M)
    / DISTAL_LINK_LENGTH_M
)
FORWARD_BIASED_REAR_HIP_RAD = math.asin(
    (-LOW_SWEPT_STANCE_FORE_AFT_M + FORWARD_BIASED_STANCE_FORWARD_M)
    / DISTAL_LINK_LENGTH_M
)
FORWARD_BIASED_FRONT_DOWN_M = (
    math.sqrt(
        DISTAL_LINK_LENGTH_M**2
        - (LOW_SWEPT_STANCE_FORE_AFT_M + FORWARD_BIASED_STANCE_FORWARD_M) ** 2
    )
    + DISTAL_LINK_LENGTH_M
    + 0.031
)
FORWARD_BIASED_REAR_DOWN_M = (
    math.sqrt(
        DISTAL_LINK_LENGTH_M**2
        - (-LOW_SWEPT_STANCE_FORE_AFT_M + FORWARD_BIASED_STANCE_FORWARD_M) ** 2
    )
    + DISTAL_LINK_LENGTH_M
    + 0.031
)
FORWARD_BIASED_BASE_HEIGHT_M = LOW_SWEPT_BASE_HEIGHT_M + 0.5 * (
    FORWARD_BIASED_FRONT_DOWN_M
    + FORWARD_BIASED_REAR_DOWN_M
    - 2.0 * LOW_SWEPT_STANCE_DOWN_M
)
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
    # Optional mixture component for profiles that need extra coverage near the
    # low end without removing high-speed commands from the training batch.
    # A fraction of environments sample [speed_min, low_band_max]; the rest
    # retain the normal [speed_min, speed_max] curriculum distribution.
    forward_speed_low_band_fraction = 0.0
    forward_speed_low_band_max_m_s = 0.0
    command_curriculum_steps = 1
    command_curriculum_offset_steps = 0
    # Actor inputs historically used raw SI command values.  Keep that contract
    # by default, while allowing a separately versioned policy to magnify nearby
    # low-speed commands without changing the 50-value observation shape.
    command_observation_scale = (1.0, 1.0, 1.0)
    backward_speed_min_m_s = 0.06
    backward_speed_max_m_s = 0.16
    turn_forward_speed_max_m_s = 0.10
    turn_rate_min_rad_s = 0.35
    turn_rate_max_rad_s = 0.80

    # Optional within-episode run -> decelerate -> hold command program.  Legacy
    # profiles sample one command per episode and leave these values unused.
    run_stop_run_duration_min_s = 2.5
    run_stop_run_duration_max_s = 4.5
    run_stop_ramp_duration_min_s = 0.35
    run_stop_ramp_duration_max_s = 0.90
    run_stop_abrupt_fraction = 0.25

    action_scale_abduction_rad = 0.12
    action_scale_hip_rad = 0.30
    action_scale_knee_rad = 0.40
    action_mode = "direct"
    # Integrated target-velocity policies may retain a bounded fraction of their
    # output while zero-command homing remains the dominant action.  Zero keeps
    # the historical deterministic slew-to-neutral behavior for every legacy
    # policy and profile.
    integrated_stop_action_scale = 0.0
    # A velocity-output policy may instead reinterpret its bounded output as a
    # small position residual around calibrated neutral once the command is
    # zero. Unlike adding policy velocity to a saturated homing command, this
    # gives the actor a reachable equilibrium for compensating foot height,
    # compliance, and payload-centre differences. The normal target-rate limiter
    # still bounds every step toward that posture.
    integrated_stop_target_residual_scale = 0.0
    # An integrated-target actor emits joint velocity while its gait clock is
    # parameterized by phase.  If cadence changes but the emitted velocity does
    # not, joint excursion per cycle changes inversely with cadence.  Opt-in
    # profiles can normalize the motion action around a reference frequency so
    # one learned phase trajectory retains its stride as cadence changes.  The
    # resulting action is still clamped to the ordinary packet target-rate limit.
    integrated_action_scale_with_gait_frequency = False
    integrated_action_reference_frequency_hz = 1.0
    # Linear gain applied to (current/reference cadence - 1).  A value of one
    # gives full phase-rate normalization; zero disables the numerical gain
    # while retaining the versioned action mode for controlled diagnostics.
    integrated_action_frequency_gain = 1.0
    # The first four exported actions are hip-abduction balance axes.  ``None``
    # uses the common gain; a profile may leave them unscaled while changing the
    # eight sagittal propulsion joints.
    integrated_action_abduction_frequency_gain = None
    # Gait-residual policies can independently remove actor authority once the
    # command reaches standstill.  A zero value makes the analytic parked pose
    # (and its packet-rate-limited transition) a controller guarantee instead
    # of relying on the learned residual to choose a safe stopped stance.
    gait_residual_stop_action_scale = 1.0
    # Simulation-only V44 ablation: attenuate sagittal residuals during the C2
    # scheduled swing envelope, in LEG_NAMES order FL, RL, FR, RR. Unit scales
    # retain the exact historical arithmetic path. This has no Pi/export parity.
    gait_swing_residual_scale_by_leg = (1.0, 1.0, 1.0, 1.0)
    # Populated only by the isolated exact-task ablation workflow, never JSON.
    gait_swing_residual_ablation_task = None
    # Isolated V44 evaluation-only rear lift witness; no onboard/export parity.
    # Zero is the exact existing target-mapping path. Marker is loader-derived.
    gait_rear_pitch_lift_m = 0.0
    gait_rear_pitch_lift_ablation_task = None
    # Separate experimental action projection; historical0/5mm behavior stays
    # exact when false. It has no onboard/export implementation.
    gait_rear_pitch_projection_enabled = False
    gait_rear_pitch_projection_ablation_task = None
    # Isolated unchanged-peak descent timing; .5 preserves the original envelope.
    gait_rear_pitch_descent_fraction = 0.5
    gait_front_stance_tracking_feedback_gain = 0.0
    gait_front_stance_tracking_feedback_ablation_task = None
    gait_rear_tracking_feedback_gain = 0.0
    gait_rear_tracking_feedback_leg_mask = (1.0, 1.0)
    gait_rear_tracking_feedback_mode = "independent"
    gait_rear_tracking_feedback_ablation_task = None
    # Legacy policies observe their previous raw network request.  A
    # hardware-feasible policy may instead observe the previous rate-limited
    # target expressed in normalized action coordinates, which tells the actor
    # what the controller actually sent when a requested step was too large.
    action_history_mode = "requested_action"
    residual_action_scale = 1.0
    # Optional per-action override in the exported/Isaac action order.  ``None``
    # preserves the legacy scalar contract for every existing task.
    residual_action_scale_by_action = None
    # The deployable controller may impose a stricter target slew rate than the
    # actuator itself.  Keeping this configurable lets training reproduce that
    # packet-level behavior exactly.
    target_velocity_limit_rad_s = SERVO_VELOCITY_LIMIT_RAD_S
    # Optional per-action target-rate contract in exported/Isaac action order.
    # ``None`` preserves the scalar limit for every existing checkpoint.  This
    # lets balance joints react more deliberately than sagittal propulsion
    # joints without slowing the complete gait.
    target_velocity_limit_rad_s_by_action = None
    # Opt-in V44 deployment envelope; legacy tasks keep the original behavior.
    target_packet_step_limit_rad = 0.0
    target_packet_hold_probability = 0.0
    target_packet_max_hold_steps = 1
    actuator_target_response_enabled = False
    actuator_target_response_nominal_tau_s = 0.0
    actuator_target_response_nominal_gain = 1.0
    actuator_target_response_tau_s_range = (0.0, 0.0)
    actuator_target_response_gain_range = (1.0, 1.0)
    actuator_target_response_tau_s_range_by_action = None
    actuator_target_response_gain_range_by_action = None
    penalty_stance_sole_tilt = 0.0
    penalty_swing_sole_tilt = 0.0
    swing_sole_tilt_ablation_task = None
    penalty_swing_corner_clearance = 0.0
    penalty_contact_point_slip = 0.0
    penalty_normalized_contact_slip = 0.0
    normalized_contact_slip_ablation_task = None
    sole_clearance_target_m = 0.012
    sole_clearance_transition_fraction = 0.15
    # Opt-in poor-swing-foot emphasis. Zero preserves historical mean-only
    # scoring; one uses the worst scheduled-swing corner deficit at each tick.
    swing_clearance_worst_leg_fraction = 0.0
    sole_corner_soft_contact_height_m = 0.004
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
    # Speed-scaled legacy profiles derive phase from elapsed time and the current
    # frequency.  A run/stop policy opts into integration so changing frequency
    # cannot teleport the clock to another point in the cycle.
    integrate_gait_phase = False
    gait_standstill_deadband_m_s = 0.0
    gait_speed_min_m_s = 0.04
    gait_speed_max_m_s = 0.10
    gait_frequency_min_hz = 1.25
    gait_frequency_max_hz = 1.25
    gait_stride_scale_min = 1.0
    gait_duty_factor = 0.65
    gait_phase_offsets = (0.0, 0.5, 0.5, 0.0)
    gait_reference_mode = "continuous"
    # Optional JSON phase table containing canonical joint-position targets in
    # the environment's exact joint order. This supports motion-prior residual
    # policies without baking a large generated table into Python source.
    gait_phase_reference_path = None
    gait_phase_reference_low_path = None
    gait_phase_reference_blend_speed_min_m_s = None
    gait_phase_reference_blend_speed_max_m_s = None
    gait_phase_reference_scale_about_neutral = 1.0
    # Isolated V44 simulation ablation only; default retains floor lookup.
    # Linear sampling does not yet have Raspberry Pi/runtime parity.
    gait_phase_reference_linear_interpolation_enabled = False
    # Loader-derived exact-task marker, not an arbitrary JSON override field.
    gait_phase_reference_interpolation_ablation_task = None
    # Opt-in planned body-left displacement metadata for new crawl ablations.
    # Legacy/production references continue to score about the original line.
    gait_phase_table_lateral_reference_enabled = False
    # Experimental fore/aft weight transfer is not net locomotion. This flag
    # corrects instantaneous speed penalties and the stop anchor only; positive
    # progress uses actual rolling net speed after a complete window.
    gait_phase_table_forward_reference_enabled = False
    gait_smooth_support_push = False
    gait_distributed_push_phase_fractions = None
    gait_contact_transition_fraction = 0.0
    gait_stance_fore_aft_m = 0.080
    gait_stance_down_m = 0.329341447
    # Keep the rectangular sole parallel to the ground as a support foot moves
    # fore/aft.  Legacy policies retain their original constant-height arc.
    gait_flat_sole_support = False
    # Optional FL, RL, FR, RR stance depths.  This lets a pitched stable pose be
    # represented exactly while preserving the scalar legacy contract.
    gait_stance_down_by_leg_m = None
    gait_weight_shift_forward_m = 0.0
    gait_rear_weight_shift_forward_m = None
    gait_weight_shift_lateral_m = 0.0
    # Optional signed coordinated side-to-side body translation for a diagonal trot.
    # Unlike independent abduction actions, this shifts all four feet together
    # so the rear-biased COM remains over the active diagonal support line.
    gait_diagonal_support_lateral_shift_m = 0.0
    gait_translate_lateral_weight_shift = False
    gait_forward_body_pitch_rad = 0.0
    gait_stance_center_offset_m = 0.0
    target_forward_pitch_rad = 0.0
    gait_stride_m = 0.080
    gait_lift_m = 0.025
    # Optional FL, RL, FR, RR swing clearances.  The scalar remains the
    # backwards-compatible default for every existing task.
    gait_lift_by_leg_m = None
    # Reserve part of swing for nearly vertical unloading and lowering.  A
    # value of (0, 1) preserves the historical full-swing fore/aft curve.
    gait_swing_translation_start_fraction = 0.0
    gait_swing_translation_end_fraction = 1.0
    gait_start_ramp_s = 0.60
    # When enabled, zero command means a real four-foot park: no frozen lifted
    # shoe and no scheduled swing contact after the clock stops.
    gait_park_at_standstill = False
    # Opt-in for phase tables: a parked robot has four scheduled support feet.
    gait_phase_table_park_contact_override = False
    # Optional park-only geometry and posture.  Leaving these unset preserves
    # the moving stance for every legacy task.  Run/stop tasks can use a taller,
    # symmetric low-effort pose once the speed command reaches zero.
    gait_standstill_fore_aft_m = None
    gait_standstill_down_m = None
    gait_standstill_center_offset_m = 0.0
    target_standstill_pitch_rad = None
    target_standstill_base_height_m = None
    gait_reference_sigma = 0.75
    # By default, retain the historical target-space, all-joint mean gait
    # reference score.  Hardware-matched profiles can blend in measured joint
    # state and emphasize the least-following leg so a delayed actuator cannot
    # earn full reference credit merely by receiving the right target packet.
    gait_reference_actual_state_weight = 0.0
    gait_reference_worst_leg_weight = 0.0
    # Optional crawl-topology constraints.  Legacy profiles retain their
    # original behavior; V26 enables these as a controlled V25 ablation.
    require_schedule_matched_support = False
    gate_progress_by_schedule_matched_support = False
    schedule_matched_progress_gate_floor = 1.0
    gate_progress_by_four_leg_release = False
    gate_progress_by_cycle_four_leg_release = False
    cycle_progress_gate_exponent = 1.0
    four_leg_release_target_rate = 0.75
    four_leg_progress_gate_floor = 1.0
    randomize_gait_start_phase_quarters = False
    # Optional differentiable contact-release shaping.  Legacy tasks keep the
    # existing 1 N binary release test; an opt-in task may reward progressively
    # unloading a scheduled swing shoe before it becomes fully airborne.
    use_soft_scheduled_swing_release = False
    scheduled_swing_release_force_sigma_n = 2.0
    use_threshold_centered_release_shaping = False
    # The historical sigmoid gives 0.5 quality exactly at the contact threshold.
    # New profiles may normalize the release half so a still-contacting shoe earns
    # no shaping credit. Pair/progress gates always use the strict normalized
    # signal; this switch affects only the auxiliary discovery reward.
    normalize_threshold_centered_release_quality = False
    scheduled_release_force_threshold_n = 1.0
    scheduled_release_shaping_width_n = 0.35
    # Optional wider sigmoid used only for discovery shaping. ``None`` reuses
    # the strict gate width and preserves all existing profiles.
    scheduled_release_discovery_width_n = None
    scheduled_release_min_consecutive_steps = 5
    use_soft_scheduled_stance_quality = False
    scheduled_stance_quality_width_n = 1.0
    use_soft_schedule_matched_progress_gate = False
    # Optional diagonal/trot contact contract.  When enabled, positive motion
    # credit during a two-foot swing is gated by the less-unloaded member of the
    # scheduled pair.  This prevents a policy from dragging one rear leg while
    # the other three manufacture forward displacement.
    gate_progress_by_scheduled_pair_release = False
    scheduled_pair_progress_gate_floor = 1.0
    # Isolated reward-credit ablation; never a production default.
    decouple_sustained_progress_from_pair_release = False
    sustained_progress_decoupling_ablation_task = None

    # Deliberately reproduce quadruped_walk_v1.yaml's successful reward instead
    # of continuing to tune a novel objective.  Net displacement remains a hard
    # evaluation metric, but adding it to the reward caused unsafe lunge exploits.
    reward_forward_velocity_tracking = 4.0
    # Optional tracking of the same trailing velocity used by sustained-motion
    # qualification.  This gives command-conditioned policies a low-noise reason
    # to distinguish nearby speeds instead of merely exceeding a minimum floor.
    reward_rolling_velocity_tracking = 0.0
    rolling_velocity_tracking_sigma_m_s = 0.020
    # Optional symmetric trailing-speed error cost.  Unlike a Gaussian reward,
    # this keeps a useful gradient when the current gait is far below command.
    penalty_rolling_velocity_error = 0.0
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
    penalty_backward_motion = 0.0
    penalty_overspeed = 0.0
    penalty_rearward_pitch = 0.0
    rearward_pitch_normalizer_rad = math.radians(3.0)
    reward_gait_reference = 2.00
    reward_scheduled_stance = 0.50
    penalty_missing_scheduled_stance = 0.0
    reward_scheduled_swing = 1.50
    reward_scheduled_release_shaping = 0.0
    reward_scheduled_pair_release = 0.0
    penalty_scheduled_pair_mismatch = 0.0
    reward_cycle_four_leg_release = 0.0
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
    # Optional world-path tracking closes the curved-walk loophole in which a
    # yawed robot earns full forward reward along its own body X axis.  A
    # heading-hold outer loop maps accumulated yaw error into the existing yaw
    # command input, preserving the 50-value observation contract.
    track_episode_world_path = False
    heading_hold_kp_s = 0.0
    heading_hold_max_correction_rad_s = 0.0
    penalty_normalized_lateral_velocity = 0.0
    normalized_lateral_velocity_speed_floor_m_s = 0.010
    reward_straight_aligned_progress = 0.0
    straight_progress_lateral_sigma_m_s = 0.010
    # Legacy profiles express lateral alignment as an absolute velocity. Newer
    # low-speed profiles may instead normalize it by commanded speed so the
    # same angular/path error receives comparable shaping across the curriculum.
    straight_progress_use_normalized_lateral_velocity = False
    straight_progress_lateral_ratio_sigma = 0.05
    straight_progress_heading_sigma_rad = 0.10
    # Optionally attenuate only positive motion rewards when the robot is not
    # travelling straight. Negative progress remains fully penalized, avoiding
    # an exploit in which a poorly aligned policy discounts its own reversal.
    gate_positive_progress_by_straight_alignment = False
    straight_progress_gate_floor = 1.0
    # Quadratic path costs are retained for historical checkpoints. A unit
    # Huber loss gives new profiles a useful near-line gradient without a large
    # delayed cliff after the robot has already left the recoverable corridor.
    use_huber_path_costs = False
    penalty_body_height = 2.0
    # Opt-in dimensionless posture costs for simulation ablations. These
    # replace the raw metre-squared/projected-gravity costs only when enabled;
    # existing tasks keep their exact legacy expressions and reward weights.
    normalized_posture_cost_enabled = False
    # Isolated V44 height-only hypothesis. During commanded translation, blend
    # legacy height into the dimensionless height cost below; keep legacy tilt
    # and exact legacy parked height. Never combine with normalized posture.
    normalized_body_height_cost_enabled = False
    posture_height_deadband_m = 0.003
    posture_height_normalizer_m = 0.010
    posture_angle_deadband_rad = math.radians(1.0)
    posture_angle_normalizer_rad = math.radians(3.0)
    penalty_normalized_body_height = 4.0
    penalty_normalized_body_roll = 4.0
    penalty_normalized_body_pitch = 4.0
    penalty_action_rate = 0.03
    penalty_action_acceleration = 0.12
    penalty_action_saturation = 0.50
    penalty_target_limiter_gap = 0.0
    # Integrated target-velocity policies can otherwise spend unnecessary
    # authority rocking the abduction joints while the sagittal joints provide
    # the actual propulsion.  These opt-in terms act on the four abduction
    # velocity requests and accumulated targets respectively.
    penalty_abduction_action_magnitude = 0.0
    penalty_abduction_target_magnitude = 0.0
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
    # Static correction for provisional fully-solid CAD masses.  A value below
    # one scales every dry link and dry-base inertia while leaving the measured
    # replacement payload mass independent.
    dry_robot_mass_scale = 1.0
    robot_mass_scale_range = (1.0, 1.0)
    rear_payload_combined_mass_scale_range = (1.0, 1.0)
    rear_payload_combined_com_jitter_m = (0.0, 0.0, 0.0)
    shoe_static_friction = 1.05
    shoe_dynamic_friction = 0.85
    shoe_restitution = 0.02
    shoe_contact_stiffness = 12_000.0
    shoe_contact_damping = 45.0
    # ``None`` preserves the authored shared shoe material exactly.  Non-None
    # ranges opt into a one-time, per-environment PhysX material assignment.
    shoe_static_friction_randomization_range = None
    shoe_dynamic_friction_randomization_range = None
    # Optional episode-consistent surface domain plus per-foot differential.
    # These are separate from the legacy absolute per-foot ranges above so old
    # profiles retain bit-for-bit material sampling.
    shoe_common_static_friction_randomization_range = None
    shoe_common_dynamic_friction_randomization_range = None
    shoe_friction_differential_scale_range = (1.0, 1.0)
    # Peak joint torque available to the implicit drive before the common and
    # per-joint capacity scales below are applied.  Keep the historical
    # 0.8826 N*m value as the default so existing tasks/checkpoints retain
    # identical physics; newer profiles may opt into a higher transient cap.
    actuator_peak_effort_nm = EFFORT_CAP_NM
    actuator_effort_scale_range = (1.0, 1.0)
    target_velocity_scale_range = (1.0, 1.0)
    correlate_common_actuator_scales = False
    actuator_individual_effort_scale_range = (1.0, 1.0)
    target_individual_velocity_scale_range = (1.0, 1.0)
    actuator_individual_stiffness_scale_range = (1.0, 1.0)
    actuator_individual_damping_scale_range = (1.0, 1.0)
    # Explicit nominal drive hypotheses, independent of randomized-domain masks.
    # Unit scales preserve legacy dynamics. These do not change effort/rate caps.
    actuator_nominal_stiffness_scale = 1.0
    actuator_nominal_damping_scale = 1.0
    actuator_sync_drive_gain_telemetry = False
    physical_randomization_nominal_fraction = 1.0
    mirror_physical_randomization_pairs = False
    joint_target_bias_abduction_rad = 0.0
    joint_target_bias_flexion_rad = 0.0
    control_delay_step_range = (0, 0)
    reset_roll_pitch_noise_rad = 0.0
    reset_yaw_noise_rad = 0.0
    imu_angular_velocity_bias_range_rad_s = 0.0
    imu_projected_gravity_noise_std = 0.0
    imu_linear_acceleration_noise_std_g = 0.0
    base_force_randomization_range_n = (0.0, 0.0, 0.0)
    base_torque_randomization_range_nm = (0.0, 0.0, 0.0)
    smoothness_curriculum_steps = 0
    smoothness_initial_scale = 1.0
    joint_acceleration_normalizer_rad_s2 = 40.0
    body_linear_acceleration_normalizer_m_s2 = 5.0
    body_angular_acceleration_normalizer_rad_s2 = 10.0
    penalty_joint_acceleration = 0.0
    penalty_body_linear_acceleration = 0.0
    penalty_body_angular_acceleration = 0.0
    # Isolated V44 effort-observable experiment. Both paired arms collect one
    # fresh native active-DOF projection after physics; only the treatment uses
    # it in the four effort costs. This is NOT identified drive-only torque.
    # Defaults retain the historical approximate-PD costs without a new getter.
    native_effort_telemetry_enabled = False
    native_effort_reward_enabled = False
    native_effort_ablation_task = None
    # Critic-only information ablation. Both 90-wide arms build the same causal
    # context; the zero arm masks all 32 new inputs. Actor/runtime stays 50-wide.
    # The isolated loader derives state_space and the exact task marker.
    critic_context_mode = "legacy58"
    critic_context_ablation_task = None
    effort_soft_limit_fraction = 0.85
    penalty_effort_soft_limit = 0.0
    # Optional continuous effort price normalized to the same sustainable
    # threshold.  The soft-limit hinge above is useful for overloads but gives
    # no preference between two sub-threshold motions; the RMS term lets a
    # hardware-focused continuation find a genuinely lower-current gait.
    penalty_effort_rms = 0.0
    # A servo cannot deliver stall torque and no-load speed simultaneously.
    # This optional linear motor-envelope approximation prices operating points
    # where normalized torque plus normalized speed exceeds the configured
    # margin.  It is disabled for historical checkpoints.
    actuator_torque_speed_no_load_rad_s = SERVO_VELOCITY_LIMIT_RAD_S
    actuator_torque_speed_margin = 1.0
    penalty_actuator_torque_speed_envelope = 0.0
    # Small load sharing cost across the four three-joint legs.  This avoids a
    # policy buying speed by repeatedly saturating one rear leg while the other
    # three remain lightly loaded.
    penalty_leg_operating_point_imbalance = 0.0

    # Run/stop-only shaping.  All terms are zero for historical profiles.
    stop_velocity_sigma_m_s = 0.035
    stop_position_tolerance_m = 0.040
    reward_stop_velocity = 0.0
    reward_stop_all_feet_contact = 0.0
    reward_stop_weakest_foot_contact = 0.0
    stop_neutral_action_sigma = 0.20
    reward_stop_neutral_action = 0.0
    # The Gaussian neutral reward can have an extremely small gradient when a
    # transferred policy initially parks far from action zero.  This optional
    # quadratic term keeps a useful corrective signal across the full range.
    penalty_stop_action_magnitude = 0.0
    # Integrated target-velocity policies can stop their output before the
    # accumulated target has returned to neutral. This optional term acts on
    # the applied normalized target rather than the network output.
    penalty_stop_target_magnitude = 0.0
    penalty_stop_position = 0.0
    penalty_stop_yaw_rate = 0.0
    penalty_stop_joint_velocity = 0.0


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
class DrobotCommandedWalkingPaddedFeetForwardBiasExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingHigherSpeedStraightCrawlExternalRearPayloadEnvCfg
):
    """V24 lower-friction Velcro-sole crawl biased against the rear payload."""

    # The new adhesive Velcro-like sole is modestly softer and may slide more
    # readily than the bare printed tread. Model that uncertainty conservatively
    # with lower friction, lower contact stiffness, and more damping.
    shoe_static_friction = 0.75
    shoe_dynamic_friction = 0.55
    shoe_restitution = 0.01
    shoe_contact_stiffness = 7_000.0
    shoe_contact_damping = 85.0

    # Low charge does not change the battery mass or COM, but voltage sag can
    # reduce servo torque and achievable target rate. Parallel robots span this
    # bounded capacity range so the actor cannot rely on ideal fresh-battery
    # response. Operation below the hardware voltage cutoff is still unsafe.
    actuator_effort_scale_range = (0.70, 1.00)
    target_velocity_scale_range = (0.70, 1.00)

    # Shift the analytic crawl support target farther toward the front feet and
    # ask the learned residual to hold a mild nose-down attitude. This counters
    # the externally mounted rear battery without changing the measured mass.
    gait_weight_shift_forward_m = 0.018
    target_forward_pitch_rad = math.radians(3.0)

    # Preserve the V23 speed range, but make reverse travel unambiguously worse
    # than slow forward progress under the new padded-foot contact regime.
    velocity_tracking_sigma_m_s = 0.012
    reward_forward_velocity_tracking = 4.00
    reward_instant_progress = 4.00
    reward_sustained_progress = 5.00
    minimum_sustained_speed_fraction = 0.55
    penalty_sustained_stall = 8.00
    penalty_backward_motion = 10.00
    penalty_overspeed = 8.00

    # The first V24 continuation reduced reverse motion but obtained extra
    # progress by exceeding the requested speed and using two-foot support too
    # often at 0.015 m/s. Keep the slow crawl genuinely commandable and retain
    # the intended one-leg-at-a-time support topology.
    reward_scheduled_stance = 2.50
    reward_three_foot_support = 3.50
    penalty_excess_airborne_feet = 12.00
    penalty_body_tilt = 8.00
    penalty_rearward_pitch = 1.50


@configclass
class DrobotCommandedWalkingRobustStraightLowStanceExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingPaddedFeetForwardBiasExternalRearPayloadEnvCfg
):
    """V25 straight-path, asymmetric-hardware, smooth aggressive crawl."""

    # A 92 mm opposed fore/hind sweep lowers the body by 7.67 mm while keeping
    # the broad rectangular soles flat.  The opposed geometry retains the
    # approximately 304 mm fore/aft foot-center footprint needed by the rear pack;
    # an all-legs-same-direction pose would reduce it to roughly 120 mm.
    def __post_init__(self) -> None:
        # ``robot`` is a configclass instance field in the installed Isaac Lab
        # release, so it is not available on the parent class while this class
        # body is being evaluated.  Replace the inherited per-instance value
        # only after config construction.
        self.robot = self.robot.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, LOW_SWEPT_BASE_HEIGHT_M),
                joint_pos={
                    ".*_hip_abduction": 0.0,
                    "front_.*_hip_flexion": LOW_SWEPT_STANCE_ANGLE_RAD,
                    "rear_.*_hip_flexion": -LOW_SWEPT_STANCE_ANGLE_RAD,
                    "front_.*_knee": -LOW_SWEPT_STANCE_ANGLE_RAD,
                    "rear_.*_knee": LOW_SWEPT_STANCE_ANGLE_RAD,
                },
                joint_vel={".*": 0.0},
            )
        )
    target_base_height_m = LOW_SWEPT_BASE_HEIGHT_M
    gait_stance_fore_aft_m = LOW_SWEPT_STANCE_FORE_AFT_M
    gait_stance_down_m = LOW_SWEPT_STANCE_DOWN_M
    gait_reference_mode = "smooth_distributed_push"
    gait_smooth_support_push = True

    # Spread propulsion through the three-foot support interval instead of a
    # one-controller-tick shove, then expose a faster but still sequential
    # crawl range.  Geometry is deliberately kept inside the 57-degree Isaac
    # soft limit even after the full residual and a randomized joint-zero bias.
    initial_forward_speed_min_m_s = 0.008
    initial_forward_speed_max_m_s = 0.025
    forward_speed_min_m_s = 0.005
    forward_speed_max_m_s = 0.040
    gait_speed_min_m_s = 0.005
    gait_speed_max_m_s = 0.040
    gait_frequency_min_hz = 0.12
    gait_frequency_max_hz = 0.62
    # Keep transfer and clearance intact at low commands.  Requested speed is
    # expressed by cadence; joint-space amplitude scaling previously dragged
    # loaded rear shoes across the floor.
    gait_stride_scale_min = 1.0
    gait_stride_m = 0.045
    gait_lift_m = 0.020
    gait_weight_shift_forward_m = 0.010
    # Six millimetres was the best zero-residual compromise in nominal Isaac:
    # it balanced unloading across all four legs without the 12 mm reference's
    # larger yaw and alternating diagonal bias. The line reward subtracts this
    # periodic zero-mean sway while continuing to punish accumulated drift.
    gait_weight_shift_lateral_m = 0.006
    gait_start_ramp_s = 1.50
    # The action contract is joint-kind-major: all four abductions, then all
    # four sagittal hips, then all four knees.  Preserve enough abduction range
    # to compensate lateral hardware asymmetry while keeping sagittal residuals
    # inside the 57-degree Isaac soft limit after joint-zero randomization.
    residual_action_scale = 0.05
    residual_action_scale_by_action = (
        0.20,
        0.20,
        0.20,
        0.20,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
        0.05,
    )
    # Four degrees per 60 Hz packet remains below the STS3215 no-load rate,
    # while per-domain rate scaling trains effective limits down to 2.64 deg.
    target_velocity_limit_rad_s = math.radians(240.0)

    # Score velocity in the episode-start world frame, not the robot's rotated
    # body frame.  Heading error drives a bounded outer-loop yaw-rate command
    # that is present in both simulation and the BNO085 runtime.
    track_episode_world_path = True
    heading_hold_kp_s = 1.50
    heading_hold_max_correction_rad_s = 0.20
    penalty_lateral_velocity = 0.0
    penalty_normalized_lateral_velocity = 6.0
    lateral_corridor_half_width_m = 0.008
    penalty_lateral_displacement = 60.0
    penalty_lateral_corridor = 10.0
    penalty_yaw_rate = 22.0
    heading_error_normalizer_rad = math.radians(3.0)
    penalty_heading_error = 8.0
    reward_straight_aligned_progress = 6.0
    straight_progress_lateral_sigma_m_s = 0.006
    straight_progress_heading_sigma_rad = math.radians(4.0)

    # Keep forward motion valuable enough that stronger line/robustness costs
    # cannot recreate the old stationary bent-knee optimum.
    reward_forward_velocity_tracking = 6.0
    reward_instant_progress = 8.0
    reward_sustained_progress = 12.0
    penalty_sustained_stall = 14.0
    penalty_backward_motion = 18.0
    penalty_overspeed = 8.0
    # The inherited 12-point pose reward let the first nominal continuation
    # stay upright while moving slightly backward.  Keep the analytic crawl as
    # a stabilizing prior, but make actual path progress decisively more valuable.
    reward_gait_reference = 5.0

    # Target the measured failure modes: rear/lateral mass uncertainty, supply
    # sag common to the robot, independent leg strength/rate mismatch, small
    # servo-zero errors, 0-1 frame latency, and persistent lateral/yaw loads.
    # The URDF's 4.526139 kg ledger is a fully-solid print estimate.  The whole
    # assembled robot measured 7 lb (3.17514659 kg); after preserving the
    # measured 0.523179545 kg pack, the dry CAD mass needs a 0.650607608 scale.
    dry_robot_mass_scale = 0.65060760808206
    robot_mass_scale_range = (0.96, 1.04)
    rear_payload_combined_mass_scale_range = (0.955, 1.055)
    rear_payload_combined_com_jitter_m = (0.008, 0.010, 0.006)
    actuator_effort_scale_range = (0.75, 1.00)
    target_velocity_scale_range = (0.75, 1.00)
    actuator_individual_effort_scale_range = (0.88, 1.00)
    target_individual_velocity_scale_range = (0.88, 1.00)
    actuator_individual_stiffness_scale_range = (0.85, 1.15)
    actuator_individual_damping_scale_range = (0.85, 1.15)

    physical_randomization_nominal_fraction = 0.25
    mirror_physical_randomization_pairs = True
    # Match the measured Velcro-like tread near the V24 nominal while exposing
    # asymmetric low-grip cases.  Dynamic friction is clamped not to exceed
    # static friction after sampling.
    shoe_static_friction_randomization_range = (0.55, 0.90)
    shoe_dynamic_friction_randomization_range = (0.35, 0.70)
    joint_target_bias_abduction_rad = math.radians(1.0)
    joint_target_bias_flexion_rad = math.radians(1.5)
    control_delay_step_range = (0, 1)
    reset_roll_pitch_noise_rad = math.radians(2.0)
    reset_yaw_noise_rad = math.radians(5.0)
    imu_angular_velocity_bias_range_rad_s = 0.020
    imu_projected_gravity_noise_std = 0.006
    imu_linear_acceleration_noise_std_g = 0.020
    base_force_randomization_range_n = (0.10, 0.40, 0.0)
    base_torque_randomization_range_nm = (0.025, 0.015, 0.040)

    # Preserve smoothness weights while asking for more stride and cadence, and
    # add only a soft effort-headroom cost for the lower stance.
    penalty_action_rate = 0.18
    penalty_action_acceleration = 0.90
    penalty_joint_acceleration = 0.30
    penalty_body_linear_acceleration = 0.55
    penalty_body_angular_acceleration = 0.45
    effort_soft_limit_fraction = 0.85
    penalty_effort_soft_limit = 0.20


@configclass
class DrobotCommandedWalkingBalancedFourLegStraightCrawlExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingRobustStraightLowStanceExternalRearPayloadEnvCfg
):
    """V26b balanced, higher-clearance four-leg straight crawl."""

    # The first controlled V26 ablation retained the V25 reference exactly.  It
    # proved that schedule-matched support alone was insufficient: rear-left
    # release still reached zero in some nominal episodes.  V26b therefore gives
    # the actor enough sagittal authority to unload a mismatched leg, increases
    # real shoe-edge clearance, and makes the least-active leg decisive.
    require_schedule_matched_support = True
    gate_progress_by_four_leg_release = True
    four_leg_release_target_rate = 0.65
    four_leg_progress_gate_floor = 0.05
    randomize_gait_start_phase_quarters = True

    gait_stride_m = 0.050
    gait_lift_m = 0.028
    gait_frequency_max_hz = 0.72
    residual_action_scale_by_action = (
        0.10,
        0.10,
        0.10,
        0.10,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
    )
    reward_scheduled_swing = 8.0
    reward_qualified_touchdown = 0.50
    reward_least_active_swing = 16.0


@configclass
class DrobotCommandedWalkingAdaptiveAsymmetricFourLegStraightCrawlExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingBalancedFourLegStraightCrawlExternalRearPayloadEnvCfg
):
    """V27 straight crawl with corrected transfer and asymmetric adaptation."""

    # Retain the low 92 mm swept stance and four-beat lateral-sequence crawl.
    # V26's reference had two independent problems: lateral transfer changed
    # stance width instead of translating the support polygon, and 50/28 mm
    # stride/lift left essentially no hip-limit margin.  The values below were
    # selected from the exact 2,048-sample reference table: the worst reference
    # hip is -55.90 degrees, 1.10 degrees inside the -57-degree soft limit, while
    # the 46 mm stride at 0.72 Hz still exposes a 0.033 m/s speed range.
    gait_stride_m = 0.046
    gait_lift_m = 0.024
    gait_frequency_max_hz = 0.72
    gait_translate_lateral_weight_shift = True
    gait_weight_shift_forward_m = 0.008
    gait_rear_weight_shift_forward_m = 0.010
    gait_weight_shift_lateral_m = 0.006

    # Keep meaningful per-leg correction authority, but reduce V26's hip
    # residual from 0.15 to 0.10.  The knee has ample limit margin and retains
    # 0.15 authority so a weak or mis-zeroed rear leg can create real shoe
    # clearance without demanding an unsafe hip target.  Reference, zero-bias,
    # and residual targets are independently clamped to the soft joint envelope.
    residual_action_scale_by_action = (
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.15,
        0.15,
        0.15,
        0.15,
    )

    # Tighten only cycle-scale/world-path shaping.  Instantaneous lateral
    # velocity remains unpenalized so the analytic crawl may perform its
    # necessary zero-mean six-millimetre weight transfer without being driven
    # toward a stationary or shuffling solution.
    use_soft_scheduled_swing_release = True
    # The first nominal shaping stage used 4 N to obtain a useful gradient at
    # the initially loaded rear shoes.  The continuation tightens to 3 N so a
    # moderate load is no longer an attractive plateau; quality approaches one
    # only as the scheduled shoe becomes genuinely unloaded.
    scheduled_swing_release_force_sigma_n = 3.0
    penalty_lateral_velocity = 0.0
    penalty_normalized_lateral_velocity = 8.0
    lateral_corridor_half_width_m = 0.005
    penalty_lateral_displacement = 72.0
    penalty_lateral_corridor = 14.0
    penalty_heading_error = 10.0
    reward_straight_aligned_progress = 8.0


@configclass
class DrobotCommandedWalkingForwardBiasedCycleGatedFourLegStraightCrawlExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingAdaptiveAsymmetricFourLegStraightCrawlExternalRearPayloadEnvCfg
):
    """V28 nose-down, cycle-gated crawl with realistic rated torque."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # Start from the same pose the analytic table describes.  The common
        # +X foot bias preserves opposed knees and the 304 mm support span while
        # shortening the front supports and lengthening the rear supports.  A
        # two-degree pitch is small enough for the compliant tread but moves the
        # rear pack load forward before either rear shoe is asked to lift.
        self.robot = self.robot.replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, FORWARD_BIASED_BASE_HEIGHT_M),
                rot=(
                    0.0,
                    math.sin(0.5 * FORWARD_BIASED_BODY_PITCH_RAD),
                    0.0,
                    math.cos(0.5 * FORWARD_BIASED_BODY_PITCH_RAD),
                ),
                joint_pos={
                    ".*_hip_abduction": 0.0,
                    "front_.*_hip_flexion": FORWARD_BIASED_FRONT_HIP_RAD,
                    "rear_.*_hip_flexion": FORWARD_BIASED_REAR_HIP_RAD,
                    "front_.*_knee": -FORWARD_BIASED_FRONT_HIP_RAD,
                    "rear_.*_knee": -FORWARD_BIASED_REAR_HIP_RAD,
                },
                joint_vel={".*": 0.0},
            )
        )

    target_base_height_m = FORWARD_BIASED_BASE_HEIGHT_M
    gait_forward_body_pitch_rad = FORWARD_BIASED_BODY_PITCH_RAD
    target_forward_pitch_rad = FORWARD_BIASED_BODY_PITCH_RAD

    # Keep the low 92 mm swept stance and 46/24 mm step geometry that remained
    # inside the joint envelope.  A modest cadence extension provides a more
    # assertive upper command without converting the statically stable crawl to
    # a two-leg dynamic gait.
    initial_forward_speed_min_m_s = 0.008
    initial_forward_speed_max_m_s = 0.030
    forward_speed_max_m_s = 0.045
    gait_speed_max_m_s = 0.045
    gait_frequency_max_hz = 0.80
    # The pitched reference centers the whole-cycle hip envelope at roughly
    # -52.51..+53.19 degrees.  Four hundredths of a radian plus the 1.5-degree
    # randomized zero offset remains inside the +/-57-degree soft limit; knees
    # keep the larger authority needed to create real shoe clearance.
    residual_action_scale_by_action = (
        0.10,
        0.10,
        0.10,
        0.10,
        0.04,
        0.04,
        0.04,
        0.04,
        0.15,
        0.15,
        0.15,
        0.15,
    )

    # V27's differentiable force score accepted partially loaded shoes and even
    # awarded transfer/settle frames as successful swing.  V28 requires every
    # foot to remain below 1 N for five consecutive 60 Hz controller ticks in a
    # complete gait cycle.  Only a small threshold-centered sigmoid remains as
    # discovery shaping; hard release and the all-four cycle event dominate.
    gate_progress_by_four_leg_release = False
    gate_progress_by_cycle_four_leg_release = True
    # Give early per-foot successes a causal discovery gradient without letting
    # a three-leg cycle earn most of the motion reward.  The fractions after
    # one/two/three/four qualified feet are approximately 0.004/0.063/0.316/1.
    cycle_progress_gate_exponent = 4.0
    four_leg_progress_gate_floor = 0.02
    use_soft_scheduled_swing_release = False
    use_threshold_centered_release_shaping = True
    normalize_threshold_centered_release_quality = True
    scheduled_release_force_threshold_n = 1.0
    scheduled_release_shaping_width_n = 0.35
    scheduled_release_min_consecutive_steps = 5
    reward_scheduled_swing = 10.0
    reward_scheduled_release_shaping = 1.0
    # A hard all-four event is paid once when the fourth foot qualifies.  Keep
    # it large enough to matter beside the integrated per-tick swing shaping,
    # while deterministic selection still requires every completed cycle.
    reward_cycle_four_leg_release = 120.0
    reward_least_active_swing = 8.0

    # A valid crawl keeps exactly three feet planted during swing and all four
    # planted during transfer/settle.  Strengthen that topology modestly rather
    # than paying for longer swing time, which would increase overlap between
    # adjacent steps and recreate the two-leg support seen on hardware.
    reward_scheduled_stance = 3.5
    reward_three_foot_support = 5.0
    penalty_excess_airborne_feet = 18.0

    # Score net cycle-scale drift in the episode-start world frame.  The known
    # six-millimetre analytic weight shift is subtracted before these terms are
    # evaluated, so the policy may sway temporarily but is not paid for walking
    # a curved or diagonally biased path.
    penalty_normalized_lateral_velocity = 10.0
    penalty_heading_error = 12.0
    reward_straight_aligned_progress = 10.0

    # The old 0.8826 N*m setting modeled a historical 30% register.  The robot
    # now runs a 90% register, so use 90% of the documented 30 kg-cm stall
    # torque as an engineering approximation of the short-duration drive cap.
    # Do not teach the policy to live there: the soft limit below starts at the
    # documented 10 kg-cm rated torque and prices sustained use above it.
    actuator_peak_effort_nm = TRANSIENT_EFFORT_CAP_NM
    # A 3S supply affects available torque and no-load rate together.  Sample
    # one episode-consistent voltage factor over an inferred 11.0-12.6 V
    # operating envelope around the 12 V rating, then add modest independent
    # servo mismatch.  Deep-undervoltage behavior belongs to a safety stop, not
    # a locomotion objective.
    actuator_effort_scale_range = (0.92, 1.05)
    target_velocity_scale_range = (0.92, 1.05)
    correlate_common_actuator_scales = True
    actuator_individual_effort_scale_range = (0.90, 1.05)
    target_individual_velocity_scale_range = (0.88, 1.05)
    effort_soft_limit_fraction = (
        SUSTAINABLE_RATED_EFFORT_NM / TRANSIENT_EFFORT_CAP_NM
    )
    penalty_effort_soft_limit = 0.55

    # The new Velcro-like tread can shift grip for every foot together, while
    # pad wear/installation also creates left-right differences.  Sample a
    # common surface first and a paired per-foot multiplier second.  These are
    # engineering priors until a tilt/pull test measures the actual materials.
    shoe_static_friction_randomization_range = None
    shoe_dynamic_friction_randomization_range = None
    shoe_common_static_friction_randomization_range = (0.40, 0.90)
    shoe_common_dynamic_friction_randomization_range = (0.25, 0.70)
    shoe_friction_differential_scale_range = (0.85, 1.15)


@configclass
class DrobotCommandedWalkingScheduleMatchedSupportStraightCrawlExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingForwardBiasedCycleGatedFourLegStraightCrawlExternalRearPayloadEnvCfg
):
    """V29 identity-matched, support-first straight crawl curriculum."""

    # Retain the selected low, opposed 92 mm stance and 46/24 mm step geometry.
    # Only phase timing changes: load transfer gets one additional 60 Hz update
    # at the 0.8 Hz ceiling without increasing total swing time or duty factor.
    gait_distributed_push_phase_fractions = (
        SCHEDULE_MATCHED_DISTRIBUTED_PUSH_PHASE_FRACTIONS
    )
    gait_weight_shift_forward_m = 0.015
    gait_rear_weight_shift_forward_m = 0.020
    # The +2 degree stance helper already supplies the complete matched
    # fore/aft bias.  An additional -5 mm offset cancelled most of that geometry
    # while reset and posture rewards still demanded +2 degrees.
    gait_stance_center_offset_m = 0.0
    # Exclude roughly one controller tick at each quintic lift/lower endpoint
    # from the strict release interval.  This favors gradual unloading and early
    # touchdown instead of demanding discontinuous contact at phase boundaries.
    gait_contact_transition_fraction = 0.08

    # A transferred V28 checkpoint starts at the conservative range below and
    # expands gradually toward the analytic 0.037 m/s ceiling. This teaches the
    # new contact identity contract before exposing the most aggressive cadence.
    initial_forward_speed_min_m_s = 0.008
    initial_forward_speed_max_m_s = 0.020
    # 46 mm * 0.80 Hz is 0.0368 m/s.  Do not ask residual actions to manufacture
    # the extra 22 percent implied by the old 0.045 m/s command ceiling.
    forward_speed_max_m_s = 0.037
    gait_speed_max_m_s = 0.037
    command_curriculum_steps = 128_000

    # During swing, all three named anchors must contact and the named swing foot
    # must release. During transfer and settle, all four must contact. Positive
    # progress receives only a discovery floor when that exact topology is wrong.
    gate_progress_by_schedule_matched_support = True
    schedule_matched_progress_gate_floor = 0.02
    use_soft_scheduled_stance_quality = True
    scheduled_stance_quality_width_n = 1.0
    use_soft_schedule_matched_progress_gate = True
    reward_three_foot_support = 8.0
    penalty_missing_scheduled_stance = 12.0
    penalty_excess_airborne_feet = 24.0

    # The inherited current-cycle gate paid most progress, plus a 120-point
    # completion event, after the fixed final FL swing.  That phase asymmetry can
    # directly teach left-biased propulsion.  Exact per-tick topology is already
    # symmetric; keep all-four cycle release as an evaluation metric only.
    gate_progress_by_cycle_four_leg_release = False
    reward_cycle_four_leg_release = 0.0

    # V28 settled around -0.4 degrees despite a +2 degree nose-down target. Keep
    # the tested geometry and penalize only the rearward error rather than rolling
    # the rectangular shoes farther onto their compliant tread edges.
    penalty_rearward_pitch = 4.0

    # V28's 0.04 normalized hip residual was only 0.04 * 0.30 rad = 0.69
    # degrees, too little to redistribute the rear-pack load. V29 uses 0.12
    # (2.06 degrees); the final joint target remains clamped to the Isaac soft
    # envelope before rate limiting.
    residual_action_scale_by_action = (
        0.10,
        0.10,
        0.10,
        0.10,
        0.12,
        0.12,
        0.12,
        0.12,
        0.15,
        0.15,
        0.15,
        0.15,
    )


@configclass
class DrobotCommandedWalkingSymmetryGatedRobustStraightCrawlExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingScheduleMatchedSupportStraightCrawlExternalRearPayloadEnvCfg
):
    """V30 speed-normalized straight-path and asymmetric-hardware curriculum."""

    # Retain the opposed 92 mm stance and 46/24 mm stride/lift envelope. The
    # reference hip already approaches the Isaac soft limit, so aggression is
    # added through a modest cadence increase rather than lower/longer geometry.
    gait_frequency_max_hz = 0.85
    # 46 mm * 0.85 Hz = 0.0391 m/s.
    forward_speed_max_m_s = 0.039
    gait_speed_max_m_s = 0.039

    # Zero-residual schedule ablations found that the opposite sign reduced net
    # lateral drift. Use the moderate value: a larger -24 mm shift improved the
    # endpoint but increased periodic sway, reverse motion, and effort.
    gait_weight_shift_lateral_m = -0.012
    # The rear-mounted pack leaves the diagonal front anchor light during a
    # rear-leg swing.  A controlled 20/25/30 mm ablation selected 25 mm: versus
    # 20 mm it improved exact contact and three/four-foot support, while 30 mm
    # added substantially more reversal and joint/body acceleration.
    gait_rear_weight_shift_forward_m = 0.025

    # Gate forward credit by cycle-scale sideslip relative to requested speed.
    # The 20% floor preserves discovery gradients; reverse progress is never
    # attenuated. Smooth Huber corridor/heading terms avoid the former delayed
    # quadratic cliff while still increasing linearly outside the corridor.
    straight_progress_use_normalized_lateral_velocity = True
    straight_progress_lateral_ratio_sigma = 0.05
    straight_progress_heading_sigma_rad = math.radians(4.0)
    gate_positive_progress_by_straight_alignment = True
    straight_progress_gate_floor = 0.20
    reward_straight_aligned_progress = 5.0
    use_huber_path_costs = True
    lateral_corridor_half_width_m = 0.010
    penalty_lateral_corridor = 3.0
    heading_error_normalizer_rad = math.radians(5.0)
    penalty_heading_error = 6.0

    # Train against plausible fixed assembly errors instead of one perfectly
    # mirrored robot: extra lateral payload-COM uncertainty, persistent roll
    # moment, and independent abduction zero offsets. Keep a nominal quarter of
    # environments so robustness does not erase the measured configuration.
    rear_payload_combined_com_jitter_m = (0.008, 0.015, 0.006)
    base_torque_randomization_range_nm = (0.040, 0.015, 0.040)
    joint_target_bias_abduction_rad = math.radians(1.5)
    # Extend the common supply-strength proxy down to about 10.6 V relative to
    # the servo's 12 V rating.  This covers a partly discharged 3S pack under
    # load without teaching the policy to continue through unsafe brownout.
    actuator_effort_scale_range = (0.88, 1.05)
    target_velocity_scale_range = (0.88, 1.05)
    mirror_physical_randomization_pairs = False
    physical_randomization_nominal_fraction = 0.25


@configclass
class DrobotCommandedRunningFastStopExternalRearPayloadEnvCfg(
    DrobotCommandedWalkingSymmetryGatedRobustStraightCrawlExternalRearPayloadEnvCfg
):
    """V31 fast diagonal trot with randomized graceful and abrupt stops."""

    # This is a separate ten-second run/stop task, not a replacement for V30's
    # statically stable crawl.  Every episode runs long enough to establish the
    # gait, decelerates at an unpredictable time, then holds the stopped stance.
    command_profile = "run_stop"
    episode_length_s = 10.0
    initial_training_horizon_s = 10.0
    final_training_horizon_s = 10.0
    episode_horizon_curriculum_steps = 1
    initial_forward_speed_min_m_s = 0.050
    initial_forward_speed_max_m_s = 0.090
    # Retain the low command range after curriculum completion.  Raising the
    # mature minimum to 0.10 m/s made a high-speed candidate forget rear-left
    # release at the 0.08 m/s guarded-trial speed.
    forward_speed_min_m_s = 0.050
    forward_speed_max_m_s = 0.200
    # Uniform full-range sampling underrepresented guarded-trial speeds and a
    # late checkpoint narrowly missed rear-left release at 0.08 m/s. Preserve
    # the full range while deliberately rehearsing the low-speed gait.
    forward_speed_low_band_fraction = 0.50
    forward_speed_low_band_max_m_s = 0.100
    command_curriculum_steps = 48_000
    run_stop_run_duration_min_s = 4.5
    run_stop_run_duration_max_s = 6.5
    run_stop_ramp_duration_min_s = 0.35
    run_stop_ramp_duration_max_s = 0.90
    # Abrupt zero commands prevent the policy from depending on a controller-side
    # ramp.  The remaining episodes teach lower-jerk commanded deceleration.
    run_stop_abrupt_fraction = 0.35

    # A 60% duty-factor diagonal trot has two short four-foot overlap intervals
    # per cycle but never requires a flight phase.  Speed comes primarily from
    # cadence, within the measured 60 Hz control and STS3215 target-rate limits.
    gait_reference_mode = "continuous"
    gait_smooth_support_push = False
    gait_distributed_push_phase_fractions = None
    gait_clock_mode = "speed_scaled"
    integrate_gait_phase = True
    gait_park_at_standstill = True
    gait_standstill_fore_aft_m = 0.080
    gait_standstill_down_m = 0.329341447
    gait_standstill_center_offset_m = 0.0
    target_standstill_pitch_rad = 0.0
    target_standstill_base_height_m = 0.3750
    gait_standstill_deadband_m_s = 0.004
    gait_speed_min_m_s = 0.060
    gait_speed_max_m_s = 0.200
    gait_frequency_min_hz = 0.90
    gait_frequency_max_hz = 2.40
    gait_stride_scale_min = 0.70
    gait_duty_factor = 0.60
    gait_phase_offsets = (0.0, 0.5, 0.5, 0.0)
    randomize_gait_start_phase_quarters = True
    gait_stance_fore_aft_m = LOW_SWEPT_STANCE_FORE_AFT_M
    gait_stance_down_m = LOW_SWEPT_STANCE_DOWN_M
    gait_stance_down_by_leg_m = (
        FORWARD_BIASED_FRONT_DOWN_M,
        FORWARD_BIASED_REAR_DOWN_M,
        FORWARD_BIASED_FRONT_DOWN_M,
        FORWARD_BIASED_REAR_DOWN_M,
    )
    gait_stance_center_offset_m = FORWARD_BIASED_STANCE_FORWARD_M
    gait_stride_m = 0.065
    gait_lift_m = 0.026
    gait_start_ramp_s = 0.50

    # Continue the proven V18 direct-action trot actor.  V30 residual transfer
    # was rejected after a zero-residual diagnostic showed that the low-stance
    # table alone mostly rocked in place.  The analytic gait remains a timing and
    # pose prior while PPO owns the final targets, bounded by the same soft limits
    # and 240 deg/s packet-level target slew limit used by the robust crawl.
    action_mode = "direct"
    residual_action_scale_by_action = (
        0.25,
        0.25,
        0.25,
        0.25,
        0.18,
        0.18,
        0.18,
        0.18,
        0.25,
        0.25,
        0.25,
        0.25,
    )
    target_velocity_limit_rad_s = math.radians(240.0)

    # A trot intentionally carries two-foot support, so remove the sequential
    # crawl's one-airborne-foot gates while retaining timing, impact, slip,
    # posture, straight-path, smoothness, and effort objectives.
    require_schedule_matched_support = False
    gate_progress_by_schedule_matched_support = False
    gate_progress_by_four_leg_release = False
    gate_progress_by_cycle_four_leg_release = False
    use_soft_scheduled_swing_release = True
    use_threshold_centered_release_shaping = False
    use_soft_scheduled_stance_quality = False
    use_soft_schedule_matched_progress_gate = False
    # The slow end of the command range otherwise permits a marginal rear-left
    # drag. Reward completion of every four-foot cycle and put a signed shaping
    # term on whichever foot has accumulated the least release activity.
    reward_cycle_four_leg_release = 0.50
    reward_least_active_swing = 2.0
    reward_three_foot_support = 0.0
    penalty_missing_scheduled_stance = 1.0
    penalty_excess_airborne_feet = 0.0
    reward_scheduled_stance = 1.5
    reward_scheduled_swing = 2.5
    gate_progress_by_scheduled_pair_release = True
    scheduled_pair_progress_gate_floor = 0.10
    reward_scheduled_pair_release = 8.0
    penalty_scheduled_pair_mismatch = 5.0
    reward_scheduled_release_shaping = 0.0
    reward_qualified_touchdown = 0.05
    penalty_diagonal_sync = 0.50
    gait_reference_sigma = 0.65
    reward_gait_reference = 4.0

    # Preserve V30's world-frame line tracking, rear-pack attitude target, fall
    # cost, Velcro contact domain, effort headroom, and asymmetric hardware
    # domains.  Motion has higher priority than in the crawl, but speed tracking
    # remains bounded so a forward lunge is not mistaken for controlled running.
    sustained_speed_window_s = 0.75
    active_translation_threshold_m_s = 0.020
    scale_sustained_speed_with_command = True
    minimum_sustained_speed_fraction = 0.55
    minimum_sustained_speed_m_s = 0.055
    velocity_tracking_sigma_m_s = 0.035
    reward_forward_velocity_tracking = 12.0
    reward_instant_progress = 14.0
    reward_sustained_progress = 18.0
    reward_straight_aligned_progress = 12.0
    penalty_sustained_stall = 20.0
    penalty_backward_motion = 26.0
    penalty_overspeed = 4.0
    straight_progress_lateral_ratio_sigma = 0.08
    straight_progress_heading_sigma_rad = math.radians(5.0)
    straight_progress_gate_floor = 0.15
    lateral_corridor_half_width_m = 0.015
    penalty_lateral_displacement = 55.0
    penalty_lateral_corridor = 8.0
    penalty_normalized_lateral_velocity = 8.0
    penalty_yaw_rate = 16.0
    heading_error_normalizer_rad = math.radians(4.0)
    penalty_heading_error = 10.0

    # Keep acceleration and impact shaping material without suppressing the
    # cadence needed for a trot.  Candidate selection still gates absolute joint
    # and body acceleration and sustained effort independently of total return.
    penalty_action_rate = 0.10
    penalty_action_acceleration = 0.50
    penalty_joint_velocity = 0.030
    penalty_joint_acceleration = 0.22
    penalty_body_linear_acceleration = 0.35
    penalty_body_angular_acceleration = 0.30
    penalty_support_foot_slip = 0.55
    penalty_touchdown_impact = 0.15
    touchdown_force_soft_limit_n = 16.0
    # Motion rewards are much stronger than the crawl's.  Scale the sustainable
    # effort term accordingly so PPO cannot buy speed by holding several servos
    # near the transient 90%-of-stall cap, especially after it has stopped.
    penalty_effort_soft_limit = 6.00

    # Once the command has reached zero, reward genuine stationary four-foot
    # support and penalize motion past the predicted stop point.  The target is
    # the integral of the sampled smooth command ramp; abrupt-stop episodes use
    # the trigger position itself.
    stop_velocity_sigma_m_s = 0.030
    stop_position_tolerance_m = 0.050
    reward_stop_velocity = 6.0
    reward_stop_all_feet_contact = 2.0
    # Direct action zero is the calibrated symmetric center stance.  A narrow
    # stopped-only objective prevents the actor from finding a quiet but
    # unnecessarily torque-heavy crouch after it has already decelerated.
    stop_neutral_action_sigma = 0.45
    reward_stop_neutral_action = 4.0
    penalty_stop_position = 5.0
    penalty_stop_yaw_rate = 6.0
    penalty_stop_joint_velocity = 2.0


@configclass
class DrobotCommandedRunningHardwareMatchedStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningFastStopExternalRearPayloadEnvCfg
):
    """V32 grounded run/stop policy matched to the first V31 hardware trace."""

    def __post_init__(self) -> None:
        super().__post_init__()
        # V31 inherited V28's approximately 35-degree, two-degree nose-down
        # reset pose even though its exported direct-action contract and the
        # physical guarded trial both used the calibrated 30-degree center
        # stance.  For a direct-action actor, that makes every output map to a
        # different absolute joint target in simulation and on hardware.  V32
        # deliberately restores one shared neutral for reset, action zero,
        # stopping, export, and the Raspberry Pi controller.
        self.robot = self.robot.replace(
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
            )
        )

    target_base_height_m = 0.3750

    # V31 requested 240 deg/s on nearly every sagittal joint. On the loaded
    # robot the encoder trace reached only about 80-120 deg/s and lagged the
    # policy by 0.20-0.47 seconds. Train only the command range we can qualify
    # on hardware, and use a cadence that leaves time for the feet to complete
    # their intended diagonal support transition.
    initial_forward_speed_min_m_s = 0.040
    initial_forward_speed_max_m_s = 0.070
    forward_speed_min_m_s = 0.040
    forward_speed_max_m_s = 0.120
    forward_speed_low_band_fraction = 0.60
    forward_speed_low_band_max_m_s = 0.080
    command_curriculum_steps = 64_000
    gait_speed_min_m_s = 0.040
    gait_speed_max_m_s = 0.120
    # One ideal no-slip gait cycle advances the body by approximately one
    # scheduled stride.  The old 0.70..1.30 Hz range therefore requested only
    # 0.034..0.085 m/s from the analytic reference for 0.040..0.120 m/s
    # commands.  PPO had to outrun its own contact/pose prior, and the first
    # faster continuation did so by dragging a rear shoe.  Match cadence to
    # command at both endpoints; interpolation stays close in between while
    # the actuator-rate randomization still teaches slower loaded hardware.
    gait_frequency_min_hz = 0.040 / (0.065 * 0.75)
    gait_frequency_max_hz = 0.120 / 0.065
    gait_stride_scale_min = 0.75
    gait_duty_factor = 0.68
    gait_stride_m = 0.065
    gait_lift_m = 0.022
    gait_start_ramp_s = 0.80

    # Two degrees per 60 Hz control interval is the deployment contract and is
    # close to the fastest sustained motion actually seen in the V31 trace.
    # Robust domains further reduce common and individual rate so the actor
    # must produce targets that remain useful on the slowest loaded joints.
    target_velocity_limit_rad_s = math.radians(120.0)
    target_velocity_scale_range = (0.70, 1.05)
    target_individual_velocity_scale_range = (0.78, 1.08)
    actuator_effort_scale_range = (0.70, 1.00)
    actuator_individual_effort_scale_range = (0.80, 1.05)
    actuator_individual_stiffness_scale_range = (0.55, 1.10)
    actuator_individual_damping_scale_range = (0.85, 1.30)
    control_delay_step_range = (0, 1)

    # V31's real heading correction sat at +/-0.20 rad/s for 41% of the trial.
    # A slower bounded outer loop avoids turning a correctable line error into
    # another alternating lateral command while the gait is already delayed.
    heading_hold_kp_s = 0.75
    heading_hold_max_correction_rad_s = 0.12
    base_torque_randomization_range_nm = (0.050, 0.025, 0.065)

    # Reject the exact failure mode observed on hardware: saturated actions
    # whose requested target runs persistently ahead of the executable target.
    # Speed remains rewarded, but not at the expense of feasible tracking,
    # diagonal coordination, smoothness, or sustainable servo effort.
    minimum_sustained_speed_fraction = 0.50
    minimum_sustained_speed_m_s = 0.035
    reward_forward_velocity_tracking = 12.0
    reward_instant_progress = 10.0
    reward_sustained_progress = 16.0
    reward_straight_aligned_progress = 12.0
    penalty_sustained_stall = 22.0
    penalty_backward_motion = 30.0
    penalty_overspeed = 6.0
    reward_gait_reference = 6.0
    gait_reference_sigma = 0.50
    penalty_diagonal_sync = 1.50
    penalty_action_rate = 0.40
    penalty_action_acceleration = 2.00
    penalty_action_saturation = 8.00
    penalty_target_limiter_gap = 18.00
    penalty_joint_velocity = 0.050
    penalty_joint_acceleration = 0.65
    penalty_body_linear_acceleration = 0.55
    penalty_body_angular_acceleration = 0.50
    penalty_support_foot_slip = 0.70
    penalty_touchdown_impact = 0.22
    penalty_effort_soft_limit = 24.0
    penalty_effort_rms = 4.0

    # The first clean V32 continuation moved and stopped well, but deterministic
    # evaluation found 24% requested/processed disagreement, 30% packet-rate
    # ceiling occupancy, and 7% saturated actions.  Preserve locomotion credit
    # while making executable, unsaturated trajectories strictly preferable.
    reward_scheduled_pair_release = 10.0
    penalty_scheduled_pair_mismatch = 8.0
    reward_least_active_swing = 4.0

    # Stop around the same calibrated neutral used by reset/export instead of
    # holding the transferred actor's asymmetric, high-torque terminal action.
    reward_stop_neutral_action = 8.0
    penalty_stop_action_magnitude = 10.0


@configclass
class DrobotCommandedRunningAppliedTargetHistoryStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningHardwareMatchedStopExternalRearPayloadEnvCfg
):
    """V33 observes the last executable target instead of the raw request."""

    # V32's deterministic 0.06 m/s trial showed that every knee continued to
    # command ahead of the packet-rate limiter.  Keeping the 50-value shape but
    # replacing the final history vector lets the actor condition its next
    # decision on the target the servo bus actually received.
    action_history_mode = "applied_target"


@configclass
class DrobotCommandedRunningIntegratedTargetVelocityStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningAppliedTargetHistoryStopExternalRearPayloadEnvCfg
):
    """V34 emits bounded target velocity instead of unreachable positions."""

    # V33 observed the executable target, but deterministic evaluation still
    # found 21.0% requested/applied disagreement and 27.0% packet-rate ceiling
    # occupancy.  Integrating a normalized velocity request makes a target jump
    # larger than the elapsed-time slew budget impossible by construction.
    action_mode = "integrated_target_velocity"
    action_history_mode = "applied_target"
    # Zero command always slews to the same calibrated center pose used by the
    # dashboard stance control. A learned 20% posture residual was evaluated in
    # two continuations: it stopped translation but repeatedly left one shoe
    # unloaded in the fixed randomized domain. The physical robot is known to
    # stand securely at calibrated center, so do not make its post-run support
    # depend on an unqualified policy output. The ordinary 120 deg/s / 2-degree
    # packet limiter still makes this a gradual transition rather than a jump.
    integrated_stop_action_scale = 0.0
    integrated_stop_target_residual_scale = 0.0

    # V32 deliberately included a 0.70 common strength/rate floor while
    # diagnosing the failed V31 controller trace.  V34's bounded, elapsed-time
    # target integration removes that controller failure mode, so inheriting the
    # same floor makes a fixed robust evaluation demand 0.08 m/s at an
    # approximately 8.4 V supply equivalent.  That is below the 10.6 V operating
    # floor modeled by V30 and belongs to battery/brownout safety, not locomotion.
    # Restore the measured 3S operating envelope.  V32 also widened individual
    # effort, rate, stiffness, and damping far beyond the earlier hardware prior
    # to imitate the same blocked-controller trace.  The V31 log cannot identify
    # those as servo-plant errors because packet catch-up bursts and synchronous
    # feedback reads dominated the observation.  Return to the independently
    # varied V25/V30 ranges while retaining per-servo mismatch, calibration bias,
    # delay, COM, friction, and external-wrench domains.  A genuinely weak or
    # stalled servo belongs to runtime health monitoring rather than a gait that
    # must propel itself through a simulated half-stiff actuator.
    actuator_effort_scale_range = (0.88, 1.05)
    target_velocity_scale_range = (0.88, 1.05)
    actuator_individual_effort_scale_range = (0.90, 1.05)
    target_individual_velocity_scale_range = (0.88, 1.05)
    actuator_individual_stiffness_scale_range = (0.85, 1.15)
    actuator_individual_damping_scale_range = (0.85, 1.15)
    # The selected actor already clears every nominal locomotion gate. Reserve
    # only a small rehearsal subset for nominal physics and spend the remaining
    # batch on the foot-friction, calibration, payload, and servo mismatches that
    # still expose one weak rear leg in deterministic robust replay.
    physical_randomization_nominal_fraction = 0.10

    # This is the separate run policy, while V30 remains the low-speed crawl
    # fallback. The earlier 60% low-band mix exposed the actor to >0.08 m/s in
    # only about one fifth of mature episodes; fixed 0.10 m/s evaluation then
    # reached just 0.053-0.057 m/s and dragged both rear shoes. Concentrate
    # adaptation on the useful run envelope while retaining 20% rehearsal at
    # 0.06..0.08 m/s.
    initial_forward_speed_min_m_s = 0.060
    initial_forward_speed_max_m_s = 0.085
    forward_speed_min_m_s = 0.060
    forward_speed_max_m_s = 0.120
    forward_speed_low_band_fraction = 0.20
    forward_speed_low_band_max_m_s = 0.080
    # A deterministic speed sweep exposed cadence collapse above 0.08 m/s: the
    # same actor produced 0.081 m/s at an 0.08 command, but slowed to 0.070 at
    # 0.10 and 0.054 at 0.12 as rear-foot release disappeared. Replace the
    # inherited 0.82..1.85 Hz clock with a longer, lower-frequency stride. At
    # 0.10 m/s this grows scheduled swing time from about 0.20 to 0.26 seconds,
    # while the 75 mm reference preserves nearly the same ideal ground speed.
    # The extra clearance is also reached at a lower peak vertical foot speed.
    gait_speed_min_m_s = 0.060
    gait_speed_max_m_s = 0.120
    gait_frequency_min_hz = 0.90
    gait_frequency_max_hz = 1.40
    gait_stride_scale_min = 0.85
    gait_stride_m = 0.075
    gait_lift_m = 0.034
    # Start each swing by lifting before the shoe travels fore/aft, finish that
    # travel before touchdown, and give both battery-loaded rear legs 6 mm more
    # clearance.  This directly addresses the held-out failures where a rear
    # shoe stayed above the strict 1 N release threshold while the actor swung
    # the leg laterally.  The left/right values remain symmetric.
    gait_lift_by_leg_m = (0.034, 0.040, 0.034, 0.040)
    gait_swing_translation_start_fraction = 0.20
    gait_swing_translation_end_fraction = 0.80

    # In this mode action magnitude is target speed, action rate is target
    # acceleration, and action acceleration is target jerk.  Keep enough range
    # for propulsion while making a continuous full-rate command expensive.
    penalty_action_magnitude = 1.50
    penalty_action_rate = 0.80
    penalty_action_acceleration = 2.00
    penalty_action_saturation = 12.00
    penalty_target_limiter_gap = 0.0

    # The first mixed-domain V34 pass exposed a reward/evaluation mismatch.  At
    # the 0.08 m/s guarded-trial command, the inherited sustained-speed threshold
    # was min(0.035, 0.50 * 0.08) = 0.035 m/s, while candidate acceptance requires
    # at least 65% of command, or 0.052 m/s.  The policy consequently learned a
    # safe 0.046 m/s randomized-domain shuffle and stopped receiving the stall
    # penalty. Use the same 65% floor during optimization throughout the complete
    # 0.06..0.12 m/s command range. Stronger tracking/progress terms ask for
    # useful ground speed without increasing the executable 120 deg/s target cap.
    minimum_sustained_speed_fraction = 0.65
    minimum_sustained_speed_m_s = 0.080
    reward_forward_velocity_tracking = 16.0
    # Fixed 0.08 and 0.10 m/s evaluations of the first release-qualified actor
    # both converged near 0.087 m/s.  Instantaneous body speed oscillates strongly
    # within a trot, so tighten command fidelity on the 0.75 s rolling value
    # rather than suppressing the gait's natural within-cycle motion.
    reward_rolling_velocity_tracking = 10.0
    rolling_velocity_tracking_sigma_m_s = 0.012
    reward_instant_progress = 12.0
    reward_sustained_progress = 20.0
    penalty_sustained_stall = 36.0
    penalty_overspeed = 10.0

    # The analytic diagonal gait anchors the integrated target so the actor
    # cannot obtain progress by drifting permanently into a joint limit.
    reward_gait_reference = 10.0
    gait_reference_sigma = 0.40
    # V34 bridge evaluation exposed a second sim-to-real reward loophole.  The
    # original integrated-target reward compared the executable servo target to
    # the analytic gait, so a physically slow rear leg could remain on the floor
    # while the policy still collected nearly full reference credit.  Score the
    # position actually reached as the dominant signal and blend the four-leg
    # mean toward the worst leg.  Target-space credit remains as a small anchor
    # so the actor does not compensate for lag with an unbounded phase lead.
    gait_reference_actual_state_weight = 0.70
    gait_reference_worst_leg_weight = 0.75

    # The first V34 continuation improved 0.08 m/s tracking from 89% to 97%,
    # but rear-left hard release fell from 90% to 48%.  The inherited 3 N
    # Gaussian still assigns exp(-(2/3)^2) ~= 0.64 quality to a shoe carrying
    # 2 N, so PPO could drag one member of a diagonal pair while retaining much
    # of its swing and progress credit.  V34 already begins from a checkpoint
    # that releases every shoe; use the real <1 N contact contract for the
    # per-frame and least-active-foot rewards, while retaining a narrow smooth
    # gradient only around that same threshold for pair-gated progress.
    use_soft_scheduled_swing_release = False
    use_threshold_centered_release_shaping = True
    normalize_threshold_centered_release_quality = False
    scheduled_release_force_threshold_n = 1.0
    scheduled_release_shaping_width_n = 0.35
    # The strict pair/progress gate remains exactly zero above 1 N. A separate
    # wider reward signal supplies a gradient while a high-friction rear shoe is
    # still carrying several newtons, so PPO can discover progressive unloading
    # without receiving any false contact-release credit.
    scheduled_release_discovery_width_n = 1.50
    # Randomized evaluation still found the rear shoes partially loaded during
    # their scheduled swing (37%/60% hard release).  Add a small continuous
    # threshold-centred discovery term, then retain hard release as the dominant
    # per-frame and least-active-leg objective.  This does not reward faster leg
    # motion; it rewards transferring load before asking either rear shoe to lift.
    reward_scheduled_swing = 6.0
    reward_scheduled_release_shaping = 4.0
    reward_least_active_swing = 12.0
    reward_cycle_four_leg_release = 6.0
    # Pair gating prevents one member of the active diagonal from being dragged,
    # but fixed randomized evaluation showed that the actor could alternate which
    # diagonal underperformed and still collect most of the episode's progress.
    # Gate every positive motion term by the weakest cumulative hard-release rate
    # across all four legs.  The small floor preserves a discovery gradient; full
    # speed credit requires every leg to meet the same 70% acceptance contract.
    gate_progress_by_four_leg_release = True
    # Keep a margin above the 70% evaluator threshold so speed tracking cannot
    # be bought by returning to a barely dragging rear shoe.
    four_leg_release_target_rate = 0.85
    four_leg_progress_gate_floor = 0.05

    # Four-foot contact is sparse under randomized calibration and rigid shoe
    # contact, so also reward the continuously measured weakest-foot load. A
    # modest posture cost keeps the learned stop close to calibrated center but
    # no longer overwhelms the few-degree residual needed to plant all shoes.
    reward_stop_all_feet_contact = 6.0
    reward_stop_weakest_foot_contact = 8.0
    reward_stop_neutral_action = 4.0
    penalty_stop_action_magnitude = 2.0
    penalty_stop_target_magnitude = 6.0
    penalty_stop_position = 3.0


@configclass
class DrobotCommandedRunningCommandScaledHighSpeedStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningIntegratedTargetVelocityStopExternalRearPayloadEnvCfg
):
    """V35 speed-conditioned, hardware-feasible high-speed run/stop policy."""

    # V34's deterministic actor converged to approximately 0.085 m/s for 0.08,
    # 0.10, and 0.12 m/s commands. Raw SI command inputs differ by only a few
    # hundredths, so V35 magnifies them while retaining the same 50 observations.
    # The bootstrap checkpoint divides the matching first-layer columns by ten,
    # which makes its initial deterministic output numerically identical.
    command_observation_scale = (10.0, 10.0, 10.0)

    # This is a separate high-speed policy, not the V30 crawl replacement. Keep
    # some 0.08-0.10 m/s rehearsal but spend most rollouts on 0.10-0.14 m/s.
    initial_forward_speed_min_m_s = 0.080
    initial_forward_speed_max_m_s = 0.140
    forward_speed_min_m_s = 0.080
    forward_speed_max_m_s = 0.140
    forward_speed_low_band_fraction = 0.20
    forward_speed_low_band_max_m_s = 0.100

    # A longer stride and longer swing window request more forward displacement
    # without restoring V31's unreachable 2.4 Hz / 240 deg/s target schedule.
    gait_speed_min_m_s = 0.080
    gait_speed_max_m_s = 0.140
    gait_frequency_min_hz = 1.00
    gait_frequency_max_hz = 1.45
    gait_stride_scale_min = 0.80
    gait_stride_m = 0.090
    gait_duty_factor = 0.64
    gait_lift_m = 0.038
    gait_lift_by_leg_m = (0.038, 0.044, 0.038, 0.044)
    gait_swing_translation_start_fraction = 0.18
    gait_swing_translation_end_fraction = 0.82

    # A Gaussian supplies precision near command. The symmetric rolling error
    # remains informative when the actor is far below command, while the strict
    # four-leg release gates continue to prevent speed by dragging a shoe.
    minimum_sustained_speed_fraction = 0.82
    minimum_sustained_speed_m_s = 0.115
    velocity_tracking_sigma_m_s = 0.030
    reward_forward_velocity_tracking = 18.0
    reward_rolling_velocity_tracking = 14.0
    rolling_velocity_tracking_sigma_m_s = 0.014
    penalty_rolling_velocity_error = 24.0
    penalty_overspeed = 12.0
    reward_gait_reference = 12.0


@configclass
class DrobotCommandedRunningFlatSoleHighSpeedStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningCommandScaledHighSpeedStopExternalRearPayloadEnvCfg
):
    """V36 rectangular-shoe-aware, speed-conditioned run/stop policy."""

    # Reference-only diagnosis of V35 showed that its constant-height contact
    # arc moved the robot backward: the 100 mm rectangular shoes rolled onto an
    # edge during stance.  Keep planted soles level and use the same 80 mm
    # opposed stance that is already proven on hardware.
    gait_flat_sole_support = True
    gait_stance_fore_aft_m = 0.080
    gait_stance_down_m = 0.329341447
    gait_stance_down_by_leg_m = None
    gait_stance_center_offset_m = stance_forward_bias_for_body_pitch_m(
        FORWARD_BIASED_BODY_PITCH_RAD,
        fore_aft_m=gait_stance_fore_aft_m,
    )

    # Keep the complete fore/aft envelope below the 60-degree hip soft limit.
    # Cadence supplies speed, while the wide swing-translation window keeps peak
    # target rate close to the executable 120 deg/s contract.
    gait_stride_m = 0.070
    gait_frequency_min_hz = 1.20
    gait_frequency_max_hz = 1.65
    gait_stride_scale_min = 0.80
    gait_duty_factor = 0.62
    gait_lift_m = 0.050
    gait_lift_by_leg_m = (0.050, 0.060, 0.050, 0.060)
    gait_swing_translation_start_fraction = 0.10
    gait_swing_translation_end_fraction = 0.90

    minimum_sustained_speed_fraction = 0.75
    minimum_sustained_speed_m_s = 0.110
    reward_gait_reference = 14.0


@configclass
class DrobotCommandedRunningFlatSoleResidualRunStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningFlatSoleHighSpeedStopExternalRearPayloadEnvCfg
):
    """V37 bounded residual corrections around a forward flat-sole run."""

    # V36's integrated-velocity actor reacts to larger commands, but the
    # accumulated targets leave the analytic gait entirely: at 0.14 m/s rear
    # release collapses and the robot slows while its legs move more violently.
    # In V37 the command directly controls a physically forward feed-forward
    # gait and PPO may make only small, bounded corrections around it.
    action_mode = "gait_residual"
    action_history_mode = "applied_target"
    gait_residual_stop_action_scale = 0.0
    residual_action_scale = 0.08
    residual_action_scale_by_action = (
        0.12,
        0.12,
        0.12,
        0.12,
        0.16,
        0.16,
        0.16,
        0.16,
        0.20,
        0.20,
        0.20,
        0.20,
    )

    # Keep this separate run profile inside the actuator-feasible cadence that
    # passed V36 evaluation.  The wider swing window avoids the catch-up bursts
    # seen in the real V31 trace, while a 56% duty factor gives each loaded rear
    # shoe enough time to unload.  Speed remains command-conditioned through
    # both stride amplitude and cadence instead of asking the actor to invent a
    # completely different pose trajectory for every requested velocity.
    initial_forward_speed_min_m_s = 0.080
    initial_forward_speed_max_m_s = 0.130
    forward_speed_min_m_s = 0.080
    forward_speed_max_m_s = 0.130
    forward_speed_low_band_fraction = 0.25
    forward_speed_low_band_max_m_s = 0.095
    command_curriculum_steps = 1
    gait_speed_min_m_s = 0.080
    gait_speed_max_m_s = 0.130
    gait_frequency_min_hz = 1.00
    gait_frequency_max_hz = 1.35
    gait_stride_scale_min = 0.80
    gait_stance_fore_aft_m = 0.070
    gait_stance_center_offset_m = stance_forward_bias_for_body_pitch_m(
        FORWARD_BIASED_BODY_PITCH_RAD,
        fore_aft_m=gait_stance_fore_aft_m,
    )
    gait_stride_m = 0.090
    gait_duty_factor = 0.56
    gait_lift_m = 0.050
    gait_lift_by_leg_m = (0.050, 0.062, 0.050, 0.062)
    gait_swing_translation_start_fraction = 0.08
    gait_swing_translation_end_fraction = 0.92
    gait_start_ramp_s = 1.00

    # The reference supplies propulsion; the actor is rewarded for the smallest
    # smooth correction that makes every shoe release, holds the heading, and
    # closes the rolling-speed error.  Strong body/joint acceleration costs make
    # a lateral flail strictly worse than a slightly slower coordinated gait.
    minimum_sustained_speed_fraction = 0.80
    minimum_sustained_speed_m_s = 0.105
    reward_forward_velocity_tracking = 20.0
    reward_rolling_velocity_tracking = 18.0
    rolling_velocity_tracking_sigma_m_s = 0.014
    penalty_rolling_velocity_error = 30.0
    penalty_overspeed = 14.0
    reward_gait_reference = 16.0
    gait_reference_sigma = 0.45
    penalty_diagonal_sync = 2.0
    penalty_action_magnitude = 0.50
    penalty_action_rate = 0.90
    penalty_action_acceleration = 2.50
    penalty_action_saturation = 10.0
    penalty_target_limiter_gap = 8.0
    penalty_joint_velocity = 0.060
    penalty_joint_acceleration = 0.80
    penalty_body_linear_acceleration = 0.70
    penalty_body_angular_acceleration = 0.70
    penalty_support_foot_slip = 0.90
    penalty_touchdown_impact = 0.30


@configclass
class DrobotCommandedRunningStraightLowAbductionLongStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningFlatSoleHighSpeedStopExternalRearPayloadEnvCfg
):
    """V38 long-horizon run with deliberately bounded lateral leg motion."""

    # V36 practiced only 4.5..6.5 seconds of motion before its stop transition.
    # Fifteen-second held-out evaluations then exposed a slow, repeatable side
    # drift that the shorter optimization horizon did not price consistently.
    # Keep several seconds after the ramp for the same graceful-stop objective,
    # but make straight locomotion the majority of every episode.
    episode_length_s = 14.0
    initial_training_horizon_s = 14.0
    final_training_horizon_s = 14.0
    episode_horizon_curriculum_steps = 1
    run_stop_run_duration_min_s = 8.0
    run_stop_run_duration_max_s = 10.0
    run_stop_ramp_duration_min_s = 0.45
    run_stop_ramp_duration_max_s = 1.00
    run_stop_abrupt_fraction = 0.20

    # The flat rectangular-sole reference has zero abduction motion. V36's
    # deterministic policy nevertheless requested 60..96 deg/s abduction peaks,
    # matching the side-to-side leg swing visible on hardware. Cap only those
    # four balance axes at 60 deg/s while preserving the qualified 120 deg/s
    # sagittal hip/knee envelope used for forward propulsion.
    target_velocity_limit_rad_s_by_action = (
        math.radians(60.0),
        math.radians(60.0),
        math.radians(60.0),
        math.radians(60.0),
        math.radians(120.0),
        math.radians(120.0),
        math.radians(120.0),
        math.radians(120.0),
        math.radians(120.0),
        math.radians(120.0),
        math.radians(120.0),
        math.radians(120.0),
    )
    penalty_abduction_action_magnitude = 8.0
    penalty_abduction_target_magnitude = 4.0

    # Make persistent drift expensive early in the longer run rather than only
    # after it has crossed a broad corridor. The policy still sees IMU attitude,
    # acceleration, and joint feedback, so it retains small asymmetric authority
    # for the real robot's fixed assembly and load differences.
    straight_progress_lateral_ratio_sigma = 0.055
    straight_progress_gate_floor = 0.10
    lateral_corridor_half_width_m = 0.010
    penalty_lateral_displacement = 70.0
    penalty_lateral_corridor = 12.0
    penalty_normalized_lateral_velocity = 12.0
    reward_straight_aligned_progress = 16.0


@configclass
class DrobotCommandedRunningReferenceDrivenStraightStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningStraightLowAbductionLongStopExternalRearPayloadEnvCfg
):
    """V39 command-faithful flat-sole run with bounded learned corrections."""

    # V38 remained upright and substantially reduced lateral drift, but its
    # integrated velocity actor converged near 0.09--0.10 m/s for every command.
    # At 0.14 m/s it moved *slower* and dragged a rear shoe.  Make the commanded
    # gait the feed-forward target so speed controls stride/cadence directly;
    # PPO is responsible only for the small corrections needed by IMU feedback,
    # the rear pack, tread friction, calibration error, and actuator mismatch.
    action_mode = "gait_residual"
    action_history_mode = "applied_target"
    gait_residual_stop_action_scale = 0.0

    # Earlier V37 reference targets were silently clipped by the legacy direct-
    # action normalization (about +/-17 deg hip and +/-23 deg knee around the
    # 30-degree neutral).  These scales cover the complete sampled V39 analytic
    # envelope, including the higher rear-foot clearance, without changing the
    # 60/120 deg/s packet-rate limits below.
    action_scale_hip_rad = 0.52
    action_scale_knee_rad = 0.64
    residual_action_scale = 1.0
    residual_action_scale_by_action = (
        0.45,
        0.45,
        0.45,
        0.45,
        0.35,
        0.35,
        0.35,
        0.35,
        0.35,
        0.35,
        0.35,
        0.35,
    )

    # Keep the requested range physically consistent with a 75 mm step.  The
    # endpoints request approximately 0.078 and 0.120 m/s from the analytic
    # gait, unlike V38's 0.14 m/s command paired with only a 0.116 m/s reference.
    initial_forward_speed_min_m_s = 0.080
    initial_forward_speed_max_m_s = 0.120
    forward_speed_min_m_s = 0.080
    forward_speed_max_m_s = 0.120
    forward_speed_low_band_fraction = 0.20
    forward_speed_low_band_max_m_s = 0.090
    command_curriculum_steps = 1
    gait_speed_min_m_s = 0.080
    gait_speed_max_m_s = 0.120
    gait_frequency_min_hz = 1.10
    gait_frequency_max_hz = 1.60
    gait_stride_scale_min = 0.95
    gait_stance_fore_aft_m = 0.080
    gait_stance_center_offset_m = stance_forward_bias_for_body_pitch_m(
        FORWARD_BIASED_BODY_PITCH_RAD,
        fore_aft_m=gait_stance_fore_aft_m,
    )
    gait_stride_m = 0.075
    # A 56% duty factor retains a two-diagonal support overlap while giving the
    # loaded rear feet 275 ms of swing at the maximum 1.6 Hz cadence.
    gait_duty_factor = 0.56
    gait_lift_m = 0.030
    gait_lift_by_leg_m = (0.030, 0.038, 0.030, 0.038)
    gait_diagonal_support_lateral_shift_m = 0.012
    gait_swing_translation_start_fraction = 0.05
    gait_swing_translation_end_fraction = 0.95
    gait_start_ramp_s = 1.00

    # Preserve V38's smooth low-abduction contract.  The policy cannot inject
    # more than about 3.1 degrees of lateral correction through the residual,
    # and the actual sent target remains capped at 60 deg/s on those four axes.
    penalty_abduction_action_magnitude = 12.0
    penalty_abduction_target_magnitude = 4.0

    # A reference-driven gait should track speed rather than finding another
    # command-independent compromise.  Hard four-leg release gating inherited
    # from V34 still prevents earning this progress by dragging one shoe.
    minimum_sustained_speed_fraction = 0.82
    minimum_sustained_speed_m_s = 0.110
    reward_forward_velocity_tracking = 20.0
    reward_rolling_velocity_tracking = 18.0
    rolling_velocity_tracking_sigma_m_s = 0.012
    penalty_rolling_velocity_error = 30.0
    reward_instant_progress = 14.0
    reward_sustained_progress = 24.0
    penalty_sustained_stall = 48.0
    penalty_overspeed = 14.0
    reward_gait_reference = 16.0
    gait_reference_sigma = 0.45
    penalty_diagonal_sync = 2.0
    penalty_action_magnitude = 1.0
    penalty_action_rate = 1.0
    penalty_action_acceleration = 2.5
    penalty_joint_acceleration = 0.80
    penalty_body_linear_acceleration = 0.70
    penalty_body_angular_acceleration = 0.70


@configclass
class DrobotCommandedRunningSpeedFaithfulStraightStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningStraightLowAbductionLongStopExternalRearPayloadEnvCfg
):
    """V40 focused 0.09--0.12 m/s continuation of the stable V38 actor."""

    # V38 safely reached about 0.098 m/s, but its 0.14 m/s command exceeded the
    # 70 mm reference's 0.116 m/s ideal ceiling.  The policy responded by moving
    # more violently, losing rear release, and slowing down.  Train only the
    # actuator-feasible useful band and align the reference endpoints with it.
    initial_forward_speed_min_m_s = 0.090
    initial_forward_speed_max_m_s = 0.120
    forward_speed_min_m_s = 0.090
    forward_speed_max_m_s = 0.120
    forward_speed_low_band_fraction = 0.15
    forward_speed_low_band_max_m_s = 0.100
    command_curriculum_steps = 1
    gait_speed_min_m_s = 0.090
    gait_speed_max_m_s = 0.120
    gait_frequency_min_hz = 1.30
    gait_frequency_max_hz = 1.65
    gait_stride_scale_min = 0.95

    # A longer active interval makes persistent speed error and drift dominate
    # the episode while still leaving a complete center-stance stop rehearsal.
    episode_length_s = 16.0
    initial_training_horizon_s = 16.0
    final_training_horizon_s = 16.0
    episode_horizon_curriculum_steps = 1
    run_stop_run_duration_min_s = 10.0
    run_stop_run_duration_max_s = 12.0
    run_stop_ramp_duration_min_s = 0.55
    run_stop_ramp_duration_max_s = 1.00
    run_stop_abrupt_fraction = 0.15

    # Make a command-independent 0.09--0.10 m/s compromise expensive.  These
    # terms act on the 0.75 s rolling speed, not the within-stride oscillation,
    # and all positive progress remains gated by the weakest leg's hard release.
    minimum_sustained_speed_fraction = 0.88
    minimum_sustained_speed_m_s = 0.110
    velocity_tracking_sigma_m_s = 0.024
    reward_forward_velocity_tracking = 26.0
    reward_rolling_velocity_tracking = 26.0
    rolling_velocity_tracking_sigma_m_s = 0.010
    penalty_rolling_velocity_error = 48.0
    reward_instant_progress = 18.0
    reward_sustained_progress = 30.0
    penalty_sustained_stall = 60.0
    penalty_overspeed = 18.0

    # Preserve clean diagonal support while asking for the final 10--20% speed.
    # In particular, reject the high-command rear drag observed in V38 rather
    # than allowing speed reward to buy a faster lateral shuffle.
    reward_scheduled_swing = 8.0
    reward_scheduled_release_shaping = 6.0
    reward_scheduled_pair_release = 14.0
    penalty_scheduled_pair_mismatch = 12.0
    reward_least_active_swing = 20.0
    reward_cycle_four_leg_release = 10.0
    four_leg_release_target_rate = 0.92
    four_leg_progress_gate_floor = 0.02

    # V38's smoother controller trajectory is a hard requirement, not a reward
    # to trade away for speed.  Keep the low-abduction caps and strengthen the
    # measured joint/body acceleration prices modestly.
    penalty_action_rate = 0.90
    penalty_action_acceleration = 2.30
    penalty_joint_acceleration = 0.78
    penalty_body_linear_acceleration = 0.68
    penalty_body_angular_acceleration = 0.68
    penalty_support_foot_slip = 0.90
    penalty_touchdown_impact = 0.30


@configclass
class DrobotCommandedRunningPhaseNormalizedSpeedStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningSpeedFaithfulStraightStopExternalRearPayloadEnvCfg
):
    """V41 keeps integrated joint excursion consistent across gait cadences."""

    # V40 checkpoint 3600 is safe and straight, but a fixed-command sweep found
    # essentially one 0.10 m/s attractor: 0.09, 0.10, 0.11, and 0.12 m/s
    # commands produced 0.100, 0.102, 0.101, and 0.098 m/s.  The integrated
    # actor emits target velocity as a function of phase.  Advancing phase more
    # quickly therefore reduced angle travelled per cycle unless the actor also
    # learned an exact frequency-dependent gain.  Scale its motion output by
    # current/reference cadence before the unchanged hard target-rate clamp.
    integrated_action_scale_with_gait_frequency = True
    integrated_action_reference_frequency_hz = 1.30 + (1.0 / 3.0) * (1.65 - 1.30)


@configclass
class DrobotCommandedRunningLearnedCycleResidualStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningSpeedFaithfulStraightStopExternalRearPayloadEnvCfg
):
    """V42 time-warps V40's stable learned cycle with small RL corrections."""

    gait_reference_mode = "phase_table"
    gait_phase_reference_path = str(
        PROJECT_ROOT
        / "simulation"
        / "isaac"
        / "rl"
        / "parallel_walking"
        / "references"
        / "v40_model3600_0p10_phase_reference.json"
    )
    # The source cycle was extracted at 0.10 m/s.  Scaling it by 1.10 and using
    # a 0.90 stride factor at that command reproduces the original excursion;
    # the full table then adds modest stride at the top of the new speed band.
    gait_phase_reference_scale_about_neutral = 1.10
    action_mode = "gait_residual"
    action_history_mode = "applied_target"
    gait_residual_stop_action_scale = 0.0
    residual_action_scale = 1.0
    residual_action_scale_by_action = (
        0.08,
        0.08,
        0.08,
        0.08,
        0.18,
        0.18,
        0.18,
        0.18,
        0.18,
        0.18,
        0.18,
        0.18,
    )

    initial_forward_speed_min_m_s = 0.090
    initial_forward_speed_max_m_s = 0.120
    forward_speed_min_m_s = 0.080
    forward_speed_max_m_s = 0.140
    forward_speed_low_band_fraction = 0.20
    forward_speed_low_band_max_m_s = 0.100
    command_curriculum_steps = 32_000
    gait_speed_min_m_s = 0.080
    gait_speed_max_m_s = 0.140
    gait_frequency_min_hz = 1.20
    gait_frequency_max_hz = 1.85
    gait_stride_scale_min = 0.85
    gait_start_ramp_s = 1.00

    # The reference remains bounded by the same deliberate lateral rate but can
    # use 180 deg/s on sagittal axes. That is below the STS3215's approximate
    # 270 deg/s no-load speed while leaving margin for the loaded rear joints.
    target_velocity_limit_rad_s = math.radians(180.0)
    target_velocity_limit_rad_s_by_action = (
        math.radians(60.0),
        math.radians(60.0),
        math.radians(60.0),
        math.radians(60.0),
        math.radians(180.0),
        math.radians(180.0),
        math.radians(180.0),
        math.radians(180.0),
        math.radians(180.0),
        math.radians(180.0),
        math.radians(180.0),
        math.radians(180.0),
    )

    # Keep residuals subordinate to the stable cycle and strongly reject the
    # lateral flail from the V31 hardware trace.
    reward_gait_reference = 20.0
    gait_reference_sigma = 0.35
    penalty_action_magnitude = 2.0
    penalty_abduction_action_magnitude = 18.0
    penalty_abduction_target_magnitude = 5.0
    penalty_action_rate = 1.20
    penalty_action_acceleration = 2.80
    penalty_joint_acceleration = 0.90
    penalty_body_linear_acceleration = 0.75
    penalty_body_angular_acceleration = 0.75


@configclass
class DrobotCommandedRunningFootWarpTrotResidualStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningLearnedCycleResidualStopExternalRearPayloadEnvCfg
):
    """V43 low-duty foot-space trot with IMU-driven residual corrections."""

    # V40's learned 0.10 m/s cycle is phase-remapped in Cartesian sagittal foot
    # space.  The body is 22 mm lower, stance travel is 20% longer, and swing
    # has 30 mm of additional smooth clearance.  A 46% stance interval produced
    # 0.126 m/s nominally and 0.110 m/s across the first five held-out hardware
    # randomizations without a fall, unlike joint-angle scaling in V41/V42.
    gait_phase_reference_path = str(
        PROJECT_ROOT
        / "simulation"
        / "isaac"
        / "rl"
        / "parallel_walking"
        / "references"
        / "v43_v40_low_foot_warp_duty0p46_scale1p20_lift0p030.json"
    )
    gait_phase_reference_scale_about_neutral = 1.0
    gait_phase_reference_low_path = str(
        PROJECT_ROOT
        / "simulation"
        / "isaac"
        / "rl"
        / "parallel_walking"
        / "references"
        / "v43_v40_low_foot_warp_duty0p46_scale1p00_lift0p030.json"
    )
    gait_phase_reference_blend_speed_min_m_s = 0.10
    gait_phase_reference_blend_speed_max_m_s = 0.14
    gait_duty_factor = 0.46
    gait_stride_scale_min = 1.0
    gait_speed_min_m_s = 0.10
    gait_speed_max_m_s = 0.14
    gait_frequency_min_hz = 1.15
    gait_frequency_max_hz = 1.60
    gait_start_ramp_s = 1.20

    initial_forward_speed_min_m_s = 0.10
    initial_forward_speed_max_m_s = 0.14
    forward_speed_min_m_s = 0.10
    forward_speed_max_m_s = 0.14
    forward_speed_low_band_fraction = 0.25
    forward_speed_low_band_max_m_s = 0.115
    command_curriculum_steps = 1

    # The generated trajectory supplies essentially all of the motion.  PPO
    # may make only a few degrees of correction from IMU/joint feedback, with
    # especially little authority on lateral abduction axes.
    residual_action_scale_by_action = (
        0.04,
        0.04,
        0.04,
        0.04,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
        0.12,
    )
    reward_gait_reference = 24.0
    gait_reference_sigma = 0.30
    penalty_action_magnitude = 3.0
    penalty_abduction_action_magnitude = 22.0
    penalty_abduction_target_magnitude = 6.0

    # Reward real command tracking, but retain hard four-leg release gating so
    # a dragging shoe cannot earn forward-progress credit.  The narrow line
    # corridor directly targets the left drift seen in the hardware videos.
    minimum_sustained_speed_fraction = 0.85
    minimum_sustained_speed_m_s = 0.115
    reward_forward_velocity_tracking = 30.0
    reward_rolling_velocity_tracking = 30.0
    rolling_velocity_tracking_sigma_m_s = 0.012
    penalty_rolling_velocity_error = 52.0
    reward_instant_progress = 20.0
    reward_sustained_progress = 34.0
    penalty_sustained_stall = 64.0
    reward_straight_aligned_progress = 18.0
    lateral_corridor_half_width_m = 0.008
    penalty_lateral_displacement = 80.0
    penalty_lateral_corridor = 14.0

    # Keep the target packet envelope inherited from V42 and explicitly price
    # the coupled torque-speed region rather than treating torque and rate as
    # independent capacities.  Load sharing is deliberately modest because the
    # real rear payload makes perfectly equal leg effort physically incorrect.
    actuator_torque_speed_margin = 0.92
    penalty_actuator_torque_speed_envelope = 6.0
    penalty_leg_operating_point_imbalance = 1.5
    penalty_action_rate = 1.35
    penalty_action_acceleration = 3.00
    penalty_joint_acceleration = 1.00
    penalty_body_linear_acceleration = 0.90
    penalty_body_angular_acceleration = 0.90


@configclass
class DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg(
    DrobotCommandedRunningFootWarpTrotResidualStopExternalRearPayloadEnvCfg
):
    """V44 corrective flat-contact candidate; not hardware qualified.

    This deliberately slow stage fixes geometry/target feasibility before any
    claim of running. The rigid feet have no independent ankle; support depth
    and body pose must adapt together. Preserve V43 as an immutable baseline.
    """

    gait_phase_reference_path = str(Path(__file__).with_name("references") / "v44_flat_contact_high.json")
    gait_phase_reference_low_path = str(Path(__file__).with_name("references") / "v44_flat_contact_low.json")
    gait_phase_reference_scale_about_neutral = 1.0
    gait_phase_reference_blend_speed_min_m_s = 0.03
    gait_phase_reference_blend_speed_max_m_s = 0.05
    gait_duty_factor = 0.55
    gait_period_s = 1.0 / 0.45
    gait_speed_min_m_s = 0.03
    gait_speed_max_m_s = 0.05
    gait_frequency_min_hz = 0.35
    gait_frequency_max_hz = 0.45
    gait_stride_scale_min = 1.0
    gait_phase_table_park_contact_override = True
    # PhysX writers do not update the implicit actuator's approximate PD torque
    # model. Keep its gains consistent during V44 gain hypotheses/randomization.
    actuator_sync_drive_gain_telemetry = True
    gait_start_ramp_s = 1.50
    gait_contact_transition_fraction = 0.0
    gait_stance_fore_aft_m = 0.080
    gait_stance_down_m = 0.329341447
    gait_stance_down_by_leg_m = None
    gait_stance_center_offset_m = 0.0
    gait_forward_body_pitch_rad = 0.0
    target_forward_pitch_rad = 0.0
    target_base_height_m = 0.375
    gait_diagonal_support_lateral_shift_m = 0.0

    initial_forward_speed_min_m_s = 0.03
    initial_forward_speed_max_m_s = 0.05
    forward_speed_min_m_s = 0.03
    forward_speed_max_m_s = 0.05
    forward_speed_low_band_fraction = 0.25
    forward_speed_low_band_max_m_s = 0.035
    command_curriculum_steps = 1
    action_scale_abduction_rad = 0.12
    action_scale_hip_rad = 0.30
    action_scale_knee_rad = 0.40
    residual_action_scale = 0.06
    residual_action_scale_by_action = (0.06,) * 4 + (0.20,) * 8
    target_velocity_limit_rad_s = math.radians(90.0)
    target_velocity_limit_rad_s_by_action = (math.radians(60.0),) * 4 + (math.radians(90.0),) * 8
    target_packet_step_limit_rad = math.radians(2.0)
    target_packet_hold_probability = 0.12
    target_packet_max_hold_steps = 1
    control_delay_step_range = (0, 0)
    # Hypotheses, not an identified servo model. Avoid adding the measured
    # ~0.3s total cyclic lag on top of existing actuator/contact physics.
    actuator_target_response_enabled = True
    actuator_target_response_nominal_tau_s = 0.030
    actuator_target_response_nominal_gain = 0.97
    actuator_target_response_tau_s_range = (0.010, 0.060)
    actuator_target_response_gain_range = (0.90, 1.0)
    actuator_target_response_tau_s_range_by_action = (
        (0.010, 0.030), (0.010, 0.030), (0.010, 0.030), (0.010, 0.030),
        (0.020, 0.070), (0.010, 0.040), (0.020, 0.070), (0.010, 0.040),
        (0.020, 0.070), (0.020, 0.060), (0.020, 0.070), (0.020, 0.060),
    )
    penalty_stance_sole_tilt = 12.0
    penalty_swing_corner_clearance = 6.0
    penalty_contact_point_slip = 4.0
    sole_clearance_target_m = 0.010
    sole_clearance_transition_fraction = 0.50
    sole_corner_soft_contact_height_m = 0.004

    # Reduce pressure to imitate a fixed table or chase the old0.115m/s
    # threshold. Preserve positive progress, four-leg participation, line
    # tracking, acceleration costs, and the existing neutral stop contract.
    reward_gait_reference = 6.0
    gait_reference_sigma = 0.45
    penalty_action_magnitude = 1.5
    penalty_abduction_action_magnitude = 14.0
    minimum_sustained_speed_m_s = 0.020
    # Cover a complete slow gait cycle instead of scoring each swing as a stall.
    sustained_speed_window_s = 3.0
    minimum_sustained_speed_fraction = 0.65
    velocity_tracking_sigma_m_s = 0.015
    reward_forward_velocity_tracking = 18.0
    reward_rolling_velocity_tracking = 18.0
    penalty_rolling_velocity_error = 30.0
    reward_instant_progress = 10.0
    reward_sustained_progress = 18.0
    penalty_sustained_stall = 32.0
    penalty_overspeed = 18.0


@configclass
class DrobotCommandedWalkingDirectionalEnvCfg(DrobotCommandedWalkingForwardEnvCfg):
    """Episode commands include forward, backward, left, right, and stop."""

    command_profile = "directional"
