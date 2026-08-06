"""Configuration for command-conditioned parallel Drobot walking."""

from __future__ import annotations

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
ROBOT_USD = PROJECT_ROOT / "exports" / "isaac" / "quadruped_robot_floating.usdc"
EFFORT_CAP_NM = 0.8825985
SERVO_VELOCITY_LIMIT_RAD_S = 4.5836625
STABLE_NEUTRAL_FRONT_HIP_RAD = -0.1544915885
STABLE_NEUTRAL_REAR_HIP_RAD = 0.1544915885
STABLE_NEUTRAL_FRONT_KNEE_RAD = 0.4699251950
STABLE_NEUTRAL_REAR_KNEE_RAD = -0.4699251950


@configclass
class DrobotCommandedWalkingForwardEnvCfg(DirectRLEnvCfg):
    """Flat-ground, forward-first walking with hardware-reproducible inputs."""

    # The validated single-environment walking policy used a 60 Hz controller.
    # Thirty hertz made the parallel policy learn discrete foot stamping instead.
    decimation = 2
    episode_length_s = 8.0
    # Preview may override this through Hydra. Training leaves it false so PPO
    # continues to receive fixed-horizon episode boundaries.
    disable_time_limit = False
    action_space = 12
    # command 3 + IMU 9 + joint position error 12 + velocity 12 + last action 12
    observation_space = 48
    # policy observation 48 + privileged base velocity 3 + height 1 + contacts 4
    state_space = 56

    viewer: ViewerCfg = ViewerCfg(
        eye=(-1.35, -1.20, 0.75),
        lookat=(0.30, 0.0, 0.28),
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
        env_spacing=3.0,
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
            pos=(0.0, 0.0, 0.3730),
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
    reset_joint_position_noise_rad = 0.015
    reset_xy_jitter_m = 0.02
    # These are the validated walking termination limits.  V9's 0.28 m height
    # cutoff and 100-point failure cost made safe standing a strong local optimum.
    minimum_base_height_m = 0.22
    minimum_upright_cosine = 0.78
    base_contact_grace_steps = 10
    maximum_distance_from_origin_m = 3.0
    target_base_height_m = 0.3730
    velocity_tracking_sigma_m_s = 0.10
    distance_success_fraction = 0.65

    # Deliberately reproduce quadruped_walk_v1.yaml's successful reward instead
    # of continuing to tune a novel objective.  Net displacement remains a hard
    # evaluation metric, but adding it to the reward caused unsafe lunge exploits.
    reward_forward_velocity_tracking = 2.0
    reward_upright = 0.50
    # In 128-env PPO, the old 0.05 survival term let a 0.3 m lunge outscore a
    # full stable episode.  At 0.50, standing beats an early fall while sustained
    # velocity tracking still pays roughly 2.5x more than standing.
    reward_alive = 0.50
    penalty_lateral_velocity = 0.50
    penalty_vertical_velocity = 0.20
    penalty_roll_pitch_rate = 0.05
    penalty_yaw_rate = 0.10
    penalty_body_height = 2.0
    penalty_action_rate = 0.02
    penalty_action_magnitude = 0.002
    penalty_joint_velocity = 0.005
    penalty_termination = 100.0
    # Keep value targets conditioned while preserving all reward ratios.
    reward_scale = 0.10
    qualified_foot_air_time_s = 0.10


@configclass
class DrobotCommandedWalkingDirectionalEnvCfg(DrobotCommandedWalkingForwardEnvCfg):
    """Episode commands include forward, backward, left, right, and stop."""

    command_profile = "directional"
