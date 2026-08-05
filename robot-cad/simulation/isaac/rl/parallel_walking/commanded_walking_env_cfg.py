"""Configuration for command-conditioned parallel Drobot walking."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
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
STABLE_NEUTRAL_HIP_RAD = 0.0872664626
STABLE_NEUTRAL_KNEE_RAD = 0.6981317008


@configclass
class DrobotCommandedWalkingForwardEnvCfg(DirectRLEnvCfg):
    """Flat-ground, forward-first walking with hardware-reproducible inputs."""

    decimation = 4
    episode_length_s = 8.0
    action_space = 12
    # command 3 + IMU 9 + joint position error 12 + velocity 12 + last action 12
    observation_space = 48
    state_space = 0

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
            pos=(0.0, 0.0, 0.3305),
            joint_pos={
                ".*_hip_abduction": 0.0,
                "front_.*_hip_flexion": STABLE_NEUTRAL_HIP_RAD,
                "rear_.*_hip_flexion": -STABLE_NEUTRAL_HIP_RAD,
                "front_.*_knee": STABLE_NEUTRAL_KNEE_RAD,
                "rear_.*_knee": -STABLE_NEUTRAL_KNEE_RAD,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=EFFORT_CAP_NM,
                effort_limit_sim=EFFORT_CAP_NM,
                velocity_limit=SERVO_VELOCITY_LIMIT_RAD_S,
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
    forward_speed_min_m_s = 0.08
    forward_speed_max_m_s = 0.22
    backward_speed_min_m_s = 0.06
    backward_speed_max_m_s = 0.16
    turn_forward_speed_max_m_s = 0.10
    turn_rate_min_rad_s = 0.35
    turn_rate_max_rad_s = 0.80

    action_scale_abduction_rad = 0.12
    action_scale_hip_rad = 0.30
    action_scale_knee_rad = 0.40
    reset_joint_position_noise_rad = 0.015
    reset_xy_jitter_m = 0.04
    minimum_base_height_m = 0.18
    minimum_upright_cosine = 0.35
    base_contact_grace_steps = 10
    maximum_distance_from_origin_m = 3.0
    target_base_height_m = 0.3305
    velocity_tracking_sigma_m_s = 0.10
    yaw_tracking_sigma_rad_s = 0.25


@configclass
class DrobotCommandedWalkingDirectionalEnvCfg(DrobotCommandedWalkingForwardEnvCfg):
    """Episode commands include forward, backward, left, right, and stop."""

    command_profile = "directional"
