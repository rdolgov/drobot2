"""Isolated 12/16-joint PPO pilot, using the existing Isaac Lab installation."""
from pathlib import Path
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

ROOT = Path(__file__).resolve().parents[4]
LEGS = ("front_left", "rear_left", "front_right", "rear_right")


@configclass
class ActiveEnvCfg(DirectRLEnvCfg):
    variant = "active"
    decimation = 2
    episode_length_s = 12.0
    action_space = 16
    observation_space = 62
    state_space = 70
    command_min = 0.06
    command_max = 0.14
    target_rates_deg_s = (60.0, 100.0, 100.0, 100.0)
    action_scales_rad = (0.18, 0.60, 0.80, 0.60)
    effort_cap_nm = 0.8825985
    no_load_speed_rad_s = 4.712389
    stall_effort_nm = 2.6477955
    pilot_strength_min = 0.85
    metrics_path = ""
    sim = SimulationCfg(
        dt=1.0 / 120.0, render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**19),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
    )
    scene = InteractiveSceneCfg(num_envs=128, env_spacing=6.0,
                                replicate_physics=True, clone_in_fabric=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=0.9, restitution=0.0),
        debug_vis=False,
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ROOT / "simulation/exports/isaac/ankle_v2/active/robot.usdc"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, retain_accelerations=True,
                max_depenetration_velocity=1.0, max_linear_velocity=5.0,
                max_angular_velocity=12.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8,
                solver_velocity_iteration_count=4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.375),
            joint_pos={".*_hip_abduction": 0.0,
                       "front_.*_hip_flexion": 0.5239596454,
                       "rear_.*_hip_flexion": -0.5239596454,
                       "front_.*_knee": -0.5239596454,
                       "rear_.*_knee": 0.5239596454,
                       ".*_ankle_pitch": 0.0},
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={"servos": ImplicitActuatorCfg(
            joint_names_expr=[".*"], stiffness=30.0, damping=4.5836625,
            effort_limit=0.8825985, effort_limit_sim=0.8825985,
            velocity_limit=4.712389, velocity_limit_sim=4.712389,
            friction=0.0, armature=0.0)},
    )
    contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/.*",
        history_length=2, update_period=1.0 / 120.0,
        track_air_time=True, force_threshold=1.0)
    imu = ImuCfg(prim_path="/World/envs/env_.*/Robot/Geometry/base_link",
                 update_period=1.0 / 120.0)


@configclass
class BaselineEnvCfg(ActiveEnvCfg):
    variant = "baseline"
    action_space = 12
    observation_space = 50
    state_space = 58

    def __post_init__(self):
        super().__post_init__()
        self.robot.spawn.usd_path = str(ROOT / f"simulation/exports/isaac/ankle_v2/{self.variant}/robot.usdc")
        self.robot.init_state.joint_pos.pop(".*_ankle_pitch", None)


@configclass
class LockedEnvCfg(BaselineEnvCfg):
    variant = "locked"
