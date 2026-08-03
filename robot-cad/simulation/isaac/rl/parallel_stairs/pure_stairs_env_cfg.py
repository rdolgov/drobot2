"""Vectorized pure-RL Drobot stair environment configuration."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.physics import PhysxCfg

from .exact_stairs_terrain import ExactStairsTerrainCfg, exact_stairs_terrain

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROBOT_USD = PROJECT_ROOT / "exports" / "isaac" / "quadruped_robot_floating.usdc"
EFFORT_CAP_NM = 0.8825985
LOW_FOLD_HIP_RAD = 0.7382742736
LOW_FOLD_KNEE_RAD = 1.4765485472


@configclass
class DrobotPureStairsEnvCfg(DirectRLEnvCfg):
    """128-way GPU task with deployable observations and no gait phase."""

    decimation = 4
    episode_length_s = 12.0
    action_space = 12
    observation_space = 70
    state_space = 0

    viewer: ViewerCfg = ViewerCfg(
        eye=(-0.90, -2.50, 0.82),
        lookat=(0.10, 0.0, 0.32),
    )
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics=PhysxCfg(gpu_max_rigid_patch_count=2**20),
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
        env_spacing=3.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=1053,
            curriculum=False,
            size=(2.75, 1.50),
            border_width=0.0,
            num_rows=8,
            num_cols=16,
            color_scheme="height",
            use_cache=True,
            sub_terrains={
                "exact_stairs": ExactStairsTerrainCfg(function=exact_stairs_terrain)
            },
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.10,
            dynamic_friction=0.90,
            restitution=0.0,
        ),
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
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.46),
            joint_pos={
                ".*_hip_abduction": 0.0,
                "front_.*_hip_flexion": -0.1544915916,
                "rear_.*_hip_flexion": 0.1544915916,
                "front_.*_knee": 0.4699252058,
                "rear_.*_knee": -0.4699252058,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "legs": IdealPDActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=EFFORT_CAP_NM,
                effort_limit_sim=EFFORT_CAP_NM,
                velocity_limit=4.5836625,
                stiffness=30.0,
                damping=4.5836625,
                friction=0.0,
                armature=0.0,
            )
        },
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/.*",
        history_length=3,
        update_period=1.0 / 120.0,
        track_air_time=True,
        force_threshold=1.0,
    )

    # 8 x 8, 45-degree VL53L5CX-like depth at the real 15 Hz limit.
    depth_sensor: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Geometry/base_link",
        update_period=1.0 / 15.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.1145, 0.0, 0.123),
            rot=(0.0, -0.3420201433, 0.0, 0.9396926208),
        ),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=8,
            vertical_fov_range=(-22.5, 22.5),
            horizontal_fov_range=(-22.5, 22.5),
            horizontal_res=6.5,
        ),
        mesh_prim_paths=["/World/ground"],
        max_distance=4.0,
        debug_vis=False,
    )

    action_scale_abduction_rad = 0.18
    action_scale_hip_rad = 0.60
    action_scale_knee_rad = 0.90
    initial_base_height_m = 0.46
    reset_yaw_deg = 0.0
    stair_rise_m = 0.18
    stair_tread_depth_m = 0.25
    stair_start_from_origin_m = 0.45
    stair_step_count = 4
    top_success_x_from_origin_m = 1.48
    minimum_top_base_height_m = 1.05
    minimum_base_height_m = 0.18
    minimum_upright_cosine = 0.55
    maximum_lateral_deviation_m = 0.55

    def play_mode(self) -> None:
        """Use one terrain tile so the playback camera and robot share world origin."""
        super().play_mode()
        self.scene.num_envs = 1
        if self.terrain.terrain_generator is not None:
            self.terrain.terrain_generator.num_rows = 1
            self.terrain.terrain_generator.num_cols = 1


@configclass
class DrobotPureStairsHipEnvCfg(DrobotPureStairsEnvCfg):
    """Forward reset with more hip authority, still bounded by real limits."""

    action_scale_abduction_rad = 0.30
    action_scale_hip_rad = 0.90
    action_scale_knee_rad = 1.20


@configclass
class DrobotPureStairsLowHipEnvCfg(DrobotPureStairsHipEnvCfg):
    """Start near the lowest centered stance allowed by soft joint limits."""

    initial_base_height_m = 0.30

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot.init_state.pos = (0.0, 0.0, self.initial_base_height_m)
        self.robot.init_state.joint_pos = {
            ".*_hip_abduction": 0.0,
            "front_.*_hip_flexion": -LOW_FOLD_HIP_RAD,
            "rear_.*_hip_flexion": LOW_FOLD_HIP_RAD,
            "front_.*_knee": LOW_FOLD_KNEE_RAD,
            "rear_.*_knee": -LOW_FOLD_KNEE_RAD,
        }


@configclass
class DrobotPureStairsSidewaysHipEnvCfg(DrobotPureStairsHipEnvCfg):
    """Start at 90 degrees to test a genuinely lateral stair approach."""

    reset_yaw_deg = 90.0

    def __post_init__(self) -> None:
        super().__post_init__()
        # Isaac Lab asset configs use quaternion order (x, y, z, w).
        self.robot.init_state.rot = (0.0, 0.0, 0.7071067812, 0.7071067812)
