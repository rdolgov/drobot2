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
    reset_forward_offset_m = 0.0
    reset_forward_jitter_m = 0.02
    first_step_curriculum = False
    first_step_min_base_gain_m = 0.06
    first_step_require_base_gain = True
    first_step_hold_steps = 8
    foot_lift_curriculum = False
    foot_lift_height_m = 0.19
    foot_lift_hold_steps = 8
    support_reward_scale = 0.06
    progress_delta_reward_scale = 60.0
    height_delta_reward_scale = 45.0
    supported_lift_reward_scale = 0.0
    tread_hold_reward_scale = 0.0
    tread_transfer_reward_scale = 0.0
    narrow_transfer_reward_scale = 0.0
    first_step_completion_reward_scale = 0.0
    success_completion_reward_scale = 25.0
    tread_height_delta_scale = 0.0
    new_tread_potential_reward_scale = 2.0
    tread_potential_reward_scale = 0.02
    new_narrow_tread_potential_reward_scale = 0.0
    narrow_tread_potential_reward_scale = 0.0
    tread_contact_reward_scale = 0.35
    reward_tread_count = 4
    base_contact_grace_steps = 30
    base_contact_failure_drop_m = 0.08
    base_contact_failure_upright_cosine = 0.75
    stair_rise_m = 0.18
    stair_tread_depth_m = 0.25
    stair_start_from_origin_m = 0.45
    stair_step_count = 4
    top_success_x_from_origin_m = 1.48
    minimum_top_base_height_m = 1.05
    minimum_base_height_m = 0.18
    minimum_upright_cosine = 0.55
    maximum_lateral_deviation_m = 0.55
    report_episode_totals_on_close = False

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


@configclass
class DrobotPureStairsFirstStepHipEnvCfg(DrobotPureStairsHipEnvCfg):
    """Pure-PPO curriculum for supported acquisition of the first tread."""

    episode_length_s = 8.0
    reset_forward_offset_m = 0.10
    reset_forward_jitter_m = 0.03
    first_step_curriculum = True
    reward_tread_count = 1
    support_reward_scale = 0.25
    supported_lift_reward_scale = 0.50
    report_episode_totals_on_close = True


@configclass
class DrobotPureStairsFirstStepLandingHipEnvCfg(DrobotPureStairsFirstStepHipEnvCfg):
    """Learn a supported, phase-free first-tread landing before body transfer."""

    reset_forward_offset_m = 0.10
    reset_forward_jitter_m = 0.03
    first_step_min_base_gain_m = 0.0
    first_step_require_base_gain = False
    first_step_hold_steps = 3
    tread_hold_reward_scale = 2.00
    tread_transfer_reward_scale = 0.0
    tread_height_delta_scale = 0.0
    new_tread_potential_reward_scale = 4.0
    tread_potential_reward_scale = 0.40
    new_narrow_tread_potential_reward_scale = 8.0
    narrow_tread_potential_reward_scale = 1.00
    tread_contact_reward_scale = 2.00
    first_step_completion_reward_scale = 10.0


@configclass
class DrobotPureStairsFirstStepLandingLowHipEnvCfg(
    DrobotPureStairsFirstStepLandingHipEnvCfg
):
    """First-tread landing from the lowest centered, nearly folded stance."""

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
class DrobotPureStairsFirstStepLandingSidewaysHipEnvCfg(
    DrobotPureStairsFirstStepLandingHipEnvCfg
):
    """First-tread landing from a true 90-degree lateral approach."""

    reset_yaw_deg = 90.0

    def __post_init__(self) -> None:
        super().__post_init__()
        self.robot.init_state.rot = (0.0, 0.0, 0.7071067812, 0.7071067812)


@configclass
class DrobotPureStairsFirstStepClose1HipEnvCfg(DrobotPureStairsFirstStepHipEnvCfg):
    """Bridge supported landing into 1 cm of body rise."""

    reset_forward_offset_m = 0.10
    reset_forward_jitter_m = 0.03
    first_step_min_base_gain_m = 0.01
    first_step_hold_steps = 4
    progress_delta_reward_scale = 10.0
    height_delta_reward_scale = 60.0
    tread_hold_reward_scale = 3.00
    tread_transfer_reward_scale = 3.00
    narrow_transfer_reward_scale = 4.00
    tread_height_delta_scale = 180.0
    new_tread_potential_reward_scale = 4.0
    tread_potential_reward_scale = 0.40
    new_narrow_tread_potential_reward_scale = 8.0
    narrow_tread_potential_reward_scale = 1.00
    tread_contact_reward_scale = 3.00
    first_step_completion_reward_scale = 15.0


@configclass
class DrobotPureStairsFirstStepClose2HipEnvCfg(DrobotPureStairsFirstStepHipEnvCfg):
    """Contact-preserving first-step stage requiring 2 cm of supported base gain."""

    reset_forward_offset_m = 0.10
    reset_forward_jitter_m = 0.03
    first_step_min_base_gain_m = 0.02
    first_step_hold_steps = 5
    tread_hold_reward_scale = 0.80
    tread_transfer_reward_scale = 2.50
    narrow_transfer_reward_scale = 3.00
    tread_height_delta_scale = 120.0
    new_tread_potential_reward_scale = 4.0
    tread_potential_reward_scale = 0.40
    new_narrow_tread_potential_reward_scale = 8.0
    narrow_tread_potential_reward_scale = 1.00
    tread_contact_reward_scale = 1.00
    first_step_completion_reward_scale = 20.0


@configclass
class DrobotPureStairsFirstStepClose4HipEnvCfg(DrobotPureStairsFirstStepHipEnvCfg):
    """Intermediate first-step stage requiring 4 cm of supported base gain."""

    reset_forward_offset_m = 0.10
    reset_forward_jitter_m = 0.03
    first_step_min_base_gain_m = 0.04
    first_step_hold_steps = 6
    tread_hold_reward_scale = 0.90
    tread_transfer_reward_scale = 3.00
    narrow_transfer_reward_scale = 2.50
    tread_height_delta_scale = 140.0
    new_tread_potential_reward_scale = 3.5
    tread_potential_reward_scale = 0.35
    tread_contact_reward_scale = 1.20
    first_step_completion_reward_scale = 25.0


@configclass
class DrobotPureStairsFirstStepClose6HipEnvCfg(DrobotPureStairsFirstStepHipEnvCfg):
    """Final stage requiring the full 6 cm first-step base gain."""

    reset_forward_offset_m = 0.10
    reset_forward_jitter_m = 0.03
    first_step_min_base_gain_m = 0.06
    first_step_hold_steps = 8
    tread_hold_reward_scale = 1.00
    tread_transfer_reward_scale = 3.50
    narrow_transfer_reward_scale = 2.00
    tread_height_delta_scale = 160.0
    new_tread_potential_reward_scale = 3.0
    tread_potential_reward_scale = 0.30
    tread_contact_reward_scale = 1.50
    first_step_completion_reward_scale = 30.0


@configclass
class DrobotPureStairsFootLiftHipEnvCfg(DrobotPureStairsHipEnvCfg):
    """Pure-PPO precursor: lift any foot 19 cm while retaining three-foot support."""

    episode_length_s = 6.0
    reset_forward_offset_m = -0.10
    reset_forward_jitter_m = 0.02
    foot_lift_curriculum = True
    reward_tread_count = 1
    support_reward_scale = 0.25
    supported_lift_reward_scale = 1.50


@configclass
class DrobotPureStairsFootLiftConsolidateHipEnvCfg(DrobotPureStairsFootLiftHipEnvCfg):
    """Move rare supported lifts into the policy mean without prescribing a leg."""

    success_completion_reward_scale = 100.0
    report_episode_totals_on_close = True


@configclass
class DrobotPureStairsFootLift10HipEnvCfg(DrobotPureStairsFootLiftHipEnvCfg):
    """First height stage for learning stable three-foot support."""

    foot_lift_height_m = 0.10
    foot_lift_hold_steps = 6


@configclass
class DrobotPureStairsFootLift14HipEnvCfg(DrobotPureStairsFootLiftHipEnvCfg):
    """Intermediate height stage before the required 19 cm hold."""

    foot_lift_height_m = 0.14
    foot_lift_hold_steps = 7
