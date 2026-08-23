"""Pure-RL vectorized commanded-walking environment for Drobot."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations

import gymnasium as gym
import numpy as np
import isaaclab.sim as sim_utils
import torch
import warp as wp
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

from . import preview_control
from .commanded_walking_env_cfg import (
    DISTAL_LINK_LENGTH_M,
    ORIGINAL_BASE_COM_M,
    ORIGINAL_BASE_INERTIA_KG_M2,
    ORIGINAL_BASE_MASS_KG,
    RECTANGULAR_SHOE_LENGTH_FORE_AFT_M,
    RECTANGULAR_SHOE_MASS_KG,
    RECTANGULAR_SHOE_SOLE_BACK_FROM_FORK_M,
    RECTANGULAR_SHOE_SOLE_THICKNESS_M,
    RECTANGULAR_SHOE_TREAD_LENGTH_M,
    RECTANGULAR_SHOE_TREAD_PROJECTION_M,
    RECTANGULAR_SHOE_TREAD_WIDTH_M,
    RECTANGULAR_SHOE_WIDTH_LATERAL_M,
    REPLACED_BATTERY_CENTER_M,
    REPLACED_BATTERY_MASS_KG,
    REPLACED_BATTERY_SIZE_M,
    SERVO_VELOCITY_LIMIT_RAD_S,
    DrobotCommandedWalkingForwardEnvCfg,
)


LEG_NAMES = ("front_left", "rear_left", "front_right", "rear_right")


def _author_rectangular_shoes(stage) -> None:
    """Replace env-0 fork-tip spheres with the latest flat CAD shoe proxies."""
    distal_prims = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.GetName() in {f"{leg}_distal_link" for leg in LEG_NAMES}
        and str(prim.GetPath()).startswith("/World/envs/env_0/Robot")
    }
    if len(distal_prims) != 4:
        raise RuntimeError(
            "Expected four env-0 distal links before cloning rectangular shoes; "
            f"found {tuple(distal_prims)}"
        )

    tread_material = UsdShade.Material.Define(
        stage, "/World/Materials/RectangularShoeTreadContact"
    )
    material_api = UsdPhysics.MaterialAPI.Apply(tread_material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(1.05)
    material_api.CreateDynamicFrictionAttr().Set(0.85)
    material_api.CreateRestitutionAttr().Set(0.02)
    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(tread_material.GetPrim())
    physx_api.CreateCompliantContactStiffnessAttr().Set(12000.0)
    physx_api.CreateCompliantContactDampingAttr().Set(45.0)

    sole_center_x = (
        DISTAL_LINK_LENGTH_M
        + RECTANGULAR_SHOE_SOLE_BACK_FROM_FORK_M
        + RECTANGULAR_SHOE_SOLE_THICKNESS_M / 2.0
    )
    tread_center_x = (
        DISTAL_LINK_LENGTH_M
        + RECTANGULAR_SHOE_SOLE_BACK_FROM_FORK_M
        + RECTANGULAR_SHOE_SOLE_THICKNESS_M
        + RECTANGULAR_SHOE_TREAD_PROJECTION_M / 2.0
    )

    # The imported distal arm is 215.137 g.  Add the CAD PLA estimate to the
    # same rigid body so the shoe does not create extra articulation bodies.
    # The diagonal inertia is a conservative box approximation around the
    # combined center of mass; the detailed printable ribs remain visual-only.
    original_mass = 0.215137
    original_com = Gf.Vec3f(0.07301168, -0.000021551, -0.000924466)
    shoe_com = Gf.Vec3f(DISTAL_LINK_LENGTH_M + 0.020, 0.0, 0.0)
    combined_mass = original_mass + RECTANGULAR_SHOE_MASS_KG
    combined_com = Gf.Vec3f(
        (original_mass * original_com[0] + RECTANGULAR_SHOE_MASS_KG * shoe_com[0])
        / combined_mass,
        (original_mass * original_com[1] + RECTANGULAR_SHOE_MASS_KG * shoe_com[1])
        / combined_mass,
        (original_mass * original_com[2] + RECTANGULAR_SHOE_MASS_KG * shoe_com[2])
        / combined_mass,
    )
    combined_diagonal_inertia = Gf.Vec3f(0.000155, 0.001192, 0.001178)

    for distal in distal_prims.values():
        for child in distal.GetChildren():
            if child.GetName().startswith("simulation_only_fork_tip_contact_proxy"):
                collision = UsdPhysics.CollisionAPI(child)
                if collision:
                    collision.CreateCollisionEnabledAttr().Set(False)

        mass_api = UsdPhysics.MassAPI.Apply(distal)
        mass_api.CreateMassAttr().Set(combined_mass)
        mass_api.CreateCenterOfMassAttr().Set(combined_com)
        mass_api.CreateDiagonalInertiaAttr().Set(combined_diagonal_inertia)
        mass_api.CreatePrincipalAxesAttr().Set(
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        )

        for label, center_x, size_xyz, color in (
            (
                "sole",
                sole_center_x,
                (
                    RECTANGULAR_SHOE_SOLE_THICKNESS_M,
                    RECTANGULAR_SHOE_LENGTH_FORE_AFT_M,
                    RECTANGULAR_SHOE_WIDTH_LATERAL_M,
                ),
                (0.18, 0.22, 0.28),
            ),
            (
                "tread",
                tread_center_x,
                (
                    RECTANGULAR_SHOE_TREAD_PROJECTION_M,
                    RECTANGULAR_SHOE_TREAD_LENGTH_M,
                    RECTANGULAR_SHOE_TREAD_WIDTH_M,
                ),
                (0.12, 0.65, 0.24),
            ),
        ):
            cube = UsdGeom.Cube.Define(
                stage,
                distal.GetPath().AppendChild(
                    f"simulation_only_rectangular_shoe_{label}_proxy"
                ),
            )
            cube.CreateSizeAttr().Set(1.0)
            cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(cube.GetPrim())
            xform.AddTranslateOp().Set(Gf.Vec3d(center_x, 0.0, 0.0))
            xform.AddScaleOp().Set(Gf.Vec3d(*size_xyz))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                tread_material,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )


def _box_inertia(mass_kg: float, size_m: np.ndarray) -> np.ndarray:
    x, y, z = size_m
    return np.diag(
        (
            mass_kg * (y * y + z * z) / 12.0,
            mass_kg * (x * x + z * z) / 12.0,
            mass_kg * (x * x + y * y) / 12.0,
        )
    )


def _parallel_axis(offset_m: np.ndarray) -> np.ndarray:
    return np.eye(3) * np.dot(offset_m, offset_m) - np.outer(
        offset_m, offset_m
    )


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, ...]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        return (
            0.25 * scale,
            (rotation[2, 1] - rotation[1, 2]) / scale,
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[1, 0] - rotation[0, 1]) / scale,
        )
    index = int(np.argmax(np.diag(rotation)))
    if index == 0:
        scale = np.sqrt(
            1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
        ) * 2.0
        return (
            (rotation[2, 1] - rotation[1, 2]) / scale,
            0.25 * scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            (rotation[0, 2] + rotation[2, 0]) / scale,
        )
    if index == 1:
        scale = np.sqrt(
            1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
        ) * 2.0
        return (
            (rotation[0, 2] - rotation[2, 0]) / scale,
            (rotation[0, 1] + rotation[1, 0]) / scale,
            0.25 * scale,
            (rotation[1, 2] + rotation[2, 1]) / scale,
        )
    scale = np.sqrt(
        1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
    ) * 2.0
    return (
        (rotation[1, 0] - rotation[0, 1]) / scale,
        (rotation[0, 2] + rotation[2, 0]) / scale,
        (rotation[1, 2] + rotation[2, 1]) / scale,
        0.25 * scale,
    )


def _author_rear_battery_payload(stage, cfg) -> None:
    """Replace the provisional centered battery with the measured rear pack."""
    if not cfg.rear_payload_enabled:
        return
    base_prims = [
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "base_link"
        and str(prim.GetPath()).startswith("/World/envs/env_0/Robot")
    ]
    if len(base_prims) != 1:
        raise RuntimeError(
            "Expected one env-0 base_link before authoring the rear payload; "
            f"found {len(base_prims)}"
        )

    original_com = np.asarray(ORIGINAL_BASE_COM_M, dtype=np.float64)
    ixx, ixy, ixz, iyy, iyz, izz = ORIGINAL_BASE_INERTIA_KG_M2
    original_inertia = np.asarray(
        ((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)),
        dtype=np.float64,
    )
    old_mass = REPLACED_BATTERY_MASS_KG
    old_center = np.asarray(REPLACED_BATTERY_CENTER_M, dtype=np.float64)
    old_size = np.asarray(REPLACED_BATTERY_SIZE_M, dtype=np.float64)
    dry_mass = ORIGINAL_BASE_MASS_KG - old_mass
    dry_com = (
        ORIGINAL_BASE_MASS_KG * original_com - old_mass * old_center
    ) / dry_mass
    dry_inertia_about_original = original_inertia - (
        _box_inertia(old_mass, old_size)
        + old_mass * _parallel_axis(old_center - original_com)
    )
    dry_inertia = dry_inertia_about_original - dry_mass * _parallel_axis(
        dry_com - original_com
    )

    payload_mass = float(cfg.rear_payload_mass_kg)
    payload_center = np.asarray(cfg.rear_payload_center_m, dtype=np.float64)
    payload_size = np.asarray(cfg.rear_payload_size_m, dtype=np.float64)
    combined_mass = dry_mass + payload_mass
    combined_com = (
        dry_mass * dry_com + payload_mass * payload_center
    ) / combined_mass
    combined_inertia = (
        dry_inertia
        + dry_mass * _parallel_axis(dry_com - combined_com)
        + _box_inertia(payload_mass, payload_size)
        + payload_mass * _parallel_axis(payload_center - combined_com)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(combined_inertia)
    axis_order = max(
        permutations(range(3)),
        key=lambda order: sum(
            abs(float(eigenvectors[axis, order[axis]])) for axis in range(3)
        ),
    )
    diagonal_inertia = eigenvalues[list(axis_order)]
    principal_axes = eigenvectors[:, list(axis_order)]
    if np.linalg.det(principal_axes) < 0.0:
        principal_axes[:, 0] *= -1.0
    quaternion = _matrix_to_quaternion_wxyz(principal_axes)

    base = base_prims[0]
    mass_api = UsdPhysics.MassAPI.Apply(base)
    mass_api.CreateMassAttr().Set(float(combined_mass))
    mass_api.CreateCenterOfMassAttr().Set(
        Gf.Vec3f(*(float(value) for value in combined_com))
    )
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(*(float(value) for value in diagonal_inertia))
    )
    mass_api.CreatePrincipalAxesAttr().Set(
        Gf.Quatf(
            float(quaternion[0]),
            Gf.Vec3f(*(float(value) for value in quaternion[1:])),
        )
    )

    proxy = UsdGeom.Cube.Define(
        stage,
        base.GetPath().AppendChild("simulation_only_rear_battery_enclosure_proxy"),
    )
    proxy.CreateSizeAttr().Set(1.0)
    proxy.CreateDisplayColorAttr().Set([Gf.Vec3f(0.32, 0.10, 0.10)])
    xform = UsdGeom.Xformable(proxy.GetPrim())
    xform.AddTranslateOp().Set(
        Gf.Vec3d(*(float(value) for value in payload_center))
    )
    xform.AddScaleOp().Set(Gf.Vec3d(*(float(value) for value in payload_size)))
    UsdPhysics.CollisionAPI.Apply(proxy.GetPrim())


class DrobotCommandedWalkingEnv(DirectRLEnv):
    """Learn smooth command tracking around a coordinated four-leg gait clock."""

    cfg: DrobotCommandedWalkingForwardEnvCfg
    def __init__(
        self,
        cfg: DrobotCommandedWalkingForwardEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        self._older_actions = torch.zeros_like(self._actions)
        self._previous_targets = self._robot.data.default_joint_pos.torch.clone()
        self._previous_joint_velocity = self._robot.data.joint_vel.torch.clone()
        self._previous_body_linear_velocity = (
            self._robot.data.root_lin_vel_w.torch.clone()
        )
        self._previous_body_angular_velocity = (
            self._robot.data.root_ang_vel_w.torch.clone()
        )
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._joint_scale = torch.tensor(
            [
                self.cfg.action_scale_abduction_rad
                if name.endswith("hip_abduction")
                else self.cfg.action_scale_hip_rad
                if name.endswith("hip_flexion")
                else self.cfg.action_scale_knee_rad
                for name in self._robot.joint_names
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if len(self._robot.joint_names) != 12:
            raise RuntimeError(f"Expected 12 joints, got {self._robot.joint_names}")
        self._leg_phase_offsets = torch.tensor(
            (0.0, 0.5, 0.5, 0.0), dtype=torch.float32, device=self.device
        )
        self._leg_front_sign = torch.tensor(
            (1.0, -1.0, 1.0, -1.0), dtype=torch.float32, device=self.device
        )
        self._leg_abduction_joint_ids = torch.tensor(
            [self._robot.joint_names.index(f"{leg}_hip_abduction") for leg in LEG_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self._leg_hip_joint_ids = torch.tensor(
            [self._robot.joint_names.index(f"{leg}_hip_flexion") for leg in LEG_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self._leg_knee_joint_ids = torch.tensor(
            [self._robot.joint_names.index(f"{leg}_knee") for leg in LEG_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self._base_sensor_ids, _ = self._contact_sensor.find_sensors("base_link")
        self._foot_sensor_ids, self._foot_sensor_names = self._contact_sensor.find_sensors(
            ".*_distal_link"
        )
        if len(self._base_sensor_ids) == 0:
            raise RuntimeError("Could not find base_link in the contact sensor")
        if len(self._foot_sensor_ids) != 4:
            raise RuntimeError(
                f"Expected four foot sensors, got {self._foot_sensor_names}"
            )
        self._foot_phase_offsets = torch.tensor(
            [
                self._leg_phase_offsets[
                    next(i for i, leg in enumerate(LEG_NAMES) if leg in name)
                ].item()
                for name in self._foot_sensor_names
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._foot_body_ids, self._foot_body_names = self._robot.find_bodies(
            ".*_distal_link"
        )
        if len(self._foot_body_ids) != 4:
            raise RuntimeError(
                f"Expected four distal rigid bodies, got {self._foot_body_names}"
            )
        self._randomize_rear_payload_estimate()

        self._steps_since_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure_nonfinite = torch.zeros_like(self._failed)
        self._failure_low_height = torch.zeros_like(self._failed)
        self._failure_tilt = torch.zeros_like(self._failed)
        self._failure_out_of_bounds = torch.zeros_like(self._failed)
        self._failure_base_contact = torch.zeros_like(self._failed)
        self._episode_start_position = torch.zeros((self.num_envs, 3), device=self.device)
        self._episode_velocity_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_yaw_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_commanded_distance = torch.zeros(self.num_envs, device=self.device)
        self._episode_action_saturation_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_swing_step_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_touchdown_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_qualified_touchdown_count = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_qualified_touchdown_by_foot = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._episode_command_speed_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_base_height_sum = torch.zeros(self.num_envs, device=self.device)
        self._sustained_window_steps = max(
            1, int(round(self.cfg.sustained_speed_window_s / self.step_dt))
        )
        self._rolling_command_speed = torch.zeros(
            (self.num_envs, self._sustained_window_steps), device=self.device
        )
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        self._episode_min_rolling_speed = torch.full(
            (self.num_envs,), torch.inf, device=self.device
        )
        self._episode_stall_step_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_joint_acceleration_squared_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_body_linear_acceleration_squared_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_body_angular_acceleration_squared_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        reward_term_names = (
            "forward_velocity_tracking",
            "instant_progress",
            "sustained_progress",
            "sustained_stall",
            "gait_reference",
            "scheduled_stance",
            "scheduled_swing",
            "upright",
            "alive",
            "lateral_velocity",
            "lateral_displacement",
            "vertical_velocity",
            "roll_pitch_rate",
            "body_tilt",
            "yaw_rate",
            "body_height",
            "action_rate",
            "action_acceleration",
            "action_saturation",
            "diagonal_sync",
            "action_magnitude",
            "joint_velocity",
            "joint_acceleration",
            "body_linear_acceleration",
            "body_angular_acceleration",
            "support_foot_slip",
            "touchdown_impact",
            "qualified_touchdown",
            "termination",
        )
        self._episode_reward_term_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in reward_term_names
        }

    def _randomize_rear_payload_estimate(self) -> None:
        """Spread the measured payload uncertainty across vectorized robots."""
        if not self.cfg.rear_payload_enabled:
            return
        base_ids, base_names = self._robot.find_bodies("base_link")
        if len(base_ids) != 1:
            raise RuntimeError(f"Expected one base_link body, got {base_names}")
        base_ids = torch.as_tensor(base_ids, dtype=torch.int32, device=self.device)
        env_ids = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        low, high = self.cfg.rear_payload_combined_mass_scale_range
        mass_scale = low + (high - low) * torch.rand(
            (self.num_envs, 1), device=self.device
        )
        masses = self._robot.data.body_mass.torch[:, base_ids.long()] * mass_scale
        inertias = (
            self._robot.data.body_inertia.torch[:, base_ids.long()] * mass_scale.unsqueeze(-1)
        )
        self._robot.set_masses_index(
            masses=masses, body_ids=base_ids, env_ids=env_ids
        )
        self._robot.set_inertias_index(
            inertias=inertias, body_ids=base_ids, env_ids=env_ids
        )

        jitter_limit = torch.tensor(
            self.cfg.rear_payload_combined_com_jitter_m,
            dtype=torch.float32,
            device=self.device,
        )
        coms = self._robot.data.body_com_pose_b.torch[:, base_ids.long()].clone()
        coms[:, :, :3] += (
            2.0 * torch.rand((self.num_envs, 1, 3), device=self.device) - 1.0
        ) * jitter_limit
        self._robot.set_coms_index(coms=coms, body_ids=base_ids, env_ids=env_ids)

    def _current_episode_horizon_steps(self) -> int:
        curriculum_step = (
            self.common_step_counter + self.cfg.command_curriculum_offset_steps
        )
        fraction = min(
            float(curriculum_step) / self.cfg.episode_horizon_curriculum_steps,
            1.0,
        )
        horizon_s = self.cfg.initial_training_horizon_s + fraction * (
            self.cfg.final_training_horizon_s
            - self.cfg.initial_training_horizon_s
        )
        return max(1, int(round(horizon_s / self.step_dt)))

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        _author_rectangular_shoes(self.scene.stage)
        _author_rear_battery_payload(self.scene.stage, self.cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._imu_sensor = Imu(self.cfg.imu_sensor)
        self.scene.sensors["imu_sensor"] = self._imu_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        src, dest = "/World/envs/env_0", "/World/envs/env_{}"
        positions = cloner.grid_transforms(
            self.scene.num_envs, self.scene.cfg.env_spacing, device=self.device
        )[0]
        plan = cloner.clone_plan_from_env_0(src, dest, self.scene.num_envs, self.device, positions)
        cloner.replicate(plan, stage=self.scene.stage)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.82, 0.82, 0.82))
        light.func("/World/Light", light)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._older_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._actions = torch.clamp(actions, -1.0, 1.0)
        desired_targets = (
            self._robot.data.default_joint_pos.torch + self._joint_scale * self._actions
        )
        limits = self._robot.data.soft_joint_pos_limits.torch
        desired_targets = torch.clamp(
            desired_targets, limits[:, :, 0], limits[:, :, 1]
        )
        max_delta = SERVO_VELOCITY_LIMIT_RAD_S * self.step_dt
        self._processed_actions = torch.clamp(
            desired_targets,
            self._previous_targets - max_delta,
            self._previous_targets + max_delta,
        )
        self._previous_targets.copy_(self._processed_actions)

    def _apply_action(self):
        self._robot.set_joint_position_target_index(target=self._processed_actions)

    def _base_contact_force(self) -> torch.Tensor:
        force = self._contact_sensor.data.net_forces_w.torch[:, self._base_sensor_ids]
        return torch.linalg.norm(force, dim=-1).amax(dim=1)

    def _foot_forces(self) -> torch.Tensor:
        force = self._contact_sensor.data.net_forces_w.torch[:, self._foot_sensor_ids]
        return torch.linalg.norm(force, dim=-1)

    def _gait_phase(self) -> torch.Tensor:
        return torch.remainder(
            self._steps_since_reset.float() * self.step_dt / self.cfg.gait_period_s,
            1.0,
        )

    def _gait_targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deployable smooth joint references and scheduled contacts."""
        phase = self._gait_phase()
        leg_phase = torch.remainder(
            phase.unsqueeze(1) + self._leg_phase_offsets.unsqueeze(0), 1.0
        )
        duty = self.cfg.gait_duty_factor
        stance_u = torch.clamp(leg_phase / duty, 0.0, 1.0)
        swing_u = torch.clamp((leg_phase - duty) / (1.0 - duty), 0.0, 1.0)
        stance_curve = stance_u * stance_u * (3.0 - 2.0 * stance_u)
        swing_curve = swing_u * swing_u * (3.0 - 2.0 * swing_u)
        half_stride = 0.5 * self.cfg.gait_stride_m
        stride_offset = torch.where(
            leg_phase < duty,
            half_stride - self.cfg.gait_stride_m * stance_curve,
            -half_stride + self.cfg.gait_stride_m * swing_curve,
        )
        lift = torch.where(
            leg_phase < duty,
            torch.zeros_like(leg_phase),
            self.cfg.gait_lift_m * torch.sin(torch.pi * swing_u),
        )
        ramp = torch.clamp(
            self._steps_since_reset.float() * self.step_dt / self.cfg.gait_start_ramp_s,
            0.0,
            1.0,
        ).unsqueeze(1)
        stride_offset = stride_offset * ramp
        lift = lift * ramp

        nominal_forward = 0.080 * self._leg_front_sign.unsqueeze(0)
        world_forward = nominal_forward + stride_offset
        default_down = (
            (DISTAL_LINK_LENGTH_M**2 - 0.080**2) ** 0.5
            + DISTAL_LINK_LENGTH_M
            + 0.031
        )
        down = default_down - lift
        distal_contact_length = DISTAL_LINK_LENGTH_M + 0.031
        cosine_knee = torch.clamp(
            (
                torch.square(down)
                + torch.square(world_forward)
                - DISTAL_LINK_LENGTH_M**2
                - distal_contact_length**2
            )
            / (2.0 * DISTAL_LINK_LENGTH_M * distal_contact_length),
            -1.0,
            1.0,
        )
        knee = -self._leg_front_sign.unsqueeze(0) * torch.acos(cosine_knee)
        hip = torch.atan2(world_forward, down) - torch.atan2(
            distal_contact_length * torch.sin(knee),
            DISTAL_LINK_LENGTH_M + distal_contact_length * torch.cos(knee),
        )

        reference = torch.zeros_like(self._actions)
        default = self._robot.data.default_joint_pos.torch
        reference[:, self._leg_abduction_joint_ids] = 0.0
        reference[:, self._leg_hip_joint_ids] = (
            hip - default[:, self._leg_hip_joint_ids]
        ) / self.cfg.action_scale_hip_rad
        reference[:, self._leg_knee_joint_ids] = (
            knee - default[:, self._leg_knee_joint_ids]
        ) / self.cfg.action_scale_knee_rad
        reference = torch.clamp(reference, -1.0, 1.0)

        foot_phase = torch.remainder(
            phase.unsqueeze(1) + self._foot_phase_offsets.unsqueeze(0), 1.0
        )
        scheduled_contact = foot_phase < duty
        scheduled_contact = torch.where(
            ramp < 0.999,
            torch.ones_like(scheduled_contact),
            scheduled_contact,
        )
        return reference, scheduled_contact

    def _get_observations(self) -> dict[str, torch.Tensor]:
        gait_angle = 2.0 * torch.pi * self._gait_phase()
        gait_clock = torch.stack((torch.sin(gait_angle), torch.cos(gait_angle)), dim=1)
        policy_observation = torch.cat(
            (
                self._commands,
                gait_clock,
                self._imu_sensor.data.ang_vel_b.torch,
                self._robot.data.projected_gravity_b.torch,
                self._imu_sensor.data.lin_acc_b.torch / 9.81,
                self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch,
                self._robot.data.joint_vel.torch / SERVO_VELOCITY_LIMIT_RAD_S,
                self._previous_actions,
            ),
            dim=-1,
        )
        foot_contact = (self._foot_forces() > 1.0).float()
        base_height = (
            self._robot.data.root_pos_w.torch[:, 2:3]
            - self._terrain.env_origins[:, 2:3]
        )
        critic_observation = torch.cat(
            (
                policy_observation,
                self._robot.data.root_lin_vel_b.torch,
                base_height,
                foot_contact,
            ),
            dim=-1,
        )
        return {
            "policy": torch.clamp(policy_observation, -20.0, 20.0),
            "critic": torch.clamp(critic_observation, -20.0, 20.0),
        }

    def _get_rewards(self) -> torch.Tensor:
        linear_velocity = self._robot.data.root_lin_vel_b.torch
        angular_velocity = self._robot.data.root_ang_vel_b.torch
        forward_error = linear_velocity[:, 0] - self._commands[:, 0]
        velocity_error_squared = torch.square(forward_error)
        yaw_error = angular_velocity[:, 2] - self._commands[:, 2]
        velocity_tracking = torch.exp(
            -velocity_error_squared / self.cfg.velocity_tracking_sigma_m_s**2
        )
        command_speed = torch.linalg.norm(self._commands[:, :2], dim=1)
        command_direction = self._commands[:, :2] / torch.clamp(
            command_speed.unsqueeze(1), min=1.0e-4
        )
        commanded_velocity = torch.sum(
            linear_velocity[:, :2] * command_direction, dim=1
        )
        rolling_slots = torch.remainder(
            self._steps_since_reset, self._sustained_window_steps
        )
        self._rolling_command_speed[self._env_indices, rolling_slots] = (
            commanded_velocity
        )
        rolling_counts = torch.clamp(
            self._steps_since_reset + 1,
            min=1,
            max=self._sustained_window_steps,
        ).float()
        rolling_speed = torch.sum(self._rolling_command_speed, dim=1) / rolling_counts
        rolling_window_ready = (
            self._steps_since_reset + 1 >= self._sustained_window_steps
        )
        displacement = self._robot.data.root_pos_w.torch - self._episode_start_position
        net_commanded_distance = torch.sum(
            displacement[:, :2] * command_direction, dim=1
        )
        lateral_direction = torch.stack(
            (-command_direction[:, 1], command_direction[:, 0]), dim=1
        )
        net_lateral_displacement = torch.sum(
            displacement[:, :2] * lateral_direction, dim=1
        )
        active_translation = command_speed > 0.03
        sustained_progress = torch.where(
            active_translation,
            torch.clamp(
                rolling_speed / torch.clamp(command_speed, min=1.0e-4),
                min=-1.0,
                max=1.0,
            ),
            torch.zeros_like(rolling_speed),
        )
        normalized_stall_deficit = torch.clamp(
            (self.cfg.minimum_sustained_speed_m_s - rolling_speed)
            / self.cfg.minimum_sustained_speed_m_s,
            min=0.0,
            max=2.0,
        )
        sustained_stall = rolling_window_ready & active_translation & (
            rolling_speed < self.cfg.minimum_sustained_speed_m_s
        )
        net_commanded_distance = torch.where(
            active_translation,
            net_commanded_distance,
            torch.zeros_like(net_commanded_distance),
        )
        upright_cosine = torch.clamp(
            -self._robot.data.projected_gravity_b.torch[:, 2], 0.0, 1.0
        )
        base_height = (
            self._robot.data.root_pos_w.torch[:, 2]
            - self._terrain.env_origins[:, 2]
        )
        height_error = base_height - self.cfg.target_base_height_m
        action_rate = torch.mean(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        action_acceleration = torch.mean(
            torch.square(
                self._actions
                - 2.0 * self._previous_actions
                + self._older_actions
            ),
            dim=1,
        )
        joint_acceleration = (
            self._robot.data.joint_vel.torch - self._previous_joint_velocity
        ) / self.step_dt
        body_linear_acceleration = (
            self._robot.data.root_lin_vel_w.torch
            - self._previous_body_linear_velocity
        ) / self.step_dt
        body_angular_acceleration = (
            self._robot.data.root_ang_vel_w.torch
            - self._previous_body_angular_velocity
        ) / self.step_dt
        joint_acceleration_squared = torch.mean(
            torch.square(joint_acceleration), dim=1
        )
        body_linear_acceleration_squared = torch.mean(
            torch.square(body_linear_acceleration), dim=1
        )
        body_angular_acceleration_squared = torch.mean(
            torch.square(body_angular_acceleration), dim=1
        )
        normalized_joint_acceleration = joint_acceleration_squared / (
            self.cfg.joint_acceleration_normalizer_rad_s2**2
        )
        normalized_body_linear_acceleration = (
            body_linear_acceleration_squared
            / self.cfg.body_linear_acceleration_normalizer_m_s2**2
        )
        normalized_body_angular_acceleration = (
            body_angular_acceleration_squared
            / self.cfg.body_angular_acceleration_normalizer_rad_s2**2
        )
        if self.cfg.smoothness_curriculum_steps > 0:
            smoothness_fraction = min(
                float(
                    self.common_step_counter
                    + self.cfg.command_curriculum_offset_steps
                )
                / float(self.cfg.smoothness_curriculum_steps),
                1.0,
            )
            smoothness_scale = self.cfg.smoothness_initial_scale + (
                1.0 - self.cfg.smoothness_initial_scale
            ) * smoothness_fraction
        else:
            smoothness_scale = 1.0
        action_magnitude = torch.mean(torch.square(self._actions), dim=1)
        action_saturation = torch.mean(
            torch.square(torch.clamp(torch.abs(self._actions) - 0.80, min=0.0)),
            dim=1,
        )
        leg_actions = torch.stack(
            (
                self._actions[:, self._leg_abduction_joint_ids],
                self._actions[:, self._leg_hip_joint_ids],
                self._actions[:, self._leg_knee_joint_ids],
            ),
            dim=2,
        )
        first_diagonal_error = torch.stack(
            (
                leg_actions[:, 0, 0] + leg_actions[:, 3, 0],
                leg_actions[:, 0, 1] - leg_actions[:, 3, 1],
                leg_actions[:, 0, 2] - leg_actions[:, 3, 2],
            ),
            dim=1,
        )
        second_diagonal_error = torch.stack(
            (
                leg_actions[:, 1, 0] + leg_actions[:, 2, 0],
                leg_actions[:, 1, 1] - leg_actions[:, 2, 1],
                leg_actions[:, 1, 2] - leg_actions[:, 2, 2],
            ),
            dim=1,
        )
        diagonal_sync_error = 0.5 * (
            torch.mean(torch.square(first_diagonal_error), dim=1)
            + torch.mean(torch.square(second_diagonal_error), dim=1)
        )
        joint_speed = torch.mean(
            torch.square(self._robot.data.joint_vel.torch / SERVO_VELOCITY_LIMIT_RAD_S),
            dim=1,
        )

        foot_forces = self._foot_forces()
        foot_contact = foot_forces > 1.0
        gait_reference, scheduled_contact = self._gait_targets()
        scheduled_swing = ~scheduled_contact
        stance_count = torch.clamp(scheduled_contact.float().sum(dim=1), min=1.0)
        swing_count = scheduled_swing.float().sum(dim=1)
        scheduled_stance_score = (
            (foot_contact & scheduled_contact).float().sum(dim=1) / stance_count
        )
        scheduled_swing_score = torch.where(
            swing_count > 0.0,
            ((~foot_contact) & scheduled_swing).float().sum(dim=1)
            / torch.clamp(swing_count, min=1.0),
            torch.ones_like(swing_count),
        )
        gait_reference_error = torch.mean(
            torch.square(self._actions - gait_reference), dim=1
        )
        gait_reference_tracking = torch.exp(
            -gait_reference_error / self.cfg.gait_reference_sigma**2
        )
        last_air_time = self._contact_sensor.data.last_air_time.torch[
            :, self._foot_sensor_ids
        ]
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt).torch[
            :, self._foot_sensor_ids
        ]
        airborne_count = torch.sum((~foot_contact).float(), dim=1)
        symmetric_swing = (airborne_count >= 1.0) & (airborne_count <= 2.0)
        qualified_touchdown = first_contact.bool() & (
            last_air_time >= self.cfg.qualified_foot_air_time_s
        )
        foot_velocity_xy = self._robot.data.body_lin_vel_w.torch[
            :, self._foot_body_ids, :2
        ]
        support_foot_slip = torch.mean(
            torch.sum(torch.square(foot_velocity_xy), dim=2)
            * foot_contact.float(),
            dim=1,
        )
        normalized_touchdown_overload = torch.clamp(
            (foot_forces - self.cfg.touchdown_force_soft_limit_n)
            / self.cfg.touchdown_force_soft_limit_n,
            min=0.0,
        )
        touchdown_impact = torch.mean(
            torch.square(normalized_touchdown_overload) * first_contact.float(),
            dim=1,
        )

        # Preserve the validated short-horizon objective, then explicitly reward
        # rolling forward progress so a policy cannot collect its return with an
        # initial burst followed by a fixed saturated stance.  Physical
        # displacement and falls remain separate metrics so exploits stay visible.
        reward_terms = {
            "forward_velocity_tracking": (
                self.cfg.reward_forward_velocity_tracking * velocity_tracking
            ),
            "instant_progress": (
                self.cfg.reward_instant_progress
                * torch.where(
                    active_translation,
                    torch.clamp(
                        commanded_velocity
                        / torch.clamp(command_speed, min=1.0e-4),
                        min=-1.0,
                        max=1.0,
                    ),
                    torch.zeros_like(commanded_velocity),
                )
            ),
            "sustained_progress": (
                self.cfg.reward_sustained_progress * sustained_progress
            ),
            "sustained_stall": (
                -self.cfg.penalty_sustained_stall
                * torch.square(normalized_stall_deficit)
                * (rolling_window_ready & active_translation).float()
            ),
            "gait_reference": (
                self.cfg.reward_gait_reference * gait_reference_tracking
            ),
            "scheduled_stance": (
                self.cfg.reward_scheduled_stance * scheduled_stance_score
            ),
            "scheduled_swing": (
                self.cfg.reward_scheduled_swing * scheduled_swing_score
            ),
            "upright": self.cfg.reward_upright * upright_cosine,
            "alive": torch.full_like(upright_cosine, self.cfg.reward_alive),
            "lateral_velocity": (
                -self.cfg.penalty_lateral_velocity * torch.square(linear_velocity[:, 1])
            ),
            "lateral_displacement": (
                -self.cfg.penalty_lateral_displacement
                * torch.square(net_lateral_displacement)
            ),
            "vertical_velocity": (
                -self.cfg.penalty_vertical_velocity * torch.square(linear_velocity[:, 2])
            ),
            "roll_pitch_rate": (
                -self.cfg.penalty_roll_pitch_rate
                * torch.sum(torch.square(angular_velocity[:, :2]), dim=1)
            ),
            "body_tilt": (
                -self.cfg.penalty_body_tilt
                * torch.sum(
                    torch.square(self._robot.data.projected_gravity_b.torch[:, :2]),
                    dim=1,
                )
            ),
            "yaw_rate": -self.cfg.penalty_yaw_rate * torch.square(yaw_error),
            "body_height": -self.cfg.penalty_body_height * torch.square(height_error),
            "action_rate": (
                -smoothness_scale * self.cfg.penalty_action_rate * action_rate
            ),
            "action_acceleration": (
                -smoothness_scale
                * self.cfg.penalty_action_acceleration
                * action_acceleration
            ),
            "action_saturation": (
                -self.cfg.penalty_action_saturation * action_saturation
            ),
            "diagonal_sync": (
                -self.cfg.penalty_diagonal_sync * diagonal_sync_error
            ),
            "action_magnitude": -self.cfg.penalty_action_magnitude * action_magnitude,
            "joint_velocity": -self.cfg.penalty_joint_velocity * joint_speed,
            "joint_acceleration": (
                -smoothness_scale
                * self.cfg.penalty_joint_acceleration
                * normalized_joint_acceleration
            ),
            "body_linear_acceleration": (
                -smoothness_scale
                * self.cfg.penalty_body_linear_acceleration
                * normalized_body_linear_acceleration
            ),
            "body_angular_acceleration": (
                -smoothness_scale
                * self.cfg.penalty_body_angular_acceleration
                * normalized_body_angular_acceleration
            ),
            "support_foot_slip": (
                -self.cfg.penalty_support_foot_slip * support_foot_slip
            ),
            "touchdown_impact": (
                -self.cfg.penalty_touchdown_impact * touchdown_impact
            ),
            "qualified_touchdown": (
                self.cfg.reward_qualified_touchdown
                * torch.sum(qualified_touchdown.float(), dim=1)
            ),
            "termination": -self.cfg.penalty_termination * self._failed.float(),
        }
        reward = (
            torch.stack(tuple(reward_terms.values()), dim=0).sum(dim=0)
            * self.cfg.reward_scale
        )

        self._episode_velocity_error_sum += torch.sqrt(velocity_error_squared)
        self._episode_yaw_error_sum += torch.abs(yaw_error)
        self._episode_commanded_distance.copy_(net_commanded_distance)
        self._episode_action_saturation_sum += torch.mean(
            (torch.abs(self._actions) >= 0.98).float(), dim=1
        )
        self._episode_swing_step_sum += symmetric_swing.float()
        self._episode_touchdown_count += torch.sum(first_contact.float(), dim=1)
        self._episode_qualified_touchdown_count += torch.sum(
            qualified_touchdown.float(), dim=1
        )
        self._episode_qualified_touchdown_by_foot += qualified_touchdown.float()
        self._episode_command_speed_sum += command_speed
        self._episode_base_height_sum += base_height
        self._episode_stall_step_sum += sustained_stall.float()
        self._episode_joint_acceleration_squared_sum += joint_acceleration_squared
        self._episode_body_linear_acceleration_squared_sum += (
            body_linear_acceleration_squared
        )
        self._episode_body_angular_acceleration_squared_sum += (
            body_angular_acceleration_squared
        )
        self._episode_min_rolling_speed = torch.where(
            rolling_window_ready & active_translation,
            torch.minimum(self._episode_min_rolling_speed, rolling_speed),
            self._episode_min_rolling_speed,
        )
        for name, term in reward_terms.items():
            self._episode_reward_term_sums[name] += term
        self._previous_joint_velocity.copy_(self._robot.data.joint_vel.torch)
        self._previous_body_linear_velocity.copy_(
            self._robot.data.root_lin_vel_w.torch
        )
        self._previous_body_angular_velocity.copy_(
            self._robot.data.root_ang_vel_w.torch
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._steps_since_reset += 1
        root_position = self._robot.data.root_pos_w.torch
        local_position = root_position - self._terrain.env_origins
        upright_cosine = -self._robot.data.projected_gravity_b.torch[:, 2]
        base_contact = (self._base_contact_force() > 5.0) & (
            self._steps_since_reset > self.cfg.base_contact_grace_steps
        )
        finite = torch.isfinite(root_position).all(dim=1) & torch.isfinite(
            self._robot.data.joint_pos.torch
        ).all(dim=1)
        self._failure_nonfinite = ~finite
        self._failure_low_height = (
            local_position[:, 2] < self.cfg.minimum_base_height_m
        )
        self._failure_tilt = upright_cosine < self.cfg.minimum_upright_cosine
        if self.cfg.disable_time_limit:
            self._failure_out_of_bounds = torch.zeros_like(self._failed)
        else:
            self._failure_out_of_bounds = (
                torch.linalg.norm(local_position[:, :2], dim=1)
                > self.cfg.maximum_distance_from_origin_m
            )
        self._failure_base_contact = base_contact
        self._failed = (
            self._failure_nonfinite
            | self._failure_low_height
            | self._failure_tilt
            | self._failure_out_of_bounds
            | self._failure_base_contact
        )
        if self.cfg.disable_time_limit:
            time_out = torch.zeros_like(self._failed)
        else:
            time_out = (
                self.episode_length_buf
                >= self._current_episode_horizon_steps() - 1
            )
        return self._failed, time_out

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        self._commands[env_ids] = 0.0
        override = preview_control.COMMAND_OVERRIDE
        if override is not None:
            self._commands[env_ids] = torch.tensor(
                override, dtype=torch.float32, device=self.device
            )
            return

        if self.cfg.command_profile == "forward":
            curriculum_fraction = min(
                float(
                    self.common_step_counter
                    + self.cfg.command_curriculum_offset_steps
                )
                / self.cfg.command_curriculum_steps,
                1.0,
            )
            speed_min = self.cfg.initial_forward_speed_min_m_s + curriculum_fraction * (
                self.cfg.forward_speed_min_m_s
                - self.cfg.initial_forward_speed_min_m_s
            )
            speed_max = self.cfg.initial_forward_speed_max_m_s + curriculum_fraction * (
                self.cfg.forward_speed_max_m_s
                - self.cfg.initial_forward_speed_max_m_s
            )
            self._commands[env_ids, 0] = torch.empty(
                count, device=self.device
            ).uniform_(speed_min, speed_max)
            return
        if self.cfg.command_profile != "directional":
            raise ValueError(f"Unknown command profile: {self.cfg.command_profile}")

        # 50% forward, 15% backward, 15% left, 15% right, 5% stop.
        selector = torch.rand(count, device=self.device)
        forward = selector < 0.50
        backward = (selector >= 0.50) & (selector < 0.65)
        left = (selector >= 0.65) & (selector < 0.80)
        right = (selector >= 0.80) & (selector < 0.95)
        random_unit = torch.rand(count, device=self.device)
        self._commands[env_ids[forward], 0] = (
            self.cfg.forward_speed_min_m_s
            + random_unit[forward]
            * (self.cfg.forward_speed_max_m_s - self.cfg.forward_speed_min_m_s)
        )
        self._commands[env_ids[backward], 0] = -(
            self.cfg.backward_speed_min_m_s
            + random_unit[backward]
            * (self.cfg.backward_speed_max_m_s - self.cfg.backward_speed_min_m_s)
        )
        turn_speed = torch.rand(count, device=self.device) * self.cfg.turn_forward_speed_max_m_s
        turn_rate = self.cfg.turn_rate_min_rad_s + torch.rand(count, device=self.device) * (
            self.cfg.turn_rate_max_rad_s - self.cfg.turn_rate_min_rad_s
        )
        turning = left | right
        self._commands[env_ids[turning], 0] = turn_speed[turning]
        self._commands[env_ids[left], 2] = turn_rate[left]
        self._commands[env_ids[right], 2] = -turn_rate[right]

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) > 0:
            completed = self._steps_since_reset[env_ids] > 0
            completed_steps = torch.clamp(self._steps_since_reset[env_ids], min=1).float()
            log = self.extras.setdefault("log", {})
            log["Metrics/mean_velocity_error_m_s"] = (
                self._episode_velocity_error_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/mean_yaw_error_rad_s"] = (
                self._episode_yaw_error_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/commanded_distance_m"] = self._episode_commanded_distance[
                env_ids
            ].mean().item()
            episode_duration = completed_steps * self.step_dt
            log["Metrics/mean_commanded_speed_m_s"] = (
                self._episode_commanded_distance[env_ids] / episode_duration
            ).mean().item()
            displacement = (
                self._robot.data.root_pos_w.torch[env_ids]
                - self._episode_start_position[env_ids]
            )
            log["Metrics/net_forward_displacement_m"] = displacement[:, 0].mean().item()
            log["Metrics/net_lateral_displacement_m"] = displacement[:, 1].mean().item()
            log["Metrics/target_speed_m_s"] = (
                self._episode_command_speed_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/mean_base_height_m"] = (
                self._episode_base_height_sum[env_ids] / completed_steps
            ).mean().item()
            valid_minimum_speed = torch.where(
                torch.isfinite(self._episode_min_rolling_speed[env_ids]),
                self._episode_min_rolling_speed[env_ids],
                torch.zeros_like(self._episode_min_rolling_speed[env_ids]),
            )
            log["Metrics/min_rolling_forward_speed_m_s"] = (
                valid_minimum_speed.mean().item()
            )
            mature_steps = torch.clamp(
                completed_steps - self._sustained_window_steps + 1, min=1.0
            )
            log["Metrics/sustained_stall_rate"] = (
                self._episode_stall_step_sum[env_ids] / mature_steps
            ).mean().item()
            log["Metrics/rms_joint_acceleration_rad_s2"] = torch.sqrt(
                self._episode_joint_acceleration_squared_sum[env_ids]
                / completed_steps
            ).mean().item()
            log["Metrics/rms_body_linear_acceleration_m_s2"] = torch.sqrt(
                self._episode_body_linear_acceleration_squared_sum[env_ids]
                / completed_steps
            ).mean().item()
            log["Metrics/rms_body_angular_acceleration_rad_s2"] = torch.sqrt(
                self._episode_body_angular_acceleration_squared_sum[env_ids]
                / completed_steps
            ).mean().item()
            if self.cfg.smoothness_curriculum_steps > 0:
                fraction = min(
                    float(
                        self.common_step_counter
                        + self.cfg.command_curriculum_offset_steps
                    )
                    / float(self.cfg.smoothness_curriculum_steps),
                    1.0,
                )
                log["Metrics/smoothness_penalty_scale"] = (
                    self.cfg.smoothness_initial_scale
                    + (1.0 - self.cfg.smoothness_initial_scale) * fraction
                )
            else:
                log["Metrics/smoothness_penalty_scale"] = 1.0
            log["Metrics/current_episode_horizon_s"] = (
                self._current_episode_horizon_steps() * self.step_dt
            )
            motion_command = torch.linalg.norm(
                self._commands[env_ids, :2], dim=1
            ) > 0.03
            success_distance = (
                self.cfg.distance_success_fraction
                * torch.linalg.norm(self._commands[env_ids, :2], dim=1)
                * episode_duration
            )
            distance_success = (
                completed
                & motion_command
                & ~self._failed[env_ids]
                & (self._episode_commanded_distance[env_ids] >= success_distance)
            )
            completed_motion = completed & motion_command
            log["Metrics/distance_success_rate"] = (
                distance_success.float().sum()
                / torch.clamp(completed_motion.float().sum(), min=1.0)
            ).item()
            log["Metrics/action_saturation_rate"] = (
                self._episode_action_saturation_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/swing_step_rate"] = (
                self._episode_swing_step_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/touchdowns_per_episode"] = self._episode_touchdown_count[
                env_ids
            ].mean().item()
            log["Metrics/qualified_touchdowns_per_episode"] = (
                self._episode_qualified_touchdown_count[env_ids].mean().item()
            )
            for foot_index, sensor_name in enumerate(self._foot_sensor_names):
                leg_name = next(leg for leg in LEG_NAMES if leg in sensor_name)
                log[f"Metrics/qualified_touchdowns_{leg_name}"] = (
                    self._episode_qualified_touchdown_by_foot[
                        env_ids, foot_index
                    ].mean().item()
                )
            log["Metrics/fall_rate"] = (
                (self._failed[env_ids] & completed).float().sum()
                / torch.clamp(completed.float().sum(), min=1.0)
            ).item()
            for label, failure in (
                ("nonfinite", self._failure_nonfinite),
                ("low_height", self._failure_low_height),
                ("tilt", self._failure_tilt),
                ("out_of_bounds", self._failure_out_of_bounds),
                ("base_contact", self._failure_base_contact),
            ):
                log[f"Metrics/failure_{label}_rate"] = (
                    (failure[env_ids] & completed).float().sum()
                    / torch.clamp(completed.float().sum(), min=1.0)
                ).item()
            for name, term_sum in self._episode_reward_term_sums.items():
                log[f"Reward/{name}"] = (
                    term_sum[env_ids] / completed_steps
                ).mean().item()

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            if self.cfg.disable_time_limit:
                self.episode_length_buf.zero_()
            else:
                self.episode_length_buf[:] = torch.randint_like(
                    self.episode_length_buf,
                    high=self._current_episode_horizon_steps(),
                )

        joint_position = self._robot.data.default_joint_pos.torch[env_ids].clone()
        joint_position += self.cfg.reset_joint_position_noise_rad * (
            2.0 * torch.rand_like(joint_position) - 1.0
        )
        joint_velocity = torch.zeros_like(self._robot.data.default_joint_vel.torch[env_ids])
        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        root_pose[:, :2] += self.cfg.reset_xy_jitter_m * (
            2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        )
        root_velocity = torch.zeros_like(self._robot.data.default_root_vel.torch[env_ids])

        self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(
            root_velocity=root_velocity, env_ids=env_ids
        )
        self._robot.write_joint_position_to_sim_index(
            position=joint_position, env_ids=env_ids
        )
        self._robot.write_joint_velocity_to_sim_index(
            velocity=joint_velocity, env_ids=env_ids
        )

        self._sample_commands(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._older_actions[env_ids] = 0.0
        self._previous_targets[env_ids] = joint_position
        self._previous_joint_velocity[env_ids] = 0.0
        self._previous_body_linear_velocity[env_ids] = 0.0
        self._previous_body_angular_velocity[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        self._failed[env_ids] = False
        self._failure_nonfinite[env_ids] = False
        self._failure_low_height[env_ids] = False
        self._failure_tilt[env_ids] = False
        self._failure_out_of_bounds[env_ids] = False
        self._failure_base_contact[env_ids] = False
        self._episode_start_position[env_ids] = root_pose[:, :3]
        self._episode_velocity_error_sum[env_ids] = 0.0
        self._episode_yaw_error_sum[env_ids] = 0.0
        self._episode_commanded_distance[env_ids] = 0.0
        self._episode_action_saturation_sum[env_ids] = 0.0
        self._episode_swing_step_sum[env_ids] = 0.0
        self._episode_touchdown_count[env_ids] = 0.0
        self._episode_qualified_touchdown_count[env_ids] = 0.0
        self._episode_qualified_touchdown_by_foot[env_ids] = 0.0
        self._episode_command_speed_sum[env_ids] = 0.0
        self._episode_base_height_sum[env_ids] = 0.0
        self._rolling_command_speed[env_ids] = 0.0
        self._episode_min_rolling_speed[env_ids] = torch.inf
        self._episode_stall_step_sum[env_ids] = 0.0
        self._episode_joint_acceleration_squared_sum[env_ids] = 0.0
        self._episode_body_linear_acceleration_squared_sum[env_ids] = 0.0
        self._episode_body_angular_acceleration_squared_sum[env_ids] = 0.0
        for term_sum in self._episode_reward_term_sums.values():
            term_sum[env_ids] = 0.0
