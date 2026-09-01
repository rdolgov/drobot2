"""Pure-RL vectorized commanded-walking environment for Drobot."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import permutations

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
import warp as wp
from isaaclab import cloner
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade

from simulation.isaac._quadruped_runtime import distributed_push_crawl_by_name

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
EXPECTED_ACTION_ORDER = tuple(
    f"{leg}_{joint_kind}"
    for joint_kind in ("hip_abduction", "hip_flexion", "knee")
    for leg in LEG_NAMES
)


def _yaw_from_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return wrapped world yaw for Isaac Lab's Warp ``xyzw`` quaternion."""
    x, y, z, w = quaternion.unbind(dim=1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _quaternion_from_rpy_xyzw(
    roll: torch.Tensor,
    pitch: torch.Tensor,
    yaw: torch.Tensor,
) -> torch.Tensor:
    """Return Isaac Lab ``xyzw`` quaternions for intrinsic RPY angles."""
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    cy, sy = torch.cos(half_yaw), torch.sin(half_yaw)
    return torch.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        dim=1,
    )


def _multiply_quaternions_xyzw(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """Hamilton-product Isaac Lab ``xyzw`` quaternion batches."""
    lx, ly, lz, lw = lhs.unbind(dim=1)
    rx, ry, rz, rw = rhs.unbind(dim=1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=1,
    )


def _author_rectangular_shoes(stage, cfg) -> None:
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
    material_api.CreateStaticFrictionAttr().Set(float(cfg.shoe_static_friction))
    material_api.CreateDynamicFrictionAttr().Set(float(cfg.shoe_dynamic_friction))
    material_api.CreateRestitutionAttr().Set(float(cfg.shoe_restitution))
    physx_api = PhysxSchema.PhysxMaterialAPI.Apply(tread_material.GetPrim())
    physx_api.CreateCompliantContactStiffnessAttr().Set(
        float(cfg.shoe_contact_stiffness)
    )
    physx_api.CreateCompliantContactDampingAttr().Set(
        float(cfg.shoe_contact_damping)
    )

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
    dry_mass_scale = float(cfg.dry_robot_mass_scale)
    if not math.isfinite(dry_mass_scale) or dry_mass_scale <= 0.0:
        raise ValueError("dry_robot_mass_scale must be finite and positive")
    dry_mass *= dry_mass_scale
    dry_inertia *= dry_mass_scale

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
        scheduled_release_force_sigma_n = float(
            self.cfg.scheduled_swing_release_force_sigma_n
        )
        if (
            not math.isfinite(scheduled_release_force_sigma_n)
            or scheduled_release_force_sigma_n <= 0.0
        ):
            raise ValueError(
                "scheduled_swing_release_force_sigma_n must be finite and positive"
            )
        self._scheduled_release_force_sigma_n = scheduled_release_force_sigma_n
        release_force_threshold_n = float(
            self.cfg.scheduled_release_force_threshold_n
        )
        release_shaping_width_n = float(
            self.cfg.scheduled_release_shaping_width_n
        )
        release_consecutive_steps = int(
            self.cfg.scheduled_release_min_consecutive_steps
        )
        if not math.isfinite(release_force_threshold_n) or release_force_threshold_n <= 0.0:
            raise ValueError(
                "scheduled_release_force_threshold_n must be finite and positive"
            )
        if not math.isfinite(release_shaping_width_n) or release_shaping_width_n <= 0.0:
            raise ValueError(
                "scheduled_release_shaping_width_n must be finite and positive"
            )
        if release_consecutive_steps <= 0:
            raise ValueError(
                "scheduled_release_min_consecutive_steps must be positive"
            )
        self._scheduled_release_force_threshold_n = release_force_threshold_n
        self._scheduled_release_shaping_width_n = release_shaping_width_n
        self._scheduled_release_min_consecutive_steps = release_consecutive_steps
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        self._older_actions = torch.zeros_like(self._actions)
        self._previous_targets = self._robot.data.default_joint_pos.torch.clone()
        self._desired_targets = self._previous_targets.clone()
        self._previous_requested_targets = self._previous_targets.clone()
        self._previous_joint_velocity = self._robot.data.joint_vel.torch.clone()
        self._previous_body_linear_velocity = (
            self._robot.data.root_lin_vel_w.torch.clone()
        )
        self._previous_body_angular_velocity = (
            self._robot.data.root_ang_vel_w.torch.clone()
        )
        self._commands = torch.zeros((self.num_envs, 3), device=self.device)
        self._nominal_yaw_command = torch.zeros(self.num_envs, device=self.device)
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
        if tuple(self._robot.joint_names) != EXPECTED_ACTION_ORDER:
            raise RuntimeError(
                "Isaac articulation joint order does not match the exported/onboard "
                f"action contract: expected {EXPECTED_ACTION_ORDER}, got "
                f"{tuple(self._robot.joint_names)}"
            )
        configured_residual_scales = self.cfg.residual_action_scale_by_action
        if configured_residual_scales is None:
            configured_residual_scales = (
                float(self.cfg.residual_action_scale),
            ) * len(EXPECTED_ACTION_ORDER)
        if len(configured_residual_scales) != len(EXPECTED_ACTION_ORDER):
            raise ValueError(
                "residual_action_scale_by_action must contain one value for each "
                f"action in {EXPECTED_ACTION_ORDER}"
            )
        self._residual_action_scale = torch.tensor(
            configured_residual_scales,
            dtype=torch.float32,
            device=self.device,
        )
        if not torch.all(torch.isfinite(self._residual_action_scale)):
            raise ValueError("residual action scales must be finite")
        if torch.any(self._residual_action_scale <= 0.0) or torch.any(
            self._residual_action_scale > 1.0
        ):
            raise ValueError("residual action scales must be within (0, 1]")
        self._leg_phase_offsets = torch.tensor(
            self.cfg.gait_phase_offsets, dtype=torch.float32, device=self.device
        )
        if self._leg_phase_offsets.shape != (4,):
            raise RuntimeError(
                "gait_phase_offsets must contain FL, RL, FR, RR offsets"
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
        self._foot_leg_names = tuple(
            next(
                (
                    leg
                    for leg in LEG_NAMES
                    if leg in str(sensor_name)
                ),
                "",
            )
            for sensor_name in self._foot_sensor_names
        )
        if set(self._foot_leg_names) != set(LEG_NAMES) or len(
            set(self._foot_leg_names)
        ) != len(LEG_NAMES):
            raise RuntimeError(
                "Foot sensors must identify front-left, rear-left, front-right, "
                f"and rear-right exactly once; got {self._foot_sensor_names}"
            )
        self._foot_phase_offsets = torch.tensor(
            [
                self._leg_phase_offsets[LEG_NAMES.index(leg_name)].item()
                for leg_name in self._foot_leg_names
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
        self._base_body_ids, base_body_names = self._robot.find_bodies("base_link")
        if len(self._base_body_ids) != 1:
            raise RuntimeError(f"Expected one base_link body, got {base_body_names}")
        self._base_body_ids = torch.as_tensor(
            self._base_body_ids, dtype=torch.int32, device=self.device
        )
        self._initialize_physical_randomization()
        self._distributed_reference_joint_pos: torch.Tensor | None = None
        self._distributed_reference_contact: torch.Tensor | None = None
        self._distributed_reference_lateral_shift: torch.Tensor | None = None
        self._distributed_reference_lateral_shift_phase_derivative: (
            torch.Tensor | None
        ) = None
        self._stance_forward_bias_m = 0.0
        # DirectRLEnv increments the episode step before asking for reward.  Cache
        # the exact phase/reference applied to physics so reward and diagnostics
        # do not accidentally score the following 60 Hz gait tick.
        self._applied_gait_reference: torch.Tensor | None = None
        self._applied_scheduled_contact: torch.Tensor | None = None
        self._applied_gait_phase: torch.Tensor | None = None
        self._applied_expected_lateral_displacement: torch.Tensor | None = None
        self._applied_expected_lateral_velocity: torch.Tensor | None = None
        if self.cfg.gait_reference_mode in (
            "distributed_push",
            "smooth_distributed_push",
        ):
            self._build_distributed_reference_table()
        elif self.cfg.gait_reference_mode != "continuous":
            raise RuntimeError(
                f"Unknown gait_reference_mode: {self.cfg.gait_reference_mode}"
            )
        self._randomize_rear_payload_estimate()
        self._randomize_actuator_capacity()
        self._randomize_foot_materials()
        self._initialize_episode_perturbation_buffers()

        self._steps_since_reset = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        # A phase-zero reset always presents the rear-right step first.  Profiles
        # that enforce four-leg participation can instead start on any quarter
        # cycle so PPO sees each physical leg as the first loaded swing leg.
        self._gait_phase_offset = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._failed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._failure_nonfinite = torch.zeros_like(self._failed)
        self._failure_low_height = torch.zeros_like(self._failed)
        self._failure_tilt = torch.zeros_like(self._failed)
        self._failure_out_of_bounds = torch.zeros_like(self._failed)
        self._failure_base_contact = torch.zeros_like(self._failed)
        self._episode_start_position = torch.zeros((self.num_envs, 3), device=self.device)
        self._episode_start_yaw = torch.zeros(self.num_envs, device=self.device)
        self._episode_forward_direction_w = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._episode_lateral_direction_w = torch.zeros_like(
            self._episode_forward_direction_w
        )
        self._episode_velocity_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_yaw_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_heading_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_straight_progress_gate_sum = torch.zeros(
            self.num_envs, device=self.device
        )
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
        self._episode_scheduled_swing_success_by_foot = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._episode_scheduled_release_quality_by_foot = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._episode_scheduled_swing_opportunity_by_foot = torch.zeros(
            (self.num_envs, 4), device=self.device
        )
        self._episode_four_leg_progress_gate_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._previous_applied_gait_phase = torch.zeros(
            self.num_envs, device=self.device
        )
        self._gait_phase_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._cycle_release_full_cycle_active = torch.zeros_like(
            self._gait_phase_initialized
        )
        self._cycle_release_consecutive_steps = torch.zeros(
            (self.num_envs, 4), dtype=torch.long, device=self.device
        )
        self._cycle_release_pass_by_foot = torch.zeros(
            (self.num_envs, 4), dtype=torch.bool, device=self.device
        )
        self._last_completed_cycle_all_four_release = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._episode_completed_gait_cycles = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_all_four_release_cycles = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_cycle_release_qualifications_by_foot = torch.zeros(
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
        minimum_gait_frequency_hz = (
            1.0 / self.cfg.gait_period_s
            if self.cfg.gait_clock_mode == "fixed"
            else self.cfg.gait_frequency_min_hz
        )
        if minimum_gait_frequency_hz <= 0.0:
            raise ValueError("minimum gait frequency must be positive")
        self._lateral_cycle_history_steps = max(
            3,
            int(math.ceil(1.0 / (minimum_gait_frequency_hz * self.step_dt)))
            + 2,
        )
        self._lateral_displacement_history = torch.zeros(
            (self.num_envs, self._lateral_cycle_history_steps),
            device=self.device,
        )
        self._episode_cycle_lateral_speed_sum = torch.zeros(
            self.num_envs,
            device=self.device,
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
        self._episode_effort_soft_limit_step_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_max_applied_effort_fraction = torch.zeros(
            self.num_envs, device=self.device
        )
        reward_term_names = (
            "forward_velocity_tracking",
            "instant_progress",
            "sustained_progress",
            "sustained_stall",
            "backward_motion",
            "overspeed",
            "rearward_pitch",
            "gait_reference",
            "scheduled_stance",
            "missing_scheduled_stance",
            "scheduled_swing",
            "scheduled_release_shaping",
            "cycle_four_leg_release",
            "upright",
            "alive",
            "lateral_velocity",
            "normalized_lateral_velocity",
            "lateral_displacement",
            "lateral_corridor",
            "vertical_velocity",
            "roll_pitch_rate",
            "body_tilt",
            "yaw_rate",
            "heading_error",
            "body_height",
            "action_rate",
            "action_acceleration",
            "action_saturation",
            "target_limiter_gap",
            "diagonal_sync",
            "action_magnitude",
            "joint_velocity",
            "joint_acceleration",
            "body_linear_acceleration",
            "body_angular_acceleration",
            "effort_soft_limit",
            "support_foot_slip",
            "touchdown_impact",
            "qualified_touchdown",
            "least_active_swing",
            "three_foot_support",
            "excess_airborne_feet",
            "termination",
            "straight_aligned_progress",
        )
        self._episode_reward_term_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in reward_term_names
        }

    def _build_distributed_reference_table(self) -> None:
        """Sample the proven hardware crawl into a GPU lookup table."""

        sample_count = 2048
        joint_positions: list[list[float]] = []
        scheduled_contacts: list[list[bool]] = []
        lateral_shifts: list[float] = []
        for sample_index in range(sample_count):
            phase = sample_index / sample_count
            pose, state = distributed_push_crawl_by_name(
                phase,
                period_s=1.0,
                stride_m=self.cfg.gait_stride_m,
                lift_m=self.cfg.gait_lift_m,
                support_extension_m=0.0,
                weight_shift_forward_m=self.cfg.gait_weight_shift_forward_m,
                weight_shift_lateral_m=self.cfg.gait_weight_shift_lateral_m,
                rear_weight_shift_forward_m=(
                    self.cfg.gait_rear_weight_shift_forward_m
                ),
                translate_lateral_weight_shift=(
                    self.cfg.gait_translate_lateral_weight_shift
                ),
                forward_body_pitch_rad=self.cfg.gait_forward_body_pitch_rad,
                stance_center_offset_m=self.cfg.gait_stance_center_offset_m,
                down_m=self.cfg.gait_stance_down_m,
                fore_aft_m=self.cfg.gait_stance_fore_aft_m,
                abduction_deg=0.0,
                smooth_support_push=self.cfg.gait_smooth_support_push,
                phase_fractions=(
                    self.cfg.gait_distributed_push_phase_fractions
                ),
                contact_transition_fraction=(
                    self.cfg.gait_contact_transition_fraction
                ),
            )
            joint_positions.append(
                [float(pose[name]) for name in self._robot.joint_names]
            )
            support_legs = set(state["expected_support_legs"])
            self._stance_forward_bias_m = float(state["stance_forward_bias_m"])
            lateral_shifts.append(float(state["body_shift_lateral_m"]))
            scheduled_contacts.append(
                [
                    next(leg for leg in LEG_NAMES if leg in foot_name)
                    in support_legs
                    for foot_name in self._foot_sensor_names
                ]
            )
        self._distributed_reference_joint_pos = torch.tensor(
            joint_positions,
            dtype=torch.float32,
            device=self.device,
        )
        self._distributed_reference_contact = torch.tensor(
            scheduled_contacts,
            dtype=torch.bool,
            device=self.device,
        )
        self._distributed_reference_lateral_shift = torch.tensor(
            lateral_shifts,
            dtype=torch.float32,
            device=self.device,
        )
        # Central periodic finite difference with respect to normalized cycle
        # phase. Multiplying this table by gait frequency yields m/s.
        self._distributed_reference_lateral_shift_phase_derivative = 0.5 * sample_count * (
            torch.roll(self._distributed_reference_lateral_shift, shifts=-1)
            - torch.roll(self._distributed_reference_lateral_shift, shifts=1)
        )

    def _initialize_physical_randomization(self) -> None:
        """Choose the fixed nominal-domain subset and mirrored joint pairs."""
        nominal_fraction = float(self.cfg.physical_randomization_nominal_fraction)
        if not 0.0 <= nominal_fraction <= 1.0:
            raise ValueError(
                "physical_randomization_nominal_fraction must be in [0, 1]"
            )
        nominal_count = min(
            self.num_envs,
            max(0, int(round(nominal_fraction * self.num_envs))),
        )
        self._physical_randomization_active = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        if nominal_count == self.num_envs:
            self._physical_randomization_active.zero_()
        elif nominal_count > 0:
            nominal_ids = torch.randperm(
                self.num_envs, device=self.device
            )[:nominal_count]
            self._physical_randomization_active[nominal_ids] = False
        # Existing V24 domains predate the nominal-subset option.  Only opt them
        # into the mask when a task explicitly reserves a nontrivial subset.
        self._mask_existing_physical_randomization = nominal_fraction < 1.0
        self._mirrored_joint_pairs = tuple(
            (
                self._robot.joint_names.index(f"{left_leg}_{joint_suffix}"),
                self._robot.joint_names.index(f"{right_leg}_{joint_suffix}"),
            )
            for left_leg, right_leg in (
                ("front_left", "front_right"),
                ("rear_left", "rear_right"),
            )
            for joint_suffix in ("hip_abduction", "hip_flexion", "knee")
        )
        self._mirrored_foot_pairs = tuple(
            (
                next(
                    index
                    for index, name in enumerate(self._foot_body_names)
                    if left_leg in name
                ),
                next(
                    index
                    for index, name in enumerate(self._foot_body_names)
                    if right_leg in name
                ),
            )
            for left_leg, right_leg in (
                ("front_left", "front_right"),
                ("rear_left", "rear_right"),
            )
        )

    def _sample_paired_scale(
        self,
        value_range: tuple[float, float],
        item_count: int,
        pairs: Sequence[tuple[int, int]],
        *,
        nominal_value: float = 1.0,
    ) -> torch.Tensor:
        """Sample per-item values, optionally mirrored around the range midpoint."""
        low, high = (float(value) for value in value_range)
        if low > high:
            raise ValueError(f"Invalid randomization range: {value_range}")
        if low == high:
            values = torch.full(
                (self.num_envs, item_count), low, device=self.device
            )
        else:
            values = low + (high - low) * torch.rand(
                (self.num_envs, item_count), device=self.device
            )
        if self.cfg.mirror_physical_randomization_pairs and low != high:
            for first_id, second_id in pairs:
                first = low + (high - low) * torch.rand(
                    self.num_envs, device=self.device
                )
                second = low + high - first
                swap = torch.rand(self.num_envs, device=self.device) < 0.5
                values[:, first_id] = torch.where(swap, second, first)
                values[:, second_id] = torch.where(swap, first, second)
        values[~self._physical_randomization_active] = nominal_value
        return values

    def _sample_joint_target_bias(self) -> torch.Tensor:
        """Return fixed per-joint zero offsets, with optional mirrored signs."""
        joint_count = len(self._robot.joint_names)
        limits = torch.tensor(
            [
                self.cfg.joint_target_bias_abduction_rad
                if name.endswith("hip_abduction")
                else self.cfg.joint_target_bias_flexion_rad
                for name in self._robot.joint_names
            ],
            dtype=torch.float32,
            device=self.device,
        )
        if torch.any(limits < 0.0):
            raise ValueError("joint target bias limits must be non-negative")
        if not torch.any(limits > 0.0):
            return torch.zeros(
                (self.num_envs, joint_count), device=self.device
            )
        biases = (
            2.0 * torch.rand((self.num_envs, joint_count), device=self.device)
            - 1.0
        ) * limits
        if self.cfg.mirror_physical_randomization_pairs:
            for first_id, second_id in self._mirrored_joint_pairs:
                limit = limits[first_id]
                first = (2.0 * torch.rand(self.num_envs, device=self.device) - 1.0) * limit
                second = -first
                swap = torch.rand(self.num_envs, device=self.device) < 0.5
                biases[:, first_id] = torch.where(swap, second, first)
                biases[:, second_id] = torch.where(swap, first, second)
        biases[~self._physical_randomization_active] = 0.0
        return biases

    def _randomize_foot_materials(self) -> None:
        """Assign per-environment friction to all shapes on each distal body."""
        static_range = self.cfg.shoe_static_friction_randomization_range
        dynamic_range = self.cfg.shoe_dynamic_friction_randomization_range
        common_static_range = (
            self.cfg.shoe_common_static_friction_randomization_range
        )
        common_dynamic_range = (
            self.cfg.shoe_common_dynamic_friction_randomization_range
        )
        self._foot_static_friction = torch.full(
            (self.num_envs, 4),
            float(self.cfg.shoe_static_friction),
            device=self.device,
        )
        self._foot_dynamic_friction = torch.full(
            (self.num_envs, 4),
            float(self.cfg.shoe_dynamic_friction),
            device=self.device,
        )
        uses_common_surface = (
            common_static_range is not None
            or common_dynamic_range is not None
        )
        if uses_common_surface and (
            common_static_range is None or common_dynamic_range is None
        ):
            raise ValueError(
                "common static and dynamic friction ranges must be configured together"
            )
        if uses_common_surface and (
            static_range is not None or dynamic_range is not None
        ):
            raise ValueError(
                "common-surface friction cannot be combined with legacy absolute "
                "per-foot friction ranges"
            )
        if static_range is None and dynamic_range is None and not uses_common_surface:
            return
        if uses_common_surface:
            static_low, static_high = (
                float(value) for value in common_static_range
            )
            dynamic_low, dynamic_high = (
                float(value) for value in common_dynamic_range
            )
            if not 0.0 < static_low <= static_high:
                raise ValueError("common static friction range must be positive and ordered")
            if not 0.0 < dynamic_low <= dynamic_high:
                raise ValueError("common dynamic friction range must be positive and ordered")
            surface_draw = torch.rand((self.num_envs, 1), device=self.device)
            common_static = static_low + (static_high - static_low) * surface_draw
            common_dynamic = dynamic_low + (dynamic_high - dynamic_low) * surface_draw
            differential = self._sample_paired_scale(
                self.cfg.shoe_friction_differential_scale_range,
                4,
                self._mirrored_foot_pairs,
            )
            self._foot_static_friction = common_static * differential
            self._foot_dynamic_friction = common_dynamic * differential
            nominal = ~self._physical_randomization_active
            self._foot_static_friction[nominal] = float(
                self.cfg.shoe_static_friction
            )
            self._foot_dynamic_friction[nominal] = float(
                self.cfg.shoe_dynamic_friction
            )
        elif static_range is not None:
            self._foot_static_friction = self._sample_paired_scale(
                static_range,
                4,
                self._mirrored_foot_pairs,
                nominal_value=float(self.cfg.shoe_static_friction),
            )
        if not uses_common_surface and dynamic_range is not None:
            self._foot_dynamic_friction = self._sample_paired_scale(
                dynamic_range,
                4,
                self._mirrored_foot_pairs,
                nominal_value=float(self.cfg.shoe_dynamic_friction),
            )
        self._foot_dynamic_friction = torch.minimum(
            self._foot_dynamic_friction,
            self._foot_static_friction,
        )

        # Isaac Lab's public articulation API does not expose per-body shape
        # ranges.  This is the same backend mapping used by its official rigid-
        # body-material randomizer and is deliberately confined to init time.
        shape_count_by_body: list[int] = []
        for link_path in self._robot.root_view.link_paths[0]:
            link_view = self._robot._physics_sim_view.create_rigid_body_view(
                link_path
            )
            shape_count_by_body.append(int(link_view.max_shapes))
        if sum(shape_count_by_body) != int(self._robot.root_view.max_shapes):
            raise RuntimeError(
                "Could not map distal-body friction to articulation shapes: "
                f"counted {sum(shape_count_by_body)}, expected "
                f"{self._robot.root_view.max_shapes}"
            )
        backend_foot_ids = self._robot.map_body_ids_to_backend(
            self._foot_body_ids
        )
        materials = wp.to_torch(
            self._robot.root_view.get_material_properties()
        ).clone()
        static_values = self._foot_static_friction.to(materials.device)
        dynamic_values = self._foot_dynamic_friction.to(materials.device)
        for foot_index, backend_body_id in enumerate(backend_foot_ids):
            body_id = int(backend_body_id)
            shape_start = sum(shape_count_by_body[:body_id])
            shape_end = shape_start + shape_count_by_body[body_id]
            materials[:, shape_start:shape_end, 0] = static_values[
                :, foot_index
            ].unsqueeze(1)
            materials[:, shape_start:shape_end, 1] = dynamic_values[
                :, foot_index
            ].unsqueeze(1)
        material_env_ids = torch.arange(
            self.num_envs, dtype=torch.int32, device=materials.device
        )
        self._robot.root_view.set_material_properties(
            wp.from_torch(materials.contiguous(), dtype=wp.float32),
            wp.from_torch(material_env_ids, dtype=wp.int32),
        )

    def _initialize_episode_perturbation_buffers(self) -> None:
        """Allocate reset-time IMU and persistent-load domains."""
        if self.cfg.imu_projected_gravity_noise_std < 0.0:
            raise ValueError("projected-gravity noise standard deviation must be non-negative")
        if self.cfg.imu_linear_acceleration_noise_std_g < 0.0:
            raise ValueError("linear-acceleration noise standard deviation must be non-negative")
        self._imu_angular_velocity_bias = torch.zeros(
            (self.num_envs, 3), device=self.device
        )
        self._base_force_b = torch.zeros(
            (self.num_envs, 1, 3), device=self.device
        )
        self._base_torque_b = torch.zeros_like(self._base_force_b)

    def _randomize_episode_perturbations(self, env_ids: torch.Tensor) -> None:
        """Resample fixed-for-an-episode sensor bias and body-frame wrench."""
        count = len(env_ids)
        active = self._physical_randomization_active[env_ids].float().unsqueeze(1)
        gyro_limit = float(self.cfg.imu_angular_velocity_bias_range_rad_s)
        if gyro_limit < 0.0:
            raise ValueError("IMU angular-velocity bias range must be non-negative")
        if gyro_limit > 0.0:
            self._imu_angular_velocity_bias[env_ids] = (
                2.0 * torch.rand((count, 3), device=self.device) - 1.0
            ) * gyro_limit * active
        else:
            self._imu_angular_velocity_bias[env_ids] = 0.0

        force_limit = torch.tensor(
            self.cfg.base_force_randomization_range_n,
            dtype=torch.float32,
            device=self.device,
        )
        torque_limit = torch.tensor(
            self.cfg.base_torque_randomization_range_nm,
            dtype=torch.float32,
            device=self.device,
        )
        if force_limit.shape != (3,) or torque_limit.shape != (3,):
            raise ValueError("base force and torque ranges must each have three axes")
        if torch.any(force_limit < 0.0) or torch.any(torque_limit < 0.0):
            raise ValueError("base force and torque ranges must be non-negative")
        if torch.any(force_limit > 0.0):
            self._base_force_b[env_ids, 0] = (
                2.0 * torch.rand((count, 3), device=self.device) - 1.0
            ) * force_limit * active
        else:
            self._base_force_b[env_ids] = 0.0
        if torch.any(torque_limit > 0.0):
            self._base_torque_b[env_ids, 0] = (
                2.0 * torch.rand((count, 3), device=self.device) - 1.0
            ) * torque_limit * active
        else:
            self._base_torque_b[env_ids] = 0.0
        if torch.any(force_limit > 0.0) or torch.any(torque_limit > 0.0):
            self._robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self._base_force_b[env_ids],
                torques=self._base_torque_b[env_ids],
                body_ids=self._base_body_ids,
                env_ids=env_ids,
            )

    def _randomize_rear_payload_estimate(self) -> None:
        """Apply measured dry mass and bounded whole-robot/payload domains."""
        base_ids, base_names = self._robot.find_bodies("base_link")
        if len(base_ids) != 1:
            raise RuntimeError(f"Expected one base_link body, got {base_names}")
        base_ids = torch.as_tensor(base_ids, dtype=torch.int32, device=self.device)
        all_body_ids = torch.arange(
            len(self._robot.body_names), dtype=torch.int32, device=self.device
        )
        env_ids = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        dry_scale = float(self.cfg.dry_robot_mass_scale)
        if not math.isfinite(dry_scale) or dry_scale <= 0.0:
            raise ValueError("dry_robot_mass_scale must be finite and positive")
        global_low, global_high = self.cfg.robot_mass_scale_range
        if not 0.0 < global_low <= global_high:
            raise ValueError("robot_mass_scale_range must be positive and ordered")
        global_scale = global_low + (global_high - global_low) * torch.rand(
            (self.num_envs, 1), device=self.device
        )
        if self._mask_existing_physical_randomization:
            global_scale[~self._physical_randomization_active] = 1.0
        # The base was already reconstructed from scaled dry inertia plus the
        # independently measured pack during USD authoring.  Scale only the
        # remaining dry links here, then apply the bounded whole-robot domain.
        static_body_scale = torch.full(
            (1, len(self._robot.body_names)),
            dry_scale,
            dtype=torch.float32,
            device=self.device,
        )
        static_body_scale[:, base_ids.long()] = 1.0
        masses = (
            self._robot.data.body_mass.torch
            * static_body_scale
            * global_scale
        )
        inertias = (
            self._robot.data.body_inertia.torch
            * static_body_scale.unsqueeze(-1)
            * global_scale.unsqueeze(-1)
        )
        if self.cfg.rear_payload_enabled:
            payload_low, payload_high = (
                self.cfg.rear_payload_combined_mass_scale_range
            )
            if not 0.0 < payload_low <= payload_high:
                raise ValueError(
                    "rear_payload_combined_mass_scale_range must be positive and ordered"
                )
            payload_scale = payload_low + (payload_high - payload_low) * torch.rand(
                (self.num_envs, 1), device=self.device
            )
            if self._mask_existing_physical_randomization:
                payload_scale[~self._physical_randomization_active] = 1.0
            masses[:, base_ids.long()] *= payload_scale
            inertias[:, base_ids.long()] *= payload_scale.unsqueeze(-1)
        self._robot.set_masses_index(
            masses=masses, body_ids=all_body_ids, env_ids=env_ids
        )
        self._robot.set_inertias_index(
            inertias=inertias, body_ids=all_body_ids, env_ids=env_ids
        )

        if not self.cfg.rear_payload_enabled:
            return
        jitter_limit = torch.tensor(
            self.cfg.rear_payload_combined_com_jitter_m,
            dtype=torch.float32,
            device=self.device,
        )
        coms = self._robot.data.body_com_pose_b.torch[:, base_ids.long()].clone()
        com_jitter = (
            2.0 * torch.rand((self.num_envs, 1, 3), device=self.device) - 1.0
        ) * jitter_limit
        if self._mask_existing_physical_randomization:
            com_jitter[~self._physical_randomization_active] = 0.0
        coms[:, :, :3] += com_jitter
        self._robot.set_coms_index(coms=coms, body_ids=base_ids, env_ids=env_ids)

    def _randomize_actuator_capacity(self) -> None:
        """Approximate common and per-joint servo strength, rate, and gain spread."""
        peak_effort_nm = float(self.cfg.actuator_peak_effort_nm)
        if not math.isfinite(peak_effort_nm) or peak_effort_nm <= 0.0:
            raise ValueError("actuator_peak_effort_nm must be finite and positive")
        effort_low, effort_high = self.cfg.actuator_effort_scale_range
        rate_low, rate_high = self.cfg.target_velocity_scale_range
        if not 0.0 < effort_low <= effort_high:
            raise ValueError("actuator_effort_scale_range must be positive and ordered")
        if not 0.0 < rate_low <= rate_high:
            raise ValueError("target_velocity_scale_range must be positive and ordered")
        self._actuator_effort_scale = effort_low + (effort_high - effort_low) * torch.rand(
            (self.num_envs, 1), device=self.device
        )
        if self.cfg.correlate_common_actuator_scales:
            common_draw = (
                self._actuator_effort_scale - effort_low
            ) / max(effort_high - effort_low, torch.finfo(torch.float32).eps)
            self._target_velocity_scale = (
                rate_low + (rate_high - rate_low) * common_draw
            )
        else:
            self._target_velocity_scale = rate_low + (rate_high - rate_low) * torch.rand(
                (self.num_envs, 1), device=self.device
            )
        if self._mask_existing_physical_randomization:
            nominal = ~self._physical_randomization_active
            self._actuator_effort_scale[nominal] = 1.0
            self._target_velocity_scale[nominal] = 1.0
        joint_count = len(self._robot.joint_names)
        individual_effort = self._sample_paired_scale(
            self.cfg.actuator_individual_effort_scale_range,
            joint_count,
            self._mirrored_joint_pairs,
        )
        individual_rate = self._sample_paired_scale(
            self.cfg.target_individual_velocity_scale_range,
            joint_count,
            self._mirrored_joint_pairs,
        )
        if torch.any(individual_effort <= 0.0):
            raise ValueError("individual actuator effort scales must be positive")
        if torch.any(individual_rate <= 0.0):
            raise ValueError("individual target velocity scales must be positive")
        self._actuator_effort_scale_by_joint = (
            self._actuator_effort_scale * individual_effort
        )
        self._target_velocity_scale_by_joint = (
            self._target_velocity_scale * individual_rate
        )
        effort_limits = peak_effort_nm * self._actuator_effort_scale_by_joint
        self._actuator_effort_limit_by_joint = effort_limits
        # For implicit actuators PhysX owns the real drive clamp, while Isaac
        # Lab's actuator object independently clips its approximate torque
        # telemetry.  Keep both limits synchronized: otherwise a >1 capacity
        # scale changes the physics but reported ``applied_torque`` remains
        # clipped at the authored 0.8826 N*m cap.
        for actuator in self._robot.actuators.values():
            joint_ids = actuator.joint_indices
            if joint_ids is None:
                joint_ids = slice(None)
            actuator_limits = effort_limits[:, joint_ids]
            actuator.effort_limit.copy_(actuator_limits)
            actuator.effort_limit_sim.copy_(actuator_limits)
        self._robot.write_joint_effort_limit_to_sim_index(limits=effort_limits)

        stiffness_range = self.cfg.actuator_individual_stiffness_scale_range
        damping_range = self.cfg.actuator_individual_damping_scale_range
        self._actuator_stiffness_scale_by_joint = self._sample_paired_scale(
            stiffness_range,
            joint_count,
            self._mirrored_joint_pairs,
        )
        self._actuator_damping_scale_by_joint = self._sample_paired_scale(
            damping_range,
            joint_count,
            self._mirrored_joint_pairs,
        )
        if torch.any(self._actuator_stiffness_scale_by_joint <= 0.0):
            raise ValueError("individual actuator stiffness scales must be positive")
        if torch.any(self._actuator_damping_scale_by_joint <= 0.0):
            raise ValueError("individual actuator damping scales must be positive")
        if tuple(stiffness_range) != (1.0, 1.0):
            self._robot.write_joint_stiffness_to_sim_index(
                stiffness=(
                    self._robot.data.default_joint_stiffness.torch
                    * self._actuator_stiffness_scale_by_joint
                )
            )
        if tuple(damping_range) != (1.0, 1.0):
            self._robot.write_joint_damping_to_sim_index(
                damping=(
                    self._robot.data.default_joint_damping.torch
                    * self._actuator_damping_scale_by_joint
                )
            )

        self._joint_target_bias = self._sample_joint_target_bias()
        delay_low, delay_high = (
            int(value) for value in self.cfg.control_delay_step_range
        )
        if not 0 <= delay_low <= delay_high <= 1:
            raise ValueError("control_delay_step_range must be within [0, 1]")
        if delay_low == delay_high:
            self._control_delay_steps = torch.full(
                (self.num_envs,),
                delay_low,
                dtype=torch.long,
                device=self.device,
            )
        else:
            self._control_delay_steps = torch.randint(
                delay_low,
                delay_high + 1,
                (self.num_envs,),
                dtype=torch.long,
                device=self.device,
            )
        self._control_delay_steps[~self._physical_randomization_active] = 0

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
        _author_rectangular_shoes(self.scene.stage, self.cfg)
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
        applied_gait_phase = self._gait_phase()
        gait_reference, scheduled_contact = self._gait_targets(
            phase=applied_gait_phase
        )
        (
            expected_lateral_displacement,
            expected_lateral_velocity,
        ) = self._gait_expected_lateral_motion()
        self._applied_gait_reference = gait_reference
        self._applied_scheduled_contact = scheduled_contact
        self._applied_gait_phase = applied_gait_phase
        self._applied_expected_lateral_displacement = (
            expected_lateral_displacement
        )
        self._applied_expected_lateral_velocity = expected_lateral_velocity
        if self.cfg.action_mode == "gait_residual":
            normalized_target = (
                gait_reference + self._residual_action_scale * self._actions
            )
        elif self.cfg.action_mode == "direct":
            normalized_target = self._actions
        else:
            raise RuntimeError(f"Unknown action_mode: {self.cfg.action_mode}")
        desired_targets = self._robot.data.default_joint_pos.torch + (
            self._joint_scale * normalized_target
        ) + self._joint_target_bias
        limits = self._robot.data.soft_joint_pos_limits.torch
        desired_targets = torch.clamp(
            desired_targets, limits[:, :, 0], limits[:, :, 1]
        )
        delayed_targets = torch.where(
            (self._control_delay_steps > 0).unsqueeze(1),
            self._previous_requested_targets,
            desired_targets,
        )
        self._previous_requested_targets.copy_(desired_targets)
        self._desired_targets.copy_(delayed_targets)
        max_delta = (
            self.cfg.target_velocity_limit_rad_s
            * self.step_dt
            * self._target_velocity_scale_by_joint
        )
        self._processed_actions = torch.clamp(
            delayed_targets,
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
        frequency_hz, _stride_scale = self._gait_schedule()
        return torch.remainder(
            self._steps_since_reset.float() * self.step_dt * frequency_hz
            + self._gait_phase_offset,
            1.0,
        )

    def _gait_schedule(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-environment clock frequency and stride scale."""

        command_speed = torch.linalg.norm(self._commands[:, :2], dim=1)
        if self.cfg.gait_clock_mode == "fixed":
            frequency_hz = torch.full_like(
                command_speed,
                1.0 / self.cfg.gait_period_s,
            )
            return frequency_hz, torch.ones_like(command_speed)
        if self.cfg.gait_clock_mode != "speed_scaled":
            raise ValueError(
                f"Unsupported gait clock mode: {self.cfg.gait_clock_mode}"
            )
        speed_fraction = torch.clamp(
            (command_speed - self.cfg.gait_speed_min_m_s)
            / (self.cfg.gait_speed_max_m_s - self.cfg.gait_speed_min_m_s),
            min=0.0,
            max=1.0,
        )
        active = command_speed > self.cfg.gait_standstill_deadband_m_s
        frequency_hz = self.cfg.gait_frequency_min_hz + speed_fraction * (
            self.cfg.gait_frequency_max_hz - self.cfg.gait_frequency_min_hz
        )
        frequency_hz = torch.where(
            active,
            frequency_hz,
            torch.zeros_like(frequency_hz),
        )
        stride_scale = self.cfg.gait_stride_scale_min + speed_fraction * (
            1.0 - self.cfg.gait_stride_scale_min
        )
        stride_scale = torch.where(
            active,
            stride_scale,
            torch.zeros_like(stride_scale),
        )
        return frequency_hz, stride_scale

    def _gait_targets(
        self,
        phase: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return deployable smooth joint references and scheduled contacts."""
        if phase is None:
            phase = self._gait_phase()
        _frequency_hz, stride_scale = self._gait_schedule()
        if self._distributed_reference_joint_pos is not None:
            sample_count = self._distributed_reference_joint_pos.shape[0]
            table_index = torch.remainder(
                torch.floor(phase * sample_count).long(),
                sample_count,
            )
            desired_joint_position = self._distributed_reference_joint_pos[
                table_index
            ]
            reference = (
                desired_joint_position - self._robot.data.default_joint_pos.torch
            ) / self._joint_scale
            ramp = torch.clamp(
                self._steps_since_reset.float()
                * self.step_dt
                / self.cfg.gait_start_ramp_s,
                0.0,
                1.0,
            )
            motion_scale = ramp * stride_scale
            reference = reference * motion_scale.unsqueeze(1)
            scheduled_contact = self._distributed_reference_contact[table_index]
            scheduled_contact = torch.where(
                (ramp < 0.5).unsqueeze(1),
                torch.ones_like(scheduled_contact),
                scheduled_contact,
            )
            # This is a normalized *base joint reference*, not an actor action.
            # It may legitimately exceed one action-scale unit before the bounded
            # residual is added.  Deployment uses the exported joint table in the
            # same way, so clipping here would train against a smaller gait than
            # the Raspberry Pi later executes.
            return reference, scheduled_contact
        leg_phase = torch.remainder(
            phase.unsqueeze(1) + self._leg_phase_offsets.unsqueeze(0), 1.0
        )
        duty = self.cfg.gait_duty_factor
        stance_u = torch.clamp(leg_phase / duty, 0.0, 1.0)
        swing_u = torch.clamp((leg_phase - duty) / (1.0 - duty), 0.0, 1.0)
        stance_curve = stance_u * stance_u * (3.0 - 2.0 * stance_u)
        swing_curve = swing_u * swing_u * (3.0 - 2.0 * swing_u)
        gait_stride = self.cfg.gait_stride_m * stride_scale.unsqueeze(1)
        half_stride = 0.5 * gait_stride
        stride_offset = torch.where(
            leg_phase < duty,
            half_stride - gait_stride * stance_curve,
            -half_stride + gait_stride * swing_curve,
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

        nominal_forward = (
            self.cfg.gait_stance_fore_aft_m
            * self._leg_front_sign.unsqueeze(0)
        )
        world_forward = nominal_forward + stride_offset
        default_down = self.cfg.gait_stance_down_m
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

    def _gait_expected_lateral_motion(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return phase-locked body sway displacement and velocity.

        Straight-path scoring removes only this zero-mean reference motion.
        Accumulated drift and any residual lateral velocity remain penalized.
        """

        zeros = torch.zeros(self.num_envs, device=self.device)
        if (
            self._distributed_reference_lateral_shift is None
            or self._distributed_reference_lateral_shift_phase_derivative is None
        ):
            return zeros, zeros
        phase = self._gait_phase()
        frequency_hz, stride_scale = self._gait_schedule()
        sample_count = self._distributed_reference_lateral_shift.shape[0]
        table_index = torch.remainder(
            torch.floor(phase * sample_count).long(),
            sample_count,
        )
        ramp = torch.clamp(
            self._steps_since_reset.float()
            * self.step_dt
            / self.cfg.gait_start_ramp_s,
            0.0,
            1.0,
        )
        motion_scale = ramp * stride_scale
        shift = self._distributed_reference_lateral_shift[table_index]
        expected_displacement = shift * motion_scale
        expected_velocity = (
            self._distributed_reference_lateral_shift_phase_derivative[table_index]
            * frequency_hz
            * motion_scale
        )
        ramp_velocity = torch.where(
            ramp < 1.0,
            stride_scale / max(self.cfg.gait_start_ramp_s, 1.0e-4),
            torch.zeros_like(ramp),
        )
        expected_velocity = expected_velocity + shift * ramp_velocity
        return expected_displacement, expected_velocity

    def _set_episode_path_frame(self, env_ids: torch.Tensor) -> None:
        """Latch the translational command in each episode's start-world frame."""
        start_yaw = self._episode_start_yaw[env_ids]
        start_forward = torch.stack(
            (torch.cos(start_yaw), torch.sin(start_yaw)), dim=1
        )
        start_lateral = torch.stack(
            (-torch.sin(start_yaw), torch.cos(start_yaw)), dim=1
        )
        translation = self._commands[env_ids, :2]
        speed = torch.linalg.norm(translation, dim=1)
        body_direction = translation / torch.clamp(
            speed.unsqueeze(1), min=1.0e-4
        )
        world_direction = (
            body_direction[:, 0:1] * start_forward
            + body_direction[:, 1:2] * start_lateral
        )
        world_direction = torch.where(
            (speed > 1.0e-4).unsqueeze(1),
            world_direction,
            start_forward,
        )
        self._episode_forward_direction_w[env_ids] = world_direction
        self._episode_lateral_direction_w[env_ids] = torch.stack(
            (-world_direction[:, 1], world_direction[:, 0]), dim=1
        )

    def _update_heading_hold_command(self) -> None:
        """Map accumulated heading drift into the existing yaw-rate command."""
        maximum = float(self.cfg.heading_hold_max_correction_rad_s)
        gain = float(self.cfg.heading_hold_kp_s)
        if maximum <= 0.0 or gain <= 0.0:
            return
        heading_error = self._episode_heading_error()
        correction = torch.clamp(
            -gain * heading_error,
            min=-maximum,
            max=maximum,
        )
        self._commands[:, 2] = self._nominal_yaw_command + correction

    def _episode_heading_error(self) -> torch.Tensor:
        """Return wrapped yaw error about the integrated nominal yaw-rate path."""
        current_yaw = _yaw_from_xyzw(self._robot.data.root_quat_w.torch)
        if (
            self.cfg.track_episode_world_path
            or (
                self.cfg.heading_hold_kp_s > 0.0
                and self.cfg.heading_hold_max_correction_rad_s > 0.0
            )
        ):
            desired_yaw = self._episode_start_yaw + self._nominal_yaw_command * (
                self._steps_since_reset.float() * self.step_dt
            )
        else:
            desired_yaw = self._episode_start_yaw
        return torch.atan2(
            torch.sin(current_yaw - desired_yaw),
            torch.cos(current_yaw - desired_yaw),
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._update_heading_hold_command()
        gait_angle = 2.0 * torch.pi * self._gait_phase()
        gait_clock = torch.stack((torch.sin(gait_angle), torch.cos(gait_angle)), dim=1)
        imu_angular_velocity = (
            self._imu_sensor.data.ang_vel_b.torch
            + self._imu_angular_velocity_bias
        )
        projected_gravity = self._robot.data.projected_gravity_b.torch
        gravity_noise_std = float(self.cfg.imu_projected_gravity_noise_std)
        if gravity_noise_std > 0.0:
            active = self._physical_randomization_active.unsqueeze(1)
            noisy_gravity = projected_gravity + gravity_noise_std * torch.randn_like(
                projected_gravity
            )
            noisy_gravity = noisy_gravity / torch.clamp(
                torch.linalg.norm(noisy_gravity, dim=1, keepdim=True),
                min=1.0e-6,
            )
            projected_gravity = torch.where(
                active,
                noisy_gravity,
                projected_gravity,
            )
        linear_acceleration_g = self._imu_sensor.data.lin_acc_b.torch / 9.81
        acceleration_noise_std = float(
            self.cfg.imu_linear_acceleration_noise_std_g
        )
        if acceleration_noise_std > 0.0:
            linear_acceleration_g = linear_acceleration_g + (
                acceleration_noise_std
                * torch.randn_like(linear_acceleration_g)
                * self._physical_randomization_active.float().unsqueeze(1)
            )
        policy_observation = torch.cat(
            (
                self._commands,
                gait_clock,
                imu_angular_velocity,
                projected_gravity,
                linear_acceleration_g,
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
        command_speed = torch.linalg.norm(self._commands[:, :2], dim=1)
        command_direction = self._commands[:, :2] / torch.clamp(
            command_speed.unsqueeze(1), min=1.0e-4
        )
        displacement = self._robot.data.root_pos_w.torch - self._episode_start_position
        if self.cfg.track_episode_world_path:
            world_velocity_xy = self._robot.data.root_lin_vel_w.torch[:, :2]
            commanded_velocity = torch.sum(
                world_velocity_xy * self._episode_forward_direction_w,
                dim=1,
            )
            lateral_velocity = torch.sum(
                world_velocity_xy * self._episode_lateral_direction_w,
                dim=1,
            )
            net_commanded_distance = torch.sum(
                displacement[:, :2] * self._episode_forward_direction_w,
                dim=1,
            )
            net_lateral_displacement = torch.sum(
                displacement[:, :2] * self._episode_lateral_direction_w,
                dim=1,
            )
            forward_error = commanded_velocity - command_speed
        else:
            # Preserve the established V24 body-frame scoring exactly when the
            # episode-world path feature is disabled.
            commanded_velocity = torch.sum(
                linear_velocity[:, :2] * command_direction, dim=1
            )
            lateral_velocity = linear_velocity[:, 1]
            net_commanded_distance = torch.sum(
                displacement[:, :2] * command_direction, dim=1
            )
            lateral_direction = torch.stack(
                (-command_direction[:, 1], command_direction[:, 0]), dim=1
            )
            net_lateral_displacement = torch.sum(
                displacement[:, :2] * lateral_direction, dim=1
            )
            forward_error = linear_velocity[:, 0] - self._commands[:, 0]
        velocity_error_squared = torch.square(forward_error)
        yaw_error = angular_velocity[:, 2] - self._commands[:, 2]
        velocity_tracking = torch.exp(
            -velocity_error_squared / self.cfg.velocity_tracking_sigma_m_s**2
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
        heading_error = self._episode_heading_error()
        if (
            self._applied_expected_lateral_displacement is None
            or self._applied_expected_lateral_velocity is None
        ):
            raise RuntimeError("Applied gait phase was not cached before reward")
        expected_lateral_displacement = (
            self._applied_expected_lateral_displacement
        )
        expected_lateral_velocity = self._applied_expected_lateral_velocity
        lateral_tracking_displacement = (
            net_lateral_displacement - expected_lateral_displacement
        )
        lateral_tracking_velocity = lateral_velocity - expected_lateral_velocity
        gait_frequency_hz, _stride_scale = self._gait_schedule()
        cycle_steps = torch.clamp(
            torch.round(
                1.0
                / (
                    torch.clamp(
                        gait_frequency_hz,
                        min=self.cfg.gait_frequency_min_hz,
                    )
                    * self.step_dt
                )
            ).long(),
            min=1,
            max=self._lateral_cycle_history_steps - 2,
        )
        lateral_history_slot = torch.remainder(
            self._steps_since_reset,
            self._lateral_cycle_history_steps,
        )
        lateral_lookback_slot = torch.remainder(
            lateral_history_slot - cycle_steps,
            self._lateral_cycle_history_steps,
        )
        prior_cycle_displacement = self._lateral_displacement_history[
            self._env_indices,
            lateral_lookback_slot,
        ]
        completed_cycle = self._steps_since_reset >= cycle_steps
        cycle_averaged_lateral_velocity = (
            net_lateral_displacement - prior_cycle_displacement
        ) / (cycle_steps.float() * self.step_dt)
        startup_lateral_velocity = lateral_tracking_displacement / torch.clamp(
            (self._steps_since_reset.float() + 1.0) * self.step_dt,
            min=self.step_dt,
        )
        cycle_averaged_lateral_velocity = torch.where(
            completed_cycle,
            cycle_averaged_lateral_velocity,
            startup_lateral_velocity,
        )
        self._lateral_displacement_history[
            self._env_indices,
            lateral_history_slot,
        ] = net_lateral_displacement
        lateral_corridor_excess = torch.clamp(
            torch.abs(lateral_tracking_displacement)
            - self.cfg.lateral_corridor_half_width_m,
            min=0.0,
        )
        active_translation = (
            command_speed > self.cfg.active_translation_threshold_m_s
        )
        normalized_lateral_velocity = torch.clamp(
            torch.abs(cycle_averaged_lateral_velocity)
            / torch.clamp(
                command_speed,
                min=self.cfg.normalized_lateral_velocity_speed_floor_m_s,
            ),
            min=0.0,
            max=2.0,
        )
        if self.cfg.straight_progress_use_normalized_lateral_velocity:
            straight_lateral_error = (
                normalized_lateral_velocity
                / max(self.cfg.straight_progress_lateral_ratio_sigma, 1.0e-4)
            )
        else:
            straight_lateral_error = (
                cycle_averaged_lateral_velocity
                / max(self.cfg.straight_progress_lateral_sigma_m_s, 1.0e-4)
            )
        straight_alignment = torch.exp(
            -torch.square(straight_lateral_error)
            -torch.square(
                heading_error
                / max(self.cfg.straight_progress_heading_sigma_rad, 1.0e-4)
            )
        )
        if self.cfg.gate_positive_progress_by_straight_alignment:
            straight_gate_floor = min(
                max(float(self.cfg.straight_progress_gate_floor), 0.0), 1.0
            )
            straight_progress_gate = torch.where(
                active_translation,
                straight_gate_floor
                + (1.0 - straight_gate_floor) * straight_alignment,
                torch.ones_like(straight_alignment),
            )
        else:
            straight_progress_gate = torch.ones_like(straight_alignment)
        instant_progress = torch.where(
            active_translation,
            torch.clamp(
                commanded_velocity / torch.clamp(command_speed, min=1.0e-4),
                min=-1.0,
                max=1.0,
            ),
            torch.zeros_like(commanded_velocity),
        )
        straight_aligned_progress = torch.where(
            active_translation,
            torch.clamp(
                commanded_velocity / torch.clamp(command_speed, min=1.0e-4),
                min=0.0,
                max=1.0,
            )
            * straight_alignment,
            torch.zeros_like(commanded_velocity),
        )
        sustained_progress = torch.where(
            active_translation,
            torch.clamp(
                rolling_speed / torch.clamp(command_speed, min=1.0e-4),
                min=-1.0,
                max=1.0,
            ),
            torch.zeros_like(rolling_speed),
        )
        gated_instant_progress = torch.where(
            instant_progress > 0.0,
            instant_progress * straight_progress_gate,
            instant_progress,
        )
        gated_sustained_progress = torch.where(
            sustained_progress > 0.0,
            sustained_progress * straight_progress_gate,
            sustained_progress,
        )
        lateral_corridor_normalized = (
            lateral_corridor_excess
            / max(self.cfg.lateral_corridor_half_width_m, 1.0e-4)
        )
        heading_error_normalized = (
            heading_error
            / max(self.cfg.heading_error_normalizer_rad, 1.0e-4)
        )
        if self.cfg.use_huber_path_costs:
            lateral_corridor_cost = torch.where(
                lateral_corridor_normalized <= 1.0,
                0.5 * torch.square(lateral_corridor_normalized),
                lateral_corridor_normalized - 0.5,
            )
            absolute_heading_error_normalized = torch.abs(
                heading_error_normalized
            )
            heading_error_cost = torch.where(
                absolute_heading_error_normalized <= 1.0,
                0.5 * torch.square(heading_error_normalized),
                absolute_heading_error_normalized - 0.5,
            )
        else:
            lateral_corridor_cost = torch.square(lateral_corridor_normalized)
            heading_error_cost = torch.square(heading_error_normalized)
        if self.cfg.scale_sustained_speed_with_command:
            sustained_speed_threshold = torch.minimum(
                torch.full_like(
                    command_speed,
                    self.cfg.minimum_sustained_speed_m_s,
                ),
                self.cfg.minimum_sustained_speed_fraction * command_speed,
            )
        else:
            sustained_speed_threshold = torch.full_like(
                command_speed,
                self.cfg.minimum_sustained_speed_m_s,
            )
        normalized_stall_deficit = torch.clamp(
            (sustained_speed_threshold - rolling_speed)
            / torch.clamp(sustained_speed_threshold, min=1.0e-4),
            min=0.0,
            max=2.0,
        )
        sustained_stall = rolling_window_ready & active_translation & (
            rolling_speed < sustained_speed_threshold
        )
        normalized_backward_velocity = torch.clamp(
            -commanded_velocity / torch.clamp(command_speed, min=1.0e-4),
            min=0.0,
            max=2.0,
        )
        normalized_overspeed = torch.clamp(
            (commanded_velocity - command_speed)
            / torch.clamp(command_speed, min=1.0e-4),
            min=0.0,
            max=2.0,
        )
        net_commanded_distance = torch.where(
            active_translation,
            net_commanded_distance,
            torch.zeros_like(net_commanded_distance),
        )
        upright_cosine = torch.clamp(
            -self._robot.data.projected_gravity_b.torch[:, 2], 0.0, 1.0
        )
        projected_gravity_xy = self._robot.data.projected_gravity_b.torch[:, :2]
        target_projected_gravity_x = math.sin(self.cfg.target_forward_pitch_rad)
        body_tilt_error = torch.stack(
            (
                projected_gravity_xy[:, 0] - target_projected_gravity_x,
                projected_gravity_xy[:, 1],
            ),
            dim=1,
        )
        normalized_rearward_pitch = torch.clamp(
            (target_projected_gravity_x - projected_gravity_xy[:, 0])
            / self.cfg.rearward_pitch_normalizer_rad,
            min=0.0,
            max=3.0,
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
        target_limiter_gap = torch.mean(
            torch.square(
                (self._desired_targets - self._processed_actions)
                / torch.clamp(
                    self._joint_scale
                    * (
                        self._residual_action_scale
                        if self.cfg.action_mode == "gait_residual"
                        else 1.0
                    ),
                    min=1.0e-4,
                )
            ),
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
        effort_capacity = torch.clamp(
            self._actuator_effort_limit_by_joint,
            min=1.0e-4,
        )
        applied_effort_fraction = torch.abs(
            self._robot.data.applied_torque.torch
        ) / effort_capacity
        effort_soft_limit = float(self.cfg.effort_soft_limit_fraction)
        if not 0.0 <= effort_soft_limit < 1.0:
            raise ValueError("effort_soft_limit_fraction must be in [0, 1)")
        normalized_effort_excess = torch.clamp(
            (applied_effort_fraction - effort_soft_limit)
            / max(1.0 - effort_soft_limit, 1.0e-4),
            min=0.0,
            max=3.0,
        )
        effort_soft_limit_penalty = torch.mean(
            torch.square(normalized_effort_excess), dim=1
        )
        effort_soft_limit_rate = torch.mean(
            (applied_effort_fraction > effort_soft_limit).float(), dim=1
        )
        maximum_applied_effort_fraction = torch.amax(
            applied_effort_fraction, dim=1
        )

        foot_forces = self._foot_forces()
        foot_contact = foot_forces > self._scheduled_release_force_threshold_n
        if (
            self._applied_gait_reference is None
            or self._applied_scheduled_contact is None
            or self._applied_gait_phase is None
        ):
            raise RuntimeError("Applied gait reference was not cached before reward")
        gait_reference = self._applied_gait_reference
        scheduled_contact = self._applied_scheduled_contact
        applied_gait_phase = self._applied_gait_phase
        scheduled_swing = ~scheduled_contact
        stance_count = torch.clamp(scheduled_contact.float().sum(dim=1), min=1.0)
        swing_count = scheduled_swing.float().sum(dim=1)
        stance_quality_width_n = float(self.cfg.scheduled_stance_quality_width_n)
        if not math.isfinite(stance_quality_width_n) or stance_quality_width_n <= 0.0:
            raise ValueError("scheduled_stance_quality_width_n must be positive")
        threshold_centered_stance_quality = torch.sigmoid(
            (foot_forces - self._scheduled_release_force_threshold_n)
            / stance_quality_width_n
        )
        # A raw sigmoid assigns 0.269 quality to a completely unloaded foot when
        # the contact threshold and width are both 1 N.  Normalize the contact
        # half of the sigmoid so force at or below the threshold maps to zero;
        # otherwise a missing anchor retains about 28% of positive progress.
        scheduled_stance_contact_quality = torch.clamp(
            2.0 * threshold_centered_stance_quality - 1.0,
            min=0.0,
            max=1.0,
        )
        if self.cfg.use_soft_scheduled_stance_quality:
            scheduled_stance_score = torch.sum(
                scheduled_stance_contact_quality * scheduled_contact.float(),
                dim=1,
            ) / stance_count
            missing_scheduled_stance = torch.sum(
                (1.0 - scheduled_stance_contact_quality)
                * scheduled_contact.float(),
                dim=1,
            ) / stance_count
        else:
            scheduled_stance_score = (
                (foot_contact & scheduled_contact).float().sum(dim=1)
                / stance_count
            )
            missing_scheduled_stance = (
                ((~foot_contact) & scheduled_contact).float().sum(dim=1)
                / stance_count
            )
        threshold_centered_release_quality = torch.sigmoid(
            (self._scheduled_release_force_threshold_n - foot_forces)
            / self._scheduled_release_shaping_width_n
        )
        scheduled_release_gate_quality = torch.clamp(
            2.0 * threshold_centered_release_quality - 1.0,
            min=0.0,
            max=1.0,
        )
        if self.cfg.use_threshold_centered_release_shaping:
            scheduled_release_quality = (
                threshold_centered_release_quality
                * scheduled_swing.float()
            )
        else:
            scheduled_release_quality = (
                torch.exp(
                    -torch.square(
                        foot_forces
                        / self._scheduled_release_force_sigma_n
                    )
                )
                * scheduled_swing.float()
            )
        scheduled_swing_success = ((~foot_contact) & scheduled_swing).float()
        scheduled_swing_opportunity = scheduled_swing.float()
        scheduled_release_shaping_score = torch.where(
            swing_count > 0.0,
            scheduled_release_quality.sum(dim=1)
            / torch.clamp(swing_count, min=1.0),
            torch.zeros_like(swing_count),
        )
        if self.cfg.gate_progress_by_cycle_four_leg_release:
            # A transfer/settle frame is not a swing success.  This closes the
            # V27 loophole where roughly 45% of a cycle earned full swing reward
            # even if no shoe ever left the floor.
            scheduled_swing_score = torch.where(
                swing_count > 0.0,
                scheduled_swing_success.sum(dim=1)
                / torch.clamp(swing_count, min=1.0),
                torch.zeros_like(swing_count),
            )
        elif self.cfg.use_soft_scheduled_swing_release:
            scheduled_swing_score = torch.where(
                swing_count > 0.0,
                scheduled_release_quality.sum(dim=1)
                / torch.clamp(swing_count, min=1.0),
                torch.ones_like(swing_count),
            )
        else:
            scheduled_swing_score = torch.where(
                swing_count > 0.0,
                scheduled_swing_success.sum(dim=1)
                / torch.clamp(swing_count, min=1.0),
                torch.ones_like(swing_count),
            )

        cycle_wrapped = self._gait_phase_initialized & (
            applied_gait_phase < self._previous_applied_gait_phase
        )
        completed_cycle_all_four_release = torch.all(
            self._cycle_release_pass_by_foot,
            dim=1,
        )
        completed_full_cycle = (
            cycle_wrapped & self._cycle_release_full_cycle_active
        )
        next_last_completed_cycle_release = torch.where(
            completed_full_cycle,
            completed_cycle_all_four_release,
            self._last_completed_cycle_all_four_release,
        )
        cycle_release_pass_before = torch.where(
            cycle_wrapped.unsqueeze(1),
            torch.zeros_like(self._cycle_release_pass_by_foot),
            self._cycle_release_pass_by_foot,
        )
        consecutive_release_steps_before = torch.where(
            cycle_wrapped.unsqueeze(1),
            torch.zeros_like(self._cycle_release_consecutive_steps),
            self._cycle_release_consecutive_steps,
        )
        release_now = (~foot_contact) & scheduled_swing
        consecutive_release_steps = torch.where(
            release_now,
            consecutive_release_steps_before + 1,
            torch.zeros_like(consecutive_release_steps_before),
        )
        qualified_cycle_release = (
            consecutive_release_steps
            >= self._scheduled_release_min_consecutive_steps
        )
        cycle_release_pass = (
            cycle_release_pass_before | qualified_cycle_release
        )
        newly_qualified_cycle_release = (
            cycle_release_pass & ~cycle_release_pass_before
        )
        current_cycle_all_four_release = torch.all(
            cycle_release_pass,
            dim=1,
        )
        current_cycle_all_four_release_before = torch.all(
            cycle_release_pass_before,
            dim=1,
        )
        cycle_four_leg_release_event = (
            current_cycle_all_four_release
            & ~current_cycle_all_four_release_before
        ).float()
        prospective_swing_success = (
            self._episode_scheduled_swing_success_by_foot
            + scheduled_swing_success
        )
        prospective_release_quality = (
            self._episode_scheduled_release_quality_by_foot
            + scheduled_release_quality
        )
        prospective_swing_opportunity = (
            self._episode_scheduled_swing_opportunity_by_foot
            + scheduled_swing_opportunity
        )
        scheduled_release_rate_by_foot = prospective_swing_success / torch.clamp(
            prospective_swing_opportunity,
            min=1.0,
        )
        scheduled_release_quality_by_foot = (
            prospective_release_quality
            / torch.clamp(prospective_swing_opportunity, min=1.0)
        )
        progress_release_rate_by_foot = (
            scheduled_release_quality_by_foot
            if self.cfg.use_soft_scheduled_swing_release
            else scheduled_release_rate_by_foot
        )
        minimum_progress_release_rate = torch.amin(
            progress_release_rate_by_foot,
            dim=1,
        )
        if self.cfg.gate_progress_by_cycle_four_leg_release:
            gate_floor = float(self.cfg.four_leg_progress_gate_floor)
            gate_exponent = float(self.cfg.cycle_progress_gate_exponent)
            if not 0.0 <= gate_floor <= 1.0:
                raise ValueError("four_leg_progress_gate_floor must be in [0, 1]")
            if not math.isfinite(gate_exponent) or gate_exponent <= 0.0:
                raise ValueError("cycle_progress_gate_exponent must be positive")
            # Gate causally from the current cycle.  Carrying the previous
            # cycle's pass forward let an alternating good/bad policy receive
            # full progress credit throughout every bad cycle.  A fourth power
            # retains a small discovery gradient for one-to-three qualified
            # feet but makes the fourth foot decisive.
            qualified_foot_fraction = torch.mean(
                cycle_release_pass.float(), dim=1
            )
            cycle_gate_pass = torch.pow(
                qualified_foot_fraction,
                gate_exponent,
            )
            four_leg_progress_gate = (
                gate_floor + (1.0 - gate_floor) * cycle_gate_pass
            )
        elif self.cfg.gate_progress_by_four_leg_release:
            gate_floor = float(self.cfg.four_leg_progress_gate_floor)
            gate_target = float(self.cfg.four_leg_release_target_rate)
            if not 0.0 <= gate_floor <= 1.0:
                raise ValueError("four_leg_progress_gate_floor must be in [0, 1]")
            if not 0.0 < gate_target <= 1.0:
                raise ValueError("four_leg_release_target_rate must be in (0, 1]")
            four_leg_progress_gate = gate_floor + (1.0 - gate_floor) * torch.clamp(
                minimum_progress_release_rate / gate_target,
                min=0.0,
                max=1.0,
            )
        else:
            four_leg_progress_gate = torch.ones_like(
                minimum_progress_release_rate
            )
        release_activity_by_foot = (
            self._episode_scheduled_release_quality_by_foot
            if self.cfg.use_soft_scheduled_swing_release
            else self._episode_scheduled_swing_success_by_foot
        )
        least_success_count = torch.amin(
            release_activity_by_foot,
            dim=1,
            keepdim=True,
        )
        least_active_foot = (
            release_activity_by_foot
            <= least_success_count + 0.5
        )
        if self.cfg.use_soft_scheduled_swing_release:
            least_active_swing = torch.sum(
                (2.0 * scheduled_release_quality - scheduled_swing.float())
                * least_active_foot.float(),
                dim=1,
            )
        else:
            least_active_swing = torch.sum(
                (
                    scheduled_swing_success
                    - (foot_contact & scheduled_swing).float()
                )
                * least_active_foot.float(),
                dim=1,
            )
        if self.cfg.action_mode == "gait_residual":
            gait_reference_error = torch.mean(torch.square(self._actions), dim=1)
        else:
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
        schedule_matched_support = torch.all(
            foot_contact == scheduled_contact,
            dim=1,
        )
        if self.cfg.require_schedule_matched_support:
            # Match contact identities, not only the number of planted feet.
            # The old count-only check gave full support credit when a scheduled
            # rear swing foot stayed planted and a front anchor unloaded instead.
            three_foot_support = schedule_matched_support.float()
        else:
            three_foot_support = (airborne_count <= 1.0).float()
        if self.cfg.gate_progress_by_schedule_matched_support:
            topology_gate_floor = float(
                self.cfg.schedule_matched_progress_gate_floor
            )
            if not 0.0 <= topology_gate_floor <= 1.0:
                raise ValueError(
                    "schedule_matched_progress_gate_floor must be in [0, 1]"
                )
            if self.cfg.use_soft_schedule_matched_progress_gate:
                desired_contact_quality = torch.where(
                    scheduled_contact,
                    scheduled_stance_contact_quality,
                    scheduled_release_gate_quality,
                )
                weakest_desired_contact_quality = torch.amin(
                    desired_contact_quality,
                    dim=1,
                )
                schedule_matched_progress_gate = (
                    topology_gate_floor
                    + (1.0 - topology_gate_floor)
                    * weakest_desired_contact_quality
                )
            else:
                schedule_matched_progress_gate = torch.where(
                    schedule_matched_support,
                    torch.ones_like(airborne_count),
                    torch.full_like(airborne_count, topology_gate_floor),
                )
        else:
            schedule_matched_progress_gate = torch.ones_like(airborne_count)
        excess_airborne_feet = torch.square(
            torch.clamp(airborne_count - 1.0, min=0.0)
        )
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
                self.cfg.reward_forward_velocity_tracking
                * velocity_tracking
                * straight_progress_gate
                * four_leg_progress_gate
                * schedule_matched_progress_gate
            ),
            "instant_progress": (
                self.cfg.reward_instant_progress
                * gated_instant_progress
                * four_leg_progress_gate
                * schedule_matched_progress_gate
            ),
            "straight_aligned_progress": (
                self.cfg.reward_straight_aligned_progress
                * straight_aligned_progress
                * four_leg_progress_gate
                * schedule_matched_progress_gate
            ),
            "sustained_progress": (
                self.cfg.reward_sustained_progress
                * gated_sustained_progress
                * four_leg_progress_gate
                * schedule_matched_progress_gate
            ),
            "sustained_stall": (
                -self.cfg.penalty_sustained_stall
                * torch.square(normalized_stall_deficit)
                * (rolling_window_ready & active_translation).float()
            ),
            "backward_motion": (
                -self.cfg.penalty_backward_motion
                * torch.square(normalized_backward_velocity)
                * active_translation.float()
            ),
            "overspeed": (
                -self.cfg.penalty_overspeed
                * torch.square(normalized_overspeed)
                * active_translation.float()
            ),
            "rearward_pitch": (
                -self.cfg.penalty_rearward_pitch
                * torch.square(normalized_rearward_pitch)
            ),
            "gait_reference": (
                self.cfg.reward_gait_reference * gait_reference_tracking
            ),
            "scheduled_stance": (
                self.cfg.reward_scheduled_stance * scheduled_stance_score
            ),
            "missing_scheduled_stance": (
                -self.cfg.penalty_missing_scheduled_stance
                * torch.square(missing_scheduled_stance)
            ),
            "scheduled_swing": (
                self.cfg.reward_scheduled_swing * scheduled_swing_score
            ),
            "scheduled_release_shaping": (
                self.cfg.reward_scheduled_release_shaping
                * scheduled_release_shaping_score
            ),
            "cycle_four_leg_release": (
                self.cfg.reward_cycle_four_leg_release
                * cycle_four_leg_release_event
            ),
            "upright": self.cfg.reward_upright * upright_cosine,
            "alive": torch.full_like(upright_cosine, self.cfg.reward_alive),
            "lateral_velocity": (
                -self.cfg.penalty_lateral_velocity
                * torch.square(lateral_tracking_velocity)
            ),
            "normalized_lateral_velocity": (
                -self.cfg.penalty_normalized_lateral_velocity
                * torch.square(normalized_lateral_velocity)
                * active_translation.float()
            ),
            "lateral_displacement": (
                -self.cfg.penalty_lateral_displacement
                * torch.square(lateral_tracking_displacement)
            ),
            "lateral_corridor": (
                -self.cfg.penalty_lateral_corridor
                * lateral_corridor_cost
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
                * torch.sum(torch.square(body_tilt_error), dim=1)
            ),
            "yaw_rate": -self.cfg.penalty_yaw_rate * torch.square(yaw_error),
            "heading_error": (
                -self.cfg.penalty_heading_error
                * heading_error_cost
            ),
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
            "target_limiter_gap": (
                -self.cfg.penalty_target_limiter_gap * target_limiter_gap
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
            "effort_soft_limit": (
                -self.cfg.penalty_effort_soft_limit
                * effort_soft_limit_penalty
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
            "least_active_swing": (
                self.cfg.reward_least_active_swing * least_active_swing
            ),
            "three_foot_support": (
                self.cfg.reward_three_foot_support * three_foot_support
            ),
            "excess_airborne_feet": (
                -self.cfg.penalty_excess_airborne_feet * excess_airborne_feet
            ),
            "termination": -self.cfg.penalty_termination * self._failed.float(),
        }
        reward = (
            torch.stack(tuple(reward_terms.values()), dim=0).sum(dim=0)
            * self.cfg.reward_scale
        )

        self._episode_velocity_error_sum += torch.sqrt(velocity_error_squared)
        self._episode_yaw_error_sum += torch.abs(yaw_error)
        self._episode_heading_error_sum += torch.abs(heading_error)
        self._episode_straight_progress_gate_sum += straight_progress_gate
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
        self._episode_scheduled_swing_success_by_foot += scheduled_swing_success
        self._episode_scheduled_release_quality_by_foot += (
            scheduled_release_quality
        )
        self._episode_scheduled_swing_opportunity_by_foot += (
            scheduled_swing_opportunity
        )
        self._episode_four_leg_progress_gate_sum += four_leg_progress_gate
        self._episode_completed_gait_cycles += completed_full_cycle.float()
        self._episode_all_four_release_cycles += (
            completed_full_cycle & completed_cycle_all_four_release
        ).float()
        self._episode_cycle_release_qualifications_by_foot += (
            newly_qualified_cycle_release.float()
        )
        self._cycle_release_consecutive_steps.copy_(
            consecutive_release_steps
        )
        self._cycle_release_pass_by_foot.copy_(cycle_release_pass)
        self._last_completed_cycle_all_four_release.copy_(
            next_last_completed_cycle_release
        )
        self._cycle_release_full_cycle_active |= cycle_wrapped
        self._previous_applied_gait_phase.copy_(applied_gait_phase)
        self._gait_phase_initialized.fill_(True)
        self._episode_command_speed_sum += command_speed
        self._episode_base_height_sum += base_height
        self._episode_cycle_lateral_speed_sum += torch.abs(
            cycle_averaged_lateral_velocity
        )
        self._episode_stall_step_sum += sustained_stall.float()
        self._episode_joint_acceleration_squared_sum += joint_acceleration_squared
        self._episode_body_linear_acceleration_squared_sum += (
            body_linear_acceleration_squared
        )
        self._episode_body_angular_acceleration_squared_sum += (
            body_angular_acceleration_squared
        )
        self._episode_effort_soft_limit_step_sum += effort_soft_limit_rate
        self._episode_max_applied_effort_fraction = torch.maximum(
            self._episode_max_applied_effort_fraction,
            maximum_applied_effort_fraction,
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
            self._nominal_yaw_command[env_ids] = self._commands[env_ids, 2]
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
            self._nominal_yaw_command[env_ids] = self._commands[env_ids, 2]
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
        self._nominal_yaw_command[env_ids] = self._commands[env_ids, 2]

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
            log["Metrics/mean_heading_error_rad"] = (
                self._episode_heading_error_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/mean_straight_progress_gate"] = (
                self._episode_straight_progress_gate_sum[env_ids]
                / completed_steps
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
            if self.cfg.track_episode_world_path:
                net_forward_displacement = torch.sum(
                    displacement[:, :2]
                    * self._episode_forward_direction_w[env_ids],
                    dim=1,
                )
                net_lateral_displacement = torch.sum(
                    displacement[:, :2]
                    * self._episode_lateral_direction_w[env_ids],
                    dim=1,
                )
            else:
                net_forward_displacement = displacement[:, 0]
                net_lateral_displacement = displacement[:, 1]
            log["Metrics/net_forward_displacement_m"] = (
                net_forward_displacement.mean().item()
            )
            log["Metrics/net_lateral_displacement_m"] = (
                net_lateral_displacement.mean().item()
            )
            log["Metrics/target_speed_m_s"] = (
                self._episode_command_speed_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/mean_base_height_m"] = (
                self._episode_base_height_sum[env_ids] / completed_steps
            ).mean().item()
            log["Metrics/mean_cycle_averaged_lateral_speed_m_s"] = (
                self._episode_cycle_lateral_speed_sum[env_ids]
                / completed_steps
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
            log["Metrics/effort_soft_limit_rate"] = (
                self._episode_effort_soft_limit_step_sum[env_ids]
                / completed_steps
            ).mean().item()
            log["Metrics/maximum_applied_effort_fraction"] = (
                self._episode_max_applied_effort_fraction[env_ids]
                .mean()
                .item()
            )
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
                opportunities = self._episode_scheduled_swing_opportunity_by_foot[
                    env_ids, foot_index
                ]
                successes = self._episode_scheduled_swing_success_by_foot[
                    env_ids, foot_index
                ]
                quality_sum = self._episode_scheduled_release_quality_by_foot[
                    env_ids, foot_index
                ]
                release_rate = torch.where(
                    opportunities > 0.0,
                    successes / torch.clamp(opportunities, min=1.0),
                    torch.zeros_like(successes),
                )
                release_quality = torch.where(
                    opportunities > 0.0,
                    quality_sum / torch.clamp(opportunities, min=1.0),
                    torch.zeros_like(quality_sum),
                )
                log[f"Metrics/scheduled_release_rate_{leg_name}"] = (
                    release_rate.mean().item()
                )
                log[f"Metrics/scheduled_release_quality_{leg_name}"] = (
                    release_quality.mean().item()
                )
                log[f"Metrics/scheduled_swing_opportunities_{leg_name}"] = (
                    opportunities.mean().item()
                )
                log[f"Metrics/cycle_release_qualifications_{leg_name}"] = (
                    self._episode_cycle_release_qualifications_by_foot[
                        env_ids, foot_index
                    ].mean().item()
                )
            all_release_rates = torch.where(
                self._episode_scheduled_swing_opportunity_by_foot[env_ids] > 0.0,
                self._episode_scheduled_swing_success_by_foot[env_ids]
                / torch.clamp(
                    self._episode_scheduled_swing_opportunity_by_foot[env_ids],
                    min=1.0,
                ),
                torch.zeros_like(
                    self._episode_scheduled_swing_success_by_foot[env_ids]
                ),
            )
            log["Metrics/minimum_scheduled_release_rate"] = torch.amin(
                all_release_rates,
                dim=1,
            ).mean().item()
            all_release_qualities = torch.where(
                self._episode_scheduled_swing_opportunity_by_foot[env_ids] > 0.0,
                self._episode_scheduled_release_quality_by_foot[env_ids]
                / torch.clamp(
                    self._episode_scheduled_swing_opportunity_by_foot[env_ids],
                    min=1.0,
                ),
                torch.zeros_like(
                    self._episode_scheduled_release_quality_by_foot[env_ids]
                ),
            )
            log["Metrics/minimum_scheduled_release_quality"] = torch.amin(
                all_release_qualities,
                dim=1,
            ).mean().item()
            log["Metrics/mean_four_leg_progress_gate"] = (
                self._episode_four_leg_progress_gate_sum[env_ids]
                / completed_steps
            ).mean().item()
            completed_gait_cycles = self._episode_completed_gait_cycles[env_ids]
            all_four_release_cycles = self._episode_all_four_release_cycles[
                env_ids
            ]
            log["Metrics/completed_gait_cycles"] = (
                completed_gait_cycles.mean().item()
            )
            log["Metrics/all_four_release_cycles"] = (
                all_four_release_cycles.mean().item()
            )
            log["Metrics/all_four_release_cycle_rate"] = torch.where(
                completed_gait_cycles > 0.0,
                all_four_release_cycles
                / torch.clamp(completed_gait_cycles, min=1.0),
                torch.zeros_like(completed_gait_cycles),
            ).mean().item()
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

        joint_position = (
            self._robot.data.default_joint_pos.torch[env_ids].clone()
            + self._joint_target_bias[env_ids]
        )
        joint_position += self.cfg.reset_joint_position_noise_rad * (
            2.0 * torch.rand_like(joint_position) - 1.0
        )
        joint_velocity = torch.zeros_like(self._robot.data.default_joint_vel.torch[env_ids])
        root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self._terrain.env_origins[env_ids]
        root_pose[:, :2] += self.cfg.reset_xy_jitter_m * (
            2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        )
        roll_pitch_limit = float(self.cfg.reset_roll_pitch_noise_rad)
        yaw_limit = float(self.cfg.reset_yaw_noise_rad)
        if roll_pitch_limit < 0.0 or yaw_limit < 0.0:
            raise ValueError("reset attitude noise limits must be non-negative")
        if roll_pitch_limit > 0.0 or yaw_limit > 0.0:
            attitude_active = self._physical_randomization_active[
                env_ids
            ].float()
            roll_pitch = (
                2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
            ) * roll_pitch_limit * attitude_active.unsqueeze(1)
            yaw = (
                2.0 * torch.rand(len(env_ids), device=self.device) - 1.0
            ) * yaw_limit * attitude_active
            attitude_noise = _quaternion_from_rpy_xyzw(
                roll_pitch[:, 0],
                roll_pitch[:, 1],
                yaw,
            )
            root_pose[:, 3:7] = _multiply_quaternions_xyzw(
                attitude_noise,
                root_pose[:, 3:7],
            )
            root_pose[:, 3:7] = root_pose[:, 3:7] / torch.clamp(
                torch.linalg.norm(root_pose[:, 3:7], dim=1, keepdim=True),
                min=1.0e-6,
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
        self._randomize_episode_perturbations(env_ids)
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._older_actions[env_ids] = 0.0
        self._previous_targets[env_ids] = joint_position
        self._desired_targets[env_ids] = joint_position
        self._previous_requested_targets[env_ids] = joint_position
        self._previous_joint_velocity[env_ids] = 0.0
        self._previous_body_linear_velocity[env_ids] = 0.0
        self._previous_body_angular_velocity[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        if self.cfg.randomize_gait_start_phase_quarters:
            self._gait_phase_offset[env_ids] = torch.randint(
                0,
                4,
                (len(env_ids),),
                device=self.device,
            ).float() * 0.25
        else:
            self._gait_phase_offset[env_ids] = 0.0
        self._previous_applied_gait_phase[env_ids] = self._gait_phase_offset[
            env_ids
        ]
        self._gait_phase_initialized[env_ids] = False
        # A phase-zero reset begins at the true cycle boundary, so its first
        # wrap completes a full four-leg cycle.  Non-zero quarter offsets begin
        # part-way through a cycle and intentionally become eligible only after
        # their first wrap.
        self._cycle_release_full_cycle_active[env_ids] = (
            self._gait_phase_offset[env_ids] == 0.0
        )
        self._cycle_release_consecutive_steps[env_ids] = 0
        self._cycle_release_pass_by_foot[env_ids] = False
        self._last_completed_cycle_all_four_release[env_ids] = False
        self._failed[env_ids] = False
        self._failure_nonfinite[env_ids] = False
        self._failure_low_height[env_ids] = False
        self._failure_tilt[env_ids] = False
        self._failure_out_of_bounds[env_ids] = False
        self._failure_base_contact[env_ids] = False
        self._episode_start_position[env_ids] = root_pose[:, :3]
        self._episode_start_yaw[env_ids] = _yaw_from_xyzw(root_pose[:, 3:7])
        self._set_episode_path_frame(env_ids)
        self._episode_velocity_error_sum[env_ids] = 0.0
        self._episode_yaw_error_sum[env_ids] = 0.0
        self._episode_heading_error_sum[env_ids] = 0.0
        self._episode_straight_progress_gate_sum[env_ids] = 0.0
        self._episode_commanded_distance[env_ids] = 0.0
        self._episode_action_saturation_sum[env_ids] = 0.0
        self._episode_swing_step_sum[env_ids] = 0.0
        self._episode_touchdown_count[env_ids] = 0.0
        self._episode_qualified_touchdown_count[env_ids] = 0.0
        self._episode_qualified_touchdown_by_foot[env_ids] = 0.0
        self._episode_scheduled_swing_success_by_foot[env_ids] = 0.0
        self._episode_scheduled_release_quality_by_foot[env_ids] = 0.0
        self._episode_scheduled_swing_opportunity_by_foot[env_ids] = 0.0
        self._episode_four_leg_progress_gate_sum[env_ids] = 0.0
        self._episode_completed_gait_cycles[env_ids] = 0.0
        self._episode_all_four_release_cycles[env_ids] = 0.0
        self._episode_cycle_release_qualifications_by_foot[env_ids] = 0.0
        self._episode_command_speed_sum[env_ids] = 0.0
        self._episode_base_height_sum[env_ids] = 0.0
        self._rolling_command_speed[env_ids] = 0.0
        self._lateral_displacement_history[env_ids] = 0.0
        self._episode_cycle_lateral_speed_sum[env_ids] = 0.0
        self._episode_min_rolling_speed[env_ids] = torch.inf
        self._episode_stall_step_sum[env_ids] = 0.0
        self._episode_joint_acceleration_squared_sum[env_ids] = 0.0
        self._episode_body_linear_acceleration_squared_sum[env_ids] = 0.0
        self._episode_body_angular_acceleration_squared_sum[env_ids] = 0.0
        self._episode_effort_soft_limit_step_sum[env_ids] = 0.0
        self._episode_max_applied_effort_fraction[env_ids] = 0.0
        for term_sum in self._episode_reward_term_sums.values():
            term_sum[env_ids] = 0.0
