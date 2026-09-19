"""Pure-RL vectorized commanded-walking environment for Drobot."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from itertools import permutations
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
import warp as wp
from isaaclab import cloner
from isaaclab.actuators import ImplicitActuator
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
    PROJECT_ROOT,
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
    DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg,
    DrobotCommandedWalkingForwardEnvCfg,
)

LEG_NAMES = ("front_left", "rear_left", "front_right", "rear_right")
EXPECTED_ACTION_ORDER = tuple(
    f"{leg}_{joint_kind}"
    for joint_kind in ("hip_abduction", "hip_flexion", "knee")
    for leg in LEG_NAMES
)
CRITIC_CONTEXT_SCHEMA = "drobot.critic-causal-context.v1"
# Names index only the appended 32 features (critic indices 58 through 89).
CRITIC_CONTEXT_FEATURE_NAMES = (
    "rolling_speed_squash", "rolling_window_fill", "next_outgoing_speed_squash",
    "episode_lateral_displacement_squash", "heading_error_over_pi", "cycle_lateral_velocity_squash",
    *(f"release_success_count_squash_{leg}" for leg in LEG_NAMES),
    *(f"release_opportunity_count_squash_{leg}" for leg in LEG_NAMES),
    *(f"cycle_release_pass_{leg}" for leg in LEG_NAMES),
    *(f"consecutive_release_fraction_{leg}" for leg in LEG_NAMES),
    "episode_age_count_squash", "until_ramp_count_squash", "ramp_fraction",
    "remaining_ramp_count_squash", "stop_settled", "stop_forward_error_squash",
    "stop_lateral_error_squash", "upcoming_smoothness_scale", "packet_hold_fraction",
    "reward_history_valid",
)


class RearPitchLiftMappingError(RuntimeError):
    """An isolated requested-target construction failed before physics advanced."""

    def __init__(self, reason: str, details: dict):
        super().__init__(f"Rear pitch lift rejected: {reason}")
        self.reason = reason
        self.details = details


def _yaw_from_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return wrapped world yaw for Isaac Lab's Warp ``xyzw`` quaternion."""
    x, y, z, w = quaternion.unbind(dim=1)
    return torch.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _deadband_unit_huber(error: torch.Tensor, deadband: float, normalizer: float) -> torch.Tensor:
    """Dimensionless Huber cost with a small free posture-tolerance band."""
    normalized = torch.clamp(torch.abs(error) - deadband, min=0.0) / normalizer
    return torch.where(
        normalized <= 1.0,
        0.5 * torch.square(normalized),
        normalized - 0.5,
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


def _rotate_vectors_xyzw(
    quaternion: torch.Tensor, vectors: torch.Tensor
) -> torch.Tensor:
    """Rotate broadcastable vectors using Isaac Lab actor-frame xyzw poses."""
    xyz, vectors = torch.broadcast_tensors(quaternion[..., :3], vectors)
    twice_cross = 2.0 * torch.cross(xyz, vectors, dim=-1)
    return (
        vectors
        + quaternion[..., 3:4] * twice_cross
        + torch.cross(xyz, twice_cross, dim=-1)
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
        amplitude = cfg.gait_rear_pitch_lift_m
        marker = cfg.gait_rear_pitch_lift_ablation_task
        if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)) or amplitude not in (0.0, 0.005):
            raise ValueError("Rear pitch lift supports only the selected0 or0.005m amplitudes")
        self._rear_pitch_lift_enabled = amplitude > 0.0
        self._rear_pitch_lift_diagnostics_enabled = marker is not None
        self._rear_pitch_projection_enabled = cfg.gait_rear_pitch_projection_enabled
        if type(self._rear_pitch_projection_enabled) is not bool:
            raise ValueError("Rear-pitch projection flag must be Boolean")
        if self._rear_pitch_projection_enabled:
            if (not self._rear_pitch_lift_diagnostics_enabled
                or cfg.gait_rear_pitch_projection_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"):
                raise ValueError("Common-F5 projection requires both isolated rear-pitch task markers")
        elif cfg.gait_rear_pitch_projection_ablation_task is not None:
            raise ValueError("Disabled common-F5 projection requires a None marker")
        descent = cfg.gait_rear_pitch_descent_fraction
        if type(descent) not in (int, float) or descent not in (0.3, 0.5):
            raise ValueError("Rear descent fraction supports only0.3 or original0.5")
        if descent != 0.5 and (not self._rear_pitch_projection_enabled or amplitude != 0.005):
            raise ValueError("Changed rear descent requires isolated fixed5mm projection")
        self._rear_pitch_descent_fraction = float(descent)
        feedback_gain = cfg.gait_rear_tracking_feedback_gain
        feedback_mask = cfg.gait_rear_tracking_feedback_leg_mask
        if (not isinstance(feedback_mask, (list, tuple)) or len(feedback_mask) != 2
            or any(type(v) not in (int, float) for v in feedback_mask)
            or tuple(feedback_mask) not in ((1., 1.), (0., 1.))
            or (tuple(feedback_mask) != (1., 1.) and feedback_gain != .25)):
            raise ValueError("Rear tracking mask supports both rear legs or isolated RR at gain0.25")
        self._rear_tracking_feedback_leg_mask = tuple(feedback_mask)
        feedback_mode = cfg.gait_rear_tracking_feedback_mode
        if feedback_mode not in ("independent", "fixed_pitch") or (
            feedback_mode == "fixed_pitch" and feedback_gain not in (.25, .5)):
            raise ValueError("Fixed-pitch feedback requires isolated rear gain0.25 or0.5")
        self._rear_tracking_feedback_mode = feedback_mode
        if type(feedback_gain) not in (int, float) or feedback_gain not in (0.0, 0.25, 0.5):
            raise ValueError("Rear tracking feedback supports only zero,gain0.25 or0.5")
        if feedback_gain == .5 and (feedback_mode != "fixed_pitch" or tuple(feedback_mask) != (1.,1.)
            or cfg.penalty_swing_sole_tilt != 4. or cfg.gait_front_stance_tracking_feedback_gain != 0.
            or cfg.normalized_contact_slip_ablation_task is not None
            or cfg.physical_randomization_nominal_fraction != 1.):
            raise ValueError("Gain0.5 requires isolated nominal38 both-rear fixed-pitch controller")
        if feedback_gain:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or cfg.gait_rear_tracking_feedback_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or not self._rear_pitch_projection_enabled or amplitude != .005 or descent != .5
                or cfg.penalty_action_acceleration != 3.
                or cfg.decouple_sustained_progress_from_pair_release is not False
                or cfg.native_effort_telemetry_enabled is not False):
                raise ValueError("Rear tracking feedback requires isolated original-reward nominal F5 task")
        elif cfg.gait_rear_tracking_feedback_ablation_task is not None:
            raise ValueError("Disabled rear feedback requires None marker")
        self._rear_tracking_feedback_gain = float(feedback_gain)
        self._rear_tracking_feedback_last = None
        swing_tilt_weight = cfg.penalty_swing_sole_tilt
        if type(swing_tilt_weight) not in (int, float) or swing_tilt_weight not in (0.,2.,4.):
            raise ValueError("Swing sole tilt supports isolated weight0,2 or4")
        if swing_tilt_weight:
            if (feedback_gain not in (.25,.5) or tuple(feedback_mask) != (1.,1.) or feedback_mode != "fixed_pitch"
                or cfg.swing_sole_tilt_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"):
                raise ValueError("Swing sole tilt requires both-rear fixed-pitch feedback study")
        elif cfg.swing_sole_tilt_ablation_task is not None:
            raise ValueError("Disabled swing sole tilt requires None marker")
        self._swing_sole_tilt_weight = float(swing_tilt_weight)
        self._swing_sole_tilt_last = None
        front_gain = cfg.gait_front_stance_tracking_feedback_gain
        if type(front_gain) not in (int, float) or front_gain not in (0., .25):
            raise ValueError("Front stance study supports only zero or gain0.25")
        if front_gain:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or cfg.gait_front_stance_tracking_feedback_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or swing_tilt_weight != 4. or feedback_gain != .25
                or feedback_mode != "fixed_pitch" or tuple(feedback_mask) != (1., 1.)
                or cfg.physical_randomization_nominal_fraction != 1.0):
                raise ValueError("Front stance correction requires isolated nominal weight4 feedback task")
        elif cfg.gait_front_stance_tracking_feedback_ablation_task is not None:
            raise ValueError("Disabled front stance feedback requires None marker")
        self._front_stance_tracking_feedback_gain = float(front_gain)
        self._front_stance_tracking_feedback_last = None
        slip_weight = cfg.penalty_normalized_contact_slip
        if type(slip_weight) not in (int, float) or slip_weight not in (0., .1):
            raise ValueError("Normalized slip study supports only weight0 or0.1")
        self._normalized_contact_slip_study = cfg.normalized_contact_slip_ablation_task is not None
        if self._normalized_contact_slip_study:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or cfg.normalized_contact_slip_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or swing_tilt_weight != 4. or feedback_gain != .25 or front_gain != 0.
                or feedback_mode != "fixed_pitch" or tuple(feedback_mask) != (1.,1.)
                or cfg.physical_randomization_nominal_fraction != 1.0):
                raise ValueError("Normalized slip study requires unchanged nominal34 controller")
        elif slip_weight:
            raise ValueError("Positive normalized slip requires isolated task marker")
        self._normalized_contact_slip_weight = float(slip_weight)
        self._normalized_contact_slip_last = None
        if self._rear_pitch_lift_enabled and marker is None:
            raise ValueError("Rear pitch lift requires an isolated task marker")
        if self._rear_pitch_lift_diagnostics_enabled:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or marker != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or cfg.action_mode != "gait_residual" or cfg.gait_reference_mode != "phase_table"
                or cfg.command_profile != "run_stop" or cfg.action_history_mode != "applied_target"
                or cfg.action_space != 12 or cfg.observation_space != 50 or cfg.state_space != 58
                or cfg.critic_context_mode != "legacy58" or cfg.native_effort_reward_enabled is not False
                or cfg.gait_phase_reference_linear_interpolation_enabled is not False
                or cfg.gait_phase_table_lateral_reference_enabled is not False
                or cfg.gait_phase_table_forward_reference_enabled is not False
                or cfg.normalized_posture_cost_enabled is not False
                or cfg.normalized_body_height_cost_enabled is not False
                or tuple(cfg.gait_swing_residual_scale_by_leg) != (1.0,)*4
                or tuple(cfg.residual_action_scale_by_action or ()) != (0.06,)*4+(0.5,)*8
                or (cfg.action_scale_abduction_rad,cfg.action_scale_hip_rad,cfg.action_scale_knee_rad) != (0.12,0.30,0.40)
                or tuple(cfg.gait_phase_offsets) != (0.0,0.5,0.5,0.0)
                or cfg.gait_duty_factor != 0.55 or cfg.sole_clearance_transition_fraction != 0.5
                or cfg.gait_start_ramp_s != 1.5 or cfg.gait_standstill_deadband_m_s != 0.004
                or cfg.gait_residual_stop_action_scale != 0.0
                or cfg.gait_phase_reference_scale_about_neutral != 1.0
                or cfg.physical_randomization_nominal_fraction != 1.0):
                raise ValueError("Rear pitch lift requires the selected nominal V44 actor50/critic58 original residual/phase/reward contract")
        from .v44_ablation_contract import validate_sustained_progress_config
        validate_sustained_progress_config(cfg)
        self._decouple_sustained_progress_from_pair_release = cfg.decouple_sustained_progress_from_pair_release
        if (self._decouple_sustained_progress_from_pair_release
            and type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg):
            raise ValueError("Sustained decoupling requires the exact isolated V44 config class")
        self._critic_context_mode = cfg.critic_context_mode
        if not isinstance(self._critic_context_mode, str) or self._critic_context_mode not in ("legacy58", "context90_zero", "context90_causal"):
            raise ValueError("Unknown isolated critic context mode")
        self._critic_context_enabled = self._critic_context_mode != "legacy58"
        if self._critic_context_enabled:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or cfg.critic_context_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or cfg.action_mode != "gait_residual" or cfg.gait_reference_mode != "phase_table"
                or cfg.command_profile != "run_stop" or cfg.action_history_mode != "applied_target"
                or cfg.action_space != 12 or cfg.observation_space != 50 or cfg.state_space != 90
                or cfg.native_effort_telemetry_enabled is not True or cfg.native_effort_reward_enabled is not False):
                raise ValueError("Critic90 requires the exact isolated V44 task, actor50, PD reward and native telemetry")
            if (cfg.track_episode_world_path is not True
                or cfg.gait_phase_table_lateral_reference_enabled is not False
                or cfg.gait_phase_table_forward_reference_enabled is not False
                or cfg.gait_phase_reference_linear_interpolation_enabled is not False
                or cfg.normalized_posture_cost_enabled is not False
                or cfg.normalized_body_height_cost_enabled is not False
                or cfg.gate_progress_by_four_leg_release is not True
                or cfg.gate_progress_by_cycle_four_leg_release is not False
                or cfg.use_soft_scheduled_swing_release is not False
                or cfg.scheduled_release_min_consecutive_steps != 5
                or cfg.target_packet_max_hold_steps != 1
                or cfg.sustained_speed_window_s != 3.0
                or cfg.smoothness_initial_scale != 0.25):
                raise ValueError("Critic causal context v1 requires its unchanged path/release/window/controller semantics")
        elif cfg.critic_context_ablation_task is not None:
            raise ValueError("Legacy58 critic requires a None context ablation marker")
        self._native_effort_telemetry_enabled = cfg.native_effort_telemetry_enabled
        self._native_effort_reward_enabled = cfg.native_effort_reward_enabled
        if (type(self._native_effort_telemetry_enabled) is not bool
            or type(self._native_effort_reward_enabled) is not bool):
            raise ValueError("Native effort telemetry and reward flags must be boolean")
        if self._native_effort_reward_enabled and not self._native_effort_telemetry_enabled:
            raise ValueError("Native effort rewards require fresh native telemetry")
        if self._native_effort_telemetry_enabled:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or cfg.native_effort_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or cfg.action_mode != "gait_residual" or cfg.gait_reference_mode != "phase_table"
                or cfg.command_profile != "run_stop" or cfg.action_history_mode != "applied_target"
                or cfg.action_space != 12 or cfg.observation_space != 50
                or cfg.state_space != (90 if self._critic_context_enabled else 58)):
                raise ValueError("Native effort telemetry requires exact isolated V44 actor50 and legacy58 or strictly marked critic90")
        elif cfg.native_effort_ablation_task is not None:
            raise ValueError("Disabled native effort telemetry requires a None ablation marker")
        self._phase_reference_linear_interpolation_enabled = cfg.gait_phase_reference_linear_interpolation_enabled
        if not isinstance(self._phase_reference_linear_interpolation_enabled, bool):
            raise ValueError("gait_phase_reference_linear_interpolation_enabled must be boolean")
        interpolation_marker = cfg.gait_phase_reference_interpolation_ablation_task
        if self._phase_reference_linear_interpolation_enabled:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or interpolation_marker != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or cfg.action_mode != "gait_residual" or cfg.gait_reference_mode != "phase_table"
                or cfg.command_profile != "run_stop"):
                raise ValueError("Linear phase interpolation requires the exact isolated V44 phase_table/gait_residual/run_stop config")
            if (cfg.gait_phase_table_lateral_reference_enabled is not False
                or cfg.gait_phase_table_forward_reference_enabled is not False):
                raise ValueError("Linear joint interpolation cannot use floor-sampled body-transfer metadata")
            scale = cfg.gait_phase_reference_scale_about_neutral
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale != 1.0:
                raise ValueError("Linear phase interpolation requires scale-about-neutral 1.0")
            if not cfg.gait_phase_reference_path or not cfg.gait_phase_reference_low_path:
                raise ValueError("Linear phase interpolation requires both high and low phase tables")
        elif interpolation_marker is not None:
            raise ValueError("Disabled phase interpolation requires a None ablation marker")
        self._normalized_body_height_cost_enabled = cfg.normalized_body_height_cost_enabled
        if not isinstance(self._normalized_body_height_cost_enabled, bool):
            raise ValueError("normalized_body_height_cost_enabled must be boolean")
        if self._normalized_body_height_cost_enabled:
            if cfg.normalized_posture_cost_enabled:
                raise ValueError("Height-only and combined normalized posture costs are mutually exclusive")
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or cfg.action_mode != "gait_residual" or cfg.gait_reference_mode != "phase_table"
                or cfg.command_profile != "run_stop"):
                raise ValueError("Height-only normalization requires the exact V44 phase_table/gait_residual/run_stop config")
            deadband = cfg.gait_standstill_deadband_m_s
            if (isinstance(deadband, bool) or not isinstance(deadband, (int, float))
                or not math.isfinite(deadband) or deadband < 0.0):
                raise ValueError("Height-only normalization requires a finite nonnegative standstill deadband")
        swing_scales = cfg.gait_swing_residual_scale_by_leg
        if not isinstance(swing_scales, (tuple, list)) or len(swing_scales) != 4:
            raise ValueError("gait_swing_residual_scale_by_leg must contain four FL, RL, FR, RR numbers")
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in swing_scales):
            raise ValueError("gait_swing_residual_scale_by_leg must contain finite numbers within [0, 1]")
        self._swing_residual_scaling_enabled = any(value != 1.0 for value in swing_scales)
        if self._swing_residual_scaling_enabled:
            if (type(cfg) is not DrobotCommandedRunningFlatContactHardwareMatchedStopExternalRearPayloadEnvCfg
                or cfg.gait_swing_residual_ablation_task != "Drobot-Commanded-Run-Flat-Contact-Hardware-Matched-Stop-External-Rear-Payload"
                or cfg.action_mode != "gait_residual" or cfg.gait_reference_mode != "phase_table"):
                raise ValueError("Nonunit swing residual scales require the exact isolated V44 phase_table/gait_residual ablation")
            for name, upper, inclusive in (("gait_duty_factor", 1.0, False),
                                           ("sole_clearance_transition_fraction", 0.5, True)):
                value = getattr(cfg, name)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0.0
                    or (value > upper if inclusive else value >= upper)):
                    raise ValueError(f"Invalid {name} for the C2 swing residual envelope")
            offsets = cfg.gait_phase_offsets
            if (not isinstance(offsets, (tuple, list)) or len(offsets) != 4
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) or not 0.0 <= value < 1.0 for value in offsets)):
                raise ValueError("Swing residual phase offsets must be finite FL, RL, FR, RR values within [0, 1)")
        super().__init__(cfg, render_mode, **kwargs)
        self._gait_swing_residual_scale_by_leg = torch.tensor(swing_scales, dtype=torch.float32, device=self.device)
        self._applied_swing_residual_multiplier = torch.ones(
            (self.num_envs, len(EXPECTED_ACTION_ORDER)), dtype=torch.float32, device=self.device
        )
        self._normalized_posture_cost_enabled = self.cfg.normalized_posture_cost_enabled
        if not isinstance(self._normalized_posture_cost_enabled, bool):
            raise ValueError("normalized_posture_cost_enabled must be boolean")
        # Bounds are also checked by the simulator-free ablation loader. Never
        # permit a zero normalizer, NaN weight, or boolean disguised as a number.
        for name, lower, upper in (
            ("posture_height_deadband_m", 0.0, 0.010),
            ("posture_height_normalizer_m", 0.001, 0.050),
            ("posture_angle_deadband_rad", 0.0, math.radians(5.0)),
            ("posture_angle_normalizer_rad", math.radians(0.5), math.radians(10.0)),
            ("penalty_normalized_body_height", 0.0, 20.0),
            ("penalty_normalized_body_roll", 0.0, 20.0),
            ("penalty_normalized_body_pitch", 0.0, 20.0),
        ):
            value = getattr(self.cfg, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"{name} must be a finite number within [{lower}, {upper}]")
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
        release_discovery_width_n = (
            release_shaping_width_n
            if self.cfg.scheduled_release_discovery_width_n is None
            else float(self.cfg.scheduled_release_discovery_width_n)
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
        if (
            not math.isfinite(release_discovery_width_n)
            or release_discovery_width_n <= 0.0
        ):
            raise ValueError(
                "scheduled_release_discovery_width_n must be finite and positive"
            )
        if release_consecutive_steps <= 0:
            raise ValueError(
                "scheduled_release_min_consecutive_steps must be positive"
            )
        self._scheduled_release_force_threshold_n = release_force_threshold_n
        self._scheduled_release_shaping_width_n = release_shaping_width_n
        self._scheduled_release_discovery_width_n = release_discovery_width_n
        self._scheduled_release_min_consecutive_steps = release_consecutive_steps
        rolling_velocity_tracking_sigma_m_s = float(
            self.cfg.rolling_velocity_tracking_sigma_m_s
        )
        if (
            not math.isfinite(rolling_velocity_tracking_sigma_m_s)
            or rolling_velocity_tracking_sigma_m_s <= 0.0
        ):
            raise ValueError(
                "rolling_velocity_tracking_sigma_m_s must be finite and positive"
            )
        self._rolling_velocity_tracking_sigma_m_s = (
            rolling_velocity_tracking_sigma_m_s
        )
        self._actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._previous_actions = torch.zeros_like(self._actions)
        self._older_actions = torch.zeros_like(self._actions)
        self._previous_targets = self._robot.data.default_joint_pos.torch.clone()
        self._processed_actions = self._previous_targets.clone()
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
        self._command_observation_scale = torch.tensor(
            self.cfg.command_observation_scale,
            dtype=torch.float32,
            device=self.device,
        )
        if self._command_observation_scale.shape != (3,) or not torch.all(
            torch.isfinite(self._command_observation_scale)
        ):
            raise ValueError(
                "command_observation_scale must contain three finite values"
            )
        if torch.any(self._command_observation_scale <= 0.0):
            raise ValueError("command observation scales must be positive")
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
        if self._native_effort_telemetry_enabled:
            self._native_effort_view = self._robot.root_view
            if (int(self._native_effort_view.count) != self.num_envs
                or int(self._native_effort_view.max_dofs) != len(EXPECTED_ACTION_ORDER)
                or tuple(self._native_effort_view.shared_metatype.dof_names) != EXPECTED_ACTION_ORDER
                or self._robot.data._has_reversed_joints is not False):
                raise RuntimeError("Native effort requires identical 12-DOF backend/user order and unreversed joints")
            if not self._robot.actuators or any(
                type(actuator) is not ImplicitActuator for actuator in self._robot.actuators.values()
            ):
                raise RuntimeError("Native effort telemetry is restricted to the current implicit actuators")
            # Copy-owned storage: a later native getter must never mutate a
            # retained reward sample. Reset slots are explicitly invalid, not
            # fictitious zero effort or data belonging to the previous episode.
            self._native_effort_last_nm = torch.full(
                (self.num_envs, len(EXPECTED_ACTION_ORDER)), torch.nan,
                dtype=torch.float32, device=self.device,
            )
            self._native_effort_last_pd_nm = torch.full_like(self._native_effort_last_nm, torch.nan)
            self._native_effort_last_physics_counter = torch.full(
                (self.num_envs,), -1, dtype=torch.int64, device=self.device,
            )
        configured_target_velocity_limits = (
            self.cfg.target_velocity_limit_rad_s_by_action
        )
        if configured_target_velocity_limits is None:
            configured_target_velocity_limits = (
                float(self.cfg.target_velocity_limit_rad_s),
            ) * len(EXPECTED_ACTION_ORDER)
        if len(configured_target_velocity_limits) != len(EXPECTED_ACTION_ORDER):
            raise ValueError(
                "target_velocity_limit_rad_s_by_action must contain one value "
                f"for each action in {EXPECTED_ACTION_ORDER}"
            )
        self._target_velocity_limit_by_joint = torch.tensor(
            configured_target_velocity_limits,
            dtype=torch.float32,
            device=self.device,
        )
        if not torch.all(torch.isfinite(self._target_velocity_limit_by_joint)):
            raise ValueError("per-action target velocity limits must be finite")
        if torch.any(self._target_velocity_limit_by_joint <= 0.0):
            raise ValueError("per-action target velocity limits must be positive")
        self._target_packet_step_limit_rad = float(self.cfg.target_packet_step_limit_rad)
        self._target_packet_hold_probability = float(self.cfg.target_packet_hold_probability)
        self._target_packet_max_hold_steps = int(self.cfg.target_packet_max_hold_steps)
        if (
            not math.isfinite(self._target_packet_step_limit_rad)
            or self._target_packet_step_limit_rad < 0.0
        ):
            raise ValueError("target_packet_step_limit_rad must be finite and nonnegative")
        if not 0.0 <= self._target_packet_hold_probability <= 1.0:
            raise ValueError("target_packet_hold_probability must be within [0, 1]")
        if self._target_packet_max_hold_steps < 1:
            raise ValueError("target_packet_max_hold_steps must be at least one")
        self._target_packet_elapsed_s = torch.zeros(self.num_envs, device=self.device)
        self._target_packet_hold_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._target_packet_held = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._episode_target_packet_hold_sum = torch.zeros(self.num_envs, device=self.device)
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
        configured_stance_depths = self.cfg.gait_stance_down_by_leg_m
        if configured_stance_depths is None:
            configured_stance_depths = (float(self.cfg.gait_stance_down_m),) * 4
        self._leg_stance_down = torch.tensor(
            configured_stance_depths,
            dtype=torch.float32,
            device=self.device,
        )
        if self._leg_stance_down.shape != (4,) or not torch.all(
            torch.isfinite(self._leg_stance_down)
        ):
            raise ValueError(
                "gait_stance_down_by_leg_m must contain four finite FL, RL, FR, RR values"
            )
        if torch.any(self._leg_stance_down <= 0.0):
            raise ValueError("gait stance depths must be positive")
        configured_lifts = self.cfg.gait_lift_by_leg_m
        if configured_lifts is None:
            configured_lifts = (float(self.cfg.gait_lift_m),) * 4
        self._leg_gait_lift = torch.tensor(
            configured_lifts,
            dtype=torch.float32,
            device=self.device,
        )
        if self._leg_gait_lift.shape != (4,) or not torch.all(
            torch.isfinite(self._leg_gait_lift)
        ):
            raise ValueError(
                "gait_lift_by_leg_m must contain four finite FL, RL, FR, RR values"
            )
        if torch.any(self._leg_gait_lift < 0.0):
            raise ValueError("gait lift values must be non-negative")
        swing_translation_start = float(
            self.cfg.gait_swing_translation_start_fraction
        )
        swing_translation_end = float(
            self.cfg.gait_swing_translation_end_fraction
        )
        if not (
            math.isfinite(swing_translation_start)
            and math.isfinite(swing_translation_end)
            and 0.0 <= swing_translation_start < swing_translation_end <= 1.0
        ):
            raise ValueError(
                "gait swing translation fractions must be finite and satisfy "
                "0 <= start < end <= 1"
            )
        self._gait_swing_translation_start = swing_translation_start
        self._gait_swing_translation_span = (
            swing_translation_end - swing_translation_start
        )
        contact_transition_fraction = float(
            self.cfg.gait_contact_transition_fraction
        )
        if not (
            math.isfinite(contact_transition_fraction)
            and 0.0 <= contact_transition_fraction < 0.5
        ):
            raise ValueError(
                "gait_contact_transition_fraction must be finite and within [0, 0.5)"
            )
        self._gait_contact_transition_fraction = contact_transition_fraction
        for name in (
            "gait_standstill_fore_aft_m",
            "gait_standstill_down_m",
            "target_standstill_pitch_rad",
            "target_standstill_base_height_m",
        ):
            value = getattr(self.cfg, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when configured")
        if (
            self.cfg.gait_standstill_fore_aft_m is not None
            and float(self.cfg.gait_standstill_fore_aft_m) < 0.0
        ):
            raise ValueError("gait_standstill_fore_aft_m must be non-negative")
        if (
            self.cfg.gait_standstill_down_m is not None
            and float(self.cfg.gait_standstill_down_m) <= 0.0
        ):
            raise ValueError("gait_standstill_down_m must be positive")
        if not math.isfinite(float(self.cfg.gait_standstill_center_offset_m)):
            raise ValueError("gait_standstill_center_offset_m must be finite")
        self._trot_lateral_shift_m = float(
            self.cfg.gait_diagonal_support_lateral_shift_m
        )
        self._phase_table_lateral_reference_enabled = self.cfg.gait_phase_table_lateral_reference_enabled
        if not isinstance(self._phase_table_lateral_reference_enabled, bool):
            raise ValueError("gait_phase_table_lateral_reference_enabled must be boolean")
        self._phase_table_forward_reference_enabled = self.cfg.gait_phase_table_forward_reference_enabled
        if not isinstance(self._phase_table_forward_reference_enabled, bool):
            raise ValueError("gait_phase_table_forward_reference_enabled must be boolean")
        if (self._phase_table_lateral_reference_enabled or self._phase_table_forward_reference_enabled) and (
            self.cfg.gait_reference_mode != "phase_table"
            or not self.cfg.track_episode_world_path
            or not self.cfg.integrate_gait_phase
            or self.cfg.gait_clock_mode != "speed_scaled"
            or float(self.cfg.gait_phase_reference_scale_about_neutral) != 1.0
            or float(self.cfg.gait_stride_scale_min) != 1.0
            or not math.isclose(self._trot_lateral_shift_m, 0.0, abs_tol=1.0e-12)
        ):
            raise ValueError(
                "Body-transfer reference metadata requires phase_table, episode-world tracking, "
                "integrated speed-scaled phase, unit reference/stride scales, and no additional sinusoidal shift"
            )
        if (
            not math.isfinite(self._trot_lateral_shift_m)
            or abs(self._trot_lateral_shift_m) > 0.040
        ):
            raise ValueError(
                "gait_diagonal_support_lateral_shift_m must be finite and within +/-0.040 m"
            )
        for name in (
            "gait_reference_actual_state_weight",
            "gait_reference_worst_leg_weight",
            "gait_residual_stop_action_scale",
            "integrated_stop_action_scale",
            "integrated_stop_target_residual_scale",
        ):
            value = float(getattr(self.cfg, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        integrated_action_reference_frequency_hz = float(
            self.cfg.integrated_action_reference_frequency_hz
        )
        if (
            not math.isfinite(integrated_action_reference_frequency_hz)
            or integrated_action_reference_frequency_hz <= 0.0
        ):
            raise ValueError(
                "integrated_action_reference_frequency_hz must be finite and positive"
            )
        self._integrated_action_reference_frequency_hz = (
            integrated_action_reference_frequency_hz
        )
        integrated_action_frequency_gain = float(
            self.cfg.integrated_action_frequency_gain
        )
        if (
            not math.isfinite(integrated_action_frequency_gain)
            or not 0.0 <= integrated_action_frequency_gain <= 2.0
        ):
            raise ValueError(
                "integrated_action_frequency_gain must be finite and within [0, 2]"
            )
        abduction_frequency_gain = (
            integrated_action_frequency_gain
            if self.cfg.integrated_action_abduction_frequency_gain is None
            else float(self.cfg.integrated_action_abduction_frequency_gain)
        )
        if (
            not math.isfinite(abduction_frequency_gain)
            or not 0.0 <= abduction_frequency_gain <= 2.0
        ):
            raise ValueError(
                "integrated_action_abduction_frequency_gain must be finite and "
                "within [0, 2] when configured"
            )
        self._integrated_action_frequency_gain_by_joint = torch.tensor(
            (abduction_frequency_gain,) * 4
            + (integrated_action_frequency_gain,) * 8,
            dtype=torch.float32,
            device=self.device,
        )
        self._leg_front_sign = torch.tensor(
            (1.0, -1.0, 1.0, -1.0), dtype=torch.float32, device=self.device
        )
        self._leg_side_sign = torch.tensor(
            (1.0, 1.0, -1.0, -1.0), dtype=torch.float32, device=self.device
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
        self._distributed_reference_low_joint_pos: torch.Tensor | None = None
        self._phase_reference_blend_speed_min_m_s = 0.0
        self._phase_reference_blend_speed_span_m_s = 1.0
        self._distributed_reference_contact: torch.Tensor | None = None
        self._distributed_reference_lateral_shift: torch.Tensor | None = None
        self._distributed_reference_lateral_shift_phase_derivative: (
            torch.Tensor | None
        ) = None
        self._phase_table_forward_shift: torch.Tensor | None = None
        self._phase_table_forward_shift_phase_derivative: torch.Tensor | None = None
        self._stance_forward_bias_m = 0.0
        # DirectRLEnv increments the episode step before asking for reward.  Cache
        # the exact phase/reference applied to physics so reward and diagnostics
        # do not accidentally score the following 60 Hz gait tick.
        self._applied_gait_reference: torch.Tensor | None = None
        self._applied_scheduled_contact: torch.Tensor | None = None
        self._applied_gait_phase: torch.Tensor | None = None
        self._applied_expected_lateral_displacement: torch.Tensor | None = None
        self._applied_expected_lateral_velocity: torch.Tensor | None = None
        self._applied_expected_forward_displacement: torch.Tensor | None = None
        self._applied_expected_forward_velocity: torch.Tensor | None = None
        if self.cfg.gait_reference_mode in (
            "distributed_push",
            "smooth_distributed_push",
        ):
            self._build_distributed_reference_table()
        elif self.cfg.gait_reference_mode == "phase_table":
            self._load_phase_reference_table()
        elif self.cfg.gait_reference_mode != "continuous":
            raise RuntimeError(
                f"Unknown gait_reference_mode: {self.cfg.gait_reference_mode}"
            )
        self._randomize_rear_payload_estimate()
        self._randomize_actuator_capacity()
        self._initialize_actuator_target_response()
        if self._rear_pitch_lift_diagnostics_enabled:
            self._initialize_rear_pitch_lift()
        self._initialize_sole_diagnostics()
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
        self._gait_phase_accumulator = torch.zeros_like(self._gait_phase_offset)
        self._run_target_speed = torch.zeros(self.num_envs, device=self.device)
        self._run_stop_start_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._run_stop_ramp_steps = torch.zeros_like(self._run_stop_start_step)
        self._run_stop_started = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._run_stop_schedule_active = torch.zeros_like(self._run_stop_started)
        self._run_stop_settled = torch.zeros_like(self._run_stop_started)
        self._run_stop_origin_w = torch.zeros(
            (self.num_envs, 2), device=self.device
        )
        self._run_stop_target_w = torch.zeros_like(self._run_stop_origin_w)
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
        self._episode_scheduled_pair_progress_gate_sum = torch.zeros(
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
        # Diagnostic denominator only: count the same ready/active samples as
        # the stall numerator instead of inferring them from episode age.
        self._episode_stall_opportunity_sum = torch.zeros(self.num_envs, device=self.device)
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
        if self._native_effort_telemetry_enabled:
            self._episode_native_effort_squared_sum = torch.zeros(self.num_envs, device=self.device)
            self._episode_native_effort_soft_limit_step_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_stop_hold_step_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_stop_speed_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_stop_position_error_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_stop_all_feet_contact_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_stop_stable_step_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        reward_term_names = (
            "forward_velocity_tracking",
            "rolling_velocity_tracking",
            "rolling_velocity_error",
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
            "scheduled_pair_release",
            "scheduled_pair_mismatch",
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
            "abduction_action_magnitude",
            "abduction_target_magnitude",
            "diagonal_sync",
            "action_magnitude",
            "joint_velocity",
            "joint_acceleration",
            "body_linear_acceleration",
            "body_angular_acceleration",
            "effort_soft_limit",
            "effort_rms",
            "actuator_torque_speed_envelope",
            "leg_operating_point_imbalance",
            "support_foot_slip",
            "stance_sole_tilt",
            "swing_corner_clearance",
            "contact_point_slip",
            "touchdown_impact",
            "qualified_touchdown",
            "least_active_swing",
            "three_foot_support",
            "excess_airborne_feet",
            "termination",
            "straight_aligned_progress",
            "stop_velocity",
            "stop_all_feet_contact",
            "stop_weakest_foot_contact",
            "stop_neutral_action",
            "stop_action_magnitude",
            "stop_target_magnitude",
            "stop_position",
            "stop_yaw_rate",
            "stop_joint_velocity",
        )
        if self._swing_sole_tilt_weight:
            reward_term_names += ("swing_sole_tilt",)
        if self._normalized_contact_slip_weight:
            reward_term_names += ("normalized_contact_slip",)
        self._episode_reward_term_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in reward_term_names
        }
        if self._critic_context_enabled:
            if self._sustained_window_steps != 180 or not math.isclose(self.step_dt, 1.0 / 60.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Critic causal context v1 requires the unchanged 60 Hz / 180-step window")
            self._critic_context_foot_indices = torch.tensor(
                [self._foot_leg_names.index(name) for name in LEG_NAMES],
                dtype=torch.long, device=self.device,
            )
            # Raw reward quantities: mean, actual denominator, next outgoing
            # ring sample, cycle-averaged lateral velocity. No extra queue.
            self._critic_context_history_cache = torch.zeros((self.num_envs, 4), device=self.device)
            self._critic_context_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._critic_context_history_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
            self._critic_context_last_raw = torch.zeros((self.num_envs, 32), device=self.device)
            self._critic_context_last_observation_step = torch.full_like(self._critic_context_history_step, -1)

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

    def _load_phase_reference_table(self) -> None:
        """Load a generated periodic joint trajectory and diagonal contact map."""

        configured_path = self.cfg.gait_phase_reference_path
        if not configured_path:
            raise ValueError("phase_table mode requires gait_phase_reference_path")
        scale = float(self.cfg.gait_phase_reference_scale_about_neutral)
        if not math.isfinite(scale) or not 0.25 <= scale <= 2.0:
            raise ValueError("phase-reference scale must be in [0.25, 2.0]")
        neutral = self._robot.data.default_joint_pos.torch[0]
        limits = self._robot.data.soft_joint_pos_limits.torch[0]

        def load_joint_positions(reference_path: str) -> tuple[
            torch.Tensor, torch.Tensor | None, torch.Tensor | None,
            torch.Tensor | None, torch.Tensor | None,
        ]:
            path = Path(reference_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"phase reference does not exist: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "drobot.phase-joint-reference.v1":
                raise ValueError(f"unsupported phase-reference schema: {path}")
            if tuple(payload.get("joint_order", ())) != tuple(
                self._robot.joint_names
            ):
                raise ValueError("phase-reference joint order does not match the robot")
            rows = payload.get("joint_position_rad")
            if not isinstance(rows, list) or len(rows) < 32:
                raise ValueError("phase reference must contain at least 32 samples")
            if self._phase_reference_linear_interpolation_enabled and (
                len(rows) != 512 or type(payload.get("sample_count")) is not int
                or payload["sample_count"] != 512
            ):
                raise ValueError("The isolated linear interpolation comparison requires exactly 512 high/low rows")
            positions = torch.tensor(rows, dtype=torch.float32, device=self.device)
            if positions.shape != (len(rows), len(self._robot.joint_names)):
                raise ValueError("phase-reference rows have the wrong joint count")
            if not torch.all(torch.isfinite(positions)):
                raise ValueError("phase reference contains non-finite joint targets")
            positions = neutral + (positions - neutral) * scale
            if torch.any(positions < limits[:, 0]) or torch.any(
                positions > limits[:, 1]
            ):
                raise ValueError("scaled phase reference exceeds a soft joint limit")
            lateral, lateral_derivative = None, None
            if self._phase_table_lateral_reference_enabled:
                if len(rows) != 1024 or payload.get("sample_count") != 1024:
                    raise ValueError("Opt-in lateral references require exactly 1024 joint/metadata rows")
                generation = payload.get("generation", {})
                if not isinstance(generation, dict):
                    raise ValueError("Lateral reference generation metadata must be an object")
                contract = generation.get("lateral_reference_contract", {})
                expected_contract = {
                    "schema": "drobot.phase-body-lateral-reference.v1",
                    "positive_direction": "body translation along episode-left (+Y at initial heading), NOT foot offset",
                    "derivative_units": "metres per unit global cycle phase; multiply by actual phase rate for m/s",
                }
                if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected_contract.items()):
                    raise ValueError("Missing or incompatible phase-table lateral reference sign/units contract")
                vectors = []
                for key, bound in (("body_lateral_shift_m", 0.040), ("body_lateral_shift_derivative_m_per_cycle", 3.0)):
                    values = payload.get(key)
                    if not isinstance(values, list) or len(values) != 1024 or any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or abs(value) > bound + 1.0e-9
                        for value in values
                    ):
                        raise ValueError(f"{key} requires 1024 finite numeric values within +/-{bound}")
                    vectors.append(torch.tensor(values, dtype=torch.float32, device=self.device))
                lateral, lateral_derivative = vectors
                if abs(float(lateral.mean().item())) > 1.0e-6:
                    raise ValueError("Planned lateral reference must be periodic zero-mean, not a drift offset")
                # Reject wrong derivative units/sign while allowing the small
                # central-difference error of a sampled C2 transfer waveform.
                numerical_derivative = 512.0 * (torch.roll(lateral, -1) - torch.roll(lateral, 1))
                if torch.any(torch.abs(numerical_derivative - lateral_derivative) > 0.010):
                    raise ValueError("Lateral derivative does not agree with the periodic displacement table")
            forward, forward_derivative = None, None
            if self._phase_table_forward_reference_enabled:
                if len(rows) != 1024 or payload.get("sample_count") != 1024:
                    raise ValueError("Opt-in forward references require exactly 1024 joint/metadata rows")
                generation = payload.get("generation", {})
                if not isinstance(generation, dict):
                    raise ValueError("Forward reference generation metadata must be an object")
                contract = generation.get("forward_reference_contract", {})
                expected_contract = {
                    "schema": "drobot.phase-body-forward-reference.v1",
                    "positive_direction": "body translation along episode-forward (+X at initial heading), NOT foot offset",
                    "derivative_units": "metres per unit global cycle phase; multiply by actual phase rate for m/s",
                }
                if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected_contract.items()):
                    raise ValueError("Missing or incompatible phase-table forward reference sign/units contract")
                vectors = []
                for key, bound in (("body_forward_shift_m", 0.040), ("body_forward_shift_derivative_m_per_cycle", 3.0)):
                    values = payload.get(key)
                    if not isinstance(values, list) or len(values) != 1024 or any(
                        isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or abs(value) > bound + 1.0e-9
                        for value in values
                    ):
                        raise ValueError(f"{key} requires 1024 finite numeric values within +/-{bound}")
                    vectors.append(torch.tensor(values, dtype=torch.float32, device=self.device))
                forward, forward_derivative = vectors
                # A bounded periodic forward bias may have nonzero cycle mean;
                # it is never integrated into commanded travel or progress.
                numerical_derivative = 512.0 * (torch.roll(forward, -1) - torch.roll(forward, 1))
                if torch.any(torch.abs(numerical_derivative - forward_derivative) > 0.010):
                    raise ValueError("Forward derivative does not agree with the periodic displacement table")
            return positions, lateral, lateral_derivative, forward, forward_derivative

        joint_positions, lateral_shift, lateral_derivative, forward_shift, forward_derivative = load_joint_positions(configured_path)
        configured_low_path = self.cfg.gait_phase_reference_low_path
        if configured_low_path:
            low_joint_positions, low_lateral_shift, low_lateral_derivative, low_forward_shift, low_forward_derivative = load_joint_positions(configured_low_path)
            if low_joint_positions.shape != joint_positions.shape:
                raise ValueError(
                    "low and high phase references must have matching sample counts"
                )
            if self._phase_table_lateral_reference_enabled and (
                not torch.allclose(lateral_shift, low_lateral_shift, rtol=0.0, atol=1.0e-7)
                or not torch.allclose(lateral_derivative, low_lateral_derivative, rtol=0.0, atol=1.0e-7)
            ):
                raise ValueError("Low/high lateral arrays must agree; speed blending may not change planned sway")
            if self._phase_table_forward_reference_enabled and (
                not torch.allclose(forward_shift, low_forward_shift, rtol=0.0, atol=1.0e-7)
                or not torch.allclose(forward_derivative, low_forward_derivative, rtol=0.0, atol=1.0e-7)
            ):
                raise ValueError("Low/high forward arrays must agree; speed blending may not change planned transfer")
            blend_min = float(self.cfg.gait_phase_reference_blend_speed_min_m_s)
            blend_max = float(self.cfg.gait_phase_reference_blend_speed_max_m_s)
            if not (
                math.isfinite(blend_min)
                and math.isfinite(blend_max)
                and 0.0 <= blend_min < blend_max
            ):
                raise ValueError(
                    "phase-reference blend speeds must be finite and ordered"
                )
            self._distributed_reference_low_joint_pos = low_joint_positions
            self._phase_reference_blend_speed_min_m_s = blend_min
            self._phase_reference_blend_speed_span_m_s = blend_max - blend_min

        sample_count = joint_positions.shape[0]
        phase = torch.arange(
            sample_count, dtype=torch.float32, device=self.device
        ) / sample_count
        leg_phase = torch.remainder(
            phase.unsqueeze(1) + self._foot_phase_offsets.unsqueeze(0), 1.0
        )
        scheduled_contact = leg_phase < float(self.cfg.gait_duty_factor)
        transition = float(self.cfg.gait_contact_transition_fraction)
        if transition > 0.0:
            swing_u = torch.clamp(
                (leg_phase - float(self.cfg.gait_duty_factor))
                / (1.0 - float(self.cfg.gait_duty_factor)),
                0.0,
                1.0,
            )
            scheduled_contact |= (swing_u < transition) | (
                swing_u > 1.0 - transition
            )

        self._distributed_reference_joint_pos = joint_positions
        self._distributed_reference_contact = scheduled_contact
        self._phase_table_forward_shift = forward_shift
        self._phase_table_forward_shift_phase_derivative = forward_derivative
        if self._phase_table_lateral_reference_enabled:
            self._distributed_reference_lateral_shift = lateral_shift
            self._distributed_reference_lateral_shift_phase_derivative = lateral_derivative
        else:
            self._distributed_reference_lateral_shift = torch.zeros(
                sample_count, dtype=torch.float32, device=self.device
            )
            self._distributed_reference_lateral_shift_phase_derivative = torch.zeros_like(
                self._distributed_reference_lateral_shift
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

        self._rear_payload_com_jitter = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device,
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
        self._rear_payload_com_jitter.copy_(com_jitter[:, 0, :])
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
        nominal_stiffness = float(self.cfg.actuator_nominal_stiffness_scale)
        nominal_damping = float(self.cfg.actuator_nominal_damping_scale)
        if not all(math.isfinite(value) and value > 0.0 for value in (nominal_stiffness, nominal_damping)):
            raise ValueError("Nominal drive gain scales must be finite and positive")
        self._actuator_stiffness_scale_by_joint *= nominal_stiffness
        self._actuator_damping_scale_by_joint *= nominal_damping
        if torch.any(self._actuator_stiffness_scale_by_joint <= 0.0):
            raise ValueError("individual actuator stiffness scales must be positive")
        if torch.any(self._actuator_damping_scale_by_joint <= 0.0):
            raise ValueError("individual actuator damping scales must be positive")
        self._actuator_drive_stiffness_by_joint = (
            self._robot.data.default_joint_stiffness.torch
            * self._actuator_stiffness_scale_by_joint
        )
        self._actuator_drive_damping_by_joint = (
            self._robot.data.default_joint_damping.torch
            * self._actuator_damping_scale_by_joint
        )
        if tuple(stiffness_range) != (1.0, 1.0) or nominal_stiffness != 1.0:
            self._robot.write_joint_stiffness_to_sim_index(
                stiffness=self._actuator_drive_stiffness_by_joint
            )
        if tuple(damping_range) != (1.0, 1.0) or nominal_damping != 1.0:
            self._robot.write_joint_damping_to_sim_index(
                damping=self._actuator_drive_damping_by_joint
            )
        if self.cfg.actuator_sync_drive_gain_telemetry:
            for actuator in self._robot.actuators.values():
                joint_ids = actuator.joint_indices
                if joint_ids is None:
                    joint_ids = slice(None)
                actuator.stiffness.copy_(self._actuator_drive_stiffness_by_joint[:, joint_ids])
                actuator.damping.copy_(self._actuator_drive_damping_by_joint[:, joint_ids])

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

    def _initialize_actuator_target_response(self) -> None:
        """Opt-in uncertainty model after the delivered controller target.

        Observed total tracking lag includes physics, loading and feedback age.
        These small response constants are additional hypotheses, not a replay
        of the entire measured delay. No sensor timestamps or observations are
        delayed here, and the existing torque/stiffness plant still runs.
        """
        self._actuator_target_response_enabled = bool(
            self.cfg.actuator_target_response_enabled
        )
        self._actuator_response_targets = self._previous_targets.clone()
        self._actuator_response_tau_s = torch.zeros_like(self._previous_targets)
        self._actuator_response_gain = torch.ones_like(self._previous_targets)
        self._episode_actuator_response_gap_squared_sum = torch.zeros(
            self.num_envs, device=self.device
        )
        if not self._actuator_target_response_enabled:
            return
        nominal_tau = float(self.cfg.actuator_target_response_nominal_tau_s)
        nominal_gain = float(self.cfg.actuator_target_response_nominal_gain)
        if not math.isfinite(nominal_tau) or nominal_tau < 0.0:
            raise ValueError("nominal actuator response tau must be finite and nonnegative")
        if not math.isfinite(nominal_gain) or not 0.0 < nominal_gain <= 1.0:
            raise ValueError("nominal actuator response gain must be within (0, 1]")
        self._actuator_response_nominal_tau_s = nominal_tau
        self._actuator_response_nominal_gain = nominal_gain
        for kind in ("tau_s", "gain"):
            common_range = getattr(self.cfg, f"actuator_target_response_{kind}_range")
            ranges = getattr(self.cfg, f"actuator_target_response_{kind}_range_by_action")
            if ranges is None:
                ranges = (tuple(common_range),) * len(EXPECTED_ACTION_ORDER)
            bounds = torch.tensor(ranges, dtype=torch.float32, device=self.device)
            if bounds.shape != (len(EXPECTED_ACTION_ORDER), 2) or not torch.all(
                torch.isfinite(bounds)
            ):
                raise ValueError(f"actuator response {kind} ranges need twelve finite (low, high) pairs")
            if torch.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError(f"actuator response {kind} ranges must be ordered")
            if kind == "tau_s" and torch.any(bounds < 0.0):
                raise ValueError("actuator response tau ranges must be nonnegative")
            if kind == "gain" and (torch.any(bounds <= 0.0) or torch.any(bounds > 1.0)):
                raise ValueError("actuator response gain ranges must be within (0, 1]")
            setattr(self, f"_actuator_response_{kind}_bounds", bounds)
        self._sample_actuator_target_response(torch.arange(self.num_envs, device=self.device))

    def _sample_actuator_target_response(self, env_ids: torch.Tensor) -> None:
        if not self._actuator_target_response_enabled:
            return
        active = self._physical_randomization_active[env_ids].unsqueeze(1)
        for kind, nominal in (
            ("tau_s", self._actuator_response_nominal_tau_s),
            ("gain", self._actuator_response_nominal_gain),
        ):
            bounds = getattr(self, f"_actuator_response_{kind}_bounds")
            draws = torch.rand((len(env_ids), len(EXPECTED_ACTION_ORDER)), device=self.device)
            values = bounds[:, 0] + draws * (bounds[:, 1] - bounds[:, 0])
            getattr(self, f"_actuator_response_{kind}")[env_ids] = torch.where(
                active, values, torch.full_like(values, nominal)
            )

    def _initialize_sole_diagnostics(self) -> None:
        """Allocate optional world-frame sole metrics; legacy tasks do no pose work."""
        fraction = self.cfg.swing_clearance_worst_leg_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("swing_clearance_worst_leg_fraction must be a finite number within [0, 1]")
        self._swing_clearance_worst_leg_fraction = float(fraction)
        weights = (
            float(self.cfg.penalty_stance_sole_tilt),
            float(self.cfg.penalty_swing_corner_clearance),
            float(self.cfg.penalty_contact_point_slip),
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("sole penalty weights must be finite and nonnegative")
        self._sole_diagnostics_enabled = any(value > 0.0 for value in weights)
        self._episode_stance_sole_tilt_sum = torch.zeros((self.num_envs, 4), device=self.device)
        self._episode_stance_sole_opportunities = torch.zeros_like(self._episode_stance_sole_tilt_sum)
        self._episode_swing_corner_height_sum = torch.zeros_like(self._episode_stance_sole_tilt_sum)
        self._episode_swing_corner_min_height = torch.full_like(self._episode_stance_sole_tilt_sum, torch.inf)
        self._episode_swing_corner_opportunities = torch.zeros_like(self._episode_stance_sole_tilt_sum)
        self._episode_contact_point_speed_squared_sum = torch.zeros_like(self._episode_stance_sole_tilt_sum)
        self._episode_contact_point_opportunities = torch.zeros_like(self._episode_stance_sole_tilt_sum)
        if not self._sole_diagnostics_enabled:
            return
        if self.cfg.terrain.terrain_type != "plane":
            raise ValueError("world sole clearance currently requires plane terrain")
        self._sole_clearance_target_m = float(self.cfg.sole_clearance_target_m)
        self._sole_clearance_transition_fraction = float(self.cfg.sole_clearance_transition_fraction)
        self._sole_corner_soft_contact_height_m = float(self.cfg.sole_corner_soft_contact_height_m)
        if not math.isfinite(self._sole_clearance_target_m) or self._sole_clearance_target_m <= 0.0:
            raise ValueError("sole_clearance_target_m must be finite and positive")
        if not 0.0 < self._sole_clearance_transition_fraction <= 0.5:
            raise ValueError("sole clearance transition fraction must be within (0, 0.5]")
        if not math.isfinite(self._sole_corner_soft_contact_height_m) or self._sole_corner_soft_contact_height_m <= 0.0:
            raise ValueError("sole corner soft-contact height must be finite and positive")
        # Align physical link tensors to the contact sensor's leg order explicitly.
        self._sole_body_ids = []
        for leg_name in self._foot_leg_names:
            matching = [
                body_id for body_id, body_name in zip(self._foot_body_ids, self._foot_body_names)
                if leg_name in body_name
            ]
            if len(matching) != 1:
                raise ValueError(f"Could not uniquely map sole body for {leg_name}")
            self._sole_body_ids.append(matching[0])
        sole_bottom = DISTAL_LINK_LENGTH_M + RECTANGULAR_SHOE_SOLE_BACK_FROM_FORK_M + RECTANGULAR_SHOE_SOLE_THICKNESS_M
        # Include both the wider structural bottom and the lower, smaller tread.
        # Either may contain the lowest corner when the shoe tilts.
        self._sole_local_corners = torch.tensor(
            [
                (x, y_sign * length / 2.0, z_sign * width / 2.0)
                for x, length, width in (
                    (sole_bottom, RECTANGULAR_SHOE_LENGTH_FORE_AFT_M, RECTANGULAR_SHOE_WIDTH_LATERAL_M),
                    (sole_bottom + RECTANGULAR_SHOE_TREAD_PROJECTION_M, RECTANGULAR_SHOE_TREAD_LENGTH_M, RECTANGULAR_SHOE_TREAD_WIDTH_M),
                )
                for y_sign in (-1.0, 1.0)
                for z_sign in (-1.0, 1.0)
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _sole_contact_costs(
        self,
        scheduled_contact: torch.Tensor,
        foot_contact: torch.Tensor,
        applied_gait_phase: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._sole_diagnostics_enabled:
            zero = torch.zeros(self.num_envs, device=self.device)
            return zero, zero, zero
        data = self._robot.data
        # Use actor/link-origin poses AND velocities. body_lin_vel_w is a COM
        # alias in this Isaac Lab version and cannot be paired with link offsets.
        position = data.body_link_pos_w.torch[:, self._sole_body_ids]
        quaternion = data.body_link_quat_w.torch[:, self._sole_body_ids]
        linear_velocity = data.body_link_lin_vel_w.torch[:, self._sole_body_ids]
        angular_velocity = data.body_link_ang_vel_w.torch[:, self._sole_body_ids]
        offset_w = _rotate_vectors_xyzw(
            quaternion.unsqueeze(2), self._sole_local_corners.unsqueeze(0).unsqueeze(0)
        )
        corners_w = position.unsqueeze(2) + offset_w
        heights = corners_w[..., 2] - self._terrain.env_origins[:, None, None, 2]
        minimum_height = torch.amin(heights, dim=2)
        corner_velocity = linear_velocity.unsqueeze(2) + torch.cross(
            angular_velocity.unsqueeze(2).expand_as(offset_w), offset_w, dim=-1
        )
        down_normal = _rotate_vectors_xyzw(
            quaternion,
            torch.tensor((1.0, 0.0, 0.0), device=self.device),
        )
        upright_cosine = torch.clamp(-down_normal[..., 2], -1.0, 1.0)
        # Score scheduled support and unexpected physical stance; unloading a
        # badly pitched planned support foot must not erase its orientation cost.
        stance = scheduled_contact | foot_contact
        stance_count = torch.clamp(stance.float().sum(dim=1), min=1.0)
        stance_tilt = torch.sum((1.0 - upright_cosine) * stance.float(), dim=1) / stance_count

        leg_phase = torch.remainder(applied_gait_phase.unsqueeze(1) + self._foot_phase_offsets, 1.0)
        duty = float(self.cfg.gait_duty_factor)
        swing_u = torch.clamp((leg_phase - duty) / (1.0 - duty), 0.0, 1.0)
        transition = self._sole_clearance_transition_fraction
        enter = torch.clamp(swing_u / transition, 0.0, 1.0)
        leave = torch.clamp((1.0 - swing_u) / transition, 0.0, 1.0)
        # Match the reference's C2 lift envelope, including the no-plateau
        # transition=0.5 profile. Position, velocity and acceleration meet the
        # stance branch continuously at lift-off and touchdown.
        enter_curve = enter**3 * (10.0 - 15.0 * enter + 6.0 * enter**2)
        leave_curve = leave**3 * (10.0 - 15.0 * leave + 6.0 * leave**2)
        envelope = enter_curve * leave_curve
        active = torch.linalg.norm(self._commands[:, :2], dim=1) > float(self.cfg.gait_standstill_deadband_m_s)
        startup = torch.clamp(self._steps_since_reset.float() * self.step_dt / self.cfg.gait_start_ramp_s, 0.0, 1.0)
        swing = (~scheduled_contact) & active.unsqueeze(1)
        if self._swing_sole_tilt_weight:
            # World sole attitude, scored even if a scheduled swing drags.
            # Retain old reward arithmetic/telemetry exactly when disabled.
            eligible = swing & (~self._run_stop_started).unsqueeze(1)
            tilt_ratio = torch.clamp((1.0-upright_cosine)/(1.0-math.cos(math.radians(12.0))), min=0.0, max=25.0)
            swing_tilt_cost = torch.sum(tilt_ratio*envelope*startup.unsqueeze(1)*eligible.float(),dim=1) / torch.clamp(eligible.float().sum(dim=1),min=1.0)
            self._swing_sole_tilt_last = {
                "physics_counter": int(self._sim_step_counter),
                "upright_cosine": upright_cosine.detach().clone(),
                "envelope": envelope.detach().clone(), "startup": startup.detach().clone(),
                "eligible": eligible.detach().clone(), "cost": swing_tilt_cost.detach().clone(),
            }
        target_clearance = self._sole_clearance_target_m * envelope * startup.unsqueeze(1)
        deficit = torch.clamp((target_clearance - minimum_height) / self._sole_clearance_target_m, min=0.0, max=5.0)
        clearance_cost = torch.sum(torch.square(deficit) * swing.float(), dim=1) / torch.clamp(swing.float().sum(dim=1), min=1.0)
        if self._swing_clearance_worst_leg_fraction > 0.0:
            # A good front member must not hide a low rear member of a swing
            # pair. Include ONLY scheduled, active swing feet, never stance.
            # No deficit/target/envelope/physical limit changes are involved.
            worst_swing_cost = torch.amax(
                torch.where(swing, torch.square(deficit), torch.zeros_like(deficit)),
                dim=1,
            )
            fraction = self._swing_clearance_worst_leg_fraction
            blended_clearance_cost = (1.0 - fraction) * clearance_cost + fraction * worst_swing_cost
            # Preserve the original arithmetic result exactly for one swing
            # foot; with no active swing feet the original result is zero.
            clearance_cost = torch.where(
                swing.float().sum(dim=1) > 1.0,
                blended_clearance_cost,
                clearance_cost,
            )

        # Approximate the active contact patch by corners near the lowest corner
        # on a force-confirmed foot. Do not penalize the high edge's rotation as
        # sliding merely because a different edge is planted.
        near_lowest = heights <= minimum_height.unsqueeze(2) + self._sole_corner_soft_contact_height_m
        point_mask = near_lowest & foot_contact.unsqueeze(2)
        point_speed_squared = torch.sum(torch.square(corner_velocity[..., :2]), dim=-1)
        point_count = torch.clamp(point_mask.float().sum(dim=2), min=1.0)
        slip_by_foot = torch.sum(point_speed_squared * point_mask.float(), dim=2) / point_count
        contact_slip = torch.sum(slip_by_foot, dim=1) / torch.clamp(foot_contact.float().sum(dim=1), min=1.0)

        tilt_angle = torch.acos(upright_cosine)
        self._episode_stance_sole_tilt_sum += tilt_angle * stance.float()
        self._episode_stance_sole_opportunities += stance.float()
        core_swing = swing & (envelope >= 0.99) & (startup >= 0.99).unsqueeze(1)
        self._episode_swing_corner_height_sum += minimum_height * core_swing.float()
        self._episode_swing_corner_min_height = torch.minimum(
            self._episode_swing_corner_min_height,
            torch.where(core_swing, minimum_height, torch.full_like(minimum_height, torch.inf)),
        )
        self._episode_swing_corner_opportunities += core_swing.float()
        self._episode_contact_point_speed_squared_sum += slip_by_foot
        self._episode_contact_point_opportunities += foot_contact.float()
        return stance_tilt, clearance_cost, contact_slip

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

    @staticmethod
    def _rear_pitch_rotation_z(angle: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(angle), torch.sin(angle)
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        return torch.stack((c,-s,zero,s,c,zero,zero,zero,one), dim=-1).reshape(*angle.shape,3,3)

    def _initialize_rear_pitch_lift(self) -> None:
        """Bind the successful offline analytic construction to exact joint frames."""
        path = PROJECT_ROOT / "cad/exports/urdf/quadruped_robot.urdf"
        with path.open("rb") as stream:
            source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if source_hash != "cfae29e48a29d5d63db99a9f4443f05c0074d7100c890b227553d35a3aa6b474":
            raise ValueError("Rear pitch witness requires its pinned canonical joint frames")
        joints = {joint.get("name"): joint for joint in ET.parse(path).getroot().findall("joint")}
        def numbers(text):
            return tuple(float(value) for value in text.split())
        def rotation(rpy):
            # URDF origin is Rz(yaw) Ry(pitch) Rx(roll), not a world joint axis.
            r,p,y = rpy
            cr,sr,cp,sp,cy,sy = math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
            return ((cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr),
                    (sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr),(-sp,cp*sr,cp*cr))
        legs = ("rear_left", "rear_right")
        rotations, hip_rotations, positions, hip_positions, signs, lengths = [],[],[],[],[],[]
        for leg in legs:
            abd,hip,knee = (joints[leg+suffix] for suffix in ("_hip_abduction","_hip_flexion","_knee"))
            axis = numbers(hip.find("axis").get("xyz"))
            knee_xyz = numbers(knee.find("origin").get("xyz"))
            if (abd.find("parent").get("link") != "base_link"
                or hip.find("parent").get("link") != abd.find("child").get("link")
                or knee.find("parent").get("link") != hip.find("child").get("link")
                or numbers(abd.find("axis").get("xyz")) != (0.,0.,1.)
                or axis != numbers(knee.find("axis").get("xyz"))
                or axis[:2] != (0.,0.) or abs(axis[2]) != 1.
                or numbers(knee.find("origin").get("rpy")) != (0.,0.,0.)
                or knee_xyz != (0.159896689,0.,0.)):
                raise ValueError("Rear pitch analytic same-axis knee-chain contract differs")
            rotations.append(rotation(numbers(abd.find("origin").get("rpy"))))
            hip_rotations.append(rotation(numbers(hip.find("origin").get("rpy"))))
            positions.append(numbers(abd.find("origin").get("xyz")))
            hip_positions.append(numbers(hip.find("origin").get("xyz")))
            signs.append(axis[2]); lengths.append(knee_xyz[0])
        tensor = lambda values: torch.tensor(values, dtype=torch.float64, device=self.device)
        self._rear_pitch_abd_rotation, self._rear_pitch_hip_rotation = tensor(rotations),tensor(hip_rotations)
        self._rear_pitch_abd_position, self._rear_pitch_hip_position = tensor(positions),tensor(hip_positions)
        self._rear_pitch_sign, self._rear_pitch_length = tensor(signs),tensor(lengths)
        self._rear_pitch_joint_ids = tuple(
            torch.tensor([EXPECTED_ACTION_ORDER.index(leg+suffix) for leg in legs], device=self.device)
            for suffix in ("_hip_abduction","_hip_flexion","_knee")
        )
        self._rear_pitch_foot_ids = torch.tensor([self._foot_leg_names.index(leg) for leg in legs], device=self.device)
        self._rear_pitch_offsets = tensor([self.cfg.gait_phase_offsets[LEG_NAMES.index(leg)] for leg in legs])
        self._rear_pitch_tread_depth_m = (DISTAL_LINK_LENGTH_M + RECTANGULAR_SHOE_SOLE_BACK_FROM_FORK_M
                                        + RECTANGULAR_SHOE_SOLE_THICKNESS_M + RECTANGULAR_SHOE_TREAD_PROJECTION_M)
        self._rear_pitch_lift_last = None
        self._rear_pitch_lift_contract = {
            "schema":"drobot.v44-rear-fixed-pitch-lift.v1", "amplitude_m":float(self.cfg.gait_rear_pitch_lift_m),
            "leg_order":list(legs), "canonical_joint_frames_sha256":source_hash,
            "joint_order":list(EXPECTED_ACTION_ORDER), "construction_max_abs_X_excursion_m":0.0175,
            "construction_max_level_root_tilt_deg":12.0, "analytic_continuation_endpoints":5,
            "phase":"actual pre-target phase, not the preserved next-phase reward alias",
            "lift_envelope_fractions":{"rise":0.5,"descent":self._rear_pitch_descent_fraction},
            "startup_overlay":"Q(clamp((pre_step_age_seconds-0.75)/0.75,0,1))",
            "command_overlay":"Q(clamp((translation_speed-0.004)/(0.01909090909090909-0.004),0,1))",
            "scheduled_stance_overlay":"zero using cached applied scheduled-contact mask",
            "preserves":"Requested rear distal orientation via hip+knee sum; abduction and all front target entries unchanged",
            "pipeline":"After residual mapping, before unchanged joint clamp, delay, packet/rate limits and120Hz response filter",
            "zero_amplitude":("O is bypassed; the separately declared common-F5 projection still applies"
                              if self._rear_pitch_projection_enabled else "Exact original target arithmetic; only detached diagnostic copies"),
            "delivered_geometry":"Reported separately; limiter/response/feedback can change pitch andX. Not guaranteed identical to mapped request.",
            "fault_policy":"Dedicated pre-physics mapping error; no fallback, no relaxed bounds",
            "production_export_allowed":False, "qualifies_hardware":False,
        }
        self._rear_pitch_projection_contract = None
        if self._rear_pitch_projection_enabled:
            module_path = Path(__file__).with_name("rear_pitch_projection.py")
            with module_path.open("rb") as stream:
                projection_sha = hashlib.file_digest(stream,"sha256").hexdigest()
            self._rear_pitch_projection_contract = {
                "schema":"drobot.v44-rear-pitch-common-F5-projection.v1",
                "enabled":True,"module_path":str(module_path.resolve()),"module_sha256":projection_sha,
                "leg_order":["rear_left","rear_right"],"joint_order":list(EXPECTED_ACTION_ORDER),
                "common_maximum_lift_m":0.005,"selected_post_projection_lift_m":float(self.cfg.gait_rear_pitch_lift_m),
                "lift_envelope_fractions":{"rise":0.5,"descent":self._rear_pitch_descent_fraction},
                "common_envelope":"Actual5mm phase/startup/command/contact envelope for BOTH zero and lift arms",
                "composition":"raw reference+scaled actor residual -> Pi_F -> selected0/5mm O -> original bounds/delay/packet/rate/response",
                "raw_actor_and_reward":("Raw actor output and rewards unchanged; declared bounded rear tracking correction precedes projection. Applied-target history reflects delivered commands."
                    if self._rear_tracking_feedback_gain else
                    "Raw actor output, PPO action/history unchanged; only the declared sustained-progress contact factor changes. Applied-target observation history reflects delivered commands."
                    if self._decouple_sustained_progress_from_pair_release else
                    "Raw actor output, PPO action/history and reward expressions unchanged. Applied-target observation history reflects delivered commands."),
                "already_feasible":"Return exact input float32 values; no round-trip through interval coordinates",
                "inactive":"common lift<=1e-12m bypasses projection; no added smoothing at activation",
                "construction":"Same-branch analytic sufficient inner domain in hip+knee and hip; original soft/residual bounds,12deg level-root tilt,17.5mm X certificate",
                "nominal_half_width_rad":{"hip":0.15,"knee":0.20},"inward_float32_margin_rad":4.e-6,
                "diagnostic_intervals":"Inactive leg rows have zero intervals plus active_by_leg=false; zeros are not a domain assertion",
                "continuum_evidence":"Finite saved/reference feasibility evidence does not prove nonemptiness for every continuous context; runtime rejection remains mandatory",
                "fault_policy":"Dedicated pre-physics mapping rejection with partial evidence; no fallback or relaxed limit",
                "qualification":"UNQUALIFIED_REAR_PITCH_PROJECTION_ABLATION",
                "production_export_allowed":False,"qualifies_hardware":False,
            }

    def _rear_pitch_common_lift(self, phase: torch.Tensor, scheduled_contact: torch.Tensor) -> torch.Tensor:
        """Same5mm eligibility envelope in BOTH projected arms; no target write."""
        def Q(value):
            u = torch.clamp(value,0.,1.)
            # The quintic is analytically in [0,1], but near saturation its
            # float64 expansion can round a few ulps above one. Bound the
            # computed weight, not the strict 0..5mm projection guard.
            return torch.clamp(u**3*(10.-15.*u+6.*u*u),0.,1.)
        speed = torch.linalg.vector_norm(self._commands[:,:2].to(torch.float64),dim=1)
        leg_phase = torch.remainder(phase.to(torch.float64).unsqueeze(1)+self._rear_pitch_offsets,1.)
        u = torch.clamp((leg_phase-.55)/.45,0.,1.)
        envelope = Q(u/.5)*Q((1.-u)/self._rear_pitch_descent_fraction)
        startup = Q((self._steps_since_reset.to(torch.float64)*self.step_dt-.75)/.75)
        command = Q((speed-.004)/(.01909090909090909-.004))
        lift = .005*envelope*startup.unsqueeze(1)*command.unsqueeze(1)
        return torch.where(scheduled_contact[:,self._rear_pitch_foot_ids],torch.zeros_like(lift),lift)

    def _rear_pitch_target_geometry(self, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Read-only body-frame requested/delivered geometry, not contact or load."""
        ai,hi,ki = self._rear_pitch_joint_ids
        q = targets.to(torch.float64)
        ra = self._rear_pitch_abd_rotation.unsqueeze(0) @ self._rear_pitch_rotation_z(q[:,ai])
        pre = ra @ self._rear_pitch_hip_rotation.unsqueeze(0)
        angle = self._rear_pitch_sign * q[:,hi]
        knee_offset = self._rear_pitch_length.unsqueeze(-1) * torch.stack((torch.cos(angle),torch.sin(angle),torch.zeros_like(angle)),dim=-1)
        knee = (self._rear_pitch_abd_position.unsqueeze(0)
                + (ra @ self._rear_pitch_hip_position.unsqueeze(0).unsqueeze(-1)).squeeze(-1)
                + (pre @ knee_offset.unsqueeze(-1)).squeeze(-1))
        angle_sum = q[:,hi]+q[:,ki]
        distal = pre @ self._rear_pitch_rotation_z(self._rear_pitch_sign * angle_sum)
        center = knee + self._rear_pitch_tread_depth_m * distal[..., :,0]
        return {"body_tread_center_m":center, "hip_knee_sum_rad":angle_sum,
                "level_root_sole_tilt_rad":torch.acos(torch.clamp(-distal[...,2,0],-1.,1.))}

    def _rear_pitch_mapping_fault(self, reason: str, details: dict) -> None:
        def safe(value):
            if isinstance(value, torch.Tensor):
                return safe(value.detach().cpu().tolist())
            if isinstance(value, dict):
                return {key:safe(item) for key,item in value.items()}
            if isinstance(value, (list,tuple)):
                return [safe(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        raise RearPitchLiftMappingError(reason, safe({
            "physics_counter_before_rejected_step":int(self._sim_step_counter),
            "nonfinite_values_encoded_as_null":True, "mapping":self._rear_pitch_lift_last,
            **details,
        }))

    def _map_rear_pitch_lift(self, requested: torch.Tensor, reference_center: torch.Tensor,
                             limits: torch.Tensor, phase: torch.Tensor, scheduled_contact: torch.Tensor) -> torch.Tensor:
        """Analytic same-branch lift only; all guards precede the existing clamp."""
        def Q(value):
            u = torch.clamp(value,0.,1.)
            # Match the bounded common-envelope weight; no geometry, target
            # authority or safety threshold is relaxed by this range repair.
            return torch.clamp(u**3*(10.-15.*u+6.*u*u),0.,1.)
        ai,hi,ki = self._rear_pitch_joint_ids
        speed = torch.linalg.vector_norm(self._commands[:,:2].to(torch.float64),dim=1)
        leg_phase = torch.remainder(phase.to(torch.float64).unsqueeze(1)+self._rear_pitch_offsets,1.)
        u = torch.clamp((leg_phase-.55)/.45,0.,1.)
        envelope = Q(u/.5)*Q((1.-u)/self._rear_pitch_descent_fraction)
        startup = Q((self._steps_since_reset.to(torch.float64)*self.step_dt-.75)/.75)
        command = Q((speed-.004)/(.01909090909090909-.004))
        lift = float(self.cfg.gait_rear_pitch_lift_m)*envelope*startup.unsqueeze(1)*command.unsqueeze(1)
        lift = torch.where(scheduled_contact[:,self._rear_pitch_foot_ids],torch.zeros_like(lift),lift)
        self._rear_pitch_lift_last.update(requested_added_height_m=lift.detach().clone(),
            startup_overlay_weight=startup.detach().clone(), command_overlay_weight=command.detach().clone())
        if not torch.all(torch.isfinite(requested)) or not torch.all(torch.isfinite(lift)):
            self._rear_pitch_mapping_fault("nonfinite_requested_geometry",{})
        env_ids,leg_ids = torch.where(lift > 1.e-12)
        if env_ids.numel() == 0:
            return requested
        q = requested[env_ids].to(torch.float64)
        selected = torch.arange(len(env_ids),device=self.device)
        h0,k0,abd = q[selected,hi[leg_ids]],q[selected,ki[leg_ids]],q[selected,ai[leg_ids]]
        pre = self._rear_pitch_abd_rotation[leg_ids] @ self._rear_pitch_rotation_z(abd) @ self._rear_pitch_hip_rotation[leg_ids]
        length,sign = self._rear_pitch_length[leg_ids],self._rear_pitch_sign[leg_ids]
        A,B = length*pre[:,2,0],sign*length*pre[:,2,1]
        C,D = length*pre[:,0,0],sign*length*pre[:,0,1]
        radius,phi = torch.hypot(A,B),torch.atan2(B,A)
        offset = torch.atan2(torch.sin(h0-phi),torch.cos(h0-phi))
        angle_sum = h0+k0
        baseline_tilt = torch.acos(torch.clamp(-pre[:,2,0]*torch.cos(angle_sum)-sign*pre[:,2,1]*torch.sin(angle_sum),-1.,1.))
        base_height = A*torch.cos(h0)+B*torch.sin(h0)
        joint_ids = torch.stack((hi[leg_ids],ki[leg_ids]),dim=1)
        center = reference_center[env_ids].to(torch.float64).gather(1,joint_ids)
        authority = (self._joint_scale*self._residual_action_scale).to(torch.float64)[joint_ids]
        lower = torch.maximum(limits[env_ids,:,0].to(torch.float64).gather(1,joint_ids),center-authority)
        upper = torch.minimum(limits[env_ids,:,1].to(torch.float64).gather(1,joint_ids),center+authority)
        evidence = {"env_ids":env_ids,"rear_leg_indices":leg_ids,"original_hip_knee_rad":torch.stack((h0,k0),dim=1),
                    "lower_bound_rad":lower,"upper_bound_rad":upper,"baseline_level_root_tilt_rad":baseline_tilt}
        if torch.any(radius <= 1.e-8) or torch.any(torch.abs(torch.sin(offset)) < 1.e-8):
            self._rear_pitch_mapping_fault("original_analytic_branch_turning_point",evidence)
        if torch.any(baseline_tilt > math.radians(12.)+1.e-8):
            self._rear_pitch_mapping_fault("baseline_requested_rear_tilt_above_12deg",evidence)
        branch_sign = torch.where(offset > 0.,torch.ones_like(offset),-torch.ones_like(offset))
        for stage in range(1,6):
            added = lift[env_ids,leg_ids]*stage/5.
            cosine = (base_height+added)/radius
            evidence.update(stage=stage,requested_lift_m=added,cosine_argument=cosine)
            if torch.any(~torch.isfinite(cosine)) or torch.any(torch.abs(cosine) > 1.):
                self._rear_pitch_mapping_fault("same_branch_vertical_reach_unavailable",evidence)
            principal = phi+branch_sign*torch.acos(cosine)
            h = h0+torch.atan2(torch.sin(principal-h0),torch.cos(principal-h0))
            k = angle_sum-h
            pair = torch.stack((h,k),dim=1)
            dx = C*(torch.cos(h)-torch.cos(h0))+D*(torch.sin(h)-torch.sin(h0))
            dz = A*(torch.cos(h)-torch.cos(h0))+B*(torch.sin(h)-torch.sin(h0))
            evidence.update(attempted_hip_knee_rad=pair,signed_body_X_change_m=dx,achieved_lift_m=dz)
            if torch.any(~torch.isfinite(pair)) or torch.any(pair < lower) or torch.any(pair > upper):
                self._rear_pitch_mapping_fault("existing_residual_or_soft_joint_bound",evidence)
            if torch.any(k*k0 <= 0.):
                self._rear_pitch_mapping_fault("signed_knee_branch_change",evidence)
            if torch.any(torch.abs(dx) > .0175+1.e-7):
                self._rear_pitch_mapping_fault("half_stroke_requested_X_excursion_guard",evidence)
            if torch.any(torch.abs(dz-added) > 1.e-7) or torch.any(torch.abs(h+k-angle_sum) > 2.e-6):
                self._rear_pitch_mapping_fault("analytic_height_or_orientation_mismatch",evidence)
        mapped = requested.clone()
        mapped[env_ids,hi[leg_ids]],mapped[env_ids,ki[leg_ids]] = h.to(requested.dtype),k.to(requested.dtype)
        cast_pair = mapped[env_ids].to(torch.float64).gather(1,joint_ids)
        evidence["float32_mapped_hip_knee_rad"] = cast_pair
        if (torch.any(cast_pair < lower-2.e-6) or torch.any(cast_pair > upper+2.e-6)
            or torch.any(torch.abs(cast_pair.sum(dim=1)-angle_sum) > 2.e-6)):
            self._rear_pitch_mapping_fault("float32_mapping_bound_or_orientation_mismatch",evidence)
        self._rear_pitch_lift_last.update(mapped_pre_bounds_target_rad=mapped.detach().clone(),
            positive_lift_env_ids=env_ids.detach().clone(),positive_lift_rear_leg_indices=leg_ids.detach().clone(),
            signed_requested_body_X_change_m=dx.detach().clone(), achieved_lift_m=dz.detach().clone(),
            original_level_root_tilt_rad=baseline_tilt.detach().clone(),branch_sign=branch_sign.detach().clone(),
            minimum_joint_bound_margin_rad=torch.minimum(pair-lower,upper-pair).amin(dim=1).detach().clone(),
            requested_hip_knee_sum_error_rad=(cast_pair.sum(dim=1)-angle_sum).detach().clone())
        return mapped

    def _pre_physics_step(self, actions: torch.Tensor):
        self._older_actions.copy_(self._previous_actions)
        self._previous_actions.copy_(self._actions)
        self._actions = torch.clamp(actions, -1.0, 1.0)
        applied_gait_phase = self._gait_phase()
        # Passive evidence only. The legacy _applied_gait_phase below aliases
        # the integrated accumulator and is advanced in-place at this method's
        # end. Preserve that reward behavior for the paired ablation, while
        # retaining the actual table/mask phase before that advance.
        self._applied_reference_phase_for_diagnostics = applied_gait_phase.clone()
        gait_reference, scheduled_contact = self._gait_targets(
            phase=applied_gait_phase
        )
        (
            expected_lateral_displacement,
            expected_lateral_velocity,
        ) = self._gait_expected_lateral_motion(phase=applied_gait_phase)
        self._applied_gait_reference = gait_reference
        self._applied_scheduled_contact = scheduled_contact
        self._applied_gait_phase = applied_gait_phase
        self._applied_expected_lateral_displacement = (
            expected_lateral_displacement
        )
        self._applied_expected_lateral_velocity = expected_lateral_velocity
        (
            self._applied_expected_forward_displacement,
            self._applied_expected_forward_velocity,
        ) = self._gait_expected_forward_motion(phase=applied_gait_phase)
        limits = self._robot.data.soft_joint_pos_limits.torch
        self._target_packet_elapsed_s += self.step_dt
        if self._target_packet_hold_probability > 0.0:
            self._target_packet_held = (
                (torch.rand(self.num_envs, device=self.device) < self._target_packet_hold_probability)
                & self._physical_randomization_active
                & (self._target_packet_hold_steps < self._target_packet_max_hold_steps)
            )
        else:
            self._target_packet_held.zero_()
        # Packet holds change delivered targets, never the physics integration
        # clock. The next packet uses elapsed time and still obeys its step cap.
        max_delta = (
            self._target_velocity_limit_by_joint.unsqueeze(0)
            * self._target_packet_elapsed_s.unsqueeze(1)
            * self._target_velocity_scale_by_joint
        )
        if self._target_packet_step_limit_rad > 0.0:
            max_delta = torch.clamp(max_delta, max=self._target_packet_step_limit_rad)
        if self.cfg.action_mode in ("gait_residual", "direct"):
            if self.cfg.action_mode == "gait_residual":
                motion_requested = (
                    torch.linalg.norm(self._commands[:, :2], dim=1)
                    > float(self.cfg.gait_standstill_deadband_m_s)
                ) | (torch.abs(self._nominal_yaw_command) > 1.0e-4)
                residual_authority = torch.where(
                    motion_requested,
                    torch.ones_like(self._commands[:, 0]),
                    torch.full_like(
                        self._commands[:, 0],
                        float(self.cfg.gait_residual_stop_action_scale),
                    ),
                )
                if self._swing_residual_scaling_enabled:
                    # Use the actual phase applied to the table, not the next
                    # observation phase. The legacy clearance-phase alias is
                    # intentionally unchanged and documented in diagnostics.
                    # Joint mapping is LEG_NAMES order, not sensor foot order.
                    leg_phase = torch.remainder(applied_gait_phase.unsqueeze(1) + self._leg_phase_offsets, 1.0)
                    duty = float(self.cfg.gait_duty_factor)
                    swing_u = torch.clamp((leg_phase - duty) / (1.0 - duty), 0.0, 1.0)
                    transition = float(self.cfg.sole_clearance_transition_fraction)
                    enter = torch.clamp(swing_u / transition, 0.0, 1.0)
                    leave = torch.clamp((1.0 - swing_u) / transition, 0.0, 1.0)
                    envelope = (enter**3 * (10.0 - 15.0 * enter + 6.0 * enter**2)
                                * leave**3 * (10.0 - 15.0 * leave + 6.0 * leave**2))
                    leg_multiplier = 1.0 - (1.0 - self._gait_swing_residual_scale_by_leg) * envelope
                    # A parked command retains its previous residual authority
                    # even if the clock happens to freeze in a swing phase.
                    leg_multiplier = torch.where(motion_requested.unsqueeze(1), leg_multiplier, torch.ones_like(leg_multiplier))
                    self._applied_swing_residual_multiplier.fill_(1.0)
                    self._applied_swing_residual_multiplier[:, self._leg_hip_joint_ids] = leg_multiplier
                    self._applied_swing_residual_multiplier[:, self._leg_knee_joint_ids] = leg_multiplier
                    normalized_target = gait_reference + (
                        self._residual_action_scale * self._actions
                        * residual_authority.unsqueeze(1) * self._applied_swing_residual_multiplier
                    )
                else:
                    # Exact default arithmetic bypass: no multiplication by a
                    # synthetic unit mask and no change to raw action history.
                    normalized_target = gait_reference + (
                        self._residual_action_scale
                        * self._actions
                        * residual_authority.unsqueeze(1)
                    )
            else:
                normalized_target = self._actions
            desired_targets = self._robot.data.default_joint_pos.torch + (
                self._joint_scale * normalized_target
            ) + self._joint_target_bias
            if self._front_stance_tracking_feedback_gain:
                # Experimental causal outer correction. Zero gain bypasses all
                # arithmetic, telemetry, and additional state reads exactly.
                previous = self._previous_targets.detach().clone()
                measured = self._robot.data.joint_pos.torch.detach().clone()
                raw = desired_targets.detach().clone()
                center = self._robot.data.default_joint_pos.torch + self._joint_scale*gait_reference + self._joint_target_bias
                half = self._joint_scale*self._residual_action_scale*residual_authority.unsqueeze(1)
                lower = torch.maximum(center-half, limits[:,:,0])
                upper = torch.minimum(center+half, limits[:,:,1])
                def c2(value):
                    u = torch.clamp(value, 0., 1.)
                    return torch.clamp(u**3*(10.-15.*u+6.*u*u), 0., 1.)
                phase = torch.remainder(applied_gait_phase.to(torch.float64).unsqueeze(1)
                    + torch.tensor([0., .5], device=self.device, dtype=torch.float64).unsqueeze(0), 1.)
                u = phase/.55
                envelope = c2(u/.15)*c2((1.-u)/.15)
                startup = c2((self._steps_since_reset.to(torch.float64)*self.step_dt-.75)/.75)
                speed = torch.linalg.vector_norm(self._commands[:,:2].to(torch.float64), dim=1)
                command_gate = c2((speed-.004)/(.01909090909090909-.004))
                elapsed = (self._steps_since_reset-self._run_stop_start_step+1).to(torch.float64)*self.step_dt
                stop_fade = c2(1.-torch.clamp(elapsed,min=0.)/.2)
                weight = (envelope*startup.unsqueeze(1)*command_gate.unsqueeze(1)*stop_fade.unsqueeze(1)).to(raw.dtype)
                weight = torch.where(scheduled_contact[:,[0,2]], weight, torch.zeros_like(weight))
                correction = torch.zeros_like(raw)
                cap = math.radians(2.)
                if not all(bool(torch.isfinite(v).all()) for v in (raw,previous,measured,lower,upper,weight)):
                    self._rear_pitch_mapping_fault("front_stance_feedback_nonfinite", {})
                for front,hip,knee in ((0,4,8),(1,6,10)):
                    if bool(torch.any(lower[:,[hip,knee]] > upper[:,[hip,knee]])):
                        self._rear_pitch_mapping_fault("front_stance_feedback_empty_bounds", {})
                    error = self._front_stance_tracking_feedback_gain*(previous-measured)
                    d = .5*(torch.clamp(error[:,knee],-cap,cap)-torch.clamp(error[:,hip],-cap,cap))*weight[:,front]
                    # Reduce only toward zero. Never enlarge actor authority or
                    # force a new offset merely to correct roundoff at a bound.
                    positive_room = torch.minimum(raw[:,hip]-lower[:,hip], upper[:,knee]-raw[:,knee])
                    negative_room = torch.minimum(upper[:,hip]-raw[:,hip], raw[:,knee]-lower[:,knee])
                    room = torch.clamp(torch.where(d>=0.,positive_room,negative_room), min=0.)
                    d = torch.sign(d)*torch.minimum(torch.abs(d),room)
                    correction[:,hip] = -d
                    correction[:,knee] = d
                desired_targets = raw+correction
                front_ids = [4,6,8,10]
                if (bool(torch.any(desired_targets[:,front_ids] < lower[:,front_ids]-2.e-7))
                    or bool(torch.any(desired_targets[:,front_ids] > upper[:,front_ids]+2.e-7))):
                    self._rear_pitch_mapping_fault("front_stance_feedback_out_of_bounds", {})
                self._front_stance_tracking_feedback_last = {
                    "physics_counter_before_step": int(self._sim_step_counter),
                    "previous_delivered_target_rad": previous, "pre_physics_measured_q_rad": measured,
                    "raw_actor_requested_target_rad": raw, "reference_center_rad": center.detach().clone(),
                    "lower_bound_rad": lower.detach().clone(), "upper_bound_rad": upper.detach().clone(),
                    "weight_by_front": weight.detach().clone(), "stop_fade": stop_fade.detach().clone(),
                    "stop_elapsed_s": elapsed.detach().clone(), "bounded_correction_rad": correction.detach().clone(),
                    "corrected_target_rad": desired_targets.detach().clone(),
                }
            if self._rear_tracking_feedback_gain:
                # Causal outer correction; inner drive gains and authority stay
                # unchanged. No integral state, added physics, or extra actor call.
                previous_delivered = self._previous_targets.detach().clone()
                measured_q = self._robot.data.joint_pos.torch.detach().clone()
                error = previous_delivered - measured_q
                common_lift = self._rear_pitch_common_lift(applied_gait_phase, scheduled_contact)
                weight = (common_lift / .005).to(desired_targets.dtype)
                weight = torch.where(self._run_stop_started.unsqueeze(1), torch.zeros_like(weight), weight)
                if self._rear_tracking_feedback_leg_mask == (0., 1.):
                    weight[:,0] = 0.
                if not bool(torch.isfinite(error).all()) or not bool(torch.isfinite(weight).all()):
                    self._rear_pitch_mapping_fault("rear_feedback_nonfinite", {})
                if not bool(((weight >= 0.) & (weight <= 1.0)).all()):
                    self._rear_pitch_mapping_fault("rear_feedback_envelope_out_of_bounds", {})
                correction = torch.zeros_like(desired_targets)
                cap = math.radians(2.0)
                for rear, hip, knee in ((0,5,9),(1,7,11)):
                    for joint in (hip,knee):
                        correction[:,joint] = weight[:,rear] * torch.clamp(
                            self._rear_tracking_feedback_gain * error[:,joint], -cap, cap)
                    if self._rear_tracking_feedback_mode == "fixed_pitch":
                        # Orthogonal projection of the already capped correction.
                        # Equal/opposite pair preserves requested hip+knee sum;
                        # actual/delivered sole angle is not guaranteed by this.
                        d = .5 * (correction[:,knee] - correction[:,hip])
                        correction[:,hip] = -d
                        correction[:,knee] = d
                self._rear_tracking_feedback_last = {
                    "physics_counter_before_step": int(self._sim_step_counter),
                    "previous_delivered_target_rad": previous_delivered,
                    "pre_physics_measured_q_rad": measured_q,
                    "raw_actor_requested_target_rad": desired_targets.detach().clone(),
                    "swing_weight_by_rear": weight.detach().clone(),
                    "bounded_correction_rad": correction.detach().clone(),
                }
                desired_targets = desired_targets + correction
                self._rear_tracking_feedback_last["corrected_pre_projection_target_rad"] = desired_targets.detach().clone()
            if self._rear_pitch_lift_diagnostics_enabled:
                self._rear_pitch_lift_last = {
                    "physics_counter_before_step":int(self._sim_step_counter),
                    "pre_step_age":self._steps_since_reset.detach().clone(),
                    "actual_target_phase_cycles":applied_gait_phase.detach().clone(),
                    "pre_overlay_requested_target_rad":desired_targets.detach().clone(),
                    "mapped_pre_bounds_target_rad":desired_targets.detach().clone(),
                    "requested_added_height_m":torch.zeros((self.num_envs,2),device=self.device,dtype=torch.float64),
                }
            if self._rear_pitch_projection_enabled:
                from .rear_pitch_projection import project_rear_pitch_targets
                reference_center = self._robot.data.default_joint_pos.torch + self._joint_scale*gait_reference + self._joint_target_bias
                common_lift = self._rear_pitch_common_lift(applied_gait_phase,scheduled_contact)
                self._rear_pitch_lift_last["feasible_projection_attempt"] = {
                    "raw_requested_target_rad":desired_targets.detach().clone(),
                    "common_lift_m":common_lift.detach().clone(),
                }
                desired_targets,projection_diagnostics = project_rear_pitch_targets(
                    self,desired_targets,reference_center,limits,common_lift)
                self._rear_pitch_lift_last["feasible_projection"] = projection_diagnostics
                # These existing fields remain the immediate input/output of O.
                # Raw actor targets are retained separately in projection data.
                self._rear_pitch_lift_last["pre_overlay_requested_target_rad"] = desired_targets.detach().clone()
                self._rear_pitch_lift_last["mapped_pre_bounds_target_rad"] = desired_targets.detach().clone()
            if self._rear_pitch_lift_enabled:
                # Amplitude0 bypasses O; an explicitly enabled common-F5
                # projection above is independent of this selected amplitude.
                # Raw actor actions/history/reference remain untouched. The
                # existing applied-target history sees what is actually sent.
                reference_center = self._robot.data.default_joint_pos.torch + self._joint_scale*gait_reference + self._joint_target_bias
                desired_targets = self._map_rear_pitch_lift(
                    desired_targets,reference_center,limits,applied_gait_phase,scheduled_contact
                )
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
        elif self.cfg.action_mode == "integrated_target_velocity":
            # Delay the bounded velocity request, not an absolute target.  An
            # absolute one-frame queue can become a two-step jump when the next
            # request is integrated from a target that has not moved yet.
            delayed_action = torch.where(
                (self._control_delay_steps > 0).unsqueeze(1),
                self._previous_actions,
                self._actions,
            )
            motion_action = delayed_action
            if self.cfg.integrated_action_scale_with_gait_frequency:
                gait_frequency_hz, _stride_scale = self._gait_schedule()
                frequency_scale = (
                    gait_frequency_hz
                    / self._integrated_action_reference_frequency_hz
                )
                frequency_scale = 1.0 + (
                    frequency_scale.unsqueeze(1) - 1.0
                ) * self._integrated_action_frequency_gain_by_joint.unsqueeze(0)
                motion_action = torch.clamp(
                    delayed_action * frequency_scale,
                    -1.0,
                    1.0,
                )
            neutral_target = (
                self._robot.data.default_joint_pos.torch + self._joint_target_bias
            )
            homing_action = torch.clamp(
                (neutral_target - self._previous_targets)
                / torch.clamp(max_delta, min=1.0e-6),
                -1.0,
                1.0,
            )
            motion_requested = (
                torch.linalg.norm(self._commands[:, :2], dim=1)
                > float(self.cfg.gait_standstill_deadband_m_s)
            ) | (torch.abs(self._nominal_yaw_command) > 1.0e-4)
            stop_target_residual_scale = float(
                self.cfg.integrated_stop_target_residual_scale
            )
            if stop_target_residual_scale > 0.0:
                stop_desired_targets = neutral_target + (
                    stop_target_residual_scale
                    * self._joint_scale
                    * delayed_action
                )
                stop_desired_targets = torch.clamp(
                    stop_desired_targets, limits[:, :, 0], limits[:, :, 1]
                )
                motion_desired_targets = (
                    self._previous_targets + max_delta * motion_action
                )
                desired_targets = torch.where(
                    motion_requested.unsqueeze(1),
                    motion_desired_targets,
                    stop_desired_targets,
                )
            else:
                # Backward-compatible zero-command behavior for existing
                # checkpoints: deterministic neutral homing, optionally with
                # the historical bounded velocity residual.
                stop_action = torch.clamp(
                    homing_action
                    + float(self.cfg.integrated_stop_action_scale)
                    * delayed_action,
                    -1.0,
                    1.0,
                )
                effective_action = torch.where(
                    motion_requested.unsqueeze(1), motion_action, stop_action
                )
                desired_targets = (
                    self._previous_targets + max_delta * effective_action
                )
            desired_targets = torch.clamp(
                desired_targets, limits[:, :, 0], limits[:, :, 1]
            )
            self._previous_requested_targets.copy_(desired_targets)
            self._desired_targets.copy_(desired_targets)
        else:
            raise RuntimeError(f"Unknown action_mode: {self.cfg.action_mode}")
        self._processed_actions = torch.clamp(
            self._desired_targets,
            self._previous_targets - max_delta,
            self._previous_targets + max_delta,
        )
        self._processed_actions = torch.where(
            self._target_packet_held.unsqueeze(1),
            self._previous_targets,
            self._processed_actions,
        )
        self._previous_targets.copy_(self._processed_actions)
        self._target_packet_hold_steps = torch.where(
            self._target_packet_held,
            self._target_packet_hold_steps + 1,
            torch.zeros_like(self._target_packet_hold_steps),
        )
        self._target_packet_elapsed_s = torch.where(
            self._target_packet_held,
            self._target_packet_elapsed_s,
            torch.zeros_like(self._target_packet_elapsed_s),
        )
        self._episode_target_packet_hold_sum += self._target_packet_held.float()
        if self.cfg.integrate_gait_phase:
            frequency_hz, _stride_scale = self._gait_schedule()
            self._gait_phase_accumulator.copy_(
                torch.remainder(
                    applied_gait_phase + self.step_dt * frequency_hz,
                    1.0,
                )
            )

    def _apply_action(self):
        if not self._actuator_target_response_enabled:
            self._robot.set_joint_position_target_index(target=self._processed_actions)
            return
        neutral = self._robot.data.default_joint_pos.torch + self._joint_target_bias
        desired_response = neutral + self._actuator_response_gain * (self._processed_actions - neutral)
        tau = self._actuator_response_tau_s
        alpha = torch.where(
            tau > 0.0,
            -torch.expm1(-float(self.cfg.sim.dt) / torch.clamp(tau, min=1.0e-6)),
            torch.ones_like(tau),
        )
        self._actuator_response_targets += alpha * (desired_response - self._actuator_response_targets)
        # History remains the controller-delivered packet; response lag appears
        # only through the physical plant and the ensuing joint/IMU feedback.
        self._robot.set_joint_position_target_index(target=self._actuator_response_targets)

    def _base_contact_force(self) -> torch.Tensor:
        force = self._contact_sensor.data.net_forces_w.torch[:, self._base_sensor_ids]
        return torch.linalg.norm(force, dim=-1).amax(dim=1)

    def _foot_forces(self) -> torch.Tensor:
        force = self._contact_sensor.data.net_forces_w.torch[:, self._foot_sensor_ids]
        return torch.linalg.norm(force, dim=-1)

    def _gait_phase(self) -> torch.Tensor:
        if self.cfg.integrate_gait_phase:
            return self._gait_phase_accumulator
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
            if self._phase_reference_linear_interpolation_enabled:
                # Position-only interpolation; floor contacts stay unchanged.
                # Disabled mode executes the historical arithmetic verbatim.
                phase_position = phase * sample_count
                phase_fraction = (phase_position - torch.floor(phase_position)).unsqueeze(1)
                next_table_index = torch.remainder(table_index + 1, sample_count)
                desired_joint_position = torch.lerp(
                    desired_joint_position,
                    self._distributed_reference_joint_pos[next_table_index],
                    phase_fraction,
                )
            if self._distributed_reference_low_joint_pos is not None:
                low_joint_position = self._distributed_reference_low_joint_pos[
                    table_index
                ]
                if self._phase_reference_linear_interpolation_enabled:
                    low_joint_position = torch.lerp(
                        low_joint_position,
                        self._distributed_reference_low_joint_pos[next_table_index],
                        phase_fraction,
                    )
                command_speed = torch.linalg.norm(self._commands[:, :2], dim=1)
                blend = torch.clamp(
                    (
                        command_speed
                        - self._phase_reference_blend_speed_min_m_s
                    )
                    / self._phase_reference_blend_speed_span_m_s,
                    0.0,
                    1.0,
                )
                desired_joint_position = torch.lerp(
                    low_joint_position,
                    desired_joint_position,
                    blend.unsqueeze(1),
                )
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
            if self.cfg.gait_phase_table_park_contact_override and self.cfg.gait_park_at_standstill:
                scheduled_contact = torch.where(
                    (stride_scale <= 0.0).unsqueeze(1),
                    torch.ones_like(scheduled_contact),
                    scheduled_contact,
                )
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
        swing_translation_u = torch.clamp(
            (swing_u - self._gait_swing_translation_start)
            / self._gait_swing_translation_span,
            0.0,
            1.0,
        )
        swing_curve = swing_translation_u * swing_translation_u * (
            3.0 - 2.0 * swing_translation_u
        )
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
            self._leg_gait_lift.unsqueeze(0) * torch.sin(torch.pi * swing_u),
        )
        if self.cfg.gait_park_at_standstill:
            motion_active = stride_scale > 0.0
            lift = lift * motion_active.unsqueeze(1)
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
        world_forward = (
            nominal_forward
            + float(self.cfg.gait_stance_center_offset_m)
            + stride_offset
        )
        if self.cfg.gait_flat_sole_support:
            # With the new rectangular shoes, holding the contact center at a
            # constant height rotates the long sole onto an edge throughout the
            # stance sweep.  Follow the exact flat-sole branch instead: the
            # distal link remains vertical at zero lift, while the same smooth
            # lift profile transitions continuously onto the two-link swing IK.
            distal_contact_length = DISTAL_LINK_LENGTH_M + 0.031
            proximal_vertical = torch.sqrt(
                torch.clamp(
                    DISTAL_LINK_LENGTH_M**2 - torch.square(world_forward),
                    min=1.0e-8,
                )
            )
            down = proximal_vertical + distal_contact_length - lift
        else:
            down = self._leg_stance_down.unsqueeze(0) - lift
        if self.cfg.gait_park_at_standstill:
            motion_active = stride_scale > 0.0
            standstill_fore_aft_m = (
                float(self.cfg.gait_stance_fore_aft_m)
                if self.cfg.gait_standstill_fore_aft_m is None
                else float(self.cfg.gait_standstill_fore_aft_m)
            )
            standstill_down_m = (
                float(self.cfg.gait_stance_down_m)
                if self.cfg.gait_standstill_down_m is None
                else float(self.cfg.gait_standstill_down_m)
            )
            standstill_forward = (
                standstill_fore_aft_m * self._leg_front_sign.unsqueeze(0)
                + float(self.cfg.gait_standstill_center_offset_m)
            )
            world_forward = torch.where(
                motion_active.unsqueeze(1),
                world_forward,
                standstill_forward,
            )
            down = torch.where(
                motion_active.unsqueeze(1),
                down,
                torch.full_like(down, standstill_down_m),
            )
        # A rear-biased body cannot keep its COM on both alternating diagonal
        # support lines without a small lateral transfer.  Translate every foot
        # together relative to the body; this produces one coordinated body sway
        # instead of the independent side-to-side leg flail seen in V31.
        if not math.isclose(self._trot_lateral_shift_m, 0.0, abs_tol=1.0e-12):
            body_lateral_shift = (
                -self._trot_lateral_shift_m
                * torch.sin(2.0 * torch.pi * phase)
                * ramp.squeeze(1)
                * stride_scale
            )
            lateral_foot_delta = -body_lateral_shift
            leg_lateral_delta = (
                self._leg_side_sign.unsqueeze(0)
                * lateral_foot_delta.unsqueeze(1)
            )
            vertical_down = down
            down = torch.hypot(vertical_down, leg_lateral_delta)
            abduction = -self._leg_side_sign.unsqueeze(0) * torch.atan2(
                leg_lateral_delta,
                vertical_down,
            )
        else:
            abduction = torch.zeros_like(down)
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
        reference[:, self._leg_abduction_joint_ids] = (
            abduction - default[:, self._leg_abduction_joint_ids]
        ) / self.cfg.action_scale_abduction_rad
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
        if self._gait_contact_transition_fraction > 0.0:
            # The continuous reference, like the distributed crawl reference,
            # needs a short contact-labelled interval at both boundaries of
            # swing.  Those frames deliberately unload from and lower onto the
            # floor; counting them as failed airborne frames made the reward and
            # evaluator disagree with the configured contact schedule.
            foot_swing_u = torch.clamp(
                (foot_phase - duty) / (1.0 - duty),
                0.0,
                1.0,
            )
            scheduled_contact = scheduled_contact | (
                (foot_swing_u < self._gait_contact_transition_fraction)
                | (
                    foot_swing_u
                    > 1.0 - self._gait_contact_transition_fraction
                )
            )
        scheduled_contact = torch.where(
            ramp < 0.999,
            torch.ones_like(scheduled_contact),
            scheduled_contact,
        )
        if self.cfg.gait_park_at_standstill:
            scheduled_contact = torch.where(
                (stride_scale <= 0.0).unsqueeze(1),
                torch.ones_like(scheduled_contact),
                scheduled_contact,
            )
        return reference, scheduled_contact

    def _gait_expected_lateral_motion(self, phase: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return phase-locked body sway displacement and velocity.

        Straight-path scoring removes only this zero-mean reference motion.
        Accumulated drift and any residual lateral velocity remain penalized.
        Opt-in table metadata is planned Cartesian motion, not measured body
        pose. Startup/stride blending follows the same joint-reference factors
        (a first-order Cartesian approximation), before packet/actuator lag.
        At standstill the planned shift becomes zero with the neutral target;
        rate-limited homing may still have a nonzero tracking-error transient.
        """

        zeros = torch.zeros(self.num_envs, device=self.device)
        if not math.isclose(self._trot_lateral_shift_m, 0.0, abs_tol=1.0e-12):
            if phase is None:
                phase = self._gait_phase()
            frequency_hz, stride_scale = self._gait_schedule()
            ramp = torch.clamp(
                self._steps_since_reset.float()
                * self.step_dt
                / self.cfg.gait_start_ramp_s,
                0.0,
                1.0,
            )
            motion_scale = ramp * stride_scale
            phase_angle = 2.0 * torch.pi * phase
            expected_displacement = (
                -self._trot_lateral_shift_m
                * torch.sin(phase_angle)
                * motion_scale
            )
            expected_velocity = (
                -self._trot_lateral_shift_m
                * 2.0
                * torch.pi
                * torch.cos(phase_angle)
                * frequency_hz
                * motion_scale
            )
            ramp_velocity = torch.where(
                ramp < 1.0,
                stride_scale / max(self.cfg.gait_start_ramp_s, 1.0e-4),
                torch.zeros_like(ramp),
            )
            expected_velocity = expected_velocity + (
                -self._trot_lateral_shift_m
                * torch.sin(phase_angle)
                * ramp_velocity
            )
            return expected_displacement, expected_velocity
        if (
            self._distributed_reference_lateral_shift is None
            or self._distributed_reference_lateral_shift_phase_derivative is None
        ):
            return zeros, zeros
        if phase is None:
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

    def _gait_expected_forward_motion(self, phase: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Planned bounded body transfer, not commanded/net forward progress.

        Use the same sampled phase, cadence, startup ramp and stride as the
        applied joint prior. Cartesian startup blending is only first-order:
        packet/actuator lag and neutral-homing transients remain real errors.
        At parked stride zero, expected transfer is zero, even while the joint
        targets are still rate-limited on their way back to neutral.
        """
        zeros = torch.zeros(self.num_envs, device=self.device)
        if not self._phase_table_forward_reference_enabled:
            return zeros, zeros
        if self._phase_table_forward_shift is None or self._phase_table_forward_shift_phase_derivative is None:
            raise RuntimeError("Forward transfer metadata was not loaded")
        if phase is None:
            phase = self._gait_phase()
        frequency_hz, stride_scale = self._gait_schedule()
        sample_count = self._phase_table_forward_shift.shape[0]
        table_index = torch.remainder(torch.floor(phase * sample_count).long(), sample_count)
        ramp = torch.clamp(
            self._steps_since_reset.float() * self.step_dt / self.cfg.gait_start_ramp_s,
            0.0, 1.0,
        )
        motion_scale = ramp * stride_scale
        shift = self._phase_table_forward_shift[table_index]
        expected_displacement = shift * motion_scale
        expected_velocity = (
            self._phase_table_forward_shift_phase_derivative[table_index]
            * frequency_hz * motion_scale
        )
        ramp_velocity = torch.where(
            ramp < 1.0,
            stride_scale / max(self.cfg.gait_start_ramp_s, 1.0e-4),
            torch.zeros_like(ramp),
        )
        return expected_displacement, expected_velocity + shift * ramp_velocity

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

    def _initialize_run_stop_schedule(
        self,
        env_ids: torch.Tensor,
        speed: torch.Tensor,
        run_duration_s: torch.Tensor | None = None,
        ramp_duration_s: torch.Tensor | None = None,
    ) -> None:
        """Initialize a per-environment run, deceleration, and stopped hold."""

        count = len(env_ids)
        if speed.shape != (count,) or not torch.all(torch.isfinite(speed)):
            raise ValueError("run/stop speed must contain one finite value per environment")
        if torch.any(speed <= 0.0):
            raise ValueError("run/stop speed must be positive")
        if self._normalized_body_height_cost_enabled and torch.any(
            speed <= float(self.cfg.gait_standstill_deadband_m_s)
        ):
            raise ValueError("Height-only normalization requires run target speed above the standstill deadband")

        if run_duration_s is None:
            minimum = float(self.cfg.run_stop_run_duration_min_s)
            maximum = float(self.cfg.run_stop_run_duration_max_s)
            if not 0.0 < minimum <= maximum:
                raise ValueError("run/stop run-duration range must be positive and ordered")
            run_duration_s = torch.empty(count, device=self.device).uniform_(
                minimum, maximum
            )
        if ramp_duration_s is None:
            minimum = float(self.cfg.run_stop_ramp_duration_min_s)
            maximum = float(self.cfg.run_stop_ramp_duration_max_s)
            if not 0.0 <= minimum <= maximum:
                raise ValueError("run/stop ramp-duration range must be non-negative and ordered")
            ramp_duration_s = torch.empty(count, device=self.device).uniform_(
                minimum, maximum
            )
            abrupt_fraction = float(self.cfg.run_stop_abrupt_fraction)
            if not 0.0 <= abrupt_fraction <= 1.0:
                raise ValueError("run_stop_abrupt_fraction must be in [0, 1]")
            abrupt = torch.rand(count, device=self.device) < abrupt_fraction
            ramp_duration_s = torch.where(
                abrupt,
                torch.zeros_like(ramp_duration_s),
                ramp_duration_s,
            )
        if (
            run_duration_s.shape != (count,)
            or ramp_duration_s.shape != (count,)
            or not torch.all(torch.isfinite(run_duration_s))
            or not torch.all(torch.isfinite(ramp_duration_s))
            or torch.any(run_duration_s <= 0.0)
            or torch.any(ramp_duration_s < 0.0)
        ):
            raise ValueError("run/stop durations must be finite and non-negative")

        self._run_target_speed[env_ids] = speed
        self._run_stop_start_step[env_ids] = torch.clamp(
            torch.round(run_duration_s / self.step_dt).long(), min=1
        )
        self._run_stop_ramp_steps[env_ids] = torch.clamp(
            torch.round(ramp_duration_s / self.step_dt).long(), min=0
        )
        self._run_stop_started[env_ids] = False
        self._run_stop_schedule_active[env_ids] = True
        self._run_stop_settled[env_ids] = False
        self._run_stop_origin_w[env_ids] = 0.0
        self._run_stop_target_w[env_ids] = 0.0
        self._commands[env_ids] = 0.0
        self._commands[env_ids, 0] = speed
        self._nominal_yaw_command[env_ids] = 0.0

    def _update_run_stop_commands(self) -> None:
        """Advance command ramps before the policy observes its next command."""

        active = self._run_stop_schedule_active
        if not torch.any(active):
            return
        elapsed_steps = self._steps_since_reset - self._run_stop_start_step + 1
        started = active & (elapsed_steps > 0)
        just_started = started & ~self._run_stop_started
        if torch.any(just_started):
            current_xy = self._robot.data.root_pos_w.torch[:, :2]
            self._run_stop_origin_w[just_started] = current_xy[just_started]
            ramp_duration_s = (
                self._run_stop_ramp_steps[just_started].float() * self.step_dt
            )
            expected_deceleration_distance = (
                0.5 * self._run_target_speed[just_started] * ramp_duration_s
            )
            self._run_stop_target_w[just_started] = (
                current_xy[just_started]
                + self._episode_forward_direction_w[just_started]
                * expected_deceleration_distance.unsqueeze(1)
            )
            if self._phase_table_lateral_reference_enabled:
                if self._applied_expected_lateral_displacement is None:
                    raise RuntimeError("Stop anchoring requires the last applied planned lateral reference")
                # Hold the de-swayed line, not the temporary +/-body offset.
                # Use the cached phase actually applied to the current state,
                # before the next command update can park/advance the gait.
                self._run_stop_target_w[just_started] -= (
                    self._episode_lateral_direction_w[just_started]
                    * self._applied_expected_lateral_displacement[just_started].unsqueeze(1)
                )
            if self._phase_table_forward_reference_enabled:
                if self._applied_expected_forward_displacement is None:
                    raise RuntimeError("Stop anchoring requires the last applied planned forward reference")
                # Do not turn a temporary forward weight transfer into an
                # additional stop distance. Actual residual drift is retained.
                self._run_stop_target_w[just_started] -= (
                    self._episode_forward_direction_w[just_started]
                    * self._applied_expected_forward_displacement[just_started].unsqueeze(1)
                )
        self._run_stop_started |= started

        ramp_steps = self._run_stop_ramp_steps
        ramp_fraction = torch.where(
            ramp_steps > 0,
            torch.clamp(
                elapsed_steps.float() / torch.clamp(ramp_steps.float(), min=1.0),
                min=0.0,
                max=1.0,
            ),
            torch.where(
                started,
                torch.ones_like(self._run_target_speed),
                torch.zeros_like(self._run_target_speed),
            ),
        )
        # Cubic smoothstep has zero slope at both ends and integrates to exactly
        # half the initial velocity over the ramp duration.
        smooth_fraction = torch.square(ramp_fraction) * (3.0 - 2.0 * ramp_fraction)
        scheduled_speed = self._run_target_speed * (1.0 - smooth_fraction)
        self._commands[:, 0] = torch.where(
            active,
            scheduled_speed,
            self._commands[:, 0],
        )
        self._commands[:, 1] = torch.where(
            active,
            torch.zeros_like(self._commands[:, 1]),
            self._commands[:, 1],
        )
        self._run_stop_settled = active & started & (
            (ramp_steps == 0) | (elapsed_steps >= ramp_steps)
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
        self._update_run_stop_commands()
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
        if self.cfg.action_history_mode == "requested_action":
            action_history = self._previous_actions
        elif self.cfg.action_history_mode == "applied_target":
            if self.cfg.action_mode not in (
                "direct",
                "gait_residual",
                "integrated_target_velocity",
            ):
                raise ValueError(
                    "applied_target action history requires direct, "
                    "gait-residual, or integrated-target-velocity actions"
                )
            action_history = torch.clamp(
                (
                    self._previous_targets
                    - self._robot.data.default_joint_pos.torch
                    - self._joint_target_bias
                )
                / self._joint_scale,
                -1.0,
                1.0,
            )
        else:
            raise ValueError(
                f"Unsupported action_history_mode: {self.cfg.action_history_mode}"
            )
        policy_observation = torch.cat(
            (
                self._commands * self._command_observation_scale,
                gait_clock,
                imu_angular_velocity,
                projected_gravity,
                linear_acceleration_g,
                self._robot.data.joint_pos.torch - self._robot.data.default_joint_pos.torch,
                self._robot.data.joint_vel.torch / SERVO_VELOCITY_LIMIT_RAD_S,
                action_history,
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
        if self._critic_context_enabled:
            context = self._build_critic_causal_context()
            # Both arms assemble and retain exactly the same causal features.
            # Only the critic receives zeros in the matched information control.
            if self._critic_context_mode == "context90_zero":
                context = torch.zeros_like(context)
            critic_observation = torch.cat((critic_observation, context), dim=-1)
        return {
            "policy": torch.clamp(policy_observation, -20.0, 20.0),
            "critic": torch.clamp(critic_observation, -20.0, 20.0),
        }

    def _build_critic_causal_context(self) -> torch.Tensor:
        """Read post-reward history and the current pre-action command state.

        Never advance reward queues, commands, phase, sensors, RNG or physics.
        Repeated observation calls are idempotent except these reporting copies.
        """
        history = self._critic_context_history_cache
        valid = self._critic_context_history_valid
        age = self._steps_since_reset
        if (history.shape != (self.num_envs, 4)
            or torch.any(valid & (self._critic_context_history_step != age))
            or torch.any(~valid & (self._critic_context_history_step != -1))
            or torch.any(~valid & (age != 0))
            or torch.any(~valid.unsqueeze(1) & (history != 0.0))
            or not torch.all(self._run_stop_schedule_active)):
            raise RuntimeError("Critic context history is stale, uninitialized or outside the run-stop schedule")
        counts = torch.cat((
            self._episode_scheduled_swing_success_by_foot,
            self._episode_scheduled_swing_opportunity_by_foot,
            self._cycle_release_consecutive_steps.float(),
            age.float().unsqueeze(1),
            self._target_packet_hold_steps.float().unsqueeze(1),
        ), dim=1)
        if (not torch.all(torch.isfinite(history)) or not torch.all(torch.isfinite(counts))
            or torch.any(counts < 0.0)
            or torch.any(self._episode_scheduled_swing_success_by_foot > self._episode_scheduled_swing_opportunity_by_foot)
            or torch.any(self._target_packet_hold_steps > self._target_packet_max_hold_steps)
            or torch.any(valid & ((history[:, 1] < 1.0) | (history[:, 1] > self._sustained_window_steps)))):
            raise RuntimeError("Critic context contains invalid reward counts or packet history")

        def count_squash(value: torch.Tensor) -> torch.Tensor:
            value = value.float()
            return value / (value + 180.0)

        def signed_squash(value: torch.Tensor, scale: float) -> torch.Tensor:
            return value / (torch.abs(value) + scale)

        foot_order = self._critic_context_foot_indices
        displacement_xy = self._robot.data.root_pos_w.torch[:, :2] - self._episode_start_position[:, :2]
        lateral_displacement = torch.sum(displacement_xy * self._episode_lateral_direction_w, dim=1)
        # Exactly the same fraction/abrupt branch as the existing command update.
        elapsed = age - self._run_stop_start_step + 1
        ramp_steps = self._run_stop_ramp_steps
        ramp_fraction = torch.where(
            ramp_steps > 0,
            torch.clamp(elapsed.float() / torch.clamp(ramp_steps.float(), min=1.0), 0.0, 1.0),
            torch.where(self._run_stop_started, torch.ones_like(self._run_target_speed), torch.zeros_like(self._run_target_speed)),
        )
        stop_error_xy = torch.where(
            self._run_stop_started.unsqueeze(1),
            self._robot.data.root_pos_w.torch[:, :2] - self._run_stop_target_w,
            torch.zeros_like(self._run_stop_target_w),
        )
        if self.cfg.smoothness_curriculum_steps > 0:
            upcoming_fraction = min(
                float(self.common_step_counter + 1 + self.cfg.command_curriculum_offset_steps)
                / float(self.cfg.smoothness_curriculum_steps), 1.0,
            )
            upcoming_smoothness = self.cfg.smoothness_initial_scale + (1.0 - self.cfg.smoothness_initial_scale) * upcoming_fraction
        else:
            upcoming_smoothness = 1.0
        context = torch.cat((
            torch.stack((
                signed_squash(history[:, 0], 0.05),
                history[:, 1] / float(self._sustained_window_steps),
                signed_squash(history[:, 2], 0.05),
                signed_squash(lateral_displacement, 0.008),
                self._episode_heading_error() / torch.pi,
                signed_squash(history[:, 3], 0.02),
            ), dim=1),
            count_squash(self._episode_scheduled_swing_success_by_foot[:, foot_order]),
            count_squash(self._episode_scheduled_swing_opportunity_by_foot[:, foot_order]),
            self._cycle_release_pass_by_foot[:, foot_order].float(),
            torch.clamp(self._cycle_release_consecutive_steps[:, foot_order].float(), max=5.0) / 5.0,
            torch.stack((
                count_squash(age),
                count_squash(torch.clamp(self._run_stop_start_step - age, min=0)),
                ramp_fraction,
                count_squash(torch.clamp(ramp_steps - torch.clamp(elapsed, min=0), min=0)),
                self._run_stop_settled.float(),
                signed_squash(torch.sum(stop_error_xy * self._episode_forward_direction_w, dim=1), 0.05),
                signed_squash(torch.sum(stop_error_xy * self._episode_lateral_direction_w, dim=1), 0.05),
                torch.full_like(self._run_target_speed, upcoming_smoothness),
                self._target_packet_hold_steps.float() / float(self._target_packet_max_hold_steps),
                valid.float(),
            ), dim=1),
        ), dim=1)
        if (context.shape != (self.num_envs, len(CRITIC_CONTEXT_FEATURE_NAMES))
            or len(CRITIC_CONTEXT_FEATURE_NAMES) != 32
            or not torch.all(torch.isfinite(context)) or torch.any(torch.abs(context) > 1.0)
            or not 0.25 <= upcoming_smoothness <= 1.0):
            raise RuntimeError("Critic causal context v1 must contain 32 finite bounded values")
        self._critic_context_last_raw.copy_(context.detach())
        self._critic_context_last_observation_step.copy_(age)
        return context

    def _cache_critic_context_history(
        self, rolling_speed: torch.Tensor, rolling_counts: torch.Tensor,
        cycle_lateral_velocity: torch.Tensor,
    ) -> None:
        """Retain actual existing reward quantities once, with no new history update."""
        if torch.any(self._critic_context_history_valid & (self._critic_context_history_step >= self._steps_since_reset)):
            raise RuntimeError("Critic reward history cannot be updated twice at one control step")
        next_slot = torch.remainder(self._steps_since_reset + 1, self._sustained_window_steps)
        values = torch.stack((
            rolling_speed, rolling_counts,
            self._rolling_command_speed[self._env_indices, next_slot],
            cycle_lateral_velocity,
        ), dim=1)
        if values.shape != (self.num_envs, 4) or not torch.all(torch.isfinite(values)):
            raise RuntimeError("Cannot retain nonfinite critic reward history")
        self._critic_context_history_cache.copy_(values.detach())
        self._critic_context_history_step.copy_(self._steps_since_reset)
        self._critic_context_history_valid.fill_(True)

    def _capture_native_effort_reward_sample(self) -> None:
        """Copy one completed-physics projection and the retained PD estimate.

        The projection is active-DOF net effort, not identified drive-only
        torque. The paired PD buffer was computed before the last implicit
        solve; sharing a capture counter does not make their timing identical.
        Getter failure, mapping mismatch or nonfinite data stops the experiment.
        """
        counter = int(self._sim_step_counter)
        if counter <= 0 or counter % self.cfg.decimation != 0:
            raise RuntimeError("Native effort reward sampling requires the last completed physics substep")
        native_effort = wp.to_torch(
            self._native_effort_view.get_dof_projected_joint_forces()
        ).detach().clone()
        approximate_pd = self._robot.data.applied_torque.torch.detach().clone()
        for name, value in (("native projected effort", native_effort), ("retained approximate PD", approximate_pd)):
            if (tuple(value.shape) != (self.num_envs, len(EXPECTED_ACTION_ORDER))
                or value.dtype != self._native_effort_last_nm.dtype
                or value.device != self._native_effort_last_nm.device
                or not bool(torch.isfinite(value).all().item())):
                raise RuntimeError(f"Invalid {name} reward sample; no fallback is permitted")
        self._native_effort_last_nm.copy_(native_effort)
        self._native_effort_last_pd_nm.copy_(approximate_pd)
        self._native_effort_last_physics_counter.fill_(counter)

    def _get_rewards(self) -> torch.Tensor:
        if self._native_effort_telemetry_enabled:
            # Exactly the same once-per-control-step getter in both paired
            # arms, after DirectRLEnv's last physics solve and before resets.
            self._capture_native_effort_reward_sample()
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
        instantaneous_commanded_velocity = commanded_velocity
        if self._phase_table_forward_reference_enabled:
            if self._applied_expected_forward_velocity is None:
                raise RuntimeError("Applied planned forward velocity was not cached before reward")
            # Only instantaneous tracking/backward/overspeed penalties remove
            # this phase-locked transfer velocity. Never add it to command,
            # net distance, raw rolling speed, stall gates or speed telemetry.
            instantaneous_commanded_velocity = (
                commanded_velocity - self._applied_expected_forward_velocity
            )
            forward_error = forward_error - self._applied_expected_forward_velocity
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
        # During speed changes a fixed-time lookback is not exactly a cycle.
        # Removing planned sway before storing history prevents that mismatch
        # from being scored as drift. Preserve legacy history semantics.
        cycle_history_displacement = (
            lateral_tracking_displacement
            if self._phase_table_lateral_reference_enabled else net_lateral_displacement
        )
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
            cycle_history_displacement - prior_cycle_displacement
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
        ] = cycle_history_displacement
        lateral_corridor_excess = torch.clamp(
            torch.abs(lateral_tracking_displacement)
            - self.cfg.lateral_corridor_half_width_m,
            min=0.0,
        )
        active_translation = (
            command_speed > self.cfg.active_translation_threshold_m_s
        )
        rolling_velocity_tracking = torch.where(
            rolling_window_ready & active_translation,
            torch.exp(
                -torch.square(rolling_speed - command_speed)
                / self._rolling_velocity_tracking_sigma_m_s**2
            ),
            torch.zeros_like(rolling_speed),
        )
        normalized_rolling_velocity_error = torch.where(
            rolling_window_ready & active_translation,
            torch.clamp(
                (rolling_speed - command_speed)
                / torch.clamp(command_speed, min=1.0e-4),
                min=-2.0,
                max=2.0,
            ),
            torch.zeros_like(rolling_speed),
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
        progress_velocity = commanded_velocity
        if self._phase_table_forward_reference_enabled:
            # Positive weight-transfer half-cycles must not earn instantaneous
            # locomotion credit. Use actual NET rolling speed, with a full
            # window warmup. A fixed window (e.g. 12 s / 1.2 cycles) can retain
            # some endpoint phase displacement; this is not an exact-cycle
            # estimator and deliberately never subtracts a planned velocity
            # from the net-speed objective.
            progress_velocity = torch.where(
                rolling_window_ready, rolling_speed, torch.zeros_like(rolling_speed)
            )
        instant_progress = torch.where(
            active_translation,
            torch.clamp(
                progress_velocity / torch.clamp(command_speed, min=1.0e-4),
                min=-1.0,
                max=1.0,
            ),
            torch.zeros_like(commanded_velocity),
        )
        straight_aligned_progress = torch.where(
            active_translation,
            torch.clamp(
                progress_velocity / torch.clamp(command_speed, min=1.0e-4),
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
        if self._phase_table_forward_reference_enabled:
            sustained_progress = torch.where(
                rolling_window_ready | (sustained_progress <= 0.0),
                sustained_progress,
                torch.zeros_like(sustained_progress),
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
            -instantaneous_commanded_velocity / torch.clamp(command_speed, min=1.0e-4),
            min=0.0,
            max=2.0,
        )
        normalized_overspeed = torch.clamp(
            (instantaneous_commanded_velocity - command_speed)
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
        target_pitch = torch.full_like(
            command_speed,
            float(self.cfg.target_forward_pitch_rad),
        )
        if self.cfg.target_standstill_pitch_rad is not None:
            target_pitch = torch.where(
                command_speed <= float(self.cfg.gait_standstill_deadband_m_s),
                torch.full_like(
                    target_pitch,
                    float(self.cfg.target_standstill_pitch_rad),
                ),
                target_pitch,
            )
        target_projected_gravity_x = torch.sin(target_pitch)
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
        target_base_height = torch.full_like(
            base_height,
            float(self.cfg.target_base_height_m),
        )
        if self.cfg.target_standstill_base_height_m is not None:
            target_base_height = torch.where(
                command_speed <= float(self.cfg.gait_standstill_deadband_m_s),
                torch.full_like(
                    target_base_height,
                    float(self.cfg.target_standstill_base_height_m),
                ),
                target_base_height,
            )
        height_error = base_height - target_base_height
        if self._normalized_posture_cost_enabled:
            # For this RPY/XYZW convention, body-projected unit gravity is
            # (sin(pitch), -sin(roll)*cos(pitch), -cos(roll)*cos(pitch)).
            # Thus positive pitch agrees with the existing target g_x=sin(p).
            # Gravity defines unambiguous roll/pitch in the existing upright
            # operating domain and avoids relying on a quaternion-order guess.
            gravity = self._robot.data.projected_gravity_b.torch
            actual_roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
            actual_pitch = torch.atan2(
                gravity[:, 0], torch.sqrt(torch.square(gravity[:, 1]) + torch.square(gravity[:, 2]))
            )
            body_height_cost_term = -self.cfg.penalty_normalized_body_height * _deadband_unit_huber(
                height_error, self.cfg.posture_height_deadband_m, self.cfg.posture_height_normalizer_m
            )
            body_tilt_cost_term = (
                -self.cfg.penalty_normalized_body_roll * _deadband_unit_huber(
                    actual_roll, self.cfg.posture_angle_deadband_rad, self.cfg.posture_angle_normalizer_rad
                )
                -self.cfg.penalty_normalized_body_pitch * _deadband_unit_huber(
                    actual_pitch - target_pitch, self.cfg.posture_angle_deadband_rad, self.cfg.posture_angle_normalizer_rad
                )
            )
            # Targets retain the authored moving/standstill switch. Planned XY
            # translation is not penalized here. Startup/homing and natural
            # attitude/height excursions remain soft costs, not hard gates or
            # new termination conditions. The separate rearward cost remains.
        else:
            # Keep legacy arithmetic, term order, keys and reward scaling exact.
            body_height_cost_term = -self.cfg.penalty_body_height * torch.square(height_error)
            body_tilt_cost_term = (
                -self.cfg.penalty_body_tilt
                * torch.sum(torch.square(body_tilt_error), dim=1)
            )
            if self._normalized_body_height_cost_enabled:
                # Motion-only reward hypothesis: actual speed/height/attitude
                # never controls authority. Preserve the existing target and
                # projected-gravity tilt; only the height cost is crossfaded.
                deadband = float(self.cfg.gait_standstill_deadband_m_s)
                span = self._run_target_speed - deadband
                scheduled = self._run_stop_schedule_active
                # Active spans are strictly positive by schedule initialization.
                # Inactive entries may still be zero; avoid division by zero
                # without perturbing any genuinely positive, even tiny, span.
                safe_span = torch.where(span > 0.0, span, torch.ones_like(span))
                scheduled_alpha = torch.clamp((command_speed - deadband) / safe_span, min=0.0, max=1.0)
                alpha = torch.where(scheduled, scheduled_alpha, (command_speed > deadband).float())
                alpha = torch.where(scheduled & self._run_stop_settled, torch.zeros_like(alpha), alpha)
                # C2 at alpha0/1; no second command clock or new motor target.
                blend = alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)
                normalized_height = -self.cfg.penalty_normalized_body_height * _deadband_unit_huber(
                    height_error, self.cfg.posture_height_deadband_m, self.cfg.posture_height_normalizer_m
                )
                body_height_cost_term = torch.where(
                    alpha > 0.0,
                    body_height_cost_term + blend * (normalized_height - body_height_cost_term),
                    body_height_cost_term,
                )
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
        applied_normalized_target = torch.clamp(
            (
                self._processed_actions
                - self._robot.data.default_joint_pos.torch
                - self._joint_target_bias
            )
            / self._joint_scale,
            -1.0,
            1.0,
        )
        action_saturation = torch.mean(
            torch.square(torch.clamp(torch.abs(self._actions) - 0.80, min=0.0)),
            dim=1,
        )
        abduction_action_magnitude = torch.mean(
            torch.square(self._actions[:, self._leg_abduction_joint_ids]),
            dim=1,
        )
        abduction_target_magnitude = torch.mean(
            torch.square(
                applied_normalized_target[:, self._leg_abduction_joint_ids]
            ),
            dim=1,
        )
        target_limiter_gap = torch.mean(
            torch.square(
                (self._desired_targets - self._processed_actions)
                / torch.clamp(
                    self._joint_scale,
                    min=1.0e-4,
                )
            ),
            dim=1,
        )
        coordination_action = (
            applied_normalized_target
            if self.cfg.action_mode == "integrated_target_velocity"
            else self._actions
        )
        leg_actions = torch.stack(
            (
                coordination_action[:, self._leg_abduction_joint_ids],
                coordination_action[:, self._leg_hip_joint_ids],
                coordination_action[:, self._leg_knee_joint_ids],
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
        reward_effort_fraction = applied_effort_fraction
        if self._native_effort_telemetry_enabled:
            native_effort_fraction = torch.abs(self._native_effort_last_nm) / effort_capacity
            native_effort_squared = torch.mean(torch.square(self._native_effort_last_nm), dim=1)
            native_effort_soft_limit_rate = torch.mean(
                (native_effort_fraction > effort_soft_limit).float(), dim=1
            )
            if self._native_effort_reward_enabled:
                # The single treatment change applies to all 12 active DOFs
                # in run, ramp AND hold. It does not alter the stop controller.
                reward_effort_fraction = native_effort_fraction
        normalized_effort_excess = torch.clamp(
            (reward_effort_fraction - effort_soft_limit)
            / max(1.0 - effort_soft_limit, 1.0e-4),
            min=0.0,
            max=3.0,
        )
        effort_soft_limit_penalty = torch.mean(
            torch.square(normalized_effort_excess), dim=1
        )
        normalized_effort = reward_effort_fraction / max(
            effort_soft_limit, 1.0e-4
        )
        effort_rms_penalty = torch.mean(torch.square(normalized_effort), dim=1)
        # Preserve the historical approximate-PD diagnostics and gates in BOTH
        # arms. Only the four reward costs use reward_effort_fraction.
        effort_soft_limit_rate = torch.mean(
            (applied_effort_fraction > effort_soft_limit).float(), dim=1
        )
        maximum_applied_effort_fraction = torch.amax(
            applied_effort_fraction, dim=1
        )
        no_load_speed = float(self.cfg.actuator_torque_speed_no_load_rad_s)
        if not math.isfinite(no_load_speed) or no_load_speed <= 0.0:
            raise ValueError(
                "actuator_torque_speed_no_load_rad_s must be finite and positive"
            )
        torque_speed_margin = float(self.cfg.actuator_torque_speed_margin)
        if not math.isfinite(torque_speed_margin) or not 0.0 < torque_speed_margin <= 1.0:
            raise ValueError(
                "actuator_torque_speed_margin must be finite and within (0, 1]"
            )
        joint_speed_fraction = torch.abs(
            self._robot.data.joint_vel.torch
        ) / no_load_speed
        actuator_operating_point = reward_effort_fraction + joint_speed_fraction
        torque_speed_envelope_penalty = torch.mean(
            torch.square(
                torch.clamp(
                    actuator_operating_point - torque_speed_margin,
                    min=0.0,
                    max=2.0,
                )
            ),
            dim=1,
        )
        leg_operating_point = torch.mean(
            torch.stack(
                (
                    actuator_operating_point[:, self._leg_abduction_joint_ids],
                    actuator_operating_point[:, self._leg_hip_joint_ids],
                    actuator_operating_point[:, self._leg_knee_joint_ids],
                ),
                dim=2,
            ),
            dim=2,
        )
        leg_operating_point_imbalance = torch.mean(
            torch.square(
                leg_operating_point
                - torch.mean(leg_operating_point, dim=1, keepdim=True)
            ),
            dim=1,
        )

        foot_forces = self._foot_forces()
        foot_contact = foot_forces > self._scheduled_release_force_threshold_n
        stop_hold = self._run_stop_settled
        stop_planar_speed = torch.linalg.norm(
            self._robot.data.root_lin_vel_w.torch[:, :2], dim=1
        )
        stop_velocity_score = torch.exp(
            -torch.square(
                stop_planar_speed / max(float(self.cfg.stop_velocity_sigma_m_s), 1.0e-4)
            )
        )
        stop_position_error = torch.linalg.norm(
            self._robot.data.root_pos_w.torch[:, :2] - self._run_stop_target_w,
            dim=1,
        )
        normalized_stop_position_error = stop_position_error / max(
            float(self.cfg.stop_position_tolerance_m), 1.0e-4
        )
        stop_position_cost = torch.where(
            normalized_stop_position_error <= 1.0,
            0.5 * torch.square(normalized_stop_position_error),
            normalized_stop_position_error - 0.5,
        )
        stop_all_feet_contact = torch.all(foot_contact, dim=1)
        stop_neutral_state = (
            applied_normalized_target
            if self.cfg.action_mode == "integrated_target_velocity"
            else self._actions
        )
        stop_neutral_action_score = torch.exp(
            -torch.mean(torch.square(stop_neutral_state), dim=1)
            / max(float(self.cfg.stop_neutral_action_sigma), 1.0e-4) ** 2
        )
        stop_target_magnitude = torch.mean(
            torch.square(applied_normalized_target), dim=1
        )
        stop_stable = (
            stop_hold
            & (stop_planar_speed <= float(self.cfg.stop_velocity_sigma_m_s))
            & (
                stop_position_error
                <= float(self.cfg.stop_position_tolerance_m)
            )
            & stop_all_feet_contact
            & (upright_cosine >= self.cfg.minimum_upright_cosine)
        )
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
        stop_weakest_foot_contact_quality = torch.amin(
            scheduled_stance_contact_quality, dim=1
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
                scheduled_release_gate_quality * scheduled_swing.float()
            )
            discovery_release_quality = torch.sigmoid(
                (self._scheduled_release_force_threshold_n - foot_forces)
                / self._scheduled_release_discovery_width_n
            )
            scheduled_release_shaping_quality = (
                (
                    scheduled_release_gate_quality
                    if self.cfg.normalize_threshold_centered_release_quality
                    else discovery_release_quality
                )
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
            scheduled_release_shaping_quality = scheduled_release_quality
        scheduled_swing_success = ((~foot_contact) & scheduled_swing).float()
        scheduled_swing_opportunity = scheduled_swing.float()
        scheduled_pair_active = swing_count > 1.5
        scheduled_pair_release_quality = torch.amin(
            torch.where(
                scheduled_swing,
                scheduled_release_quality,
                torch.ones_like(scheduled_release_quality),
            ),
            dim=1,
        )
        scheduled_pair_mean_quality = torch.sum(
            scheduled_release_quality, dim=1
        ) / torch.clamp(swing_count, min=1.0)
        scheduled_pair_release_mismatch = torch.sum(
            torch.square(
                scheduled_release_quality
                - scheduled_pair_mean_quality.unsqueeze(1)
            )
            * scheduled_swing.float(),
            dim=1,
        ) / torch.clamp(swing_count, min=1.0)
        if self.cfg.gate_progress_by_scheduled_pair_release:
            pair_gate_floor = float(
                self.cfg.scheduled_pair_progress_gate_floor
            )
            if not 0.0 <= pair_gate_floor <= 1.0:
                raise ValueError(
                    "scheduled_pair_progress_gate_floor must be in [0, 1]"
                )
            scheduled_pair_progress_gate = torch.where(
                scheduled_pair_active & active_translation,
                pair_gate_floor
                + (1.0 - pair_gate_floor)
                * scheduled_pair_release_quality,
                torch.ones_like(scheduled_pair_release_quality),
            )
        else:
            scheduled_pair_progress_gate = torch.ones_like(
                scheduled_pair_release_quality
            )
        scheduled_release_shaping_score = torch.where(
            swing_count > 0.0,
            scheduled_release_shaping_quality.sum(dim=1)
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
            gait_reference_error_by_joint = torch.square(self._actions)
        elif self.cfg.action_mode == "integrated_target_velocity":
            target_reference_error_by_joint = torch.square(
                applied_normalized_target - gait_reference
            )
            actual_normalized_position = torch.clamp(
                (
                    self._robot.data.joint_pos.torch
                    - self._robot.data.default_joint_pos.torch
                    - self._joint_target_bias
                )
                / self._joint_scale,
                -1.0,
                1.0,
            )
            actual_reference_error_by_joint = torch.square(
                actual_normalized_position - gait_reference
            )
            actual_state_weight = float(
                self.cfg.gait_reference_actual_state_weight
            )
            gait_reference_error_by_joint = torch.lerp(
                target_reference_error_by_joint,
                actual_reference_error_by_joint,
                actual_state_weight,
            )
        else:
            gait_reference_error_by_joint = torch.square(
                self._actions - gait_reference
            )
        mean_gait_reference_error = torch.mean(
            gait_reference_error_by_joint, dim=1
        )
        per_leg_gait_reference_error = torch.mean(
            torch.stack(
                (
                    gait_reference_error_by_joint[
                        :, self._leg_abduction_joint_ids
                    ],
                    gait_reference_error_by_joint[:, self._leg_hip_joint_ids],
                    gait_reference_error_by_joint[:, self._leg_knee_joint_ids],
                ),
                dim=2,
            ),
            dim=2,
        )
        worst_leg_gait_reference_error = torch.amax(
            per_leg_gait_reference_error, dim=1
        )
        gait_reference_error = torch.lerp(
            mean_gait_reference_error,
            worst_leg_gait_reference_error,
            float(self.cfg.gait_reference_worst_leg_weight),
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
        stance_sole_tilt, swing_corner_clearance, contact_point_slip = self._sole_contact_costs(
            scheduled_contact, foot_contact, applied_gait_phase
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
                * scheduled_pair_progress_gate
            ),
            "rolling_velocity_tracking": (
                self.cfg.reward_rolling_velocity_tracking
                * rolling_velocity_tracking
                * straight_progress_gate
                * four_leg_progress_gate
                * schedule_matched_progress_gate
                * scheduled_pair_progress_gate
            ),
            "rolling_velocity_error": (
                -self.cfg.penalty_rolling_velocity_error
                * torch.square(normalized_rolling_velocity_error)
            ),
            "instant_progress": (
                self.cfg.reward_instant_progress
                * gated_instant_progress
                * four_leg_progress_gate
                * schedule_matched_progress_gate
                * scheduled_pair_progress_gate
            ),
            "straight_aligned_progress": (
                self.cfg.reward_straight_aligned_progress
                * straight_aligned_progress
                * four_leg_progress_gate
                * schedule_matched_progress_gate
                * scheduled_pair_progress_gate
            ),
            "sustained_progress": (
                self.cfg.reward_sustained_progress
                * gated_sustained_progress
                * four_leg_progress_gate
                * schedule_matched_progress_gate
                * (1.0 if self._decouple_sustained_progress_from_pair_release else scheduled_pair_progress_gate)
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
            "scheduled_pair_release": (
                self.cfg.reward_scheduled_pair_release
                * scheduled_pair_release_quality
                * scheduled_pair_active.float()
            ),
            "scheduled_pair_mismatch": (
                -self.cfg.penalty_scheduled_pair_mismatch
                * scheduled_pair_release_mismatch
                * scheduled_pair_active.float()
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
            "body_tilt": body_tilt_cost_term,
            "yaw_rate": -self.cfg.penalty_yaw_rate * torch.square(yaw_error),
            "heading_error": (
                -self.cfg.penalty_heading_error
                * heading_error_cost
            ),
            "body_height": body_height_cost_term,
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
            "abduction_action_magnitude": (
                -self.cfg.penalty_abduction_action_magnitude
                * abduction_action_magnitude
            ),
            "abduction_target_magnitude": (
                -self.cfg.penalty_abduction_target_magnitude
                * abduction_target_magnitude
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
            "effort_rms": (
                -self.cfg.penalty_effort_rms * effort_rms_penalty
            ),
            "actuator_torque_speed_envelope": (
                -self.cfg.penalty_actuator_torque_speed_envelope
                * torque_speed_envelope_penalty
            ),
            "leg_operating_point_imbalance": (
                -self.cfg.penalty_leg_operating_point_imbalance
                * leg_operating_point_imbalance
            ),
            "support_foot_slip": (
                -self.cfg.penalty_support_foot_slip * support_foot_slip
            ),
            "stance_sole_tilt": -self.cfg.penalty_stance_sole_tilt * stance_sole_tilt,
            "swing_corner_clearance": -self.cfg.penalty_swing_corner_clearance * swing_corner_clearance,
            "contact_point_slip": -self.cfg.penalty_contact_point_slip * contact_point_slip,
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
            "stop_velocity": (
                self.cfg.reward_stop_velocity
                * stop_velocity_score
                * stop_hold.float()
            ),
            "stop_all_feet_contact": (
                self.cfg.reward_stop_all_feet_contact
                * stop_all_feet_contact.float()
                * stop_hold.float()
            ),
            "stop_weakest_foot_contact": (
                self.cfg.reward_stop_weakest_foot_contact
                * stop_weakest_foot_contact_quality
                * stop_hold.float()
            ),
            "stop_neutral_action": (
                self.cfg.reward_stop_neutral_action
                * stop_neutral_action_score
                * stop_hold.float()
            ),
            "stop_action_magnitude": (
                -self.cfg.penalty_stop_action_magnitude
                * action_magnitude
                * stop_hold.float()
            ),
            "stop_target_magnitude": (
                -self.cfg.penalty_stop_target_magnitude
                * stop_target_magnitude
                * stop_hold.float()
            ),
            "stop_position": (
                -self.cfg.penalty_stop_position
                * stop_position_cost
                * stop_hold.float()
            ),
            "stop_yaw_rate": (
                -self.cfg.penalty_stop_yaw_rate
                * torch.square(angular_velocity[:, 2])
                * stop_hold.float()
            ),
            "stop_joint_velocity": (
                -self.cfg.penalty_stop_joint_velocity
                * joint_speed
                * stop_hold.float()
            ),
            "termination": -self.cfg.penalty_termination * self._failed.float(),
        }
        reward = (
            torch.stack(tuple(reward_terms.values()), dim=0).sum(dim=0)
            * self.cfg.reward_scale
        )
        if self._swing_sole_tilt_weight:
            if self._swing_sole_tilt_last["physics_counter"] != int(self._sim_step_counter):
                raise RuntimeError("Stale swing sole tilt reward snapshot")
            term = -self._swing_sole_tilt_weight*self._swing_sole_tilt_last["cost"]
            reward = reward + term*self.cfg.reward_scale
            reward_terms["swing_sole_tilt"] = term
        if self._normalized_contact_slip_study:
            # Retain the original raw-m^2/s^2 slip term and its reduction order.
            # Explicit weight0 records the identical cost without adding zero
            # to the reward. The added objective is dimensionless and run-only.
            eligible = (~self._run_stop_started) & (
                torch.linalg.vector_norm(self._commands[:,:2],dim=1) > self.cfg.gait_standstill_deadband_m_s)
            normalized_cost = torch.clamp(contact_point_slip/(.015**2),min=0.,max=25.)*eligible.float()
            self._normalized_contact_slip_last = {
                "physics_counter": int(self._sim_step_counter),
                "raw_slip_m2_s2": contact_point_slip.detach().clone(),
                "eligible": eligible.detach().clone(), "cost": normalized_cost.detach().clone(),
            }
            if self._normalized_contact_slip_weight:
                term = -self._normalized_contact_slip_weight*normalized_cost
                reward = reward+term*self.cfg.reward_scale
                reward_terms["normalized_contact_slip"] = term

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
        self._episode_scheduled_pair_progress_gate_sum += (
            scheduled_pair_progress_gate
        )
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
        self._episode_stall_opportunity_sum += (rolling_window_ready & active_translation).float()
        if self._actuator_target_response_enabled:
            self._episode_actuator_response_gap_squared_sum += torch.mean(
                torch.square(self._processed_actions - self._actuator_response_targets), dim=1
            )
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
        if self._native_effort_telemetry_enabled:
            self._episode_native_effort_squared_sum += native_effort_squared
            self._episode_native_effort_soft_limit_step_sum += native_effort_soft_limit_rate
        self._episode_stop_hold_step_sum += stop_hold.float()
        self._episode_stop_speed_sum += stop_planar_speed * stop_hold.float()
        self._episode_stop_position_error_sum += (
            stop_position_error * stop_hold.float()
        )
        self._episode_stop_all_feet_contact_sum += (
            stop_all_feet_contact.float() * stop_hold.float()
        )
        self._episode_stop_stable_step_sum += stop_stable.float()
        self._episode_min_rolling_speed = torch.where(
            rolling_window_ready & active_translation,
            torch.minimum(self._episode_min_rolling_speed, rolling_speed),
            self._episode_min_rolling_speed,
        )
        for name, term in reward_terms.items():
            self._episode_reward_term_sums[name] += term
        if self._critic_context_enabled:
            self._cache_critic_context_history(rolling_speed, rolling_counts, cycle_averaged_lateral_velocity)
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
        self._run_stop_schedule_active[env_ids] = False
        self._run_stop_started[env_ids] = False
        self._run_stop_settled[env_ids] = False
        run_stop_override = preview_control.RUN_STOP_OVERRIDE
        if run_stop_override is not None:
            speed_m_s, run_duration_s, ramp_duration_s = run_stop_override
            self._initialize_run_stop_schedule(
                env_ids,
                speed=torch.full(
                    (count,), float(speed_m_s), dtype=torch.float32, device=self.device
                ),
                run_duration_s=torch.full(
                    (count,), float(run_duration_s), dtype=torch.float32, device=self.device
                ),
                ramp_duration_s=torch.full(
                    (count,), float(ramp_duration_s), dtype=torch.float32, device=self.device
                ),
            )
            return
        override = preview_control.COMMAND_OVERRIDE
        if override is not None:
            self._commands[env_ids] = torch.tensor(
                override, dtype=torch.float32, device=self.device
            )
            self._nominal_yaw_command[env_ids] = self._commands[env_ids, 2]
            return

        if self.cfg.command_profile in ("forward", "run_stop"):
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
            sampled_speed = torch.empty(count, device=self.device).uniform_(
                speed_min, speed_max
            )
            low_band_fraction = float(
                self.cfg.forward_speed_low_band_fraction
            )
            if not 0.0 <= low_band_fraction <= 1.0:
                raise ValueError("forward_speed_low_band_fraction must be in [0, 1]")
            if low_band_fraction > 0.0:
                low_band_max = min(
                    float(self.cfg.forward_speed_low_band_max_m_s),
                    float(speed_max),
                )
                if low_band_max < float(speed_min):
                    raise ValueError(
                        "forward_speed_low_band_max_m_s must not be below the "
                        "current curriculum minimum"
                    )
                use_low_band = (
                    torch.rand(count, device=self.device) < low_band_fraction
                )
                low_band_speed = torch.empty(count, device=self.device).uniform_(
                    speed_min, low_band_max
                )
                sampled_speed = torch.where(
                    use_low_band, low_band_speed, sampled_speed
                )
            if self.cfg.command_profile == "run_stop":
                self._initialize_run_stop_schedule(env_ids, speed=sampled_speed)
                return
            self._commands[env_ids, 0] = sampled_speed
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
            episode_duration = completed_steps * self.step_dt
            if self._target_packet_hold_probability > 0.0:
                log["Metrics/target_packet_hold_rate"] = (
                    self._episode_target_packet_hold_sum[env_ids] / completed_steps
                ).mean().item()
            if self._actuator_target_response_enabled:
                log["Metrics/actuator_response_gap_rms_deg"] = torch.rad2deg(torch.sqrt(
                    self._episode_actuator_response_gap_squared_sum[env_ids] / completed_steps
                )).mean().item()
                for joint_index, joint_name in enumerate(EXPECTED_ACTION_ORDER):
                    log[f"Domain/response_tau_s_{joint_name}"] = self._actuator_response_tau_s[env_ids, joint_index].mean().item()
                    log[f"Domain/response_gain_{joint_name}"] = self._actuator_response_gain[env_ids, joint_index].mean().item()
            if self._sole_diagnostics_enabled:
                for foot_index, leg_name in enumerate(self._foot_leg_names):
                    stance_count = torch.clamp(self._episode_stance_sole_opportunities[env_ids, foot_index], min=1.0)
                    swing_count = torch.clamp(self._episode_swing_corner_opportunities[env_ids, foot_index], min=1.0)
                    contact_count = torch.clamp(self._episode_contact_point_opportunities[env_ids, foot_index], min=1.0)
                    log[f"Metrics/stance_sole_tilt_deg_{leg_name}"] = torch.rad2deg(
                        self._episode_stance_sole_tilt_sum[env_ids, foot_index] / stance_count
                    ).mean().item()
                    log[f"Metrics/core_swing_lowest_corner_mean_m_{leg_name}"] = (
                        self._episode_swing_corner_height_sum[env_ids, foot_index] / swing_count
                    ).mean().item()
                    minimum_height = self._episode_swing_corner_min_height[env_ids, foot_index]
                    log[f"Metrics/core_swing_lowest_corner_min_m_{leg_name}"] = torch.where(
                        torch.isfinite(minimum_height), minimum_height, torch.zeros_like(minimum_height)
                    ).mean().item()
                    log[f"Metrics/contact_point_slip_rms_m_s_{leg_name}"] = torch.sqrt(
                        self._episode_contact_point_speed_squared_sum[env_ids, foot_index] / contact_count
                    ).mean().item()
                    log[f"Metrics/core_swing_samples_{leg_name}"] = self._episode_swing_corner_opportunities[env_ids, foot_index].mean().item()
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
            # Logging-only correction for future processes; the running V44
            # training process retains its already imported logger. The reward
            # deliberately zeros _episode_commanded_distance at standstill, so
            # reading it at a run/stop episode's end falsely reported no travel.
            # Leave that reward/success state untouched. These legacy names
            # describe ACTUAL displacement/speed along the commanded world path,
            # not requested velocity; explicit aliases below remove ambiguity.
            logged_path_distance = (
                net_forward_displacement if self.cfg.track_episode_world_path
                else self._episode_commanded_distance[env_ids]
            )
            log["Metrics/commanded_distance_m"] = logged_path_distance.mean().item()
            log["Metrics/mean_commanded_speed_m_s"] = (
                logged_path_distance / episode_duration
            ).mean().item()
            log["Metrics/mean_actual_forward_speed_m_s"] = (
                net_forward_displacement / episode_duration
            ).mean().item()
            log["Metrics/requested_travel_distance_m"] = (
                self._episode_command_speed_sum[env_ids] * self.step_dt
            ).mean().item()
            log["Metrics/mean_requested_speed_m_s"] = (
                self._episode_command_speed_sum[env_ids] / completed_steps
            ).mean().item()
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
            # _get_dones increments the age before _get_rewards evaluates its
            # ready mask. A reconstructed age denominator was off by one on
            # short episodes and could report rates above 100%. This counter
            # changes only diagnostics, not the reward's readiness or values.
            mature_steps = torch.clamp(self._episode_stall_opportunity_sum[env_ids], min=1.0)
            log["Metrics/sustained_stall_rate"] = (
                self._episode_stall_step_sum[env_ids] / mature_steps
            ).mean().item()
            log["Metrics/sustained_stall_opportunity_samples"] = (
                self._episode_stall_opportunity_sum[env_ids].mean().item()
            )
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
            if self._native_effort_telemetry_enabled:
                log["Metrics/native_projected_effort_rms_nm"] = torch.sqrt(
                    self._episode_native_effort_squared_sum[env_ids] / completed_steps
                ).mean().item()
                log["Metrics/native_projected_effort_soft_limit_rate"] = (
                    self._episode_native_effort_soft_limit_step_sum[env_ids] / completed_steps
                ).mean().item()
            stop_hold_steps = self._episode_stop_hold_step_sum[env_ids]
            completed_stop = stop_hold_steps > 0.0
            safe_stop_steps = torch.clamp(stop_hold_steps, min=1.0)
            log["Metrics/stop_transition_completion_rate"] = (
                completed_stop.float().mean().item()
            )
            log["Metrics/mean_stop_hold_duration_s"] = (
                stop_hold_steps * self.step_dt
            ).mean().item()
            log["Metrics/mean_post_stop_speed_m_s"] = torch.where(
                completed_stop,
                self._episode_stop_speed_sum[env_ids] / safe_stop_steps,
                torch.zeros_like(stop_hold_steps),
            ).mean().item()
            log["Metrics/mean_post_stop_position_error_m"] = torch.where(
                completed_stop,
                self._episode_stop_position_error_sum[env_ids] / safe_stop_steps,
                torch.zeros_like(stop_hold_steps),
            ).mean().item()
            log["Metrics/post_stop_all_feet_contact_rate"] = torch.where(
                completed_stop,
                self._episode_stop_all_feet_contact_sum[env_ids] / safe_stop_steps,
                torch.zeros_like(stop_hold_steps),
            ).mean().item()
            log["Metrics/post_stop_stable_rate"] = torch.where(
                completed_stop,
                self._episode_stop_stable_step_sum[env_ids] / safe_stop_steps,
                torch.zeros_like(stop_hold_steps),
            ).mean().item()
            final_stop_position_error = torch.linalg.norm(
                self._robot.data.root_pos_w.torch[env_ids, :2]
                - self._run_stop_target_w[env_ids],
                dim=1,
            )
            log["Metrics/final_stop_position_error_m"] = torch.where(
                completed_stop,
                final_stop_position_error,
                torch.zeros_like(final_stop_position_error),
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
            log["Metrics/mean_scheduled_pair_progress_gate"] = (
                self._episode_scheduled_pair_progress_gate_sum[env_ids]
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
        self._processed_actions[env_ids] = joint_position
        self._actuator_response_targets[env_ids] = joint_position
        self._sample_actuator_target_response(env_ids)
        self._target_packet_elapsed_s[env_ids] = 0.0
        self._target_packet_hold_steps[env_ids] = 0
        self._target_packet_held[env_ids] = False
        self._episode_target_packet_hold_sum[env_ids] = 0.0
        self._episode_actuator_response_gap_squared_sum[env_ids] = 0.0
        self._episode_stance_sole_tilt_sum[env_ids] = 0.0
        self._episode_stance_sole_opportunities[env_ids] = 0.0
        self._episode_swing_corner_height_sum[env_ids] = 0.0
        self._episode_swing_corner_min_height[env_ids] = torch.inf
        self._episode_swing_corner_opportunities[env_ids] = 0.0
        self._episode_contact_point_speed_squared_sum[env_ids] = 0.0
        self._episode_contact_point_opportunities[env_ids] = 0.0
        self._desired_targets[env_ids] = joint_position
        self._previous_requested_targets[env_ids] = joint_position
        self._previous_joint_velocity[env_ids] = 0.0
        self._previous_body_linear_velocity[env_ids] = 0.0
        self._previous_body_angular_velocity[env_ids] = 0.0
        self._steps_since_reset[env_ids] = 0
        if self._rear_pitch_lift_diagnostics_enabled:
            self._rear_pitch_lift_last = None
        if self._critic_context_enabled:
            self._critic_context_history_cache[env_ids] = 0.0
            self._critic_context_history_valid[env_ids] = False
            self._critic_context_history_step[env_ids] = -1
            self._critic_context_last_raw[env_ids] = 0.0
            self._critic_context_last_observation_step[env_ids] = -1
        if self.cfg.randomize_gait_start_phase_quarters:
            self._gait_phase_offset[env_ids] = torch.randint(
                0,
                4,
                (len(env_ids),),
                device=self.device,
            ).float() * 0.25
        else:
            self._gait_phase_offset[env_ids] = 0.0
        self._gait_phase_accumulator[env_ids] = self._gait_phase_offset[env_ids]
        preview_stop_alignment = (
            preview_control.RUN_STOP_PHASE_ALIGNMENT_OVERRIDE
        )
        if (
            preview_stop_alignment is not None
            and preview_control.RUN_STOP_OVERRIDE is not None
        ):
            target_phase, tolerance, maximum_wait_s = preview_stop_alignment
            speed = self._run_target_speed[env_ids]
            speed_fraction = torch.clamp(
                (speed - float(self.cfg.gait_speed_min_m_s))
                / max(
                    float(self.cfg.gait_speed_max_m_s)
                    - float(self.cfg.gait_speed_min_m_s),
                    1.0e-6,
                ),
                0.0,
                1.0,
            )
            frequency_hz = float(self.cfg.gait_frequency_min_hz) + (
                speed_fraction
                * (
                    float(self.cfg.gait_frequency_max_hz)
                    - float(self.cfg.gait_frequency_min_hz)
                )
            )
            phase_at_nominal_stop = torch.remainder(
                self._gait_phase_offset[env_ids]
                + frequency_hz
                * self._run_stop_start_step[env_ids].float()
                * self.step_dt,
                1.0,
            )
            forward_cycles = torch.remainder(
                float(target_phase) - phase_at_nominal_stop,
                1.0,
            )
            circular_error = torch.minimum(
                forward_cycles,
                1.0 - forward_cycles,
            )
            alignment_wait_s = torch.where(
                circular_error <= float(tolerance),
                torch.zeros_like(forward_cycles),
                torch.minimum(
                    forward_cycles / torch.clamp(frequency_hz, min=1.0e-6),
                    torch.full_like(forward_cycles, float(maximum_wait_s)),
                ),
            )
            self._run_stop_start_step[env_ids] += torch.clamp(
                torch.round(alignment_wait_s / self.step_dt).long(),
                min=0,
            )
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
        self._episode_scheduled_pair_progress_gate_sum[env_ids] = 0.0
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
        self._episode_stall_opportunity_sum[env_ids] = 0.0
        self._episode_joint_acceleration_squared_sum[env_ids] = 0.0
        self._episode_body_linear_acceleration_squared_sum[env_ids] = 0.0
        self._episode_body_angular_acceleration_squared_sum[env_ids] = 0.0
        self._episode_effort_soft_limit_step_sum[env_ids] = 0.0
        self._episode_max_applied_effort_fraction[env_ids] = 0.0
        if self._native_effort_telemetry_enabled:
            self._episode_native_effort_squared_sum[env_ids] = 0.0
            self._episode_native_effort_soft_limit_step_sum[env_ids] = 0.0
            self._native_effort_last_nm[env_ids] = torch.nan
            self._native_effort_last_pd_nm[env_ids] = torch.nan
            self._native_effort_last_physics_counter[env_ids] = -1
        self._episode_stop_hold_step_sum[env_ids] = 0.0
        self._episode_stop_speed_sum[env_ids] = 0.0
        self._episode_stop_position_error_sum[env_ids] = 0.0
        self._episode_stop_all_feet_contact_sum[env_ids] = 0.0
        self._episode_stop_stable_step_sum[env_ids] = 0.0
        for term_sum in self._episode_reward_term_sums.values():
            term_sum[env_ids] = 0.0
