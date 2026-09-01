"""Evaluate whether a walking checkpoint sustains motion beyond reset transients."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--seconds", type=int, default=30)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--window-seconds", type=int, default=5)
parser.add_argument("--seed", type=int, default=4401)
parser.add_argument(
    "--domain-mode",
    choices=("task", "nominal", "randomized"),
    default="task",
    help=(
        "Use the task's nominal/randomized mix, force fully nominal physics, "
        "or force the V25 randomized domain."
    ),
)
parser.add_argument(
    "--task", default="Drobot-Commanded-Walk-Forward-Direct"
)
parser.add_argument("--forward-speed", type=float, default=0.15)
parser.add_argument(
    "--reference-weight-shift-forward-m",
    type=float,
    default=None,
    help="Optionally override the task's analytic gait-reference forward shift.",
)
parser.add_argument(
    "--reference-rear-weight-shift-forward-m",
    type=float,
    default=None,
    help="Optionally override the rear-swing-only forward weight transfer.",
)
parser.add_argument(
    "--reference-weight-shift-lateral-m",
    type=float,
    default=None,
    help=(
        "Optionally override the task's phase-timed shift away from the "
        "scheduled swing leg."
    ),
)
parser.add_argument(
    "--reference-stride-m",
    type=float,
    default=None,
    help="Optionally override the task's analytic gait-reference stride.",
)
parser.add_argument(
    "--reference-lift-m",
    type=float,
    default=None,
    help="Optionally override the task's analytic swing-foot lift.",
)
parser.add_argument(
    "--reference-stance-fore-aft-m",
    type=float,
    default=None,
    help="Optionally override the analytic stance's opposed fore/aft sweep.",
)
parser.add_argument(
    "--reference-stance-down-m",
    type=float,
    default=None,
    help="Optionally override the matching analytic flat-sole stance depth.",
)
parser.add_argument(
    "--reference-forward-body-pitch-rad",
    type=float,
    default=None,
    help="Optionally override the analytic stance's positive nose-down pitch.",
)
parser.add_argument(
    "--reference-stance-center-offset-m",
    type=float,
    default=None,
    help=(
        "Optionally translate the complete fore/aft support polygon relative "
        "to the body without changing its span."
    ),
)
parser.add_argument(
    "--actuator-effort-scale",
    type=float,
    default=None,
    help="Optionally evaluate at one fixed effective actuator-effort scale.",
)
parser.add_argument(
    "--target-velocity-scale",
    type=float,
    default=None,
    help="Optionally evaluate at one fixed effective target-rate scale.",
)
parser.add_argument(
    "--zero-policy-actions",
    action="store_true",
    help=(
        "Force the learned residual action to zero while retaining the analytic "
        "gait reference. This isolates reference-gait mechanics from the policy."
    ),
)
parser.add_argument(
    "--disable-rear-payload",
    action="store_true",
    help=(
        "Use the asset's centered provisional battery instead of authoring the "
        "external rear pack. Intended only for gait-mechanics diagnosis."
    ),
)
args = parser.parse_args()
if args.seconds < 10 or args.episodes < 1:
    parser.error("evaluation needs at least 10 seconds and one episode")
if not 1 <= args.window_seconds <= args.seconds:
    parser.error("--window-seconds must be within the evaluation duration")

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from importlib import metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import (  # noqa: E402
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

package_parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(package_parent))

import parallel_walking  # noqa: E402, F401
from parallel_walking import preview_control  # noqa: E402
from parallel_walking.commanded_walking_env import (  # noqa: E402
    LEG_NAMES,
    _yaw_from_xyzw,
)
task = args.task
env_cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
agent_cfg = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
env_cfg.scene.num_envs = 1
env_cfg.seed = args.seed
env_cfg.reset_joint_position_noise_rad = 0.0
env_cfg.reset_xy_jitter_m = 0.0
env_cfg.disable_time_limit = True
if args.disable_rear_payload:
    env_cfg.rear_payload_enabled = False
if args.domain_mode == "nominal":
    env_cfg.physical_randomization_nominal_fraction = 1.0
    # V24's common payload and supply domains are older than the nominal mask.
    env_cfg.rear_payload_combined_mass_scale_range = (1.0, 1.0)
    env_cfg.rear_payload_combined_com_jitter_m = (0.0, 0.0, 0.0)
    env_cfg.robot_mass_scale_range = (1.0, 1.0)
    env_cfg.actuator_effort_scale_range = (1.0, 1.0)
    env_cfg.target_velocity_scale_range = (1.0, 1.0)
elif args.domain_mode == "randomized":
    env_cfg.physical_randomization_nominal_fraction = 0.0
if args.reference_weight_shift_forward_m is not None:
    env_cfg.gait_weight_shift_forward_m = args.reference_weight_shift_forward_m
if args.reference_rear_weight_shift_forward_m is not None:
    env_cfg.gait_rear_weight_shift_forward_m = (
        args.reference_rear_weight_shift_forward_m
    )
if args.reference_weight_shift_lateral_m is not None:
    env_cfg.gait_weight_shift_lateral_m = args.reference_weight_shift_lateral_m
if args.reference_stride_m is not None:
    env_cfg.gait_stride_m = args.reference_stride_m
if args.reference_lift_m is not None:
    env_cfg.gait_lift_m = args.reference_lift_m
if args.reference_stance_fore_aft_m is not None:
    env_cfg.gait_stance_fore_aft_m = args.reference_stance_fore_aft_m
if args.reference_stance_down_m is not None:
    env_cfg.gait_stance_down_m = args.reference_stance_down_m
if args.reference_forward_body_pitch_rad is not None:
    env_cfg.gait_forward_body_pitch_rad = args.reference_forward_body_pitch_rad
    env_cfg.target_forward_pitch_rad = args.reference_forward_body_pitch_rad
if args.reference_stance_center_offset_m is not None:
    env_cfg.gait_stance_center_offset_m = args.reference_stance_center_offset_m
if args.actuator_effort_scale is not None:
    env_cfg.actuator_effort_scale_range = (
        args.actuator_effort_scale,
        args.actuator_effort_scale,
    )
if args.target_velocity_scale is not None:
    env_cfg.target_velocity_scale_range = (
        args.target_velocity_scale,
        args.target_velocity_scale,
    )
preview_control.COMMAND_OVERRIDE = (args.forward_speed, 0.0, 0.0)

env = gym.make(task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
checkpoint = str(Path(args.checkpoint).resolve())
runner.load(checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

results: list[dict[str, object]] = []
for _ in range(args.episodes):
    env.unwrapped._reset_idx(torch.tensor([0], device=env.unwrapped.device))
    env.unwrapped.episode_length_buf.zero_()
    actuator_effort_scale = float(
        env.unwrapped._actuator_effort_scale[0, 0].item()
    )
    target_velocity_scale = float(
        env.unwrapped._target_velocity_scale[0, 0].item()
    )
    actuator_effort_scale_by_joint = getattr(
        env.unwrapped,
        "_actuator_effort_scale_by_joint",
        env.unwrapped._actuator_effort_scale.expand(-1, 12),
    )[0].clone()
    actuator_effort_limit_by_joint = env.unwrapped._actuator_effort_limit_by_joint[
        0
    ].clone()
    target_velocity_scale_by_joint = getattr(
        env.unwrapped,
        "_target_velocity_scale_by_joint",
        env.unwrapped._target_velocity_scale.expand(-1, 12),
    )[0].clone()
    control_delay_steps = int(
        getattr(
            env.unwrapped,
            "_control_delay_steps",
            torch.zeros((1,), dtype=torch.int64, device=env.unwrapped.device),
        )[0].item()
    )
    joint_target_bias = getattr(
        env.unwrapped,
        "_joint_target_bias",
        torch.zeros((1, 12), device=env.unwrapped.device),
    )[0].clone()
    actuator_stiffness_scale_by_joint = getattr(
        env.unwrapped,
        "_actuator_stiffness_scale_by_joint",
        torch.ones((1, 12), device=env.unwrapped.device),
    )[0].clone()
    actuator_damping_scale_by_joint = getattr(
        env.unwrapped,
        "_actuator_damping_scale_by_joint",
        torch.ones((1, 12), device=env.unwrapped.device),
    )[0].clone()
    foot_static_friction = getattr(
        env.unwrapped,
        "_foot_static_friction",
        torch.full(
            (1, 4),
            float(env_cfg.shoe_static_friction),
            device=env.unwrapped.device,
        ),
    )[0].clone()
    foot_dynamic_friction = getattr(
        env.unwrapped,
        "_foot_dynamic_friction",
        torch.full(
            (1, 4),
            float(env_cfg.shoe_dynamic_friction),
            device=env.unwrapped.device,
        ),
    )[0].clone()
    obs = env.get_observations()
    start_x = float(env.unwrapped._robot.data.root_pos_w.torch[0, 0].item())
    start_y = float(env.unwrapped._robot.data.root_pos_w.torch[0, 1].item())
    start_yaw = float(
        _yaw_from_xyzw(env.unwrapped._robot.data.root_quat_w.torch)[0].item()
    )
    positions = [start_x]
    lateral_positions = [start_y]
    path_forward_positions = [0.0]
    path_lateral_positions = [0.0]
    saturated_action_sum = 0.0
    action_rate_sum = 0.0
    action_acceleration_sum = 0.0
    joint_acceleration_squared_sum = 0.0
    body_linear_acceleration_squared_sum = 0.0
    body_angular_acceleration_squared_sum = 0.0
    target_limiter_gap_sum = 0.0
    three_foot_support_sum = 0.0
    excess_airborne_sum = 0.0
    swing_schedule_step_sum = 0.0
    swing_schedule_three_plus_support_sum = 0.0
    transfer_schedule_step_sum = 0.0
    transfer_schedule_four_support_sum = 0.0
    transfer_schedule_three_plus_support_sum = 0.0
    support_count_histogram = torch.zeros(
        (5,), device=env.unwrapped.device
    )
    absolute_yaw_travel_rad = 0.0
    backward_step_sum = 0.0
    meaningful_backward_step_sum = 0.0
    backward_speed_sum_m_s = 0.0
    backward_speed_squared_sum_m2_s2 = 0.0
    reverse_motion_deadband_m_s = max(0.001, 0.10 * args.forward_speed)
    forward_pitch_sum_rad = 0.0
    absolute_path_heading_error_sum_rad = 0.0
    path_heading_error_squared_sum_rad2 = 0.0
    effort_soft_limit_joint_step_sum = 0.0
    maximum_applied_effort_fraction = 0.0
    absolute_applied_effort_sum_by_joint = torch.zeros(
        (12,), device=env.unwrapped.device
    )
    squared_applied_effort_sum_by_joint = torch.zeros(
        (12,), device=env.unwrapped.device
    )
    effort_fraction_sum_by_joint = torch.zeros(
        (12,), device=env.unwrapped.device
    )
    squared_effort_fraction_sum_by_joint = torch.zeros(
        (12,), device=env.unwrapped.device
    )
    effort_soft_limit_exceedance_by_joint = torch.zeros(
        (12,), device=env.unwrapped.device
    )
    maximum_applied_effort_fraction_by_joint = torch.zeros(
        (12,), device=env.unwrapped.device
    )
    previous_action = torch.zeros((12,), device=env.unwrapped.device)
    older_action = torch.zeros((12,), device=env.unwrapped.device)
    previous_joint_velocity = env.unwrapped._robot.data.joint_vel.torch[0].clone()
    previous_body_linear_velocity = (
        env.unwrapped._robot.data.root_lin_vel_w.torch[0].clone()
    )
    previous_body_angular_velocity = (
        env.unwrapped._robot.data.root_ang_vel_w.torch[0].clone()
    )
    foot_names = [
        next(leg for leg in LEG_NAMES if leg in name)
        for name in env.unwrapped._foot_sensor_names
    ]
    foot_body_id_by_leg = {
        next(leg for leg in LEG_NAMES if leg in name): int(body_id)
        for body_id, name in zip(
            env.unwrapped._foot_body_ids,
            env.unwrapped._foot_body_names,
            strict=True,
        )
    }
    start_body_positions = env.unwrapped._robot.data.body_pos_w.torch[0]
    start_root_position = env.unwrapped._robot.data.root_pos_w.torch[0]
    start_forward_direction = torch.tensor(
        [math.cos(start_yaw), math.sin(start_yaw)],
        device=env.unwrapped.device,
    )
    initial_distal_forward_from_base_m = {
        leg: float(
            torch.dot(
                start_body_positions[foot_body_id_by_leg[leg], :2]
                - start_root_position[:2],
                start_forward_direction,
            ).item()
        )
        for leg in foot_names
    }
    previous_contact = (
        env.unwrapped._foot_forces()[0]
        > env_cfg.scheduled_release_force_threshold_n
    )
    touchdown_counts = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_contact_matches = torch.zeros((4,), device=env.unwrapped.device)
    schedule_matched_contact_sum = 0.0
    contact_step_counts = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_swing_step_counts = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_swing_contact_counts = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_swing_force_sums = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_stance_step_counts = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_stance_airborne_counts = torch.zeros((4,), device=env.unwrapped.device)
    scheduled_swing_frame_counts = torch.zeros((4,), device=env.unwrapped.device)
    missing_stance_counts_by_swing = torch.zeros(
        (4, 4), device=env.unwrapped.device
    )
    maximum_consecutive_scheduled_release_steps = torch.zeros(
        (4,), device=env.unwrapped.device
    )
    maximum_distal_height_m = torch.full(
        (4,), -torch.inf, device=env.unwrapped.device
    )
    fell = False
    completed_steps = 0
    last_current_x = start_x
    last_path_forward = 0.0
    last_path_lateral = 0.0
    last_yaw = start_yaw
    last_cycle_release_qualifications = torch.zeros(
        (4,), device=env.unwrapped.device
    )
    last_completed_gait_cycles = 0
    last_all_four_release_cycles = 0
    for step in range(args.seconds * 60):
        with torch.inference_mode():
            actions = policy(obs)
        if args.zero_policy_actions:
            actions = torch.zeros_like(actions)
        # Keep environment state as ordinary tensors so another episode can
        # reset it in-place after deterministic inference completes.
        actions = actions.clone()
        saturated_action_sum += float(
            torch.mean((torch.abs(actions[0]) >= 0.98).float()).item()
        )
        action_rate_sum += float(torch.mean(torch.abs(actions[0] - previous_action)).item())
        action_acceleration_sum += float(
            torch.mean(torch.abs(actions[0] - 2.0 * previous_action + older_action)).item()
        )
        older_action.copy_(previous_action)
        previous_action.copy_(actions[0])
        obs, _, dones, _ = env.step(actions)
        policy.reset(dones)
        if bool(dones[0].item()):
            # DirectRLEnv resets a terminated environment inside ``step``.
            # Do not mix that reset pose, cleared cycle counters, or velocity
            # discontinuity into the terminal trial's measurements.
            fell = True
            break
        joint_velocity = env.unwrapped._robot.data.joint_vel.torch[0]
        body_linear_velocity = env.unwrapped._robot.data.root_lin_vel_w.torch[0]
        body_angular_velocity = env.unwrapped._robot.data.root_ang_vel_w.torch[0]
        projected_gravity_x = env.unwrapped._robot.data.projected_gravity_b.torch[0, 0]
        current_x = float(env.unwrapped._robot.data.root_pos_w.torch[0, 0].item())
        current_y = float(env.unwrapped._robot.data.root_pos_w.torch[0, 1].item())
        delta_x = current_x - start_x
        delta_y = current_y - start_y
        path_forward = math.cos(start_yaw) * delta_x + math.sin(start_yaw) * delta_y
        path_lateral = -math.sin(start_yaw) * delta_x + math.cos(start_yaw) * delta_y
        path_forward_velocity = (
            math.cos(start_yaw) * float(body_linear_velocity[0].item())
            + math.sin(start_yaw) * float(body_linear_velocity[1].item())
        )
        current_yaw = float(
            _yaw_from_xyzw(env.unwrapped._robot.data.root_quat_w.torch)[0].item()
        )
        last_current_x = current_x
        last_path_forward = path_forward
        last_path_lateral = path_lateral
        last_yaw = current_yaw
        path_heading_error = math.atan2(
            math.sin(current_yaw - start_yaw), math.cos(current_yaw - start_yaw)
        )
        absolute_path_heading_error_sum_rad += abs(path_heading_error)
        path_heading_error_squared_sum_rad2 += path_heading_error * path_heading_error
        absolute_applied_effort = torch.abs(
            env.unwrapped._robot.data.applied_torque.torch[0]
        )
        effort_fraction = absolute_applied_effort / torch.clamp(
            actuator_effort_limit_by_joint,
            min=1.0e-6,
        )
        absolute_applied_effort_sum_by_joint += absolute_applied_effort
        squared_applied_effort_sum_by_joint += torch.square(
            absolute_applied_effort
        )
        effort_fraction_sum_by_joint += effort_fraction
        squared_effort_fraction_sum_by_joint += torch.square(effort_fraction)
        effort_soft_limit_exceedance_by_joint += (
            effort_fraction > env_cfg.effort_soft_limit_fraction
        ).float()
        maximum_applied_effort_fraction_by_joint = torch.maximum(
            maximum_applied_effort_fraction_by_joint,
            effort_fraction,
        )
        effort_soft_limit_joint_step_sum += float(
            torch.mean(
                (effort_fraction > env_cfg.effort_soft_limit_fraction).float()
            ).item()
        )
        maximum_applied_effort_fraction = max(
            maximum_applied_effort_fraction,
            float(torch.max(effort_fraction).item()),
        )
        # V25 is selected in the episode-start path frame.  A robot that yaws
        # left must not make its own body-X velocity look like valid forward
        # progress (or hide backward travel) in the evaluation report.
        backward_speed_m_s = max(-path_forward_velocity, 0.0)
        backward_step_sum += float(backward_speed_m_s > 0.0)
        meaningful_backward_step_sum += float(
            backward_speed_m_s > reverse_motion_deadband_m_s
        )
        backward_speed_sum_m_s += backward_speed_m_s
        backward_speed_squared_sum_m2_s2 += backward_speed_m_s**2
        forward_pitch_sum_rad += math.asin(
            max(-1.0, min(1.0, float(projected_gravity_x.item())))
        )
        target_limiter_gap_sum += float(
            torch.mean(
                torch.abs(
                    env.unwrapped._desired_targets[0]
                    - env.unwrapped._processed_actions[0]
                )
            ).item()
        )
        joint_acceleration_squared_sum += float(
            torch.mean(torch.square((joint_velocity - previous_joint_velocity) * 60.0)).item()
        )
        body_linear_acceleration_squared_sum += float(
            torch.mean(
                torch.square((body_linear_velocity - previous_body_linear_velocity) * 60.0)
            ).item()
        )
        body_angular_acceleration_squared_sum += float(
            torch.mean(
                torch.square((body_angular_velocity - previous_body_angular_velocity) * 60.0)
            ).item()
        )
        previous_joint_velocity.copy_(joint_velocity)
        previous_body_linear_velocity.copy_(body_linear_velocity)
        previous_body_angular_velocity.copy_(body_angular_velocity)
        foot_force = env.unwrapped._foot_forces()[0]
        foot_contact = foot_force > env_cfg.scheduled_release_force_threshold_n
        support_count = int(torch.count_nonzero(foot_contact).item())
        three_foot_support_sum += float(support_count >= 3)
        excess_airborne_sum += float(support_count <= 2)
        support_count_histogram[support_count] += 1.0
        absolute_yaw_travel_rad += (
            abs(float(body_angular_velocity[2].item())) / 60.0
        )
        touchdown_counts += (foot_contact & ~previous_contact).float()
        previous_contact.copy_(foot_contact)
        scheduled_contact = env.unwrapped._applied_scheduled_contact
        if scheduled_contact is None:
            raise RuntimeError("Applied gait schedule was not cached during env.step")
        scheduled_swing = ~scheduled_contact[0]
        schedule_matched_contact_sum += float(
            torch.all(foot_contact == scheduled_contact[0]).item()
        )
        has_scheduled_swing = bool(torch.any(scheduled_swing).item())
        if has_scheduled_swing:
            swing_schedule_step_sum += 1.0
            swing_schedule_three_plus_support_sum += float(support_count >= 3)
        else:
            transfer_schedule_step_sum += 1.0
            transfer_schedule_four_support_sum += float(support_count == 4)
            transfer_schedule_three_plus_support_sum += float(support_count >= 3)
        contact_step_counts += foot_contact.float()
        scheduled_swing_step_counts += scheduled_swing.float()
        scheduled_swing_contact_counts += (
            scheduled_swing & foot_contact
        ).float()
        scheduled_swing_force_sums += torch.where(
            scheduled_swing,
            foot_force,
            torch.zeros_like(foot_force),
        )
        scheduled_stance = ~scheduled_swing
        scheduled_stance_step_counts += scheduled_stance.float()
        scheduled_stance_airborne_counts += (
            scheduled_stance & ~foot_contact
        ).float()
        missing_scheduled_stance = scheduled_stance & ~foot_contact
        for swing_index in range(4):
            if bool(scheduled_swing[swing_index].item()):
                scheduled_swing_frame_counts[swing_index] += 1.0
                missing_stance_counts_by_swing[swing_index] += (
                    missing_scheduled_stance.float()
                )
        maximum_consecutive_scheduled_release_steps = torch.maximum(
            maximum_consecutive_scheduled_release_steps,
            env.unwrapped._cycle_release_consecutive_steps[0].float(),
        )
        last_cycle_release_qualifications.copy_(
            env.unwrapped._episode_cycle_release_qualifications_by_foot[0]
        )
        last_completed_gait_cycles = int(
            env.unwrapped._episode_completed_gait_cycles[0].item()
        )
        last_all_four_release_cycles = int(
            env.unwrapped._episode_all_four_release_cycles[0].item()
        )
        body_positions = env.unwrapped._robot.data.body_pos_w.torch[0]
        maximum_distal_height_m = torch.maximum(
            maximum_distal_height_m,
            torch.tensor(
                [
                    body_positions[foot_body_id_by_leg[leg], 2]
                    for leg in foot_names
                ],
                device=env.unwrapped.device,
            ),
        )
        scheduled_contact_matches += (foot_contact == scheduled_contact[0]).float()
        completed_steps = step + 1
        if completed_steps % 60 == 0:
            positions.append(current_x)
            lateral_positions.append(current_y)
            path_forward_positions.append(path_forward)
            path_lateral_positions.append(path_lateral)

    duration_s = completed_steps / 60.0
    distance_m = last_current_x - start_x
    path_forward_distance_m = last_path_forward
    path_lateral_displacement_m = last_path_lateral
    final_yaw = last_yaw
    heading_error_rad = math.atan2(
        math.sin(final_yaw - start_yaw), math.cos(final_yaw - start_yaw)
    )
    window_speeds = [
        (positions[index] - positions[index - args.window_seconds])
        / args.window_seconds
        for index in range(args.window_seconds, len(positions))
    ]
    path_window_speeds = [
        (
            path_forward_positions[index]
            - path_forward_positions[index - args.window_seconds]
        )
        / args.window_seconds
        for index in range(args.window_seconds, len(path_forward_positions))
    ]
    stall_speed_threshold_m_s = max(0.001, 0.4 * args.forward_speed)
    results.append(
        {
            "duration_s": duration_s,
            "distance_m": distance_m,
            "lateral_displacement_m": lateral_positions[-1] - lateral_positions[0],
            "path_forward_distance_m": path_forward_distance_m,
            "path_lateral_displacement_m": path_lateral_displacement_m,
            "absolute_path_lateral_per_forward_m": abs(path_lateral_displacement_m)
            / max(abs(path_forward_distance_m), 0.010),
            "final_heading_error_rad": heading_error_rad,
            "mean_abs_path_heading_error_rad": absolute_path_heading_error_sum_rad
            / max(completed_steps, 1),
            "rms_path_heading_error_rad": math.sqrt(
                path_heading_error_squared_sum_rad2 / max(completed_steps, 1)
            ),
            "mean_speed_m_s": distance_m / max(duration_s, 1.0 / 60.0),
            "path_mean_speed_m_s": path_forward_distance_m
            / max(duration_s, 1.0 / 60.0),
            "minimum_window_speed_m_s": min(window_speeds, default=0.0),
            "final_window_speed_m_s": window_speeds[-1] if window_speeds else 0.0,
            "minimum_path_window_speed_m_s": min(
                path_window_speeds, default=0.0
            ),
            "final_path_window_speed_m_s": (
                path_window_speeds[-1] if path_window_speeds else 0.0
            ),
            "stall_window_fraction": (
                sum(speed < stall_speed_threshold_m_s for speed in window_speeds)
                / max(len(window_speeds), 1)
            ),
            "path_stall_window_fraction": (
                sum(
                    speed < stall_speed_threshold_m_s
                    for speed in path_window_speeds
                )
                / max(len(path_window_speeds), 1)
            ),
            "stall_speed_threshold_m_s": stall_speed_threshold_m_s,
            "action_saturation_fraction": saturated_action_sum
            / max(completed_steps, 1),
            "mean_abs_action_rate_per_step": action_rate_sum
            / max(completed_steps, 1),
            "mean_abs_action_acceleration_per_step2": action_acceleration_sum
            / max(completed_steps, 1),
            "rms_joint_acceleration_rad_s2": math.sqrt(
                joint_acceleration_squared_sum / max(completed_steps, 1)
            ),
            "rms_body_linear_acceleration_m_s2": math.sqrt(
                body_linear_acceleration_squared_sum / max(completed_steps, 1)
            ),
            "rms_body_angular_acceleration_rad_s2": math.sqrt(
                body_angular_acceleration_squared_sum / max(completed_steps, 1)
            ),
            "mean_target_limiter_gap_rad": target_limiter_gap_sum
            / max(completed_steps, 1),
            "three_or_four_foot_support_fraction": three_foot_support_sum
            / max(completed_steps, 1),
            "two_or_fewer_foot_support_fraction": excess_airborne_sum
            / max(completed_steps, 1),
            "schedule_matched_contact_fraction": schedule_matched_contact_sum
            / max(completed_steps, 1),
            "support_count_fraction": {
                str(count): float(support_count_histogram[count].item())
                / max(completed_steps, 1)
                for count in range(5)
            },
            "swing_schedule_fraction": swing_schedule_step_sum
            / max(completed_steps, 1),
            "swing_schedule_three_or_four_foot_support_fraction": (
                swing_schedule_three_plus_support_sum
                / max(swing_schedule_step_sum, 1.0)
            ),
            "transfer_schedule_fraction": transfer_schedule_step_sum
            / max(completed_steps, 1),
            "transfer_schedule_four_foot_support_fraction": (
                transfer_schedule_four_support_sum
                / max(transfer_schedule_step_sum, 1.0)
            ),
            "transfer_schedule_three_or_four_foot_support_fraction": (
                transfer_schedule_three_plus_support_sum
                / max(transfer_schedule_step_sum, 1.0)
            ),
            "absolute_yaw_travel_rad": absolute_yaw_travel_rad,
            "backward_step_fraction": backward_step_sum / max(completed_steps, 1),
            "meaningful_backward_step_fraction": meaningful_backward_step_sum
            / max(completed_steps, 1),
            "reverse_motion_deadband_m_s": reverse_motion_deadband_m_s,
            "mean_backward_speed_m_s": backward_speed_sum_m_s
            / max(completed_steps, 1),
            "rms_backward_speed_m_s": math.sqrt(
                backward_speed_squared_sum_m2_s2 / max(completed_steps, 1)
            ),
            "mean_forward_pitch_rad": forward_pitch_sum_rad / max(completed_steps, 1),
            "actuator_effort_scale": actuator_effort_scale,
            "actuator_peak_effort_nm": env_cfg.actuator_peak_effort_nm,
            "actuator_effort_limit_by_joint_nm": [
                float(value) for value in actuator_effort_limit_by_joint.tolist()
            ],
            "target_velocity_scale": target_velocity_scale,
            "actuator_effort_scale_by_joint": [
                float(value) for value in actuator_effort_scale_by_joint.tolist()
            ],
            "target_velocity_scale_by_joint": [
                float(value) for value in target_velocity_scale_by_joint.tolist()
            ],
            "control_delay_steps": control_delay_steps,
            "joint_target_bias_rad": [
                float(value) for value in joint_target_bias.tolist()
            ],
            "actuator_stiffness_scale_by_joint": [
                float(value) for value in actuator_stiffness_scale_by_joint.tolist()
            ],
            "actuator_damping_scale_by_joint": [
                float(value) for value in actuator_damping_scale_by_joint.tolist()
            ],
            "foot_static_friction": [
                float(value) for value in foot_static_friction.tolist()
            ],
            "foot_dynamic_friction": [
                float(value) for value in foot_dynamic_friction.tolist()
            ],
            "effort_soft_limit_fraction": env_cfg.effort_soft_limit_fraction,
            "effort_soft_limit_by_joint_nm": [
                float(value)
                for value in (
                    env_cfg.effort_soft_limit_fraction
                    * actuator_effort_limit_by_joint
                ).tolist()
            ],
            "effort_soft_limit_exceedance_fraction": effort_soft_limit_joint_step_sum
            / max(completed_steps, 1),
            "maximum_applied_effort_fraction": maximum_applied_effort_fraction,
            "mean_abs_applied_effort_nm_by_joint": {
                name: float(
                    absolute_applied_effort_sum_by_joint[index].item()
                    / max(completed_steps, 1)
                )
                for index, name in enumerate(env.unwrapped._robot.joint_names)
            },
            "rms_applied_effort_nm_by_joint": {
                name: math.sqrt(
                    float(squared_applied_effort_sum_by_joint[index].item())
                    / max(completed_steps, 1)
                )
                for index, name in enumerate(env.unwrapped._robot.joint_names)
            },
            "mean_applied_effort_fraction_by_joint": {
                name: float(
                    effort_fraction_sum_by_joint[index].item()
                    / max(completed_steps, 1)
                )
                for index, name in enumerate(env.unwrapped._robot.joint_names)
            },
            "rms_applied_effort_fraction_by_joint": {
                name: math.sqrt(
                    float(squared_effort_fraction_sum_by_joint[index].item())
                    / max(completed_steps, 1)
                )
                for index, name in enumerate(env.unwrapped._robot.joint_names)
            },
            "effort_soft_limit_exceedance_fraction_by_joint": {
                name: float(
                    effort_soft_limit_exceedance_by_joint[index].item()
                    / max(completed_steps, 1)
                )
                for index, name in enumerate(env.unwrapped._robot.joint_names)
            },
            "maximum_applied_effort_fraction_by_joint": {
                name: float(
                    maximum_applied_effort_fraction_by_joint[index].item()
                )
                for index, name in enumerate(env.unwrapped._robot.joint_names)
            },
            "touchdowns_by_leg": {
                name: int(touchdown_counts[index].item())
                for index, name in enumerate(foot_names)
            },
            "scheduled_contact_match_by_leg": {
                name: float(scheduled_contact_matches[index].item())
                / max(completed_steps, 1)
                for index, name in enumerate(foot_names)
            },
            "contact_fraction_by_leg": {
                name: float(contact_step_counts[index].item())
                / max(completed_steps, 1)
                for index, name in enumerate(foot_names)
            },
            "scheduled_swing_contact_fraction_by_leg": {
                name: float(scheduled_swing_contact_counts[index].item())
                / max(float(scheduled_swing_step_counts[index].item()), 1.0)
                for index, name in enumerate(foot_names)
            },
            "scheduled_swing_mean_force_n_by_leg": {
                name: float(scheduled_swing_force_sums[index].item())
                / max(float(scheduled_swing_step_counts[index].item()), 1.0)
                for index, name in enumerate(foot_names)
            },
            "scheduled_stance_airborne_fraction_by_leg": {
                name: float(scheduled_stance_airborne_counts[index].item())
                / max(float(scheduled_stance_step_counts[index].item()), 1.0)
                for index, name in enumerate(foot_names)
            },
            "missing_scheduled_stance_fraction_by_active_swing": {
                swing_name: {
                    stance_name: float(
                        missing_stance_counts_by_swing[
                            swing_index, stance_index
                        ].item()
                    )
                    / max(
                        float(scheduled_swing_frame_counts[swing_index].item()),
                        1.0,
                    )
                    for stance_index, stance_name in enumerate(foot_names)
                }
                for swing_index, swing_name in enumerate(foot_names)
            },
            "maximum_consecutive_scheduled_release_steps_by_leg": {
                name: int(
                    maximum_consecutive_scheduled_release_steps[index].item()
                )
                for index, name in enumerate(foot_names)
            },
            "cycle_release_qualifications_by_leg": {
                name: int(
                    last_cycle_release_qualifications[index].item()
                )
                for index, name in enumerate(foot_names)
            },
            "completed_gait_cycles": last_completed_gait_cycles,
            "all_four_release_cycles": last_all_four_release_cycles,
            "maximum_distal_height_m_by_leg": {
                name: float(maximum_distal_height_m[index].item())
                for index, name in enumerate(foot_names)
            },
            "initial_distal_forward_from_base_m_by_leg": (
                initial_distal_forward_from_base_m
            ),
            "fell": fell,
            "one_second_positions_m": positions,
            "one_second_lateral_positions_m": lateral_positions,
            "one_second_path_forward_positions_m": path_forward_positions,
            "one_second_path_lateral_positions_m": path_lateral_positions,
        }
    )

summary = {
    "checkpoint": checkpoint,
    "seconds": args.seconds,
    "window_seconds": args.window_seconds,
    "episodes": results,
    "task": task,
    "seed": args.seed,
    "domain_mode": args.domain_mode,
    "joint_order": list(env.unwrapped._robot.joint_names),
    "foot_order": foot_names,
    "forward_speed_m_s": args.forward_speed,
    "zero_policy_actions": args.zero_policy_actions,
    "rear_payload_enabled": env_cfg.rear_payload_enabled,
    "reference_weight_shift_forward_m": env_cfg.gait_weight_shift_forward_m,
    "reference_rear_weight_shift_forward_m": (
        env_cfg.gait_rear_weight_shift_forward_m
    ),
    "reference_weight_shift_lateral_m": env_cfg.gait_weight_shift_lateral_m,
    "reference_stride_m": env_cfg.gait_stride_m,
    "reference_lift_m": env_cfg.gait_lift_m,
    "reference_forward_body_pitch_rad": env_cfg.gait_forward_body_pitch_rad,
    "reference_stance_center_offset_m": env_cfg.gait_stance_center_offset_m,
    "stance_forward_bias_m": float(env.unwrapped._stance_forward_bias_m),
    "mean_distance_m": sum(float(item["distance_m"]) for item in results)
    / len(results),
    "mean_abs_lateral_displacement_m": sum(
        abs(float(item["lateral_displacement_m"])) for item in results
    )
    / len(results),
    "mean_path_forward_distance_m": sum(
        float(item["path_forward_distance_m"]) for item in results
    )
    / len(results),
    "mean_path_speed_m_s": sum(
        float(item["path_mean_speed_m_s"]) for item in results
    )
    / len(results),
    "mean_abs_path_lateral_displacement_m": sum(
        abs(float(item["path_lateral_displacement_m"])) for item in results
    )
    / len(results),
    "mean_absolute_path_lateral_per_forward_m": sum(
        float(item["absolute_path_lateral_per_forward_m"]) for item in results
    )
    / len(results),
    "mean_abs_final_heading_error_rad": sum(
        abs(float(item["final_heading_error_rad"])) for item in results
    )
    / len(results),
    "mean_abs_path_heading_error_rad": sum(
        float(item["mean_abs_path_heading_error_rad"]) for item in results
    )
    / len(results),
    "mean_rms_path_heading_error_rad": sum(
        float(item["rms_path_heading_error_rad"]) for item in results
    )
    / len(results),
    "mean_final_window_speed_m_s": sum(
        float(item["final_window_speed_m_s"]) for item in results
    )
    / len(results),
    "mean_final_path_window_speed_m_s": sum(
        float(item["final_path_window_speed_m_s"]) for item in results
    )
    / len(results),
    "mean_stall_window_fraction": sum(
        float(item["stall_window_fraction"]) for item in results
    )
    / len(results),
    "mean_path_stall_window_fraction": sum(
        float(item["path_stall_window_fraction"]) for item in results
    )
    / len(results),
    "fall_rate": sum(bool(item["fell"]) for item in results) / len(results),
    "mean_rms_joint_acceleration_rad_s2": sum(
        float(item["rms_joint_acceleration_rad_s2"]) for item in results
    ) / len(results),
    "mean_rms_body_linear_acceleration_m_s2": sum(
        float(item["rms_body_linear_acceleration_m_s2"]) for item in results
    ) / len(results),
    "mean_rms_body_angular_acceleration_rad_s2": sum(
        float(item["rms_body_angular_acceleration_rad_s2"]) for item in results
    ) / len(results),
    "mean_target_limiter_gap_rad": sum(
        float(item["mean_target_limiter_gap_rad"]) for item in results
    ) / len(results),
    "mean_three_or_four_foot_support_fraction": sum(
        float(item["three_or_four_foot_support_fraction"]) for item in results
    ) / len(results),
    "mean_two_or_fewer_foot_support_fraction": sum(
        float(item["two_or_fewer_foot_support_fraction"]) for item in results
    ) / len(results),
    "mean_schedule_matched_contact_fraction": sum(
        float(item["schedule_matched_contact_fraction"]) for item in results
    ) / len(results),
    "mean_support_count_fraction": {
        str(count): sum(
            float(item["support_count_fraction"][str(count)])
            for item in results
        )
        / len(results)
        for count in range(5)
    },
    "mean_swing_schedule_fraction": sum(
        float(item["swing_schedule_fraction"]) for item in results
    ) / len(results),
    "mean_swing_schedule_three_or_four_foot_support_fraction": sum(
        float(item["swing_schedule_three_or_four_foot_support_fraction"])
        for item in results
    ) / len(results),
    "mean_transfer_schedule_fraction": sum(
        float(item["transfer_schedule_fraction"]) for item in results
    ) / len(results),
    "mean_transfer_schedule_four_foot_support_fraction": sum(
        float(item["transfer_schedule_four_foot_support_fraction"])
        for item in results
    ) / len(results),
    "mean_transfer_schedule_three_or_four_foot_support_fraction": sum(
        float(item["transfer_schedule_three_or_four_foot_support_fraction"])
        for item in results
    ) / len(results),
    "completed_gait_cycles": sum(
        int(item["completed_gait_cycles"]) for item in results
    ),
    "all_four_release_cycles": sum(
        int(item["all_four_release_cycles"]) for item in results
    ),
    "all_four_release_cycle_rate": (
        sum(int(item["all_four_release_cycles"]) for item in results)
        / max(sum(int(item["completed_gait_cycles"]) for item in results), 1)
    ),
    "minimum_maximum_consecutive_scheduled_release_steps_by_leg": {
        name: min(
            int(item["maximum_consecutive_scheduled_release_steps_by_leg"][name])
            for item in results
        )
        for name in foot_names
    },
    "mean_scheduled_stance_airborne_fraction_by_leg": {
        name: sum(
            float(item["scheduled_stance_airborne_fraction_by_leg"][name])
            for item in results
        )
        / len(results)
        for name in foot_names
    },
    "mean_missing_scheduled_stance_fraction_by_active_swing": {
        swing_name: {
            stance_name: sum(
                float(
                    item[
                        "missing_scheduled_stance_fraction_by_active_swing"
                    ][swing_name][stance_name]
                )
                for item in results
            )
            / len(results)
            for stance_name in foot_names
        }
        for swing_name in foot_names
    },
    "mean_scheduled_swing_contact_fraction_by_leg": {
        name: sum(
            float(item["scheduled_swing_contact_fraction_by_leg"][name])
            for item in results
        )
        / len(results)
        for name in foot_names
    },
    "mean_absolute_yaw_travel_rad": sum(
        float(item["absolute_yaw_travel_rad"]) for item in results
    ) / len(results),
    "mean_backward_step_fraction": sum(
        float(item["backward_step_fraction"]) for item in results
    ) / len(results),
    "mean_meaningful_backward_step_fraction": sum(
        float(item["meaningful_backward_step_fraction"]) for item in results
    )
    / len(results),
    "reverse_motion_deadband_m_s": max(
        float(item["reverse_motion_deadband_m_s"]) for item in results
    ),
    "mean_backward_speed_m_s": sum(
        float(item["mean_backward_speed_m_s"]) for item in results
    )
    / len(results),
    "mean_rms_backward_speed_m_s": sum(
        float(item["rms_backward_speed_m_s"]) for item in results
    )
    / len(results),
    "mean_forward_pitch_rad": sum(
        float(item["mean_forward_pitch_rad"]) for item in results
    ) / len(results),
    "minimum_actuator_effort_scale": min(
        float(item["actuator_effort_scale"]) for item in results
    ),
    "minimum_target_velocity_scale": min(
        float(item["target_velocity_scale"]) for item in results
    ),
    "minimum_actuator_effort_scale_by_joint": min(
        min(float(value) for value in item["actuator_effort_scale_by_joint"])
        for item in results
    ),
    "minimum_target_velocity_scale_by_joint": min(
        min(float(value) for value in item["target_velocity_scale_by_joint"])
        for item in results
    ),
    "maximum_control_delay_steps": max(
        int(item["control_delay_steps"]) for item in results
    ),
    "minimum_actuator_stiffness_scale_by_joint": min(
        min(float(value) for value in item["actuator_stiffness_scale_by_joint"])
        for item in results
    ),
    "maximum_actuator_stiffness_scale_by_joint": max(
        max(float(value) for value in item["actuator_stiffness_scale_by_joint"])
        for item in results
    ),
    "minimum_actuator_damping_scale_by_joint": min(
        min(float(value) for value in item["actuator_damping_scale_by_joint"])
        for item in results
    ),
    "maximum_actuator_damping_scale_by_joint": max(
        max(float(value) for value in item["actuator_damping_scale_by_joint"])
        for item in results
    ),
    "minimum_foot_static_friction": min(
        min(float(value) for value in item["foot_static_friction"])
        for item in results
    ),
    "maximum_foot_static_friction": max(
        max(float(value) for value in item["foot_static_friction"])
        for item in results
    ),
    "minimum_foot_dynamic_friction": min(
        min(float(value) for value in item["foot_dynamic_friction"])
        for item in results
    ),
    "maximum_foot_dynamic_friction": max(
        max(float(value) for value in item["foot_dynamic_friction"])
        for item in results
    ),
    "mean_effort_soft_limit_exceedance_fraction": sum(
        float(item["effort_soft_limit_exceedance_fraction"]) for item in results
    )
    / len(results),
    # Explicit alias: this is the fraction of joint-timestep samples, not the
    # fraction of controller frames in which the whole robot was overloaded.
    "effort_soft_limit_exceedance_joint_timestep_fraction": sum(
        float(item["effort_soft_limit_exceedance_fraction"]) for item in results
    )
    / len(results),
    "maximum_applied_effort_fraction": max(
        float(item["maximum_applied_effort_fraction"]) for item in results
    ),
    "mean_abs_applied_effort_nm_by_joint": {
        name: sum(
            float(item["mean_abs_applied_effort_nm_by_joint"][name])
            for item in results
        )
        / len(results)
        for name in env.unwrapped._robot.joint_names
    },
    "mean_rms_applied_effort_nm_by_joint": {
        name: sum(
            float(item["rms_applied_effort_nm_by_joint"][name])
            for item in results
        )
        / len(results)
        for name in env.unwrapped._robot.joint_names
    },
    "mean_applied_effort_fraction_by_joint": {
        name: sum(
            float(item["mean_applied_effort_fraction_by_joint"][name])
            for item in results
        )
        / len(results)
        for name in env.unwrapped._robot.joint_names
    },
    "mean_rms_applied_effort_fraction_by_joint": {
        name: sum(
            float(item["rms_applied_effort_fraction_by_joint"][name])
            for item in results
        )
        / len(results)
        for name in env.unwrapped._robot.joint_names
    },
    "mean_effort_soft_limit_exceedance_fraction_by_joint": {
        name: sum(
            float(
                item["effort_soft_limit_exceedance_fraction_by_joint"][name]
            )
            for item in results
        )
        / len(results)
        for name in env.unwrapped._robot.joint_names
    },
    "maximum_applied_effort_fraction_by_joint": {
        name: max(
            float(item["maximum_applied_effort_fraction_by_joint"][name])
            for item in results
        )
        for name in env.unwrapped._robot.joint_names
    },
    "gait_reference": {
        "mode": env_cfg.gait_reference_mode,
        "smooth_support_push": env_cfg.gait_smooth_support_push,
        "distributed_push_phase_fractions": (
            list(env_cfg.gait_distributed_push_phase_fractions)
            if env_cfg.gait_distributed_push_phase_fractions is not None
            else None
        ),
        "contact_transition_fraction": (
            env_cfg.gait_contact_transition_fraction
        ),
        "stance_fore_aft_m": env_cfg.gait_stance_fore_aft_m,
        "stance_down_m": env_cfg.gait_stance_down_m,
        "stride_m": env_cfg.gait_stride_m,
        "lift_m": env_cfg.gait_lift_m,
        "weight_shift_forward_m": env_cfg.gait_weight_shift_forward_m,
        "rear_weight_shift_forward_m": (
            env_cfg.gait_rear_weight_shift_forward_m
        ),
        "weight_shift_lateral_m": env_cfg.gait_weight_shift_lateral_m,
        "translate_lateral_weight_shift": (
            env_cfg.gait_translate_lateral_weight_shift
        ),
        "forward_body_pitch_rad": env_cfg.gait_forward_body_pitch_rad,
        "stance_center_offset_m": env_cfg.gait_stance_center_offset_m,
        "stance_forward_bias_m": float(env.unwrapped._stance_forward_bias_m),
        "frequency_range_hz": [
            env_cfg.gait_frequency_min_hz,
            env_cfg.gait_frequency_max_hz,
        ],
    },
    "straight_path_control": {
        "enabled": env_cfg.track_episode_world_path,
        "heading_hold_kp_s": env_cfg.heading_hold_kp_s,
        "heading_hold_max_correction_rad_s": (
            env_cfg.heading_hold_max_correction_rad_s
        ),
    },
    "physical_randomization": {
        "actuator_peak_effort_nm": env_cfg.actuator_peak_effort_nm,
        "nominal_environment_fraction": (
            env_cfg.physical_randomization_nominal_fraction
        ),
        "mirror_left_right_pairs": env_cfg.mirror_physical_randomization_pairs,
        "global_effort_scale_range": list(env_cfg.actuator_effort_scale_range),
        "global_target_rate_scale_range": list(env_cfg.target_velocity_scale_range),
        "correlate_common_actuator_scales": (
            env_cfg.correlate_common_actuator_scales
        ),
        "individual_effort_scale_range": list(
            env_cfg.actuator_individual_effort_scale_range
        ),
        "individual_target_rate_scale_range": list(
            env_cfg.target_individual_velocity_scale_range
        ),
        "individual_stiffness_scale_range": list(
            env_cfg.actuator_individual_stiffness_scale_range
        ),
        "individual_damping_scale_range": list(
            env_cfg.actuator_individual_damping_scale_range
        ),
        "control_delay_step_range": list(env_cfg.control_delay_step_range),
        "reset_roll_pitch_noise_rad": env_cfg.reset_roll_pitch_noise_rad,
        "reset_yaw_noise_rad": env_cfg.reset_yaw_noise_rad,
        "joint_target_bias_abduction_rad": env_cfg.joint_target_bias_abduction_rad,
        "joint_target_bias_flexion_rad": env_cfg.joint_target_bias_flexion_rad,
        "imu_angular_velocity_bias_range_rad_s": (
            env_cfg.imu_angular_velocity_bias_range_rad_s
        ),
        "imu_projected_gravity_noise_std": (
            env_cfg.imu_projected_gravity_noise_std
        ),
        "imu_linear_acceleration_noise_std_g": (
            env_cfg.imu_linear_acceleration_noise_std_g
        ),
        "base_force_randomization_range_n": list(
            env_cfg.base_force_randomization_range_n
        ),
        "base_torque_randomization_range_nm": list(
            env_cfg.base_torque_randomization_range_nm
        ),
        "rear_payload_mass_scale_range": list(
            env_cfg.rear_payload_combined_mass_scale_range
        ),
        "dry_robot_mass_scale": env_cfg.dry_robot_mass_scale,
        "robot_mass_scale_range": list(env_cfg.robot_mass_scale_range),
        "rear_payload_com_jitter_m": list(
            env_cfg.rear_payload_combined_com_jitter_m
        ),
        "shoe_static_friction": env_cfg.shoe_static_friction,
        "shoe_dynamic_friction": env_cfg.shoe_dynamic_friction,
        "shoe_static_friction_randomization_range": list(
            env_cfg.shoe_static_friction_randomization_range or ()
        ),
        "shoe_dynamic_friction_randomization_range": list(
            env_cfg.shoe_dynamic_friction_randomization_range or ()
        ),
        "shoe_common_static_friction_randomization_range": list(
            env_cfg.shoe_common_static_friction_randomization_range or ()
        ),
        "shoe_common_dynamic_friction_randomization_range": list(
            env_cfg.shoe_common_dynamic_friction_randomization_range or ()
        ),
        "shoe_friction_differential_scale_range": list(
            env_cfg.shoe_friction_differential_scale_range
        ),
        "effort_soft_limit_fraction": env_cfg.effort_soft_limit_fraction,
        "nominal_effort_soft_limit_nm": (
            env_cfg.actuator_peak_effort_nm
            * env_cfg.effort_soft_limit_fraction
        ),
    },
}
print("SUSTAINED_EVALUATION_JSON=" + json.dumps(summary, sort_keys=True), flush=True)
env.close()
simulation_app.close()
