"""Complete diagonal steps before changing pairs, with hardware-available gates.

Only encoders, IMU, and controller state gate the reference. Simulator contacts,
sole corners, forces, and COM qualify rewards/critic metrics, never the actor or
scheduler. A proprioceptive settle is not a measurement of floor contact.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from ankle_v2.env import AnkleEnv, rotate
from ankle_v2.env_cfg import LEGS
from .env_cfg import DiagonalCfg
from .progress_io import best_effort_json


def smooth(value):
    u = value.clamp(0., 1.)
    return u**3 * (10. + u * (-15. + 6. * u))


class DiagonalEnv(AnkleEnv):
    cfg: DiagonalCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, device = self.num_envs, self.device
        self.index = torch.arange(n, device=device)
        self.pairs = torch.tensor([[True, False, False, True],
                                   [False, True, True, False]], device=device)
        self.support_ids = torch.tensor([[1, 2], [0, 3]], device=device)
        self.pair = torch.zeros(n, dtype=torch.long, device=device)
        self.stage = torch.zeros_like(self.pair)
        self.stage_time = torch.zeros(n, device=device)
        self.settle_time = torch.zeros_like(self.stage_time)
        self.gate_wait = torch.zeros_like(self.stage_time)
        self.gate_failed = torch.zeros(n, dtype=torch.bool, device=device)
        self.parked = torch.zeros_like(self.gate_failed)
        self.started = torch.zeros_like(self.gate_failed)
        self.first_placement = torch.ones_like(self.gate_failed)
        self.stage_fractions = torch.tensor(cfg.stage_fractions, device=device)
        self.stage_starts = torch.cat((torch.zeros(1, device=device),
                                       self.stage_fractions.cumsum(0)[:-1]))
        self.stride = torch.zeros(n, device=device)
        self.period = torch.ones(n, device=device)
        self.filtered = torch.zeros_like(self.actions)
        self.residual_scale = torch.tensor([s for s in cfg.residual_scales for _ in LEGS], device=device)
        self.accels = torch.tensor([math.radians(s) for s in cfg.target_accel_deg_s2 for _ in LEGS], device=device)
        self.output_velocity = torch.zeros_like(self.targets)
        self.last_output_velocity = torch.zeros_like(self.targets)
        self.skipped_output = torch.zeros(n, dtype=torch.bool, device=device)
        self.ankle_reference = self.neutral[:, 12:16].clone()
        self.ref = self.neutral.clone()
        self.requested_target = self.neutral.clone()
        self.unconstrained_target = self.neutral.clone()
        self.residual_projected = torch.zeros_like(self.targets, dtype=torch.bool)
        self.reference_clipped = torch.zeros_like(self.targets, dtype=torch.bool)
        self.ref_forward = torch.zeros((n, 4), device=device)
        self.ref_down = torch.full_like(self.ref_forward, cfg.stance_down_m)
        self.ref_clearance = torch.zeros_like(self.ref_forward)
        self.expected_stance = torch.ones((n, 4), dtype=torch.bool, device=device)
        self.front_sign = torch.tensor([1., -1., 1., -1.], device=device)
        self.corners = torch.tensor([[cfg.contact_x_m, y, z]
                                    for y in (-.047, .047) for z in (-.027, .027)], device=device)
        # All directed edges of the two supporting sole patches. The hull keeps
        # real finite foot area; two contact centers alone have zero area.
        edges = [(a, b) for a in range(8) for b in range(8) if a != b]
        self.edge_a = torch.tensor([a for a, _ in edges], device=device)
        self.edge_b = torch.tensor([b for _, b in edges], device=device)
        self.history_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.liftoff_sole = torch.zeros((n, 4, 3), device=device)
        self.flight_peak = torch.zeros((n, 4), device=device)
        self.flight_valid = torch.zeros((n, 4), dtype=torch.bool, device=device)
        self.flight_supported = torch.ones((n, 4), dtype=torch.bool, device=device)
        self.landed = torch.zeros((n, 4), dtype=torch.bool, device=device)
        self.landed_advance = torch.zeros((n, 4), device=device)
        self.touchdown_paid = torch.zeros((n, 4), dtype=torch.bool, device=device)
        self.qualified_touchdowns = torch.zeros((n, 4), device=device)
        self.lift_best_score = torch.zeros(n, device=device)
        self.foot_lift_best_score = torch.zeros((n, 4), device=device)
        self.lift_samples_by_leg = torch.zeros((n, 4), device=device)
        self.lift_clear_by_leg = torch.zeros((n, 4), device=device)
        self.failed_pair_event = torch.zeros(n, dtype=torch.bool, device=device)
        self.pair_paid = torch.zeros(n, dtype=torch.bool, device=device)
        self.plant_time = torch.zeros(n, device=device)
        self.step_counts = torch.zeros((n, 4), device=device)
        self.stride_sums = torch.zeros_like(self.step_counts)
        self.previous_angular = torch.zeros((n, 3), device=device)
        self.previous_qd = torch.zeros_like(self.targets)
        self.previous_target_delta = torch.zeros_like(self.targets)
        self.last_diagnostics = {}
        self.diagonal_log_step = -640
        self.gate_log_step = -640
        self.gate_names = ("tracking", "delivery", "forward", "height", "speed", "tilt", "angular")
        self.stage_names = ("transfer", "lift", "advance", "lower", "settle", "push")
        self.gate_values = torch.zeros((n, len(self.gate_names)), device=device)
        self.gate_blocked = torch.zeros_like(self.gate_values, dtype=torch.bool)
        self.gate_seen_blockers = torch.zeros_like(self.gate_blocked)
        self.gate_waiting = torch.zeros(n, dtype=torch.bool, device=device)
        self.gate_failure_stage = torch.full((n,), -1, dtype=torch.long, device=device)
        self.gate_failure_blocked = torch.zeros_like(self.gate_blocked)
        self.stage_entries = torch.zeros((n, 6), device=device)
        self.stage_wait_s = torch.zeros_like(self.stage_entries)
        self.accel_limited = torch.zeros_like(self.targets, dtype=torch.bool)
        self.diagnostic_totals = {
            "completed_episodes": 0, "completed_moving_episodes": 0,
            "timeouts_by_stage": {name: 0 for name in self.stage_names},
            "timeouts_by_pair": {"front_left_rear_right": 0, "front_right_rear_left": 0},
            "timeout_blockers": {name: 0 for name in self.gate_names},
            "stage_entries": {name: 0 for name in self.stage_names},
            "qualified_touchdowns_by_leg": {name: 0 for name in LEGS},
        }
        self.diagonal_sums = {key: torch.zeros(n, device=device) for key in (
            "samples", "swing_samples", "clear_samples", "stance_samples", "flat_samples",
            "slip", "angular_square", "angular_accel_square", "target_accel_square",
            "support_margin", "support_good", "paired_landings", "failed_pairs", "gate_wait_s",
            "control_samples", "accel_limited", "reference_clipped", "output_holds", "lift_samples", "lift_clear_samples", "swing_projection")}
        self._apply_calibrated_ankle_bounds()
        contract = {
            "schema": "drobot.ankle-v24-diagonal.training.v4",
            "pairs": [["front_left", "rear_right"], ["front_right", "rear_left"]],
            "stages": ["transfer", "lift", "advance", "lower", "settle", "all_foot_push"],
            "stage_fractions": cfg.stage_fractions,
            "actor_inputs": 82, "critic_inputs": 101,
            "warm_start_compatibility": "same dimensions; new clock/reference semantics",
            "scheduler_gate": "delivered/requested target tracking, planar encoder foot-path proxy, joint speed and IMU",
            "gated_stages": ["lift", "advance", "settle"],
            "floor_contact_gate": False, "timeout": "terminate failed episode; never force next pair",
            "final_stride_m": [cfg.stride_min_m, cfg.stride_max_m],
            "lift_m": cfg.lift_m, "target_accel_deg_s2": cfg.target_accel_deg_s2,
            "stance_forward_down_m": [cfg.stance_forward_m, cfg.stance_down_m],
            "body_forward_shift_m": cfg.body_forward_shift_m,
            "maximum_leg_reach_m": cfg.maximum_leg_reach_m,
            "startup": "half-stride placement from reset stance; excluded from full paired-step counts",
            "swing_residual_projection": {
                "forward_window_m": cfg.swing_forward_window_m,
                "extra_lift_m": cfg.swing_extra_lift_m,
                "maximum_lift_cancellation_m": cfg.swing_lift_cancellation_m,
                "stages": ["lift", "advance", "lower"],
                "coordinates": "planar ankle pivot; does not certify ground clearance"},
            "lift_shaping": "monotonic supported per-foot clearance and bounded per-foot progress bonuses",
            "velocity_reward_stage": "all-foot push only",
            "gate_thresholds": {"tracking_deg": cfg.gate_position_deg,
                "delivery_deg": cfg.gate_delivery_deg, "forward_m": cfg.gate_forward_m,
                "height_m": cfg.gate_height_m, "settle_height_m": cfg.gate_settle_height_m},
            "calibration": self.calibration_record,
            "limits": "intersection of unchanged USD soft limits and captured ankle bounds minus 5 deg",
            "evaluation_mode": cfg.evaluation, "hardware_qualified": False,
            "runtime_compatible_with_deployed_v23": False,
        }
        if self.metrics_file:
            self.metrics_file.with_suffix(".contract.json").write_text(json.dumps(contract, indent=2) + "\n")
        print("DIAGONAL_TRAINING_CONTRACT=" + json.dumps(contract), flush=True)

    def _apply_calibrated_ankle_bounds(self):
        calibration_path, travel_path = Path(self.cfg.hardware_calibration), Path(self.cfg.hardware_travel)
        calibration = json.loads(calibration_path.read_text())
        travel = json.loads(travel_path.read_text())
        by_name = {row["name"]: row for row in calibration["motors"]}
        travel_by_name = {row["name"]: row for row in travel["motors"]}
        rows = []
        for leg_index, leg in enumerate(LEGS):
            name = leg + "_ankle_pitch"
            center, observed = by_name[name], travel_by_name[name]
            if any(center[key] != observed[key] for key in ("id", "center_tick", "direction")):
                raise ValueError("Ankle travel coordinates disagree with calibration: " + name)
            low, high = observed["min_angle_deg"] + 5., observed["max_angle_deg"] - 5.
            if not low < high:
                raise ValueError("Empty ankle travel interval: " + name)
            joint = 12 + leg_index
            self.limits[:, joint, 0].clamp_(min=math.radians(low))
            self.limits[:, joint, 1].clamp_(max=math.radians(high))
            if (self.limits[:, joint, 0] >= self.limits[:, joint, 1]).any():
                raise ValueError("Calibrated ankle and model intervals do not intersect")
            rows.append({**center, "observed_inward_limits_deg": [low, high],
                         "effective_limits_rad": self.limits[0, joint].tolist()})
        self.calibration_record = {
            "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            "travel_sha256": hashlib.sha256(travel_path.read_bytes()).hexdigest(), "ankles": rows}

    def _curriculum(self):
        return 1. if self.cfg.evaluation else min(1., self.common_step_counter / self.cfg.curriculum_steps)

    def _age(self):
        return self.episode_length_buf.float() * self.step_dt

    def _stop_requested(self):
        stop = self.cfg.evaluation_stop_s if self.cfg.evaluation else self.cfg.episode_length_s - 8.
        return self._age() >= stop

    def _set_commands(self):
        cruise = torch.full_like(self.cruise, self.cfg.evaluation_speed) if self.cfg.evaluation else self.cruise
        self.commands[:, 0] = cruise * (.6 + .4 * self._curriculum())
        # Finish the current placement, including landing, before parking.
        self.commands[:, 0] *= smooth(self._age() / self.cfg.startup_s)
        self.commands[:, 0] *= ~self.parked

    def _stage_progress(self):
        duration = self.period * self.stage_fractions[self.stage]
        return (self.stage_time / duration).clamp(0., 1.), duration

    @staticmethod
    def _planar_ankle_position(joints):
        """Existing two-link reference coordinates, not world sole clearance.

        No simulator foot poses or contact signals enter the controller gate.
        This proxy ignores body translation/roll and must not certify contact.
        """
        hip, knee = joints[:, 4:8], joints[:, 8:12]
        forward = .159896689 * (torch.sin(hip) + torch.sin(hip + knee))
        down = .159896689 * (torch.cos(hip) + torch.cos(hip + knee))
        return forward, down

    def _advance_scheduler(self):
        active = (self._age() >= self.cfg.startup_s) & (self.cruise > 0.) & ~self.parked
        if self.cfg.evaluation:
            active = (self._age() >= self.cfg.startup_s) & ~self.parked
        newly_started = active & ~self.started
        self.stage_entries[:, 0] += newly_started
        self.started |= active
        self.stage_time += active * self.step_dt
        _, duration = self._stage_progress()
        at_end = self.stage_time >= duration
        q, qd = self.robot.data.joint_pos.torch[:, self.ids], self.robot.data.joint_vel.torch[:, self.ids]
        # A residual may intentionally differ from the nominal joint reference
        # by more than five degrees. Check delivery/tracking against the actual
        # command, while an independent foot-path proxy prevents cancelling the
        # requested lift or forward advance and passing on target tracking alone.
        leg_mask = self.pairs[self.pair]
        leg_mask = torch.where((self.stage == 4)[:, None], torch.ones_like(leg_mask), leg_mask)
        joint_mask = leg_mask.repeat(1, 4)
        error = torch.where(joint_mask, (q - self.targets).abs(), 0.).amax(1)
        delivery = torch.where(joint_mask, (self.requested_target - self.targets).abs(), 0.).amax(1)
        forward, down = self._planar_ankle_position(q)
        forward_error = torch.where(leg_mask, (forward - self.ref_forward).abs(), 0.).amax(1)
        height_error = torch.where(leg_mask, (down - self.ref_down).abs(), 0.).amax(1)
        speed = torch.where(joint_mask, qd.abs(), 0.).amax(1)
        tilt = self.robot.data.projected_gravity_b.torch[:, :2].norm(dim=1)
        angular = self.imu.data.ang_vel_b.torch.norm(dim=1)
        height_limit = torch.where(self.stage == 4, self.cfg.gate_settle_height_m, self.cfg.gate_height_m)
        self.gate_values = torch.stack((error, delivery, forward_error, height_error, speed, tilt, angular), 1)
        self.gate_blocked = torch.stack((
            error >= math.radians(self.cfg.gate_position_deg),
            delivery >= math.radians(self.cfg.gate_delivery_deg),
            forward_error >= self.cfg.gate_forward_m, height_error >= height_limit,
            speed >= math.radians(self.cfg.gate_velocity_deg_s),
            tilt >= self.cfg.gate_tilt_rad, angular >= self.cfg.gate_angular_rad_s), 1)
        self.gate_blocked |= ~torch.isfinite(self.gate_values)
        ready = ~self.gate_blocked.any(1)
        gated = (self.stage == 1) | (self.stage == 2) | (self.stage == 4)
        waiting = active & at_end & gated
        self.gate_waiting = waiting
        self.gate_seen_blockers = torch.where(waiting[:, None], self.gate_seen_blockers | self.gate_blocked, False)
        self.stage_wait_s[self.index, self.stage] += waiting * self.step_dt
        self.settle_time = torch.where(waiting & ready, self.settle_time + self.step_dt, 0.)
        self.gate_wait = torch.where(waiting, self.gate_wait + self.step_dt, 0.)
        failed = waiting & (self.gate_wait > self.cfg.gate_timeout_s)
        self.gate_failure_stage = torch.where(failed, self.stage, self.gate_failure_stage)
        self.gate_failure_blocked = torch.where(failed[:, None], self.gate_seen_blockers, self.gate_failure_blocked)
        self.gate_failed |= failed
        advance = active & at_end & (~gated | (self.settle_time >= self.cfg.settle_hold_s)) & ~self.gate_failed
        finish = advance & (self.stage == 5)
        # A failed physical landing is counted even if proprioception appeared settled.
        self.failed_pair_event = finish & ~self.pair_paid
        self.diagonal_sums["failed_pairs"] += self.failed_pair_event
        park = advance & (self.stage == 4) & self._stop_requested()
        self.parked |= park
        self.stage = torch.where(advance & ~park, (self.stage + 1) % 6, self.stage)
        self.stage_entries[self.index, self.stage] += advance & ~park
        self.stage_time = torch.where(advance & ~park, 0., self.stage_time)
        self.pair = torch.where(finish, 1 - self.pair, self.pair)
        self.first_placement[finish] = False
        self.landed[finish] = False
        self.landed_advance[finish] = 0.
        self.touchdown_paid[finish] = False
        self.lift_best_score[finish] = 0.
        self.foot_lift_best_score[finish] = 0.
        self.pair_paid[finish] = False
        self.plant_time[finish] = 0.
        self.flight_valid[finish] = False
        self.flight_peak[finish] = 0.
        self.air_time[finish] = 0.
        self.diagonal_sums["gate_wait_s"] += waiting * self.step_dt

    def _reference(self):
        u, _ = self._stage_progress()
        blend = smooth(u)
        selected = self.pairs[self.pair]
        after_advance = self.stage >= 3
        advance = torch.where(after_advance, 1., torch.where(self.stage == 2, blend, 0.))
        lift = torch.where(self.stage == 1, blend,
                           torch.where(self.stage == 2, 1., torch.where(self.stage == 3, 1. - blend, 0.)))
        push = torch.where(self.stage == 5, blend, 0.)
        startup = smooth(self._age() / self.cfg.startup_s)
        # Start with all feet in their real reset stance. Preloading half a
        # stride backward while all four feet are planted creates incompatible
        # closed-chain targets and loads the pair before it can lift.
        start_offset = torch.where(self.first_placement, 0., -.5)
        offset = selected * (start_offset[:, None] + (0.5 - start_offset[:, None]) * advance[:, None]) * self.stride[:, None]
        offset -= .5 * push[:, None] * self.stride[:, None]
        offset *= startup[:, None]
        clearance = selected * lift[:, None] * self.cfg.lift_m
        # Move the trunk slightly forward over its real rear-biased payload.
        shift = self.cfg.body_forward_shift_m * startup
        forward = self.front_sign * self.cfg.stance_forward_m + offset - shift[:, None]
        down = self.cfg.stance_down_m - clearance
        down = torch.minimum(down, (self.cfg.maximum_leg_reach_m**2 - forward.square()).clamp(min=.01).sqrt())
        length = .159896689
        cosine = ((forward.square() + down.square() - 2 * length**2) / (2 * length**2)).clamp(-.999, .999)
        knee = -self.front_sign * torch.acos(cosine)
        hip = torch.atan2(forward, down) - torch.atan2(torch.sin(knee), 1 + torch.cos(knee))
        ref = self.neutral.clone()
        ref[:, 4:8], ref[:, 8:12] = hip, knee
        actual = self.robot.data.joint_pos.torch[:, self.ids]
        pitch = torch.asin(self.robot.data.projected_gravity_b.torch[:, 0].clamp(-1, 1))
        # Correct measured hip/knee lag using active ankle pitch. Filter this
        # measurement-derived reference before the rate/acceleration limiter.
        ankle = pitch[:, None] - actual[:, 4:8] - actual[:, 8:12]
        ankle -= .04 * self.imu.data.ang_vel_b.torch[:, 1:2]
        self.ankle_reference += self.step_dt / (self.cfg.ankle_reference_filter_s + self.step_dt) * (ankle - self.ankle_reference)
        ref[:, 12:16] = self.ankle_reference
        self.reference_clipped = (ref < self.limits[:, :, 0]) | (ref > self.limits[:, :, 1])
        self.ref = torch.maximum(torch.minimum(ref, self.limits[:, :, 1]), self.limits[:, :, 0])
        self.ref_forward, self.ref_clearance = forward, clearance
        self.ref_down = down
        self.expected_stance = ~(selected & ((self.stage >= 1) & (self.stage <= 3))[:, None])
        self.expected_stance |= (~self.started | self.parked)[:, None]
        return self.ref

    def _project_swing_residual(self, desired):
        """Allow path refinements without cancelling the scheduled swing.

        Supporting legs keep their existing joint residual authority. The
        unchanged joint bounds and output limiter remain the final constraint.
        """
        selected = self.pairs[self.pair]
        swing = selected & ((self.stage >= 1) & (self.stage <= 3))[:, None]
        forward, down = self._planar_ankle_position(desired)
        forward = torch.maximum(torch.minimum(forward, self.ref_forward + self.cfg.swing_forward_window_m),
                                self.ref_forward - self.cfg.swing_forward_window_m)
        down = torch.maximum(torch.minimum(down, self.ref_down + self.cfg.swing_lift_cancellation_m),
                             self.ref_down - self.cfg.swing_extra_lift_m)
        length = .159896689
        cosine = ((forward.square() + down.square() - 2 * length**2) / (2 * length**2)).clamp(-.999, .999)
        knee = -self.front_sign * torch.acos(cosine)
        hip = torch.atan2(forward, down) - knee / 2.
        projected = desired.clone()
        projected[:, 4:8] = torch.where(swing, hip, desired[:, 4:8])
        projected[:, 8:12] = torch.where(swing, knee, desired[:, 8:12])
        self.residual_projected = (projected - desired).abs() > 1.e-5
        return projected

    def _pre_physics_step(self, actions):
        self._set_commands()
        self._advance_scheduler()
        self.last_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1., 1.))
        self.filtered += self.step_dt / (self.cfg.filter_time_s + self.step_dt) * (self.actions - self.filtered)
        desired = self._reference() + self.filtered * self.residual_scale
        self.unconstrained_target.copy_(desired)
        desired = self._project_swing_residual(desired)
        desired = torch.maximum(torch.minimum(desired, self.limits[:, :, 1]), self.limits[:, :, 0])
        self.requested_target.copy_(desired)
        # Random single-cycle holds teach tolerance to 16.7/33.3 ms deliveries.
        deliver = torch.ones_like(self.skipped_output) if self.cfg.evaluation else (
            (torch.rand(self.num_envs, device=self.device) >= self.cfg.output_skip_probability) | self.skipped_output)
        elapsed = self.step_dt * (1. + self.skipped_output.float())[:, None]
        delta = desired - self.targets
        rate_cap = self.rates * self.rate_strength
        braking_speed = torch.sqrt(2 * self.accels * delta.abs())
        requested_velocity = delta.sign() * torch.minimum(rate_cap, torch.minimum(braking_speed, delta.abs() / elapsed))
        velocity = self.output_velocity + (requested_velocity - self.output_velocity).clamp(-self.accels * elapsed, self.accels * elapsed)
        change = (velocity * elapsed).clamp(-math.radians(2.), math.radians(2.))
        next_target = torch.maximum(torch.minimum(self.targets + change, self.limits[:, :, 1]), self.limits[:, :, 0])
        self.previous_target_delta.copy_(self.target_delta)
        self.last_output_velocity.copy_(self.output_velocity)
        self.target_delta = torch.where(deliver[:, None], next_target - self.targets, 0.)
        self.output_velocity = torch.where(deliver[:, None], self.target_delta / elapsed, self.output_velocity)
        # Residual target error is not evidence that a velocity cap was hit.
        # Distinguish rate/packet limits, acceleration shaping, and bus holds.
        step_cap = torch.minimum(rate_cap * elapsed, torch.full_like(delta, math.radians(2.)))
        self.rate_limited = (delta.abs() > step_cap + 1.e-5) & deliver[:, None]
        self.accel_limited = ((requested_velocity - velocity).abs() > 1.e-5) & deliver[:, None]
        self.targets += self.target_delta
        self.skipped_output = ~deliver

    def _geometry(self):
        quat = self.robot.data.body_link_quat_w.torch[:, self.feet]
        foot = self.robot.data.body_link_pos_w.torch[:, self.feet]
        corners = foot.unsqueeze(2) + rotate(quat.unsqueeze(2), self.corners)
        heights = corners[..., 2] - self.scene.env_origins[:, None, 2:3]
        return corners, corners.mean(2), heights.amin(2), heights.amax(2)

    def _support(self, corners):
        mass = self.robot.data.body_mass.torch
        com = (self.robot.data.body_com_pos_w.torch * mass[:, :, None]).sum(1) / mass.sum(1)[:, None]
        support = corners[self.index[:, None], self.support_ids[self.pair], :, :2].reshape(self.num_envs, 8, 2)
        a = support[:, self.edge_a]
        edge = support[:, self.edge_b] - a
        offsets = support[:, None, :, :] - a[:, :, None, :]
        crosses = edge[:, :, None, 0] * offsets[..., 1] - edge[:, :, None, 1] * offsets[..., 0]
        length = edge.norm(dim=2)
        boundary = (crosses >= -1.e-7).all(2) & (length > 1.e-5)
        d = com[:, None, :2] - a
        signed = (edge[..., 0] * d[..., 1] - edge[..., 1] * d[..., 0]) / length.clamp(min=1.e-5)
        margin = torch.where(boundary, signed, torch.inf).amin(1)
        margin = torch.where(torch.isfinite(margin), margin, -.1)
        force = self.contacts.data.net_forces_w.torch[:, self.foot_sensors, 2].clamp(min=0.)
        return com, margin, force, mass.sum(1) * 9.81

    def _get_observations(self):
        self._set_commands()
        u, _ = self._stage_progress()
        clock_phase = (self.pair.float() + self.stage_starts[self.stage] + u * self.stage_fractions[self.stage]) * torch.pi
        clock = torch.stack((clock_phase.sin(), clock_phase.cos()), dim=1)
        q, qd = self.robot.data.joint_pos.torch[:, self.ids], self.robot.data.joint_vel.torch[:, self.ids]
        policy = torch.cat((self.commands * 10, clock, self.imu.data.ang_vel_b.torch,
            self.robot.data.projected_gravity_b.torch, self.imu.data.lin_acc_b.torch / 9.81,
            q - self.neutral, qd * .25, ((self.targets - self.neutral) / self.scales).clamp(-1, 1),
            self.filtered, self.expected_stance.float()), dim=1)
        corners, sole, _, _ = self._geometry()
        com, margin, force, weight = self._support(corners)
        critic = torch.cat((policy, self.robot.data.root_lin_vel_b.torch,
            self.robot.data.root_pos_w.torch[:, 2:3], self._contact().float(), force / weight[:, None],
            (com[:, :2] - sole.mean(1)[:, :2]) * 10, margin[:, None] * 10, self.pairs[self.pair].float()), dim=1)
        return {"policy": policy, "critic": critic}

    def _get_dones(self):
        fallen, timeout = super()._get_dones()
        return fallen | self.gate_failed, timeout

    def _get_rewards(self):
        corners, sole, low, high = self._geometry()
        com, margin, force, weight = self._support(corners)
        contact = self._contact()
        selected = self.pairs[self.pair]
        support = ~selected
        span = high - low
        flat = contact & (span < .005) & (high < .004)
        pos, vel = self.robot.data.root_pos_w.torch, self.robot.data.root_lin_vel_b.torch
        angular = self.imu.data.ang_vel_b.torch
        joints, qd = self.robot.data.joint_pos.torch[:, self.ids], self.robot.data.joint_vel.torch[:, self.ids]
        effort = self.robot.data.applied_torque.torch[:, self.ids]
        posture = self.robot.data.projected_gravity_b.torch[:, :2].square().sum(1)
        x, y, z, w = self.robot.data.root_quat_w.torch.unbind(1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        valid = self.history_valid
        moving = self.started & ~self.parked & (self.commands[:, 0] > .005)
        swing = selected & ((self.stage >= 1) & (self.stage <= 3))[:, None]
        mid_swing = selected & (self.stage == 2)[:, None]
        supporting_contact = ((contact | selected).all(1)
                              & ((force * support).sum(1) > .60 * weight)
                              & (posture < .16**2) & (margin > 0.) & ~self.fallen)
        slip_speed = (sole - self.previous_sole)[..., :2].norm(dim=2) / self.step_dt * valid[:, None]
        stance_count = self.expected_stance.sum(1).clamp(min=1)
        stance_contact = self.expected_stance & contact
        stance_score = stance_contact.sum(1) / stance_count
        flat_score = (torch.exp(-(span / .008).square()) * stance_contact).sum(1) / stance_count
        slip = (((slip_speed / .015).square().clamp(max=16)) * contact).sum(1) / contact.sum(1).clamp(min=1)
        clear = mid_swing & ~contact & (low > .008)
        # Shape lift before the lift gate. Previously the only height reward
        # was in stage 2, which failing stage-1 episodes could never reach.
        height_active = swing & (self.ref_clearance > .004)
        # A few mm of real rear-foot lift now earns useful shaping instead of
        # almost zero Gaussian reward. Floor scraping still earns no clearance.
        clearance_fraction = (low / self.ref_clearance.clamp(min=.004)).clamp(0., 1.)
        clearance_score = (clearance_fraction * height_active).sum(1) / 2.
        clearance_score *= supporting_contact
        excess_clearance = (((low - self.ref_clearance - .010).clamp(min=0.) / .020).square().clamp(max=4.) * height_active).sum(1) / 2.
        unload_fraction = (1. - force / (.25 * weight[:, None]).clamp(min=1.e-6)).clamp(0., 1.)
        unload_score = (unload_fraction * height_active).sum(1) / 2. * supporting_contact
        lift_stage = (self.stage == 1)[:, None] & selected
        lift_clear = lift_stage & ~contact & (low > .008)
        pair_height = torch.where(selected, low, torch.inf).amin(1)
        lift_progress = (pair_height / self.cfg.lift_m).clamp(0., 1.)
        lift_progress *= supporting_contact & ((self.stage == 1) | (self.stage == 2))
        lift_best = torch.maximum(self.lift_best_score, lift_progress)
        lift_progress_reward = lift_best - self.lift_best_score
        self.lift_best_score.copy_(lift_best)
        foot_progress = (low / self.cfg.lift_m).clamp(0., 1.) * selected
        foot_progress *= (supporting_contact & ((self.stage == 1) | (self.stage == 2)))[:, None]
        foot_best = torch.maximum(self.foot_lift_best_score, foot_progress)
        foot_progress_reward = (foot_best - self.foot_lift_best_score).sum(1)
        self.foot_lift_best_score.copy_(foot_best)
        self.lift_samples_by_leg += lift_stage
        self.lift_clear_by_leg += lift_clear & supporting_contact[:, None]
        pair_clear = (clear | ~selected).all(1) & (self.stage == 2)
        wrong_lift = (~contact & self.expected_stance).sum(1).float()
        scuff = (mid_swing & contact).sum(1).float()
        support_score = torch.sigmoid((margin - self.cfg.sole_contact_margin_m) / .008) * supporting_contact
        # Flight evidence starts at real liftoff and is invalidated by loss of
        # the opposite diagonal. It cannot accumulate across floor scraping.
        liftoff = ~contact & self.previous_contact & valid[:, None]
        self.liftoff_sole = torch.where(liftoff[:, :, None], self.previous_sole, self.liftoff_sole)
        self.flight_valid = torch.where(liftoff, swing, self.flight_valid)
        self.flight_supported = torch.where(liftoff, supporting_contact[:, None], self.flight_supported)
        self.flight_supported &= contact | supporting_contact[:, None]
        self.air_time += (~contact) * self.step_dt
        self.flight_peak = torch.maximum(self.flight_peak, low * ~contact)
        touchdown = contact & ~self.previous_contact & valid[:, None]
        distance = sole[:, :, 0] - self.liftoff_sole[:, :, 0]
        sideways = (sole[:, :, 1] - self.liftoff_sole[:, :, 1]).abs()
        placement_stride = self.stride * torch.where(self.first_placement, .5, 1.)
        minimum_stride = .65 * placement_stride
        qualified = (touchdown & selected & self.flight_valid & self.flight_supported
                     & (self.air_time >= .10) & (self.flight_peak >= .008)
                     & (distance >= minimum_stride[:, None]) & (distance < .18)
                     & (sideways < .035) & (self.stage >= 3)[:, None])
        first_qualified = qualified & ~self.touchdown_paid
        self.touchdown_paid |= first_qualified
        self.qualified_touchdowns += first_qualified
        self.landed = torch.where(liftoff, False, self.landed) | qualified
        self.landed_advance = torch.where(qualified, distance, self.landed_advance)
        planted_pair = ((self.landed | ~selected).all(1) & flat.all(1)
                        & (posture < .12**2) & (angular.norm(dim=1) < .35)
                        & (slip_speed.amax(1) < .020) & (self.stage >= 4) & moving)
        self.plant_time = torch.where(planted_pair, self.plant_time + self.step_dt, 0.)
        completed = (self.plant_time >= .12) & ~self.pair_paid
        minimum_before = self.step_counts.amin(1)
        lagging = ((self.step_counts <= minimum_before[:, None]) & selected).sum(1) / 2.
        # Standing episodes intentionally have zero stride. They must still
        # produce finite rewards even though their completion mask is false.
        advance_score = ((self.landed_advance / placement_stride[:, None].clamp(min=1.e-6)).clamp(0, 1) * selected).sum(1) / 2.
        # Small, once-per-placement progress rewards bridge the sparse paired
        # landing event. They do not increment successful pair/step counts.
        full_completed = completed & ~self.first_placement
        event_reward = (full_completed * (2. + 2. * advance_score + lagging)
                        + (completed & self.first_placement).float()
                        + .8 * lift_progress_reward + .4 * foot_progress_reward
                        + .25 * first_qualified.sum(1))
        self.step_counts += full_completed[:, None] * selected
        self.stride_sums += full_completed[:, None] * selected * self.landed_advance
        event_reward += 2. * (self.step_counts.amin(1) - minimum_before)
        self.pair_paid |= completed
        self.air_time[contact] = 0.
        self.flight_peak[contact] = 0.
        self.flight_valid[contact] = False
        self.flight_supported[contact] = True
        # Dense shaping is deliberately subordinate to completed paired steps.
        # Body velocity is pulsed by the all-foot push, so track episode-average
        # progress rather than penalizing the stationary lift/landing phases.
        velocity_reward = (vel[:, 0] / self.commands[:, 0].clamp(min=.012)).clamp(-1., 2.)
        path_forward, path_down = self._planar_ankle_position(joints)
        path_score = torch.exp(-((path_forward - self.ref_forward) / .025).square()
                              - ((path_down - self.ref_down) / .015).square()).mean(1)
        angular_accel = (angular - self.previous_angular) / self.step_dt * valid[:, None]
        target_accel = (self.output_velocity - self.last_output_velocity) / self.step_dt * valid[:, None]
        smooth_action = (self.actions - self.last_actions).square().mean(1)
        ankle_limit = ((joints[:, 12:16] - self.limits[:, 12:16, 0] < .035)
                       | (self.limits[:, 12:16, 1] - joints[:, 12:16] < .035)).float().mean(1)
        power = (effort * qd).clamp(min=0).sum(1)
        lateral = pos[:, 1] - self.start_position[:, 1]
        lifting_or_advancing = moving & ((self.stage == 1) | (self.stage == 2))
        # Quiet contact alone must not dominate learning while the selected
        # feet are supposed to leave the floor. Quiet standing stays rewarded.
        support_reward_scale = torch.where(lifting_or_advancing, .25 + .75 * clearance_score, 1.)
        reward = (
            1.5 * velocity_reward * moving * (self.stage == 5)
            + (stance_score + flat_score + 2. * support_score * moving) * support_reward_scale
            + 4. * clearance_score * moving + .75 * unload_score * moving
            - .5 * excess_clearance * moving
            + 2. * pair_clear * supporting_contact + path_score * moving
            - 2. * slip - 3. * wrong_lift - 2. * scuff
            - 1.5 * (posture / .10**2).clamp(max=9.)
            - .8 * (angular / .35).square().sum(1).clamp(max=12.)
            - .15 * (angular_accel / 6.).square().mean(1).clamp(max=12.)
            - .15 * (target_accel / self.accels).square().mean(1).clamp(max=4.)
            - .20 * smooth_action - .06 * self.filtered.square().mean(1)
            - .5 * (lateral / .05).square().clamp(max=9.)
            - .5 * (yaw / .12).square().clamp(max=9.)
            - .3 * ((pos[:, 2] - self.cfg.target_base_height_m) / .03).square()
            - .03 * power - .3 * ankle_limit - 2. * vel[:, 2].square()
            - 2. * (self.gate_wait / self.cfg.gate_timeout_s).clamp(0., 1.) * self.gate_waiting
            - 20. * vel[:, :2].square().sum(1) * self.parked
        ) * self.step_dt + event_reward - self.failed_pair_event.float() - 8. * self.fallen - 5. * self.gate_failed
        delta = pos - self.previous_position
        self.sums["move_time"] += moving * self.step_dt
        self.sums["move_distance"] += delta[:, 0] * moving * valid
        self.sums["command_integral"] += self.commands[:, 0] * moving * self.step_dt
        self.sums["rate_limited"] += self.rate_limited.float().mean(1)
        self.sums["work"] += power * self.step_dt
        self.sums["flight_time"] += ~contact.any(1) * self.step_dt
        self.contact_counts += touchdown
        ds = self.diagonal_sums
        ds["samples"] += moving
        ds["swing_samples"] += mid_swing.sum(1)
        ds["clear_samples"] += (clear & supporting_contact[:, None]).sum(1)
        ds["stance_samples"] += self.expected_stance.sum(1) * moving
        ds["flat_samples"] += (flat & self.expected_stance).sum(1) * moving
        ds["slip"] += (slip_speed * stance_contact).sum(1) * moving
        ds["angular_square"] += angular.square().sum(1) * moving
        ds["angular_accel_square"] += angular_accel.square().sum(1) * moving
        ds["target_accel_square"] += target_accel.square().mean(1) * moving
        ds["support_margin"] += margin * moving
        ds["support_good"] += supporting_contact & moving
        ds["paired_landings"] += full_completed
        ds["control_samples"] += 1.
        ds["swing_projection"] += self.residual_projected.float().mean(1)
        ds["accel_limited"] += self.accel_limited.float().mean(1)
        ds["reference_clipped"] += self.reference_clipped.float().mean(1)
        ds["output_holds"] += self.skipped_output
        ds["lift_samples"] += lift_stage.sum(1)
        ds["lift_clear_samples"] += (lift_clear & supporting_contact[:, None]).sum(1)
        self.last_diagnostics = {key: value.detach().clone() for key, value in {
            "pair": self.pair, "stage": self.stage, "stage_time_s": self.stage_time,
            "gate_wait_s": self.gate_wait, "gate_failed": self.gate_failed, "parked": self.parked,
            "gate_values": self.gate_values, "gate_blocked": self.gate_blocked,
            "gate_waiting": self.gate_waiting, "gate_ready_hold_s": self.settle_time,
            "gate_failure_stage": self.gate_failure_stage,
            "gate_blockers_seen_during_wait": self.gate_seen_blockers,
            "contacts": contact, "flat": flat, "sole_clearance_m": low,
            "paired_landing": full_completed, "startup_landing": completed & self.first_placement,
            "first_placement": self.first_placement, "qualified_steps_by_leg": self.step_counts,
            "landed_advance_m": self.landed_advance, "support_margin_m": margin,
            "normal_force_n": force, "com_w": com, "body_velocity": vel,
            "body_angular_velocity": angular, "target_acceleration": target_accel,
            "joint_target": self.targets, "joint_position": joints}.items()}
        self.last_diagnostics.update({key: value.detach().clone() for key, value in {
            "requested_target": self.requested_target, "nominal_joint_reference": self.ref,
            "unconstrained_target": self.unconstrained_target, "residual_projected": self.residual_projected,
            "joint_velocity": qd, "reference_clipped": self.reference_clipped,
            "joint_effort_nm": effort, "body_position": pos,
            "reference_forward_m": self.ref_forward, "reference_clearance_m": self.ref_clearance,
            "reference_down_m": self.ref_down,
            "encoder_forward_m": path_forward, "encoder_down_m": path_down,
            "qualified_touchdowns_by_leg": self.qualified_touchdowns,
        }.items()})
        self.previous_position.copy_(pos)
        self.previous_sole.copy_(sole)
        self.previous_contact.copy_(contact)
        self.previous_angular.copy_(angular)
        self.previous_qd.copy_(qd)
        self.history_valid[:] = True
        return reward

    def write_diagnostic_summary(self):
        """Persist completed-episode counts; this does not execute a rollout."""
        if not self.metrics_file:
            return
        path = self.metrics_file.with_suffix(".gate-totals.json")
        best_effort_json(path, {
            "schema": "drobot.ankle-v24-diagonal.gate-diagnostics.v2",
            "step": self.common_step_counter,
            "evidence": "evaluation_diagnostic" if self.cfg.evaluation else "training_only",
            "blocker_counting": "each blocker observed at least once during a timed-out wait; reasons can overlap",
            "episode_scope": "completed episodes only; current unfinished episodes excluded",
            **self.diagnostic_totals,
        })

    def _record_completed_diagnostics(self, env_ids):
        env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        if not len(env_ids):
            return
        totals = self.diagnostic_totals
        totals["completed_episodes"] += len(env_ids)
        totals["completed_moving_episodes"] += int((self.cruise[env_ids] > 0.).sum().item())
        stages = self.gate_failure_stage[env_ids]
        for stage, name in enumerate(self.stage_names):
            totals["timeouts_by_stage"][name] += int((stages == stage).sum().item())
            totals["stage_entries"][name] += int(self.stage_entries[env_ids, stage].sum().item())
        for pair, name in enumerate(totals["timeouts_by_pair"]):
            totals["timeouts_by_pair"][name] += int(((stages >= 0) & (self.pair[env_ids] == pair)).sum().item())
        for index, name in enumerate(self.gate_names):
            totals["timeout_blockers"][name] += int(self.gate_failure_blocked[env_ids, index].sum().item())
        for index, name in enumerate(LEGS):
            totals["qualified_touchdowns_by_leg"][name] += int(self.qualified_touchdowns[env_ids, index].sum().item())
        if not self.metrics_file or self.common_step_counter - self.gate_log_step < 640:
            return
        failures = env_ids[self.gate_failed[env_ids]][:2].tolist()
        with self.metrics_file.with_suffix(".gate-events.jsonl").open("a", encoding="utf-8") as stream:
            for env_id in failures:
                stage = int(self.gate_failure_stage[env_id].item())
                row = {
                    "step": self.common_step_counter, "env_id": env_id,
                    "stage": self.stage_names[stage], "pair": int(self.pair[env_id].item()),
                    "action_order": self.action_order,
                    "blockers_seen_during_wait": [name for index, name in enumerate(self.gate_names)
                        if self.gate_failure_blocked[env_id, index].item()],
                    "gate_value_order_and_units": ["tracking_rad", "delivery_rad", "forward_m",
                        "height_m", "joint_speed_rad_s", "tilt_sine", "angular_speed_rad_s"],
                    "gate_sampling": "gate values precede physics; pose/contact snapshot follows that step",
                    "snapshot": {key: value[env_id].cpu().tolist()
                                 for key, value in self.last_diagnostics.items()},
                }
                stream.write(json.dumps(row) + "\n")
        self.write_diagnostic_summary()
        self.gate_log_step = self.common_step_counter

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        log = {}
        if (hasattr(self, "diagonal_sums") and len(env_ids)
                and (self.episode_length_buf[env_ids] > 0).any()):
            self._record_completed_diagnostics(env_ids)
            ds = self.diagonal_sums
            count = ds["samples"][env_ids].clamp(min=1.)
            stance = ds["stance_samples"][env_ids].clamp(min=1.)
            log = {
                "Metrics/train_paired_landings": ds["paired_landings"][env_ids].mean().item(),
                "Metrics/train_failed_pairs": ds["failed_pairs"][env_ids].mean().item(),
                "Metrics/train_min_steps_per_leg": self.step_counts[env_ids].amin(1).mean().item(),
                "Metrics/train_gate_timeout_fraction": self.gate_failed[env_ids].float().mean().item(),
                "Metrics/train_planted_stance_fraction": (ds["flat_samples"][env_ids] / stance).mean().item(),
                "Metrics/train_stance_slip_m_s": (ds["slip"][env_ids] / stance).mean().item(),
                "Metrics/train_supported_swing_clear": (ds["clear_samples"][env_ids] / ds["swing_samples"][env_ids].clamp(min=1)).mean().item(),
                "Metrics/train_angular_rms_rad_s": (ds["angular_square"][env_ids] / count).sqrt().mean().item(),
                "Metrics/train_angular_accel_rms_rad_s2": (ds["angular_accel_square"][env_ids] / count).sqrt().mean().item(),
                "Metrics/train_target_accel_rms_rad_s2": (ds["target_accel_square"][env_ids] / count).sqrt().mean().item(),
                "Metrics/train_support_margin_m": (ds["support_margin"][env_ids] / count).mean().item(),
                "Metrics/train_supported_fraction": (ds["support_good"][env_ids] / count).mean().item(),
                "Metrics/train_gate_wait_s": ds["gate_wait_s"][env_ids].mean().item(),
                "Metrics/train_lift_stage_clear": (ds["lift_clear_samples"][env_ids] / ds["lift_samples"][env_ids].clamp(min=1)).mean().item(),
                "Metrics/train_target_accel_limited": (ds["accel_limited"][env_ids] / ds["control_samples"][env_ids].clamp(min=1)).mean().item(),
                "Metrics/train_reference_clipped": (ds["reference_clipped"][env_ids] / ds["control_samples"][env_ids].clamp(min=1)).mean().item(),
                "Metrics/train_output_hold_fraction": (ds["output_holds"][env_ids] / ds["control_samples"][env_ids].clamp(min=1)).mean().item(),
                "Metrics/train_swing_projection_fraction": (ds["swing_projection"][env_ids] / ds["control_samples"][env_ids].clamp(min=1)).mean().item(),
            }
            moving_episodes = self.cruise[env_ids] > 0.
            log["Metrics/train_gate_timeout_moving_episodes"] = (
                (self.gate_failed[env_ids] & moving_episodes).sum() / moving_episodes.sum().clamp(min=1)).item()
            for stage, name in enumerate(self.stage_names):
                log[f"Metrics/train_gate_timeout_{name}"] = (self.gate_failure_stage[env_ids] == stage).float().mean().item()
                log[f"Metrics/train_entries_{name}"] = self.stage_entries[env_ids, stage].mean().item()
                log[f"Metrics/train_wait_{name}_s"] = self.stage_wait_s[env_ids, stage].mean().item()
            for index, name in enumerate(self.gate_names):
                log[f"Metrics/train_timeout_blocker_{name}"] = self.gate_failure_blocked[env_ids, index].float().mean().item()
            for index, leg in enumerate(LEGS):
                log[f"Metrics/train_{leg}_steps"] = self.step_counts[env_ids, index].mean().item()
                log[f"Metrics/train_{leg}_stride_m"] = (self.stride_sums[env_ids, index] / self.step_counts[env_ids, index].clamp(min=1.)).mean().item()
                log[f"Metrics/train_{leg}_qualified_touchdowns"] = self.qualified_touchdowns[env_ids, index].mean().item()
                log[f"Metrics/train_{leg}_lift_clear"] = (self.lift_clear_by_leg[env_ids, index] / self.lift_samples_by_leg[env_ids, index].clamp(min=1.)).mean().item()
            if self.metrics_file and self.common_step_counter - self.diagonal_log_step >= 640:
                with self.metrics_file.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"step": self.common_step_counter, "evidence": "training_only", **log}) + "\n")
                self.diagonal_log_step = self.common_step_counter
        super()._reset_idx(env_ids)
        if not hasattr(self, "stage"):
            return
        for tensor in (self.stage, self.stage_time, self.settle_time, self.gate_wait, self.filtered,
                       self.output_velocity, self.last_output_velocity, self.flight_peak,
                       self.liftoff_sole, self.landed_advance, self.plant_time, self.step_counts,
                       self.stride_sums, self.previous_angular, self.previous_qd, self.previous_target_delta,
                       self.qualified_touchdowns, self.lift_best_score, self.gate_values,
                       self.foot_lift_best_score, self.lift_samples_by_leg, self.lift_clear_by_leg,
                       self.stage_entries, self.stage_wait_s):
            tensor[env_ids] = 0
        for tensor in (self.gate_failed, self.parked, self.started, self.skipped_output, self.history_valid,
                       self.flight_valid, self.landed, self.pair_paid, self.touchdown_paid, self.failed_pair_event,
                       self.gate_blocked, self.gate_seen_blockers, self.gate_waiting, self.gate_failure_blocked,
                       self.reference_clipped, self.accel_limited):
            tensor[env_ids] = False
        self.residual_projected[env_ids] = False
        self.gate_failure_stage[env_ids] = -1
        self.flight_supported[env_ids] = True
        self.first_placement[env_ids] = True
        # Random pair starts remove a systematic front-right/rear-left disadvantage.
        self.pair[env_ids] = 0 if self.cfg.evaluation else torch.randint(0, 2, (len(env_ids),), device=self.device)
        cruise = torch.full_like(self.cruise[env_ids], self.cfg.evaluation_speed) if self.cfg.evaluation else self.cruise[env_ids]
        fraction = ((cruise - self.cfg.command_min) / (self.cfg.command_max - self.cfg.command_min)).clamp(0, 1)
        final_stride = self.cfg.stride_min_m + fraction * (self.cfg.stride_max_m - self.cfg.stride_min_m)
        self.stride[env_ids] = final_stride * (.6 + .4 * self._curriculum()) * (cruise > 0.)
        self.period[env_ids] = final_stride / (2 * cruise.clamp(min=self.cfg.command_min))
        self.ref[env_ids] = self.neutral[env_ids]
        self.requested_target[env_ids] = self.neutral[env_ids]
        self.unconstrained_target[env_ids] = self.neutral[env_ids]
        self.ankle_reference[env_ids] = self.neutral[env_ids, 12:16]
        self.expected_stance[env_ids] = True
        self.ref_forward[env_ids] = self.front_sign * self.cfg.stance_forward_m
        self.ref_down[env_ids] = self.cfg.stance_down_m
        self.ref_clearance[env_ids] = 0.
        for values in self.diagonal_sums.values():
            values[env_ids] = 0.
        self.extras.setdefault("log", {}).update(log)
