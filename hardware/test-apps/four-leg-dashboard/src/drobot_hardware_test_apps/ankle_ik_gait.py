"""Flat-ground 16-joint IK references shared by the dashboard and simulation.

Angles are calibrated logical degrees, not servo ticks. The foot trajectory
targets the ankle pivot: the V2.3 sole is a separate 43.2117 mm offset. Encoder
gates confirm joint tracking only; they cannot prove ground contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping

LEGS = ("front_left", "front_right", "rear_left", "rear_right")
JOINTS = ("hip_abduction", "hip_flexion", "knee", "ankle_pitch")
LINK_M = .159896689
SOLE_OFFSET_M = .0432117
MAX_REACH_M = .317
ANKLE_LIMIT_DEG = 33.25
ORDER = (("rear_right",), ("front_right",), ("rear_left",), ("front_left",))
PAIRS = (("front_left", "rear_right"), ("front_right", "rear_left"))
# Each placement completes before another leg/pair is selected.
PHASES = (("weight_transfer", .18), ("lift", .18), ("swing", .22),
          ("lower", .18), ("firm_plant", .10), ("all_feet_push", .10),
          ("step_settle", .04))
GATED_PHASES = {"lift", "swing", "firm_plant", "step_settle"}
SUPPORT_ONSET_LIFT_PROGRESS = .10


@dataclass(frozen=True)
class AnkleIKConfig:
    period_s: float = 16.
    stride_m: float = .060
    lift_m: float = .025
    down_m: float = .305
    fore_aft_m: float = .060
    forward_transfer_m: float = .010
    lateral_transfer_m: float = .030
    max_cycles: int = 0
    support_push_m: float = 0.
    support_lift_fraction: float = .25
    support_onset_fraction: float = 0.
    ramp_rate_deg_s: float = 100.
    max_command_step_deg: float = 2.

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("IK configuration must be finite")
        if not (4. <= self.period_s <= 40. and .01 <= self.stride_m <= .10
                and .008 <= self.lift_m <= .040 and .27 <= self.down_m <= .305
                and .06 <= self.fore_aft_m <= .09 and 0. <= self.forward_transfer_m <= .03
                and 0. <= self.lateral_transfer_m <= .04
                and 0. <= self.support_push_m <= min(.025, self.stride_m*.5)
                and 0. <= self.support_lift_fraction <= 1.
                and 0. <= self.support_onset_fraction <= .35
                and 1. <= self.ramp_rate_deg_s <= 200.
                and .1 <= self.max_command_step_deg <= 4.
                and type(self.max_cycles) is int and 0 <= self.max_cycles <= 10):
            raise ValueError("IK gait parameters exceed the bounded commissioning envelope")


def smooth(value: float) -> float:
    """Quintic profile with zero endpoint velocity and acceleration."""
    t = min(1., max(0., value))
    return t*t*t*(10. + t*(-15. + 6.*t))


def lift_push_progress(value: float) -> float:
    """Continuous lift/push with immediate velocity and a stopped endpoint.

    Unlike the quintic profile, the first positive frame requests meaningful
    support-ankle movement. Positions never jump; the shared motor rate and
    packet limits still bound every command. Used only for propulsive lift.
    """
    t = min(1., max(0., value))
    return t+t*t-t*t*t


def support_push_progress(value: float, onset_fraction: float) -> float:
    """Lead the lift with a bounded early push, then complete it together."""
    t = min(1., max(0., value))
    if onset_fraction <= 0.:
        return lift_push_progress(t)
    if t <= SUPPORT_ONSET_LIFT_PROGRESS:
        return onset_fraction*t/SUPPORT_ONSET_LIFT_PROGRESS
    remainder = (t-SUPPORT_ONSET_LIFT_PROGRESS)/(1.-SUPPORT_ONSET_LIFT_PROGRESS)
    return onset_fraction+(1.-onset_fraction)*lift_push_progress(remainder)


def ankle_leg_ik(corner: str, forward_m: float, down_m: float) -> tuple[float, float, float]:
    if corner not in LEGS or not all(math.isfinite(v) for v in (forward_m, down_m)):
        raise ValueError("Invalid IK corner or coordinate")
    distance = math.hypot(forward_m, down_m)
    if down_m <= 0. or not .05 < distance <= MAX_REACH_M + 1e-10:
        raise ValueError(f"Unreachable ankle-pivot target for {corner}: {forward_m}, {down_m}")
    cosine = (distance*distance - 2.*LINK_M*LINK_M) / (2.*LINK_M*LINK_M)
    knee = (-1. if corner.startswith("front_") else 1.) * math.acos(cosine)
    hip = math.atan2(forward_m, down_m) - knee*.5
    ankle = -hip-knee
    values = tuple(math.degrees(v) for v in (hip, knee, ankle))
    if abs(values[2]) > ANKLE_LIMIT_DEG + 1e-8:
        raise ValueError(f"Flat-sole IK exceeds the retained ankle limit for {corner}")
    return values


def shifted_ankle_leg_ik(corner: str, forward_m: float, down_m: float,
                         body_lateral_m: float) -> tuple[float, float, float, float]:
    """Exact roll-plane geometry from the retained URDF hip offsets.

Abduction axes are +X on the left and -X on the right. The hip-flexion pivot
is 28.6117 mm outward and 95.696689 mm below the abduction pivot. Ankle pitch
cannot cancel foot roll; this small roll is reported rather than called flat.
"""
    side = 1. if corner.endswith("_left") else -1.
    local_y = side*.0286117
    desired_y = local_y-body_lateral_m
    desired_down = .095696689+down_m+SOLE_OFFSET_M
    local_down = math.sqrt(desired_y*desired_y+desired_down*desired_down-local_y*local_y)
    theta = math.atan2(desired_y, desired_down)-math.atan2(local_y, local_down)
    hip, knee, ankle = ankle_leg_ik(corner, forward_m, local_down-.095696689-SOLE_OFFSET_M)
    return math.degrees(theta/side), hip, knee, ankle


def _order(mode):
    if mode == "distributed":
        return ORDER
    if mode == "diagonal_pair":
        return PAIRS
    raise ValueError("Unknown ankle IK gait mode")


def ankle_ik_gait_degrees(time_s: float, config: AnkleIKConfig = AnkleIKConfig(),
                          mode: str = "distributed"):
    if not math.isfinite(time_s) or time_s < 0.:
        raise ValueError("IK time must be finite and nonnegative")
    order = _order(mode)
    placement_s = config.period_s / len(order)
    index = int(time_s / placement_s) % len(order)
    u = (time_s / placement_s) % 1.
    start = 0.
    phase, progress = PHASES[-1][0], 1.
    for name, fraction in PHASES:
        if u < start + fraction:
            phase, progress = name, (u-start)/fraction
            break
        start += fraction
    moving = order[index]
    increment = config.stride_m / len(order)
    offsets = {leg: config.stride_m*.5 - (len(order)-i)*increment
               for i, group in enumerate(order) for leg in group}
    for completed in range(index):
        for leg in order[completed]:
            offsets[leg] += config.stride_m
        offsets = {leg: offset-increment for leg, offset in offsets.items()}
    # Advance the body over its planted diagonal while the other pair moves.
    # Carry this advance through landing; do not
    # undo it when support swaps or add it again to the four-foot push.
    support_push = config.support_push_m if mode == "diagonal_pair" else 0.
    lift_fraction = lift_push_progress(progress) if support_push > 0. else smooth(progress)
    lift_push = support_push*config.support_lift_fraction
    if phase == "lift":
        early_push = lift_push*support_push_progress(progress, config.support_onset_fraction)
    elif phase == "swing":
        early_push = support_push*(config.support_lift_fraction
                                  +(1.-config.support_lift_fraction)*smooth(progress))
    elif phase in ("lower", "firm_plant", "all_feet_push", "step_settle"):
        early_push = support_push
    else:
        early_push = 0.
    push = (support_push+(increment-support_push)*smooth(progress)
            if phase == "all_feet_push" else increment
            if phase == "step_settle" else early_push)
    # During lift, carry the raised pair forward with the body. Holding their
    # world X fixed would pull the front raised ankle beyond its retained
    # flat-sole limit with the full early push. The subsequent swing starts
    # from this carried position and still ends at the original world stride.
    if phase == "lift":
        advance = early_push
    elif phase == "swing":
        advance = lift_push+(config.stride_m-lift_push)*smooth(progress)
    elif phase in ("lower", "firm_plant", "all_feet_push", "step_settle"):
        advance = config.stride_m
    else:
        advance = 0.
    for leg in moving:
        offsets[leg] += advance
    offsets = {leg: x-push for leg, x in offsets.items()}
    lift = (config.lift_m*lift_fraction if phase == "lift" else config.lift_m
            if phase == "swing" else config.lift_m*(1.-smooth(progress)) if phase == "lower" else 0.)
    # Forward transfer unloads rear swing feet. Front swing keeps the rear
    # payload inside its three-foot triangle. Diagonal mode uses a forward bias.
    transfer = config.forward_transfer_m if mode == "diagonal_pair" or moving[0].startswith("rear_") else 0.
    if phase == "weight_transfer":
        transfer_blend = smooth(progress)
    elif phase == "all_feet_push":
        transfer_blend = 1.-smooth(progress)
    elif phase == "step_settle":
        transfer_blend = 0.
    else:
        transfer_blend = 1.
    transfer *= transfer_blend
    lateral = (-(1. if moving[0].endswith("_left") else -1.)
               * config.lateral_transfer_m * transfer_blend if mode == "distributed" else 0.)
    forwards = {leg: (config.fore_aft_m if leg.startswith("front_") else -config.fore_aft_m)
                + offsets[leg] - transfer for leg in LEGS}
    # Lower the entire body reference when necessary; never clip one foot away
    # from the common support plane to hide an unreachable target.
    # Solve the roll transform's reach bound exactly for every leg.
    offset_down = .095696689+SOLE_OFFSET_M
    down_bounds = []
    for leg, x in forwards.items():
        local_y = (.0286117 if leg.endswith("_left") else -.0286117)
        planar_max = math.sqrt(MAX_REACH_M**2-x*x)
        # A raised diagonal foot has a shorter vertical reach than a stance
        # foot. Include its lift before deriving the common body-height bound
        # so the stronger synchronized push does not add a needless body dip.
        raised = lift if mode == "diagonal_pair" and leg in moving else 0.
        down_bounds.append(math.sqrt((offset_down+planar_max)**2
                           + local_y*local_y-(local_y-lateral)**2)-offset_down+raised)
    down = min(config.down_m, *down_bounds)
    targets = {}
    for leg in LEGS:
        abduction, hip, knee, ankle = shifted_ankle_leg_ik(leg, forwards[leg], down-(lift if leg in moving else 0.), lateral)
        targets.update({(leg, "hip_abduction"): abduction, (leg, "hip_flexion"): hip,
                        (leg, "knee"): knee, (leg, "ankle_pitch"): ankle})
    airborne = phase in ("lift", "swing", "lower")
    # The planned support ankles must participate in propulsion instead of
    # having their feedforward cancelled by temporary hip/knee tracking lag.
    # Select the upcoming support pair during weight transfer, before lift.
    # Otherwise measured-angle correction acquired in transfer decays during
    # lift and can initially cancel or reverse the small commanded ankle push.
    # Keep this reference through each encoder settle, including handover.
    ankle_feedforward = (
        list(LEGS) if phase in ("all_feet_push", "step_settle") else
        [leg for leg in LEGS if leg not in moving]
    ) if support_push > 0. else []
    return targets, {
        "phase": phase, "phase_progress": progress, "placement_index": index,
        "swing_corner": moving[0] if len(moving) == 1 else None,
        "swing_pair": list(moving) if len(moving) == 2 else None,
        "expected_support_corners": [leg for leg in LEGS if not airborne or leg not in moving],
        "ankle_pivot_forward_m": forwards, "ankle_pivot_down_m": down,
        "requested_lift_m": lift, "body_forward_transfer_m": transfer,
        "body_lateral_transfer_m": lateral,
        "support_push_m": early_push,
        "body_forward_advance_m": push,
        "ankle_feedforward_corners": ankle_feedforward,
        "sole_roll_target_deg": {leg: targets[(leg, "hip_abduction")]
            * (1. if leg.endswith("_left") else -1.) for leg in LEGS},
        "sole_pitch_target_deg": {leg: 0. for leg in LEGS},
        "encoder_gate_is_not_contact_detection": True,
    }


def validate_leveling_gravity(gravity_body):
    gx, gy, gz = (float(v) for v in gravity_body)
    if not all(math.isfinite(v) for v in (gx, gy, gz)):
        raise ValueError("Nonfinite IK gravity")
    norm = math.sqrt(gx*gx+gy*gy+gz*gz)
    if not .8 <= norm <= 1.2 or -gz/norm < math.cos(math.radians(20.)):
        raise RuntimeError("IK leveling requires valid gravity and body tilt below 20 degrees")
    return gx, gy, gz


def level_sole_pitch_degrees(targets, measured, gravity_body, correction, dt,
                             *, feedforward_legs=()):
    """Bounded ankle-only leveling using body gravity and actual leg angles.

    Swing ankles use measured hip/knee angles. Propulsive support ankles use
    planned hip/knee angles plus measured body gravity, so leg tracking lag
    does not cancel their coordinated IK push. Their actual sole pitch can
    differ until the leg tracks; the encoder gates still apply to all joints.
    The pitch ankle cannot independently correct lateral foot roll.
    No support-force/contact claim follows from these proprioceptive inputs.
    """
    if not 0. < dt <= .2 or set(measured) != set(targets):
        raise ValueError("Sole leveling requires complete, fresh joint feedback")
    feedforward_legs = frozenset(feedforward_legs)
    if not feedforward_legs.issubset(LEGS):
        raise ValueError("Unknown support ankle corner")
    gx, gy, gz = validate_leveling_gravity(gravity_body)
    if not all(math.isfinite(v) for v in measured.values()):
        raise ValueError("Nonfinite sole-leveling feedback")
    result, next_correction, limited = dict(targets), {}, []
    for leg in LEGS:
        side = 1. if leg.endswith("_left") else -1.
        alpha = math.radians(side*measured[(leg, "hip_abduction")])
        plane_pitch = math.degrees(math.atan2(gx, gy*math.sin(alpha)-gz*math.cos(alpha)))
        key = (leg, "ankle_pitch")
        reference = targets if leg in feedforward_legs else measured
        level_angle = plane_pitch-reference[(leg, "hip_flexion")]-reference[(leg, "knee")]
        requested = level_angle-targets[key]
        bounded = min(8., max(-8., requested))
        old = correction.get(leg, 0.)
        change = (bounded-old)*dt/(.10+dt)
        adjusted = old+min(30.*dt, max(-30.*dt, change))
        angle = min(ANKLE_LIMIT_DEG, max(-ANKLE_LIMIT_DEG, targets[key]+adjusted))
        result[key] = angle
        next_correction[leg] = angle-targets[key]
        if abs(requested-bounded) > .01 or abs(angle-targets[key]-adjusted) > .01:
            limited.append(leg)
    return result, next_correction, limited


class AnkleIKClock:
    """No catch-up: require support onset, then settle motion endpoints."""
    def __init__(self, config: AnkleIKConfig, mode: str):
        self.config, self.mode = config, mode
        _order(mode)
        self.time_s = 0.
        self.gate: str | None = None
        self.wait_s = self.ready_s = 0.
        self.stop_requested = self.stopped = False
        self.last_tracking_error_deg = 0.
        self.support_ankle_start_deg: dict[str, float] = {}
        self.support_ankle_direction: dict[str, float] = {}
        self.support_ankle_required_deg: dict[str, float] = {}
        self.support_ankle_progress_deg: dict[str, float] = {}
        self.support_onset_confirmed = False

    def tick(self, dt: float, measured: Mapping, velocity: Mapping, delivered: Mapping,
             tracking_targets: Mapping | None = None):
        if not math.isfinite(dt) or not 0. < dt <= .05:
            raise ValueError("IK clock accepts only bounded control ticks")
        nominal_target, state = ankle_ik_gait_degrees(self.time_s, self.config, self.mode)
        target = nominal_target
        if tracking_targets is not None:
            if set(tracking_targets) != set(target) or not all(math.isfinite(v) for v in tracking_targets.values()):
                raise ValueError("IK tracking targets must contain sixteen finite values")
            for key in target:
                if key[1] != "ankle_pitch" and abs(tracking_targets[key]-target[key]) > 1e-7:
                    raise ValueError("Leveling may change only ankle tracking targets")
                if key[1] == "ankle_pitch" and abs(tracking_targets[key]) > ANKLE_LIMIT_DEG+1e-8:
                    raise ValueError("Leveling target exceeds the retained ankle limit")
            target = dict(tracking_targets)
        if set(measured) != set(target) or set(velocity) != set(target) or set(delivered) != set(target):
            raise ValueError("IK clock requires all sixteen feedback and delivered targets")
        if not all(math.isfinite(v) for source in (measured, velocity, delivered) for v in source.values()):
            raise ValueError("Nonfinite IK feedback")
        if self.stopped:
            return target
        if self.stop_requested and not self.gate and state["phase"] in (
                "weight_transfer", "all_feet_push", "step_settle"):
            self.stopped = True
            return target
        onset_enabled = (self.mode == "diagonal_pair" and self.config.support_push_m > 0.
                         and self.config.support_lift_fraction > 0.
                         and self.config.support_onset_fraction > 0.)
        if onset_enabled and state["phase"] == "weight_transfer":
            # Actual angles just before lift, not previously sent commands.
            # Each placement gets a fresh baseline for its two planted ankles.
            self.support_ankle_start_deg = {
                leg: measured[(leg, "ankle_pitch")]
                for leg in LEGS if leg not in state["swing_pair"]}
            self.support_ankle_progress_deg = {}
            self.support_ankle_required_deg = {}
            self.support_onset_confirmed = False
        if self.gate:
            self.wait_s += dt
            if self.gate == "support_ankle_onset":
                self.support_ankle_progress_deg = {
                    leg: (measured[(leg, "ankle_pitch")]-start)*self.support_ankle_direction[leg]
                    for leg, start in self.support_ankle_start_deg.items()}
                ready = (all(self.support_ankle_progress_deg[leg] >= required
                             for leg, required in self.support_ankle_required_deg.items())
                         and max(abs(target[k]-delivered[k]) for k in target) <= .3)
                # This confirms onset, not endpoint settling. Requiring zero
                # speed here would intentionally separate push from lift.
                if not ready:
                    if self.wait_s >= 2.:
                        raise RuntimeError("IK planted ankles did not begin the forward push; stop motion")
                    return target
                self.support_onset_confirmed = True
                self.gate = None
                self.wait_s = self.ready_s = 0.
            else:
                error = max(abs(target[k]-measured[k]) for k in target)
                self.last_tracking_error_deg = error
                ready = (error <= 5. and max(abs(target[k]-delivered[k]) for k in target) <= .3
                         and max(abs(v) for v in velocity.values()) <= 18.)
                self.ready_s = self.ready_s+dt if ready else 0.
                if self.ready_s < .15:
                    if self.wait_s >= 2.:
                        raise RuntimeError(f"IK {self.gate} tracking timeout ({error:.1f} deg); stop motion")
                    return target
                if self.stop_requested and self.gate == "firm_plant":
                    self.stopped = True
                    return target
                self.gate = None
                self.wait_s = self.ready_s = 0.
        duration = self.config.period_s / len(_order(self.mode))
        placement = math.floor((self.time_s+1e-9) / duration)
        boundary = 0.
        for phase, fraction in PHASES:
            phase_start = boundary
            boundary += fraction
            if onset_enabled and phase == "lift":
                onset = (placement+phase_start+fraction*SUPPORT_ONSET_LIFT_PROGRESS)*duration
                if self.time_s+1e-9 < onset <= self.time_s+dt+1e-9:
                    if len(self.support_ankle_start_deg) != 2:
                        raise RuntimeError("IK support ankle onset requires a pre-lift encoder baseline")
                    # Use the nominal pre-lift pose to determine direction;
                    # the IMU correction must not reverse this definition.
                    start_pose, _ = ankle_ik_gait_degrees(
                        (placement+phase_start)*duration, self.config, self.mode)
                    onset_pose, _ = ankle_ik_gait_degrees(onset, self.config, self.mode)
                    for leg in self.support_ankle_start_deg:
                        key = (leg, "ankle_pitch")
                        change = onset_pose[key]-start_pose[key]
                        if abs(change) < .2:
                            raise RuntimeError("IK support ankle onset target is too small to observe")
                        self.support_ankle_direction[leg] = math.copysign(1., change)
                        self.support_ankle_required_deg[leg] = min(.5, abs(change)*.5)
                    self.time_s = onset
                    self.gate = "support_ankle_onset"
                    self.wait_s = self.ready_s = 0.
                    return onset_pose
            endpoint = (placement + boundary)*duration
            if phase in GATED_PHASES and self.time_s+1e-9 < endpoint <= self.time_s+dt+1e-9:
                self.time_s = endpoint
                self.gate = phase
                return ankle_ik_gait_degrees(self.time_s, self.config, self.mode)[0]
        self.time_s += dt
        if self.config.max_cycles and self.time_s >= self.config.period_s*self.config.max_cycles:
            self.time_s = self.config.period_s*self.config.max_cycles
            self.stopped = True
        return ankle_ik_gait_degrees(self.time_s, self.config, self.mode)[0]
