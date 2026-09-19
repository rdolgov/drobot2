"""Offline-only V44 study35 mapping. No Torch, simulator, ROS or bus dependency.

Reference -> front/rear causal feedback -> common-F5 projection -> 5mm lift
-> soft limits -> elapsed-time/packet cap. Physical actuator response is NOT
applied here: the simulated response filter models the plant, not software.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from ._v44_geometry import GEOMETRY
from ._v44_projection import build_domain, mm, rz, acos
from .contract import ACTION_NAMES, GaitClockConfig, JointTargetConfig, RunStopConfig

SCHEMA = "drobot.v44-study35-target-mapping.v1"
PROFILE = "v44-study35-offline"
CHECKPOINT_SHA256 = "2dcfc9327852635618d505ae8f0c13ea3eaa1a0cb7457e38cf000bf9134b537f"
HIGH_SHA256 = "86030cb4de7248453d9f8d8e42be739c581215ee0074febfd9ceba5b94594864"
LOW_SHA256 = "757efe413813930ba9db87f6af7735a932bbb3532b33adca9dacbcff022415eb"
LOW_SPEED = .01909090909090909
HIGH_SPEED = .02227272727272727


def c2(value):
    u = float(np.clip(value, 0., 1.))
    return float(np.clip(u**3 * (10. - 15.*u + 6.*u*u), 0., 1.))


def vector(value, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (12,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain 12 finite values")
    return value


def validate_metadata(payload):
    marker = payload.get("experimental_target_mapping")
    if marker is None:
        if payload.get("training_profile") == PROFILE:
            raise ValueError("Study35 requires its target-mapping contract")
        return False
    if not isinstance(marker, Mapping) or dict(marker) != {
        "schema": SCHEMA, "offline_only": True, "production_deployment_allowed": False,
        "front_stance_gain": .25, "rear_swing_gain": .25,
        "rear_projection": "common-F5", "rear_lift_m": .005,
        "plant_response_filter_on_pi": False,
    }:
        raise ValueError("Unsupported or altered experimental target mapping")
    if payload.get("training_profile") != PROFILE or payload.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Study35 policy identity mismatch")
    if payload.get("action_history_mode") != "applied_target" or payload.get("observation_scales") != {"command_forward_lateral_yaw": [10., 10., 10.]}:
        raise ValueError("Study35 observation semantics mismatch")
    if payload.get("heading_hold") != {"enabled":True,"mode":"relative_yaw_feedback","kp_s":.75,"max_correction_rad_s":.12}:
        raise ValueError("Study35 requires its evaluated heading feedback")
    cfg = JointTargetConfig.from_metadata(payload)
    expected_neutral = np.asarray([0.]*4 + [.5239596366882324,-.5239596366882324]*2 + [-.5239596366882324,.5239596366882324]*2, dtype=np.float32)
    if (cfg.action_mode != "gait_residual" or not np.array_equal(cfg.neutral_array, expected_neutral)
        or not np.array_equal(cfg.action_scale_array, np.array([.12]*4+[.3]*4+[.4]*4, dtype=np.float32))
        or not np.array_equal(cfg.residual_action_scale_array, np.array([.06]*4+[.5]*8, dtype=np.float32))
        or not np.array_equal(cfg.target_velocity_limit_array, np.array([math.radians(60.)]*4+[math.radians(90.)]*8, dtype=np.float32))
        or cfg.max_target_step_rad != math.radians(2.) or cfg.reference_start_ramp_s != 1.5
        or cfg.gait_residual_stop_action_scale != 0.
        or cfg.reference_blend_speed_min_m_s != LOW_SPEED or cfg.reference_blend_speed_max_m_s != HIGH_SPEED
        or cfg.reference_array.shape != (512,12) or cfg.reference_low_array.shape != (512,12)):
        raise ValueError("Study35 reference/action/target dynamics mismatch")
    if payload.get("gait_clock") != {
        "mode":"speed_scaled", "period_s":1/.35, "standstill_deadband_m_s":.004,
        "speed_min_m_s":LOW_SPEED, "speed_max_m_s":HIGH_SPEED,
        "frequency_min_hz":.3, "frequency_max_hz":.35, "stride_scale_min":1.,
    }:
        raise ValueError("Study35 gait clock mismatch")
    if payload.get("run_stop_contract") != {
        "recommended_stop_ramp_s":.9, "recommended_stopped_hold_s":3., "ramp_shape":"cubic_smoothstep",
        "phase_alignment":{"enabled":True,"ramp_start_phase_cycles":.05,"phase_tolerance_cycles":.03,"maximum_wait_s":.5},
    }:
        raise ValueError("Study35 stop schedule mismatch")
    return True


class Study35Mapper:
    """Stateless target transform; caller owns acknowledged history and clock."""
    def __init__(self, metadata):
        if not validate_metadata(metadata):
            raise ValueError("Missing study35 mapping")
        self.config = JointTargetConfig.from_metadata(metadata)
        self.clock = GaitClockConfig.from_metadata(metadata)
        self.stop = RunStopConfig.from_metadata(metadata)
        self.neutral = self.config.neutral_array
        self.scale = self.config.action_scale_array
        self.residual = self.config.residual_action_scale_array
        self.high = self.config.reference_array
        self.low = self.config.reference_low_array
        self.limits = np.asarray(GEOMETRY['limits'], dtype=np.float32)
        self.rates = self.config.target_velocity_limit_array
        phase = np.arange(512, dtype=np.float32)/np.float32(512)
        self.contacts = (phase[:,None] + np.array([0.,.5,.5,0.], dtype=np.float32)) % np.float32(1.) < np.float32(.55)

    def stop_start_age_s(self, duration_s, forward_m_s):
        """Same nominal-clock, nearest-control-tick alignment used at reset."""
        steps = max(1, round(duration_s*60.))
        frequency = np.float32(self.clock.frequency_hz(forward_m_s))
        if frequency <= 0.:
            return steps/60.
        phase = float(np.remainder(frequency*np.float32(steps)*np.float32(1./60.),np.float32(1.)))
        forward = float(np.remainder(np.float32(.05)-np.float32(phase),np.float32(1.)))
        wait = 0. if min(forward,1.-forward)<=.03 else min(forward/float(frequency),.5)
        return (steps+int(np.rint(np.float32(wait)/np.float32(1./60.))))/60.

    def map(self, action, *, phase_cycles, age_s, command, previous_delivered,
            measured_q, elapsed_s, stop_elapsed_s=None):
        action = np.clip(vector(action,"action"), -1., 1.)
        previous = vector(previous_delivered,"previous delivered")
        measured = vector(measured_q,"measured joints")
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("command must contain three finite values")
        if command[1] != 0. or command[2] != 0. or not 0. <= command[0] <= np.float32(HIGH_SPEED):
            raise ValueError("Study35 supports only forward/stop nominal commands up to its selected high command")
        if (not all(math.isfinite(x) for x in (phase_cycles, age_s, elapsed_s))
            or age_s < 0. or elapsed_s < 0. or not 0. <= phase_cycles < 1.
            or stop_elapsed_s is not None and (not math.isfinite(stop_elapsed_s) or stop_elapsed_s < 0.)):
            raise ValueError("Invalid mapping time/phase")
        phase = np.float32(phase_cycles)
        index = int(np.floor(phase*np.float32(512))) % 512
        speed = np.float32(np.linalg.norm(command[:2]))
        blend = np.clip((speed-np.float32(LOW_SPEED))/np.float32(HIGH_SPEED-LOW_SPEED),np.float32(0.),np.float32(1.))
        desired = self.low[index] + blend*(self.high[index]-self.low[index])
        ramp = np.clip(np.float32(age_s)/np.float32(1.5),np.float32(0.),np.float32(1.))
        stride = np.float32(speed > np.float32(.004))
        reference = ((desired-self.neutral)/self.scale) * (ramp*stride)
        center = self.neutral + self.scale*reference
        moving = speed > np.float32(.004) or abs(float(command[2])) > 1.e-4
        authority = np.float32(1. if moving else 0.)
        raw = self.neutral + self.scale*(reference+self.residual*action*authority)
        scheduled = self.contacts[index].copy()
        if stride == 0. or ramp < .5:
            scheduled[:] = True
        half = self.scale*self.residual*authority
        lower = np.maximum(center-half,self.limits[:,0])
        upper = np.minimum(center+half,self.limits[:,1])
        startup = c2((age_s-.75)/.75)
        command_gate = c2((float(np.linalg.norm(command[:2].astype(np.float64)))-.004)/(LOW_SPEED-.004))
        error = np.clip(np.float32(.25)*(previous-measured), -np.float32(math.radians(2.)), np.float32(math.radians(2.)))
        front = raw.copy()
        front_weights = []
        for leg, hip, knee, offset in ((0,4,8,0.),(2,6,10,.5)):
            u = ((float(phase)+offset)%1.)/.55
            fade = 1. if stop_elapsed_s is None else c2(1.-stop_elapsed_s/.2)
            weight = np.float32(c2(u/.15)*c2((1.-u)/.15)*startup*command_gate*fade if scheduled[leg] else 0.)
            d = np.float32(.5)*(error[knee]-error[hip])*weight
            room = min(raw[hip]-lower[hip], upper[knee]-raw[knee]) if d >= 0 else min(upper[hip]-raw[hip], raw[knee]-lower[knee])
            d = np.float32(np.sign(d)*min(abs(d),max(np.float32(0.),room)))
            front[hip] += -d
            front[knee] += d
            front_weights.append(float(weight))
        if np.any(front[[4,6,8,10]] < lower[[4,6,8,10]]-2.e-7) or np.any(front[[4,6,8,10]] > upper[[4,6,8,10]]+2.e-7):
            raise ValueError("Front feedback exceeds original authority")
        lifts = []
        rear = front.copy()
        for leg,hip,knee,offset in ((1,5,9,.5),(3,7,11,0.)):
            u = np.clip((((float(phase)+offset)%1.)-.55)/.45,0.,1.)
            lift = .005*c2(u/.5)*c2((1.-u)/.5)*startup*command_gate if not scheduled[leg] else 0.
            lifts.append(lift)
            weight = np.float32(lift/.005 if stop_elapsed_s is None else 0.)
            d = np.float32(.5)*(weight*error[knee]-weight*error[hip])
            rear[hip] += -d
            rear[knee] += d
        projected, mapped = rear.copy(), rear.copy()
        domains = []
        for i,(leg,abd,hip,knee) in enumerate((('rear_left',1,5,9),('rear_right',3,7,11))):
            if lifts[i] <= 1.e-12:
                domains.append(None)
                continue
            frame = GEOMETRY['frames'][leg]
            domain = build_domain(frame,float(rear[abd]),float(center[hip]),float(center[knee]),
                self.limits[hip].astype(float),self.limits[knee].astype(float),lifts[i])
            pair, identity = domain.project(float(rear[hip]),float(rear[knee]))
            # The Torch projector checks even identity points at maximum lift.
            domain.assert_geometry(*pair)
            projected[hip],projected[knee] = pair
            ar,hr,sign,length = frame
            pre = mm(mm(ar,rz(float(rear[abd]))),hr)
            A,B = length*pre[2][0],sign*length*pre[2][1]
            radius,phi = math.hypot(A,B),math.atan2(B,A)
            h0,k0 = pair
            offset = math.atan2(math.sin(h0-phi),math.cos(h0-phi))
            branch = 1. if offset > 0. else -1.
            base_height = A*math.cos(h0)+B*math.sin(h0)
            for stage in range(1,6):
                # Multiply by an exactly-one final fraction; (L*5)/5 can
                # round above L and correctly trip the strict domain guard.
                added = lifts[i]*(stage/5.)
                cosine = (base_height+added)/radius
                if abs(cosine) > 1.:
                    raise ValueError("Rear lift is unreachable")
                principal = phi+branch*acos(cosine)
                h = h0+math.atan2(math.sin(principal-h0), math.cos(principal-h0))
                k = h0+k0-h
                domain.assert_geometry(h0,k0,added)
            mapped[hip],mapped[knee] = np.float32(h),np.float32(k)
            actual_lower = np.maximum(self.limits[[hip,knee],0].astype(float), center[[hip,knee]].astype(float)-(self.scale*self.residual)[[hip,knee]].astype(float))
            actual_upper = np.minimum(self.limits[[hip,knee],1].astype(float), center[[hip,knee]].astype(float)+(self.scale*self.residual)[[hip,knee]].astype(float))
            if np.any(mapped[[hip,knee]] < actual_lower-2.e-6) or np.any(mapped[[hip,knee]] > actual_upper+2.e-6) or abs(float(mapped[hip])+float(mapped[knee])-h0-k0)>2.e-6:
                raise ValueError("Rear lift cast exceeds original bounds")
            domains.append({'identity':identity,'sum_interval':domain.S})
        requested = np.clip(mapped,self.limits[:,0],self.limits[:,1]).astype(np.float32)
        delta = np.minimum(self.rates*np.float32(elapsed_s), np.float32(self.config.max_target_step_rad))
        delivered = np.clip(requested,previous-delta,previous+delta).astype(np.float32)
        return {'raw':raw,'reference_center':center,'front_corrected':front,'rear_corrected':rear,
                'projected':projected,'requested':requested,'delivered':delivered,
                'scheduled_contact':scheduled,'common_lift_m':lifts,'front_weights':front_weights,'domains':domains}
