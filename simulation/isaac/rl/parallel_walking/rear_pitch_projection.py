"""Strictly opt-in, common-F5 rear sagittal target projection.

Vectorized port of the reviewed study-only analytic inner-domain construction.
This module makes no actuator, simulation, observation, policy or RNG calls.
The caller supplies the same common positive lift envelope in both arms, even
when the subsequent physical lift treatment amplitude is zero. It must supply
the hash-bound canonical geometry already initialized by the environment.

A successful fixed context is not a continuous-context nonemptiness proof.
Empty domains and numerical failures raise through the caller before physics;
they never choose a smaller lift, wider joint box or fallback target.
"""
from __future__ import annotations

import math

import torch

PROJECTION_SCHEMA = "drobot.v44-rear-pitch-common-F5-projection.v1"
OFFLINE_PROJECTOR_SOURCE_SHA256 = "859eb2355e30d485572a001d6b2afae868a1483c9cef84ad7f79506d309ec19e"
MARGIN = 4.e-6
BRANCH_EPS = 1.e-7
INACTIVE_LIFT_M = 1.e-12
MAX_X_M = .0175
TILT_RAD = math.radians(12.)


@torch.no_grad()
def project_rear_pitch_targets(
    env,
    requested: torch.Tensor,
    reference_center: torch.Tensor,
    limits: torch.Tensor,
    common_lift_m: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """Project only active rear hip/knee entries, preserving feasible bits.

    Inputs are N x12 float32 targets/centers, N x12 x2 float32 soft limits,
    and N x2 float64 common lift in rear-left/rear-right order. The scalar
    residual half-widths .15/.20 rad match the frozen sufficient inner domain,
    conservatively inside the existing nominal float32 authority products.
    Inactive diagnostic intervals are zero with active_by_leg=False.
    """
    evidence: dict = {"projection_schema": PROJECTION_SCHEMA}

    def fail(reason: str, details: dict | None = None) -> None:
        env._rear_pitch_mapping_fault(
            "feasible_projection:" + reason, {**evidence, **(details or {})}
        )
        raise RuntimeError("Rear pitch mapping fault callback unexpectedly returned")

    def require(condition, reason: str, details: dict | None = None) -> None:
        if not bool(condition):
            fail(reason, details)

    require(isinstance(requested, torch.Tensor) and requested.ndim == 2
            and requested.shape[1] == 12 and requested.dtype == torch.float32,
            "requested_shape_or_dtype")
    n, device = requested.shape[0], requested.device
    require(isinstance(reference_center, torch.Tensor)
            and reference_center.shape == requested.shape and reference_center.dtype == torch.float32
            and reference_center.device == device, "reference_shape_dtype_or_device")
    require(isinstance(limits, torch.Tensor) and limits.shape == (n, 12, 2)
            and limits.dtype == torch.float32 and limits.device == device,
            "limits_shape_dtype_or_device")
    require(isinstance(common_lift_m, torch.Tensor) and common_lift_m.shape == (n, 2)
            and common_lift_m.dtype == torch.float64 and common_lift_m.device == device,
            "common_lift_shape_dtype_or_device")
    require(torch.all(torch.isfinite(requested)) and torch.all(torch.isfinite(reference_center))
            and torch.all(torch.isfinite(limits)) and torch.all(torch.isfinite(common_lift_m)),
            "nonfinite_input")
    require(torch.all((common_lift_m >= 0.) & (common_lift_m <= .005)),
            "common_lift_outside_selected_range")
    require(torch.all(limits[:, :, 0] < limits[:, :, 1]), "reversed_soft_joint_limits")
    active = common_lift_m > INACTIVE_LIFT_M
    changed = torch.zeros((n, 2), dtype=torch.bool, device=device)
    sum_intervals = torch.zeros((n, 2, 2), dtype=torch.float64, device=device)
    hip_intervals = torch.zeros_like(sum_intervals)

    def diagnostics(projected: torch.Tensor) -> dict:
        return {
            "schema": PROJECTION_SCHEMA,
            "offline_projector_source_sha256": OFFLINE_PROJECTOR_SOURCE_SHA256,
            "physics_counter_before_step": int(env._sim_step_counter),
            "common_lift_m": common_lift_m.detach().clone(),
            "raw_requested_target_rad": requested.detach().clone(),
            "projected_requested_target_rad": projected.detach().clone(),
            "active_by_leg": active.detach().clone(),
            "changed_by_leg": changed.detach().clone(),
            "delta_rad": (projected - requested).detach().clone(),
            "sum_interval_rad": sum_intervals.detach().clone(),
            "conditional_hip_interval_rad": hip_intervals.detach().clone(),
            "inactive_interval_semantics": "Zero entries with active_by_leg=False; not an empty active domain.",
            "full_context_continuum_nonemptiness_certified": False,
        }

    env_ids, leg_ids = torch.where(active)
    if env_ids.numel() == 0:
        return requested, diagnostics(requested)

    ai, hi, ki = env._rear_pitch_joint_ids
    for ids, expected in zip((ai, hi, ki), ((1, 3), (5, 7), (9, 11))):
        require(isinstance(ids, torch.Tensor) and ids.shape == (2,)
                and ids.dtype == torch.int64 and ids.device == device,
                "rear_joint_index_contract")
        require(torch.equal(ids, torch.tensor(expected, dtype=torch.int64, device=device)),
                "rear_joint_order_contract")
    for frame, shape in (
        (env._rear_pitch_abd_rotation, (2, 3, 3)),
        (env._rear_pitch_hip_rotation, (2, 3, 3)),
        (env._rear_pitch_sign, (2,)),
        (env._rear_pitch_length, (2,)),
    ):
        require(isinstance(frame, torch.Tensor) and frame.shape == shape
                and frame.dtype == torch.float64 and frame.device == device
                and torch.all(torch.isfinite(frame)), "canonical_frame_tensor_contract")
    require(torch.equal(env._rear_pitch_sign, torch.tensor((-1., 1.), dtype=torch.float64, device=device))
            and torch.all(env._rear_pitch_length == .159896689), "canonical_axis_or_length_contract")

    hip_ids, knee_ids = hi[leg_ids], ki[leg_ids]
    h0 = requested[env_ids, hip_ids].to(torch.float64)
    k0 = requested[env_ids, knee_ids].to(torch.float64)
    abd = requested[env_ids, ai[leg_ids]].to(torch.float64)
    hc = reference_center[env_ids, hip_ids].to(torch.float64)
    kc = reference_center[env_ids, knee_ids].to(torch.float64)
    lift = common_lift_m[env_ids, leg_ids]
    pre = (env._rear_pitch_abd_rotation[leg_ids] @ env._rear_pitch_rotation_z(abd)
           @ env._rear_pitch_hip_rotation[leg_ids])
    length, sign = env._rear_pitch_length[leg_ids], env._rear_pitch_sign[leg_ids]
    A, B = length * pre[:, 2, 0], sign * length * pre[:, 2, 1]
    C, D = length * pre[:, 0, 0], sign * length * pre[:, 0, 1]
    radius, phi = torch.hypot(A, B), torch.atan2(B, A)
    offset = torch.atan2(torch.sin(hc - phi), torch.cos(hc - phi))
    evidence.update(env_ids=env_ids, rear_leg_indices=leg_ids,
                    original_hip_knee_rad=torch.stack((h0, k0), dim=1),
                    reference_hip_knee_rad=torch.stack((hc, kc), dim=1),
                    common_lift_m=lift, abduction_rad=abd)
    require(torch.all(radius > 1.e-8) and torch.all(torch.abs(torch.sin(offset)) > BRANCH_EPS),
            "reference_monotone_branch_unavailable")
    branch = torch.where(offset > 0., torch.ones_like(offset), -torch.ones_like(offset))
    origin = hc - offset
    hlo = torch.maximum(limits[env_ids, hip_ids, 0].to(torch.float64), hc - .15) + MARGIN
    hhi = torch.minimum(limits[env_ids, hip_ids, 1].to(torch.float64), hc + .15) - MARGIN
    klo = torch.maximum(limits[env_ids, knee_ids, 0].to(torch.float64), kc - .20) + MARGIN
    khi = torch.minimum(limits[env_ids, knee_ids, 1].to(torch.float64), kc + .20) - MARGIN
    klo = torch.where(kc > 0., torch.clamp(klo, min=BRANCH_EPS), klo)
    khi = torch.where(kc > 0., khi, torch.clamp(khi, max=-BRANCH_EPS))
    delta = MAX_X_M / length - 4. * MARGIN
    evidence.update(hip_interval_before_lift_rad=torch.stack((hlo, hhi), dim=1),
                    knee_interval_rad=torch.stack((klo, khi), dim=1), branch=branch)
    require(torch.all(hlo < hhi) and torch.all(khi - klo >= delta + 8. * MARGIN),
            "original_hip_knee_box_width")

    def checked_acos(value: torch.Tensor, label: str) -> torch.Tensor:
        require(torch.all(torch.isfinite(value)) and torch.all(torch.abs(value) <= 1. + 8.e-15),
                label, {"acos_argument": value})
        # Only the same transcendental roundoff allowance as the scalar source.
        return torch.acos(torch.clamp(value, -1., 1.))

    t0, t1 = branch * (hlo - origin), branch * (hhi - origin)
    tl = torch.clamp(torch.minimum(t0, t1), min=BRANCH_EPS)
    th = torch.clamp(torch.maximum(t0, t1), max=math.pi - BRANCH_EPS)
    z = lift / radius
    require(torch.all(tl < th), "hip_box_misses_selected_branch")
    tl = torch.maximum(tl, checked_acos(torch.cos(tl) - z, "pre_post_hip_reach"))
    angular_restriction = z > 1. - torch.cos(delta)
    if torch.any(angular_restriction):
        ratio = z[angular_restriction] / (2. * torch.sin(delta[angular_restriction] / 2.))
        require(torch.all(torch.isfinite(ratio)) and torch.all((ratio >= 0.) & (ratio <= 1.)),
                "conservative_X_certificate_unavailable", {"sine_ratio": ratio})
        angle = torch.asin(ratio)
        tl[angular_restriction] = torch.maximum(tl[angular_restriction], delta[angular_restriction] / 2. + angle)
        th[angular_restriction] = torch.minimum(th[angular_restriction], delta[angular_restriction] / 2. + math.pi - angle)
    require(torch.all(th - tl > 8. * MARGIN), "pre_post_hip_interval_empty")
    j0, j1 = origin + branch * tl, origin + branch * th
    jlo, jhi = torch.minimum(j0, j1), torch.maximum(j0, j1)
    nC, nD = -pre[:, 2, 0], -sign * pre[:, 2, 1]
    normal_radius = torch.hypot(nC, nD)
    require(torch.all(normal_radius >= math.cos(TILT_RAD - 4. * MARGIN)),
            "abduction_has_no_sole_tilt_interval")
    pitch_center = torch.atan2(nD, nC)
    pitch_center = pitch_center + 2. * math.pi * torch.round((hc + kc - pitch_center) / (2. * math.pi))
    pitch_half = checked_acos(math.cos(TILT_RAD - 4. * MARGIN) / normal_radius, "sole_tilt_interval")

    def mapped_hip(h: torch.Tensor, added: torch.Tensor = lift) -> torch.Tensor:
        t = branch * (h - origin)
        return origin + branch * checked_acos(torch.cos(t) + added / radius, "mapped_hip_reach")

    def inverse_hip(h: torch.Tensor) -> torch.Tensor:
        t = branch * (h - origin)
        return origin + branch * checked_acos(torch.cos(t) - lift / radius, "inverse_hip_reach")

    image_lo, image_hi = mapped_hip(jlo), mapped_hip(jhi)
    slo = torch.maximum(klo + torch.maximum(jlo, image_lo), pitch_center - pitch_half) + 8. * MARGIN
    shi = torch.minimum(khi + torch.minimum(jhi, image_hi), pitch_center + pitch_half) - 8. * MARGIN
    evidence.update(pre_hip_interval_rad=torch.stack((jlo, jhi), dim=1),
                    sum_interval_rad=torch.stack((slo, shi), dim=1))
    require(torch.all(shi - slo > 2. * MARGIN), "sum_interval_empty")

    def hip_interval(total: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.maximum(jlo, total - khi)
        high = torch.minimum(jhi, total - klo)
        ml = torch.maximum(image_lo, total - khi)
        mh = torch.minimum(image_hi, total - klo)
        require(torch.all(ml <= mh) and torch.all(lo <= high), "conditional_hip_knee_empty",
                {"conditional_total_rad": total})
        lo = torch.maximum(lo, inverse_hip(ml))
        high = torch.minimum(high, inverse_hip(mh))
        require(torch.all(high - lo > 6. * MARGIN), "conditional_hip_float32_interior",
                {"conditional_total_rad": total, "raw_conditional_interval_rad": torch.stack((lo, high), dim=1)})
        return lo + MARGIN, high - MARGIN

    # Same fixed-context endpoint checks as the scalar constructor.
    hip_interval(slo)
    hip_interval(shi)

    def contains(h: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total = h + k
        in_sum = (total >= slo) & (total <= shi)
        # The scalar returns False before invoking inverse trig outside S.
        # Clamp only the scratch membership query; do not change h or k.
        safe_total = torch.minimum(torch.maximum(total, slo), shi)
        lo, high = hip_interval(safe_total)
        return in_sum & (h >= lo) & (h <= high), lo, high

    identity, _, _ = contains(h0, k0)
    total = torch.minimum(torch.maximum(h0 + k0, slo + MARGIN), shi - MARGIN)
    lo, high = hip_interval(total)
    chosen = torch.minimum(torch.maximum(h0, lo + MARGIN), high - MARGIN)
    candidate_h = chosen.to(torch.float32).to(torch.float64)
    candidate_k = (total - chosen).to(torch.float32).to(torch.float64)
    h = torch.where(identity, h0, candidate_h)
    k = torch.where(identity, k0, candidate_k)
    contained, final_lo, final_hi = contains(h, k)
    require(torch.all(contained), "float32_projected_pair_left_inner_domain",
            {"projected_hip_knee_rad": torch.stack((h, k), dim=1), "identity_by_active_leg": identity})

    # Maximum-lift endpoint plus the domain's monotone-branch certificate covers
    # every lower amplitude for this fixed context; no continuum-context claim.
    mh = mapped_hip(h)
    mk = h + k - mh
    dx = C * (torch.cos(mh) - torch.cos(h)) + D * (torch.sin(mh) - torch.sin(h))
    normal_z = nC * torch.cos(h + k) + nD * torch.sin(h + k)
    valid = ((h >= hlo - MARGIN) & (h <= hhi + MARGIN) & (mh >= hlo - MARGIN) & (mh <= hhi + MARGIN)
             & (k >= klo - MARGIN) & (k <= khi + MARGIN) & (mk >= klo - MARGIN) & (mk <= khi + MARGIN)
             & (k * mk > 0.) & (normal_z >= math.cos(TILT_RAD) - 1.e-12)
             & (torch.abs(dx) <= MAX_X_M + 1.e-12)
             & torch.isfinite(mh) & torch.isfinite(mk) & torch.isfinite(dx) & torch.isfinite(normal_z))
    require(torch.all(valid), "projected_geometry_outside_original_bounds_tilt_X_or_sign",
            {"projected_hip_knee_rad": torch.stack((h, k), dim=1),
             "maximum_lift_hip_knee_rad": torch.stack((mh, mk), dim=1), "signed_X_change_m": dx,
             "sole_normal_z": normal_z})

    sum_intervals[env_ids, leg_ids] = torch.stack((slo, shi), dim=1)
    hip_intervals[env_ids, leg_ids] = torch.stack((final_lo, final_hi), dim=1)
    selected_changed = ~identity
    changed[env_ids, leg_ids] = selected_changed
    if not torch.any(selected_changed):
        return requested, diagnostics(requested)
    projected = requested.clone()
    changed_envs = env_ids[selected_changed]
    projected[changed_envs, hip_ids[selected_changed]] = h[selected_changed].to(torch.float32)
    projected[changed_envs, knee_ids[selected_changed]] = k[selected_changed].to(torch.float32)
    return projected, diagnostics(projected)
