"""Pure-Python controller checks that do not require Isaac Sim."""

from __future__ import annotations

import math

from simulation.isaac import _quadruped_runtime as runtime


def test_stance_commands_are_mirrored_in_world_with_common_urdf_signs():
    stance = runtime.stance_by_name()

    for joint_kind in runtime.JOINT_KINDS:
        assert math.isclose(
            stance[f"front_left_{joint_kind}"],
            stance[f"front_right_{joint_kind}"],
            abs_tol=1e-12,
        )
        assert math.isclose(
            stance[f"rear_left_{joint_kind}"],
            stance[f"rear_right_{joint_kind}"],
            abs_tol=1e-12,
        )

    assert stance["front_left_hip_flexion"] < 0.0
    assert stance["front_left_knee"] > 0.0
    assert stance["rear_left_hip_flexion"] > 0.0
    assert stance["rear_left_knee"] < 0.0


def test_crawl_targets_stay_inside_urdf_limits_over_full_cycle():
    limits = {
        "hip_abduction": math.radians(25.0),
        "hip_flexion": math.radians(60.0),
        "knee": math.radians(90.0),
    }
    minimum_margin = math.inf

    for phase_index in range(401):
        pose = runtime.crawl_by_name(
            phase_index / 400.0 * 2.8,
            period_s=2.8,
            stride_m=0.025,
            lift_m=0.012,
        )
        assert set(pose) == runtime.EXPECTED_DOF_NAMES
        for name, value in pose.items():
            joint_kind = next(
                kind for kind in runtime.JOINT_KINDS if name.endswith(kind)
            )
            margin = limits[joint_kind] - abs(value)
            minimum_margin = min(minimum_margin, margin)
            assert margin > 0.0, (name, value)

    assert minimum_margin >= math.radians(19.0)


def test_servo_torque_profiles_match_verified_values():
    assert runtime.torque_cap_nm("rated", None) == (
        "rated",
        runtime.RATED_TORQUE_NM,
    )
    assert runtime.torque_cap_nm("stall", None) == (
        "stall",
        runtime.STALL_TORQUE_NM,
    )
    assert runtime.torque_cap_nm("rated", 1.25) == ("custom", 1.25)


def test_quasistatic_crawl_has_one_swing_foot_and_safe_joint_targets():
    limits = {
        "hip_abduction": math.radians(25.0),
        "hip_flexion": math.radians(60.0),
        "knee": math.radians(90.0),
    }
    observed_swing_legs = set()
    observed_all_feet_advance = False

    for phase_index in range(801):
        pose, state = runtime.quasistatic_crawl_by_name(
            phase_index / 800.0 * 10.0,
            period_s=10.0,
            stride_m=0.025,
            lift_m=0.012,
            weight_shift_forward_m=0.018,
            weight_shift_lateral_m=0.018,
        )
        assert set(pose) == runtime.EXPECTED_DOF_NAMES
        for name, value in pose.items():
            joint_kind = next(
                kind for kind in runtime.JOINT_KINDS if name.endswith(kind)
            )
            assert abs(value) < limits[joint_kind], (name, value, state)

        if state["phase"] in {"lift", "swing", "lower"}:
            assert state["swing_leg"] is not None
            assert state["swing_leg"] not in state["expected_support_legs"]
            assert len(state["expected_support_legs"]) == 3
            observed_swing_legs.add(state["swing_leg"])
        else:
            assert len(state["expected_support_legs"]) == 4
        observed_all_feet_advance |= state["phase"] == "all_feet_advance"

    assert observed_swing_legs == set(runtime.LEGS)
    assert observed_all_feet_advance


def test_quasistatic_weight_transfer_uses_opposite_support_corner():
    step_midpoint = runtime.QUASISTATIC_STEP_FRACTION * 0.10
    _, state = runtime.quasistatic_crawl_by_name(
        step_midpoint * 10.0,
        period_s=10.0,
        stride_m=0.025,
        lift_m=0.012,
        weight_shift_forward_m=0.018,
        weight_shift_lateral_m=0.018,
    )

    assert state["swing_leg"] == "rear_right"
    assert state["body_shift_forward_m"] > 0.0
    assert state["body_shift_lateral_m"] > 0.0


def test_tuned_quasistatic_commands_are_periodic_and_below_servo_speed():
    period_s = 20.0
    sample_hz = 120
    ordered_names = sorted(runtime.EXPECTED_DOF_NAMES)
    samples = []

    for sample_index in range(int(period_s * sample_hz) + 1):
        pose, _ = runtime.quasistatic_crawl_by_name(
            sample_index / sample_hz,
            period_s=period_s,
            stride_m=0.015,
            lift_m=0.010,
            weight_shift_forward_m=0.030,
            weight_shift_lateral_m=0.0,
            down_m=0.310,
            fore_aft_m=0.025,
            abduction_deg=0.0,
        )
        samples.append([pose[name] for name in ordered_names])

    assert all(
        math.isclose(start, end, abs_tol=1e-12)
        for start, end in zip(samples[0], samples[-1], strict=True)
    )
    maximum_command_speed = max(
        abs(
            (samples[index][joint] - samples[index - 1][joint])
            * sample_hz
        )
        for index in range(1, len(samples))
        for joint in range(len(ordered_names))
    )
    assert maximum_command_speed < runtime.MAX_NO_LOAD_VELOCITY_RAD_S
