"""A separate flat-ground training contract; not the deployed V23 runtime."""
from isaaclab.utils.configclass import configclass
from ankle_v23_transfer.env_cfg import TransferCfg, ROOT


@configclass
class DiagonalCfg(TransferCfg):
    contract_revision = 4
    observation_space = 82
    state_space = 101
    episode_length_s = 48.0
    command_min = .012
    command_max = .018
    evaluation_speed = .015
    evaluation_stop_s = 40.0
    curriculum_steps = 96000
    stride_min_m = .080
    stride_max_m = .100
    lift_m = .022
    # Retain the broader stance: narrower upright reference diagnostics fell.
    # Cap radial reach explicitly if a future reference approaches extension.
    stance_forward_m = .080
    stance_down_m = .280
    maximum_leg_reach_m = .317
    body_forward_shift_m = .014
    target_base_height_m = .368908389
    filter_time_s = .16
    residual_scales = (.10, .18, .22, .14)
    swing_forward_window_m = .008
    swing_extra_lift_m = .010
    swing_lift_cancellation_m = .002
    target_accel_deg_s2 = (160., 240., 260., 200.)
    # Transfer, lift, advance, lower, settle, all-foot push.
    stage_fractions = (.10, .18, .24, .18, .10, .20)
    startup_s = 2.5
    settle_hold_s = .15
    gate_timeout_s = 1.5
    gate_position_deg = 5.0
    gate_delivery_deg = 2.0
    gate_forward_m = .015
    gate_height_m = .008
    gate_settle_height_m = .010
    gate_velocity_deg_s = 18.0
    gate_angular_rad_s = .40
    gate_tilt_rad = .16
    # Position targets remain within the same URDF and calibrated ankle bounds.
    ankle_reference_filter_s = .06
    output_skip_probability = .08
    sole_contact_margin_m = .004
    hardware_calibration = str(ROOT / "hardware/robot-runtime/servos/calibration-ankles.json")
    hardware_travel = str(ROOT / "hardware/robot-runtime/servos/ankle-travel-observations.json")
