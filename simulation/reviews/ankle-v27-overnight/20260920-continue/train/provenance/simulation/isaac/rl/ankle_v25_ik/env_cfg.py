from isaaclab.utils.configclass import configclass
from ankle_v24_diagonal.env_cfg import DiagonalCfg


@configclass
class IKDiagonalCfg(DiagonalCfg):
    contract_revision = 1
    episode_length_s = 32.
    evaluation_stop_s = 24.
    command_min = .020
    command_max = .025
    evaluation_speed = .025
    curriculum_steps = 12800
    stance_forward_m = .080
    stance_down_m = .288
    target_base_height_m = .376908389
    body_forward_shift_m = 0.
    lift_m = .035
    stride_min_m = .080
    stride_max_m = .100
    # V5/V6 transfer/lift/swing/lower/plant, then push+settle combined.
    stage_fractions = (.18, .18, .22, .18, .10, .14)
    target_rates_deg_s = (60., 200., 200., 200.)
    residual_scales = (.08, .14, .18, .10)
    filter_time_s = .10
    target_accel_deg_s2 = (160., 360., 420., 360.)
    estimated_total_mass_kg = 3.39514659
    support_push_m = .020
    support_onset_fraction = .30
    evaluation_pair = 0

    def __post_init__(self):
        super().__post_init__()
        self.nominal_mass_kg = self.estimated_total_mass_kg
