import json
import math
from isaaclab.utils.configclass import configclass
from ankle_v2.env_cfg import ActiveEnvCfg, ROOT

@configclass
class TransferCfg(ActiveEnvCfg):
    variant = 'active'
    observation_space = 82
    state_space = 101
    episode_length_s = 40.0
    command_min = .015
    command_max = .021
    duty_factor = .82
    stance_forward_m = .070
    stance_down_m = .275
    lift_m = .018
    filter_time_s = .10
    residual_scales = (.14,.24,.30,.20)
    action_scales_rad = (.18,.70,.90,.60)
    curriculum_steps = 96000
    evaluation = False
    evaluation_speed = .018
    evaluation_stop_s = 40.0
    ankle_mode = 'learned'
    nominal_mass_kg = 0.0
    contact_x_m = .0432117
    # XML FK: base-to-tread height = stance_down_m + 0.088908389 m.
    # Bend the legs 10 mm more to retain reach during the body shift.
    target_base_height_m = .363908389
    transfer_start_phase = .75
    support_margin_m = .008

    def __post_init__(self):
        super().__post_init__()
        manifest=ROOT/'simulation/exports/isaac/ankle_v23_stride/assets.json'
        if manifest.exists(): self.nominal_mass_kg=json.loads(manifest.read_text())['nominal_mass_kg']
        self.robot.spawn.usd_path=str(ROOT/'simulation/exports/isaac/ankle_v23_stride/active/robot.usdc')
        L=.159896689
        knee=-math.acos((self.stance_forward_m**2+self.stance_down_m**2-2*L*L)/(2*L*L))
        hip=math.atan2(self.stance_forward_m,self.stance_down_m)-math.atan2(math.sin(knee),1+math.cos(knee))
        self.robot.init_state.pos=(0.,0.,self.target_base_height_m+.014)
        self.robot.init_state.joint_pos={'.*_hip_abduction':0.,
            'front_.*_hip_flexion':hip,'rear_.*_hip_flexion':-hip,
            'front_.*_knee':knee,'rear_.*_knee':-knee,
            'front_.*_ankle_pitch':-hip-knee,'rear_.*_ankle_pitch':hip+knee}
