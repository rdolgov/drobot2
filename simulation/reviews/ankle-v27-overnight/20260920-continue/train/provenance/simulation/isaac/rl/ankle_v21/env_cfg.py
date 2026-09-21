import math
from isaaclab.utils.configclass import configclass
from ankle_v2.env_cfg import ActiveEnvCfg, ROOT

@configclass
class PlantedCfg(ActiveEnvCfg):
    # Retain 16-actuator mapping in the shared mechanics implementation.
    variant = 'active'
    observation_space = 82
    state_space = 90
    episode_length_s = 24.0
    command_min = .025
    command_max = .065
    frequency_hz = .60
    duty_factor = .78
    stance_forward_m = .070
    stance_down_m = .285
    lift_m = .018
    filter_time_s = .12
    residual_scales = (.10, .20, .24, .22)
    action_scales_rad = (.18, .70, .90, .60)
    curriculum_steps = 96000  # 1500 iterations of 64 control steps.
    evaluation = False
    evaluation_speed = .04
    evaluation_stop_s = 30.0
    ankle_mode = 'learned'  # recording ablation: locked_target or reference_only
    nominal_mass_kg = 0.0   # populated from the asset manifest

    def __post_init__(self):
        super().__post_init__()
        import json
        manifest = ROOT / 'simulation/exports/isaac/ankle_v21/assets.json'
        if manifest.exists():
            self.nominal_mass_kg = json.loads(manifest.read_text())['nominal_mass_kg']
        self.robot.spawn.usd_path = str(ROOT / 'simulation/exports/isaac/ankle_v21/active/robot.usdc')
        L = .159896689
        knee = -math.acos((self.stance_forward_m**2+self.stance_down_m**2-2*L*L)/(2*L*L))
        hip = math.atan2(self.stance_forward_m,self.stance_down_m)-math.atan2(math.sin(knee),1+math.cos(knee))
        self.robot.init_state.pos = (0,0,.344)
        self.robot.init_state.joint_pos = {'.*_hip_abduction':0.,
            'front_.*_hip_flexion':hip,'rear_.*_hip_flexion':-hip,
            'front_.*_knee':knee,'rear_.*_knee':-knee,
            'front_.*_ankle_pitch':-hip-knee,'rear_.*_ankle_pitch':hip+knee}
