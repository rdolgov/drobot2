from isaaclab.utils.configclass import configclass
from ankle_v23_footplan.env_cfg import FootplanCfg


@configclass
class RealStairsStumbleCfg(FootplanCfg):
    phase_count = 1
    phase_s = 12.
    episode_length_s = 320.
    shaping_gamma = .999687548824
    support_weight = 0.
    frequency_hz = 1/12.
    observation_space = 504  # append desired joint-reference error
    state_space = 583
    residual_scales = (.06, .10, .12, .10)
    ground_start_fraction = .5
    privileged_actor = False
    stair_rise_m = .18
    stair_tread_m = .25
    stair_count = 4
    servo_damping = .30

    def __post_init__(self):
        super().__post_init__()
        self.robot.actuators['servos'].damping = self.servo_damping

        self.decimation = 12
        self.sim.dt = 1./240.
        self.sim.render_interval = 12
        self.robot.spawn.articulation_props.solver_position_iteration_count = 24
        self.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
