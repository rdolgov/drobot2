from isaaclab.utils.configclass import configclass
from ankle_v23_footplan.env_cfg import FootplanCfg


@configclass
class RealStairsMeanPolicyCfg(FootplanCfg):
    phase_count = 1
    phase_s = 36.
    episode_length_s = 40.
    frequency_hz = 1/36.
    observation_space = 485  # append desired joint-reference error
    state_space = 528
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

        self.decimation = 4
        self.sim.dt = 1./240.
        self.sim.render_interval = 4
        self.robot.spawn.articulation_props.solver_position_iteration_count = 24
        self.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
