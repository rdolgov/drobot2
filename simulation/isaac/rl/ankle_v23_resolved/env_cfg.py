from isaaclab.utils.configclass import configclass
from ankle_v23_footplan.env_cfg import FootplanCfg


@configclass
class ResolvedCfg(FootplanCfg):
    phase_count = 8
    phase_s = 12.
    episode_length_s = 100.
    frequency_hz = 1/12.
    observation_space = 485  # append desired joint-reference error
    state_space = 524
    residual_scales = (.06, .10, .12, .10)
    ground_start_fraction = .30
    privileged_actor = False
    servo_damping = .30

    def __post_init__(self):
        super().__post_init__()
        self.robot.actuators['servos'].damping = self.servo_damping

        # Finer integration and tighter constraint convergence, same mechanics.
        self.decimation = 4
        self.sim.dt = 1. / 240.
        self.sim.render_interval = 4
        self.robot.spawn.articulation_props.solver_position_iteration_count = 24
        self.robot.spawn.articulation_props.solver_velocity_iteration_count = 1
