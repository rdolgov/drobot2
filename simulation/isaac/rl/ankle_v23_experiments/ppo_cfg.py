from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
from ankle_v23_stairs.ppo_cfg import StairsPPOCfg


@configclass
class StrategyPPOCfg(StairsPPOCfg):
    max_iterations = 256
    save_interval = 128
    experiment_name = "drobot_ankle_v23_strategy_screen"
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=.5, use_clipped_value_loss=True, clip_param=.2,
        entropy_coef=.005, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3e-4, schedule="adaptive", gamma=.999, lam=.97,
        desired_kl=.015, max_grad_norm=.5)
