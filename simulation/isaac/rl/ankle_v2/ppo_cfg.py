from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class BoundedDistributionCfg(RslRlMLPModelCfg.DistributionCfg):
    class_name = "BetaDistribution"
    action_range = (-1.0, 1.0)


@configclass
class ActivePPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 500
    save_interval = 50
    experiment_name = "drobot_ankle_v2_active_pilot"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(hidden_dims=[256, 256], activation="elu",
                           obs_normalization=False, distribution_cfg=BoundedDistributionCfg())
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5, use_clipped_value_loss=True, clip_param=0.2,
        entropy_coef=0.003, num_learning_epochs=5, num_mini_batches=4,
        learning_rate=3e-4, schedule="adaptive", gamma=0.995, lam=0.95,
        desired_kl=0.015, max_grad_norm=0.5)


@configclass
class BaselinePPOCfg(ActivePPOCfg):
    experiment_name = "drobot_ankle_v2_baseline_pilot"


@configclass
class LockedPPOCfg(ActivePPOCfg):
    experiment_name = "drobot_ankle_v2_locked_pilot"
