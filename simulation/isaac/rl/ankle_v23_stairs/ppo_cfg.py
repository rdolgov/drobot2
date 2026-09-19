from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlRNNModelCfg, RslRlMLPModelCfg
from ankle_v2.ppo_cfg import ActivePPOCfg, BoundedDistributionCfg


@configclass
class StairsPPOCfg(ActivePPOCfg):
    max_iterations = 1000
    save_interval = 100
    experiment_name = "drobot_ankle_v23_stairs_d405"
    # Memory retains the front-camera evidence as the rear feet approach it.
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128], activation="elu", obs_normalization=True,
        rnn_type="gru", rnn_hidden_dim=128, rnn_num_layers=1,
        distribution_cfg=BoundedDistributionCfg())
    critic = RslRlMLPModelCfg(hidden_dims=[256, 256, 128], activation="elu",
                            obs_normalization=True)
