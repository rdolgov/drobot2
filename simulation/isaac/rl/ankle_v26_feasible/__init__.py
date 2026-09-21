import gymnasium as gym
from . import env_cfg, ppo_cfg

gym.register(id='Drobot-Ankle-V26-Feasible',
    entry_point=f'{__name__}.env:IKDiagonalEnv', disable_env_checker=True,
    kwargs={'env_cfg_entry_point': env_cfg.IKDiagonalCfg,
            'rsl_rl_cfg_entry_point': ppo_cfg.IKPPOCfg})
