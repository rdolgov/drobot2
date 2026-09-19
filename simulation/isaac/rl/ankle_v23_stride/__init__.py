"""Longer strides on the V2.3 R3 retained-leg, solid-sole robot."""
import gymnasium as gym
gym.register(id='Drobot-Ankle-V23-Stride',entry_point=__name__+'.env:StrideEnv',
    disable_env_checker=True,kwargs={'env_cfg_entry_point':__name__+'.env_cfg:StrideCfg',
                                   'rsl_rl_cfg_entry_point':__name__+'.ppo_cfg:StridePPOCfg'})
