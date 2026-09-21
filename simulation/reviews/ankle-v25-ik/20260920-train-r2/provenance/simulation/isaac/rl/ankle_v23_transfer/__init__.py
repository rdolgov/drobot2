"""Longer strides on the V2.3 R3 retained-leg, solid-sole robot."""
import gymnasium as gym
gym.register(id='Drobot-Ankle-V23-Transfer',entry_point=__name__+'.env:TransferEnv',
    disable_env_checker=True,kwargs={'env_cfg_entry_point':__name__+'.env_cfg:TransferCfg',
                                   'rsl_rl_cfg_entry_point':__name__+'.ppo_cfg:TransferPPOCfg'})
