"""Printable ankle V2.1: independent flat-stance residual PPO task."""
import gymnasium as gym
gym.register(id='Drobot-Ankle-V21-Planted', entry_point=__name__+'.env:PlantedEnv',
             disable_env_checker=True,
             kwargs={'env_cfg_entry_point':__name__+'.env_cfg:PlantedCfg',
                     'rsl_rl_cfg_entry_point':__name__+'.ppo_cfg:PlantedPPOCfg'})
