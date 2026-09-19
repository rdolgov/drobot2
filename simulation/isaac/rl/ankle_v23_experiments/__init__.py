"""Matched PPO screens of stair approach and foot-placement strategies."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-Strategy", entry_point=__name__ + ".env:StrategyEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:StrategyCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:StrategyPPOCfg"})
