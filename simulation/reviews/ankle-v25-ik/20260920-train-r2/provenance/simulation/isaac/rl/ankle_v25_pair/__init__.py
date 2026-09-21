"""IK-guided flat-ground diagonal residual learning; no hardware deployment."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V25-Pair", entry_point=__name__+".env:IKDiagonalEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": __name__+".env_cfg:IKDiagonalCfg",
                     "rsl_rl_cfg_entry_point": __name__+".ppo_cfg:IKPPOCfg"})
