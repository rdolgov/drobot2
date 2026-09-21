"""Flat-ground, completed diagonal-pair stepping with active ankles."""
import gymnasium as gym

gym.register(
    id="Drobot-Ankle-V24-Diagonal",
    entry_point=__name__ + ".env:DiagonalEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:DiagonalCfg",
            "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:DiagonalPPOCfg"},
)
