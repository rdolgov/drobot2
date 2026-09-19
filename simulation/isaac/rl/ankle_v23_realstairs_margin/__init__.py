"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsMargin", entry_point=__name__ + ".env:RealStairsMarginEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsMarginCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsMarginPPOCfg"})
