"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsOblique", entry_point=__name__ + ".env:RealStairsObliqueEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsObliqueCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsObliquePPOCfg"})
