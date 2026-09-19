"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsPostureDamping", entry_point=__name__ + ".env:RealStairsPostureDampingEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsPostureDampingCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsPostureDampingPPOCfg"})
