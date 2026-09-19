"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsDetour", entry_point=__name__ + ".env:RealStairsDetourEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsDetourCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsDetourPPOCfg"})
