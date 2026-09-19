"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsFootTracking", entry_point=__name__ + ".env:RealStairsFootTrackingEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsFootTrackingCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsFootTrackingPPOCfg"})
