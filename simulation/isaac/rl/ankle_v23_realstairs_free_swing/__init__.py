"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsFreeSwing", entry_point=__name__ + ".env:RealStairsFreeSwingEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsFreeSwingCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsFreeSwingPPOCfg"})
