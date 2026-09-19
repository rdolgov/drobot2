"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-Supported", entry_point=__name__ + ".env:SupportedEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:SupportedCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:SupportedPPOCfg"})
