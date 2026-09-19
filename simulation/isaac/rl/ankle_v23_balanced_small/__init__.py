"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-BalancedSmall", entry_point=__name__ + ".env:BalancedSmallEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:BalancedSmallCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:BalancedSmallPPOCfg"})
