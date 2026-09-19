"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsBoundedIK", entry_point=__name__ + ".env:RealStairsBoundedIKEnv",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsBoundedIKCfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsBoundedIKPPOCfg"})
