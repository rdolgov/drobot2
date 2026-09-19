"""One-foot curriculum with corrected reset geometry and optional IK guidance."""
import gymnasium as gym

gym.register(id="Drobot-Ankle-V23-RealStairsClearance10", entry_point=__name__ + ".env:RealStairsClearance10Env",
    disable_env_checker=True, kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:RealStairsClearance10Cfg",
    "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:RealStairsClearance10PPOCfg"})
