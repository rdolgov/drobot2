"""V2.3 solid-sole stair climbing with a D405-like depth observation."""
import gymnasium as gym

gym.register(
    id="Drobot-Ankle-V23-Stairs-D405",
    entry_point=__name__ + ".env:StairsEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": __name__ + ".env_cfg:StairsCfg",
            "rsl_rl_cfg_entry_point": __name__ + ".ppo_cfg:StairsPPOCfg"},
)
