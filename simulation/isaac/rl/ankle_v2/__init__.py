"""Separate ankle-study tasks; no imports or mutations of V43/V44 tasks."""
import gymnasium as gym

for variant in ("Active", "Baseline", "Locked"):
    gym.register(
        id=f"Drobot-Ankle-V2-{variant}",
        entry_point=f"{__name__}.env:AnkleEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.env_cfg:{variant}EnvCfg",
                "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:{variant}PPOCfg"},
    )
