"""Pure-RL, vectorized Drobot stair task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Drobot-Pure-Stairs-Direct",
    entry_point=f"{__name__}.pure_stairs_env:DrobotPureStairsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pure_stairs_env_cfg:DrobotPureStairsEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:DrobotPureStairsPPORunnerCfg"
        ),
    },
)
